"""Build and summarize the isolated minWM two-native-block sensitivity run.

This run never replaces the primary 13-pixel-frame minWM evaluation.  It uses
29 conditioning pixel frames (eight native latent frames) and the same 480
generated pixel frames, 32 contexts, nine commands, and five six-second
evaluation windows as the primary ICLR fleet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import fleet30s_common as fc

MODEL = "minwm_seed29"
FAMILY = "minwm"
WINDOW_STARTS = (0, 6, 12, 18, 24)
ENDPOINTS = (6, 12, 18, 24, 30)
SEAT_MODELS = ("lingbot", "dreamx", "matrixgame2", "ours_recovery_base")


def probe(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    if decoded != n:
        raise ValueError(f"header/decode mismatch for {path}: {n} versus {decoded}")
    return decoded, fps, width, height


def build_manifest(out: Path) -> None:
    rows = []
    for uid in fc.UIDS:
        seed = Path(fc.SEED_CLIP(uid))
        if not seed.exists():
            raise FileNotFoundError(seed)
        for direction in fc.DIRS:
            scene = f"{uid}_{direction}"
            path = Path(fc.PATH[MODEL](uid, direction))
            if not path.exists():
                raise FileNotFoundError(path)
            sidecar = Path(str(path) + ".json")
            if not sidecar.exists():
                raise FileNotFoundError(sidecar)
            meta = json.loads(sidecar.read_text())
            n, fps, width, height = probe(path)
            expected = {
                "seed_frames": 29,
                "seed_latents": 8,
                "generated_frames": 480,
            }
            for key, value in expected.items():
                if int(meta.get(key, -1)) != value:
                    raise ValueError(f"{path}: {key}={meta.get(key)!r}, expected {value}")
            if n != 509 or abs(fps - 16.0) > 0.01:
                raise ValueError(f"{path}: decoded={n}, fps={fps}, expected 509 at 16 fps")
            rows.append(dict(
                scene=scene, uid=uid, direction=direction, model=MODEL,
                family=FAMILY, path=str(path), real_path=str(seed),
                local_video=True, metadata_source="decoded_and_sidecar_validated",
                container_frames=n, fps=fps, width=width, height=height,
                context_frames=29, generated_frames=480,
                generated_duration_s=30.0, last_generated_timestamp_s=29.9375,
                directional=direction != "N", decoded_frames=n,
                bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                sidecar_path=str(sidecar), checkpoint_identity="MIN-Lab/minWM DMD",
                checkpoint_identity_source="official_huggingface_checkpoint",
                seed_start_frame=0, seed_start_source="seed65_e1 frames 0--28",
                sidecar_generated_frames=int(meta["generated_frames"]),
                available_video=True, storage_site="u6qf"))
    manifest = pd.DataFrame(rows).sort_values(["uid", "direction"])
    if len(manifest) != 288 or manifest.scene.nunique() != 288:
        raise AssertionError(f"manifest is incomplete: {len(manifest)} rows")
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "video_manifest.csv", index=False)
    (out / "MANIFEST_VALIDATION.md").write_text(
        "# minWM 29-frame sensitivity manifest\n\n"
        "Validated 288/288 videos by full decode. Every video contains 29 "
        "conditioning frames and 480 generated frames at 16 fps (509 total).\n")
    print(f"validated {len(manifest)} videos -> {out / 'video_manifest.csv'}")


def flatten_json(directory: Path, metric: str) -> pd.DataFrame:
    rows = []
    files = sorted(directory.glob("*.json"))
    for path in files:
        obj = json.loads(path.read_text())
        if obj.get("metric") != metric:
            raise ValueError(f"wrong metric in {path}")
        rows.extend(obj["rows"])
    data = pd.DataFrame(rows)
    if len(data) != 288 * 6:
        raise AssertionError(f"{metric}: expected 1728 producer rows, got {len(data)}")
    return data


def hf_scores(out: Path, old: Path) -> pd.DataFrame:
    candidate_ep = pd.read_csv(out / "cpu_endpoints.csv")
    candidate_win = pd.read_csv(out / "cpu_windows.csv")
    old_ep = pd.read_csv(old / "cpu_endpoints.csv")
    old_win = pd.read_csv(old / "cpu_windows.csv")
    rows = []
    for endpoint, start in zip(ENDPOINTS, WINDOW_STARTS):
        if endpoint == 6:
            cand = candidate_ep[candidate_ep.horizon_s == 6][["scene", "d_blur"]]
            fixed = old_ep[(old_ep.horizon_s == 6) & old_ep.model.isin(SEAT_MODELS)][
                ["scene", "model", "d_blur"]]
        else:
            cand = candidate_win[candidate_win.window_start_s == start][
                ["scene", "d_blur_from_early_base"]].rename(
                    columns={"d_blur_from_early_base": "d_blur"})
            fixed = old_win[(old_win.window_start_s == start) & old_win.model.isin(SEAT_MODELS)][
                ["scene", "model", "d_blur_from_early_base"]].rename(
                    columns={"d_blur_from_early_base": "d_blur"})
        if len(cand) != 288 or len(fixed) != 4 * 288:
            raise AssertionError((endpoint, len(cand), len(fixed)))
        panels = fixed.groupby("scene").d_blur.apply(list).to_dict()
        for r in cand.itertuples(index=False):
            panel = panels[r.scene] + [float(r.d_blur)]
            median = float(np.median(panel))
            b = median - float(r.d_blur)
            rows.append(dict(scene=r.scene, model=MODEL, endpoint_s=endpoint,
                             candidate_d_blur=float(r.d_blur), panel_median=median,
                             hf_B=b, hf_failure=int(b > 150)))
    data = pd.DataFrame(rows)
    data.to_csv(out / "hf_five_family_sensitivity.csv", index=False)
    return data


def summarize(out: Path, old: Path) -> None:
    manifest = pd.read_csv(out / "video_manifest.csv")
    if len(manifest) != 288:
        raise AssertionError("manifest is not complete")
    style = pd.read_csv(out / "style_windows.csv")
    control = pd.read_csv(out / "control_windows.csv")
    geometry = flatten_json(out / "geometry", "geometry")
    conj = flatten_json(out / "conjuration", "conjuration")
    for name, data in (("style", style), ("control", control),
                       ("geometry", geometry), ("conjuration", conj)):
        data = data[data.window_start_s.isin(WINDOW_STARTS)]
        if len(data) != 288 * 5:
            raise AssertionError(f"{name}: expected 1440 final rows, got {len(data)}")

    keys = ["scene", "model", "window_start_s"]
    d = control[control.window_start_s.isin(WINDOW_STARTS)].copy()
    directional = d.direction != "N"
    d["control_failure"] = np.where(
        directional, d.wrong_direction_60.fillna(0), d.noop_motion_010.fillna(0)).astype(int)
    d = d[keys + ["direction", "control_failure", "cosine", "magnitude"]]
    s = style[style.window_start_s.isin(WINDOW_STARTS)].copy()
    s["style_failure"] = (s.drift_from_real > 0.72).astype(int)
    d = d.merge(s[keys + ["drift_from_real", "local_adjacent_drift", "style_failure"]], on=keys)
    g = geometry[geometry.window_start_s.isin(WINDOW_STARTS)]
    d = d.merge(g[keys + ["p_uncanny", "geometry_flag"]], on=keys)
    c = conj[conj.window_start_s.isin(WINDOW_STARTS)][keys + ["conjuration_flag"]].copy()
    c = c.sort_values(["scene", "window_start_s"])
    c["conjuration_cumulative"] = c.groupby("scene").conjuration_flag.cummax()
    d = d.merge(c, on=keys)
    d["endpoint_s"] = d.window_start_s + 6
    hf = hf_scores(out, old)
    d = d.merge(hf[["scene", "model", "endpoint_s", "hf_B", "hf_failure"]],
                on=["scene", "model", "endpoint_s"])
    if len(d) != 288 * 5 or d.isna().any().any():
        missing = d.columns[d.isna().any()].tolist()
        # cosine and local drift are deliberately undefined for no-op/first window.
        allowed = {"cosine", "local_adjacent_drift"}
        if set(missing) - allowed:
            raise AssertionError((len(d), missing))
    d.to_csv(out / "per_video_five_window_scores.csv", index=False)

    metrics = {
        "control_failure_pct": "control_failure",
        "style_failure_pct": "style_failure",
        "geometry_failure_pct": "geometry_flag",
        "conjuration_cumulative_pct": "conjuration_cumulative",
        "hf_failure_pct": "hf_failure",
    }
    summary = d.groupby("endpoint_s").agg(
        videos=("scene", "size"),
        **{name: (column, lambda x: 100 * x.mean()) for name, column in metrics.items()},
        dino_drift_mean=("drift_from_real", "mean"),
        geometry_probability_mean=("p_uncanny", "mean"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    # Directly paired comparison with the primary 13-frame minWM row.
    prior = pd.read_csv(old / "metric_trajectory_rates.csv")
    prior = prior[prior.model == "minwm"]
    lookup = {
        "control_failure_rate": "control_failure_pct",
        "geometry_failure_rate": "geometry_failure_pct",
        "conjuration_failure_rate": "conjuration_cumulative_pct",
        "hf_failure_rate": "hf_failure_pct",
    }
    comparisons = []
    for r in prior.itertuples(index=False):
        if r.metric not in lookup or r.endpoint_s not in ENDPOINTS:
            continue
        new = float(summary.loc[summary.endpoint_s == r.endpoint_s, lookup[r.metric]].iloc[0])
        old_pct = 100 * float(r.value)
        comparisons.append(dict(endpoint_s=r.endpoint_s, metric=r.metric,
                                seed13_pct=old_pct, seed29_pct=new,
                                seed29_minus_seed13_pp=new-old_pct))
    old_style = pd.read_csv(old / "style_dino_trajectory_summary.csv")
    old_style = old_style[old_style.model == "minwm"]
    for r in old_style.itertuples(index=False):
        if r.endpoint_s not in ENDPOINTS:
            continue
        new = float(summary.loc[summary.endpoint_s == r.endpoint_s,
                                "style_failure_pct"].iloc[0])
        old_pct = 100 * float(r.style_failure)
        comparisons.append(dict(endpoint_s=r.endpoint_s, metric="style_failure_rate",
                                seed13_pct=old_pct, seed29_pct=new,
                                seed29_minus_seed13_pp=new-old_pct))
    pd.DataFrame(comparisons).to_csv(out / "paired_primary_comparison.csv", index=False)

    lines = [
        "# minWM two-native-block conditioning sensitivity",
        "",
        "This is a separate 29-frame conditioning sensitivity run. The primary paper row uses 13 frames.",
        "All rates pool all 288 videos, including no-op, at every endpoint.",
        "Conjuration is cumulative after its first detection in a rollout.",
        "",
        "| Endpoint | Control | Style | Geometry | Conjuration | HF |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {int(r.endpoint_s)} s | {r.control_failure_pct:.1f}% | "
            f"{r.style_failure_pct:.1f}% | {r.geometry_failure_pct:.1f}% | "
            f"{r.conjuration_cumulative_pct:.1f}% | {r.hf_failure_pct:.1f}% |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("manifest", "summarize"):
        q = sub.add_parser(name)
        q.add_argument("--out", required=True, type=Path)
        if name == "summarize":
            q.add_argument("--old", required=True, type=Path)
    a = p.parse_args()
    if a.command == "manifest":
        build_manifest(a.out.resolve())
    else:
        summarize(a.out.resolve(), a.old.resolve())


if __name__ == "__main__":
    main()
