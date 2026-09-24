"""Auditable, read-only inventory and artifact checks for the ICLR 30 s fleet.

Writes only below --out. It does not promote the old pooled scores to v2 results.
"""
from __future__ import annotations

import argparse
import csv
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import fleet30s_common as fc  # noqa: E402

DEFAULT_MODELS = ["lingbot", "dreamx", "minwm", "matrixgame2", "minwm_ode",
          "ours_kl4rung", "ours_mse4rung", "ours_recovery_base", "ours_no_commit",
          "ours_no_aux", "ours_no_gan", "ours_no_carn", "ours_stat_mean_only",
          "ours_stat_nonmean_only", "ours_base_v2_BROKEN"]
MODELS = [x for x in os.environ.get("FLEET30S_MODELS", "").split(",") if x] or DEFAULT_MODELS
FAMILY = (
    {m: fc.MODEL_FAMILY[m] for m in MODELS}
    if fc.PANEL32_MODE else
    {m: ("ours" if m.startswith("ours_") else "minwm" if m.startswith("minwm") else m)
     for m in MODELS}
)
SEAT = (
    dict(fc.FAMILY_SEATS) if fc.PANEL32_MODE else
    {"lingbot": "lingbot", "dreamx": "dreamx", "minwm": "minwm",
     "matrixgame2": "matrixgame2", "ours": "ours_recovery_base"}
)
if not fc.PANEL32_MODE and "yume5b" in MODELS:
    SEAT["yume5b"] = "yume5b"
EXPECTED = {"lingbot": (493, 16, 1), "dreamx": (489, 16, 1),
            "yume5b": (481, 16, 1),
            "minwm": (493, 16, 13), "minwm_ode": (493, 16, 13),
            "matrixgame2": (753, 25, 1)}
if os.environ.get("FLEET30S_ALIGNED32_DIR"):
    EXPECTED.update({"lingbot": (493, 16, 1), "dreamx": (489, 16, 1),
                     "minwm": (509, 16, 29), "matrixgame2": (753, 25, 1)})


def frame_index(ctx: int, fps: float, n: int, seconds: float) -> int:
    """Generated frame zero is ctx; last generated frame is ctx+round(H*fps)-1.

    A 480-frame 16 fps rollout has last index ctx+479, timestamp 29.9375 s.
    No out-of-range clamp is permitted.
    """
    frames = int(round(seconds * fps))
    if frames < 1 or not math.isclose(frames / fps, seconds, abs_tol=0.03):
        raise ValueError(f"non-integral horizon {seconds}s at {fps} fps")
    i = ctx + frames - 1
    if i >= n:
        raise IndexError(f"need index {i}, clip has {n} frames")
    return i


def probe(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def manifest(out: Path):
    rows = []
    selected_uids = fc.eval_uids()
    for uid in selected_uids:
        for direction in fc.DIRS:
            scene = f"{uid}_{direction}"
            for model in MODELS:
                p = fc._path(scene, model)
                exists = Path(p).exists()
                v = probe(p) if exists else None
                n, fps, w, h = v if v else (*EXPECTED.get(model, (513, 16, 33))[:2], None, None)
                ctx = fc.ctx_of(model) if exists else EXPECTED.get(model, (513, 16, 33))[2]
                row = dict(scene=scene, uid=uid, direction=direction, model=model,
                           family=FAMILY[model], path=p, real_path=str(Path(p).resolve()) if exists else "",
                           local_video=bool(v), metadata_source="local_cv2_header" if v else "handoff_expected_unverified",
                           container_frames=n, decoded_frames="", fps=fps, width=w, height=h,
                           context_frames=ctx, generated_frames=n-ctx,
                           generated_duration_s=round((n-ctx)/fps, 5),
                           last_generated_timestamp_s=round((n-ctx-1)/fps, 5),
                           directional=direction != "N")
                for horizon in (6, 15, 30):
                    try:
                        row[f"end_{horizon}_index"] = frame_index(ctx, fps, n, horizon)
                        row[f"end_{horizon}_available"] = bool(v)
                    except (IndexError, ValueError):
                        row[f"end_{horizon}_index"] = ""
                        row[f"end_{horizon}_available"] = False
                rows.append(row)
    d = pd.DataFrame(rows)
    expected_rows = len(selected_uids) * len(fc.DIRS) * len(MODELS)
    assert len(d) == expected_rows and not d.duplicated(["scene", "model"]).any()
    d.to_csv(out / "video_manifest.csv", index=False)
    d.groupby("model").agg(listed=("scene", "size"), local=("local_video", "sum"),
                           directional=("directional", "sum")).to_csv(out / "manifest_counts.csv")
    return d


def reproduce_builder(out: Path):
    """Run the actual v2 builder with copies of its inputs, preserving shipped output."""
    stage = out / "aaai_reproduction"
    data = stage / "out"
    data.mkdir(parents=True, exist_ok=True)
    names = ["fleet_legit.csv", "reloc_family_bias.csv", "fleet_hf.csv", "fleet_obey_pca.csv",
             "nocritic_legit.csv", "nocritic_reloc_v2.csv", "nocritic_final_eval.csv", "nocritic_dir_ctrl.csv"]
    for name in names:
        shutil.copy2(HERE / "out" / name, data / name)
    shutil.copy2(HERE / "fleet_static_inl.csv", stage / "fleet_static_inl.csv")
    shutil.copy2(HERE / "build_final_flags.py", stage / "build_final_flags.py")
    env = os.environ.copy()
    env.update(CTRL_RULE="orb", CTRL_ORB_T="500", CTRL_COS_DEG="60")
    p = subprocess.run([sys.executable, str(stage / "build_final_flags.py")],
                       cwd=stage, env=env, capture_output=True, text=True)
    (stage / "builder_stdout.txt").write_text(p.stdout)
    (stage / "builder_stderr.txt").write_text(p.stderr)
    if p.returncode:
        raise RuntimeError(f"AAAI builder failed: {p.stderr[-1000:]}")
    got = pd.read_csv(data / "final_flags_v2.csv").sort_values(["scene", "model"]).reset_index(drop=True)
    ref = pd.read_csv(HERE / "out" / "final_flags_v2.csv").sort_values(["scene", "model"]).reset_index(drop=True)
    assert len(got) == len(ref) and got[["scene", "model"]].equals(ref[["scene", "model"]])
    diffs = {}
    for c in ref.columns:
        if c in ("scene", "model"):
            continue
        a, b = got[c], ref[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diffs[c] = dict(max_abs=float(np.nanmax(np.abs(a.astype(float)-b.astype(float)))),
                            unequal=int((~np.isclose(a, b, equal_nan=True, rtol=0, atol=1e-9)).sum()))
        else:
            diffs[c] = dict(unequal=int((a.fillna("<NA>") != b.fillna("<NA>")).sum()))
    result = dict(rows=len(ref), keys_equal=True, column_diffs=diffs,
                  scope="final builder using shipped AAAI measurements; producer reruns are separate")
    leg = pd.read_csv(HERE / "out" / "fleet_legit.csv")
    ob = pd.read_csv(HERE / "out" / "fleet_obey_pca.csv")
    joined = leg[["scene", "model", "cos"]].merge(ob[["scene", "model", "g0", "g1"]],
                                                        on=["scene", "model"], validate="one_to_one")
    commands = {"F": (1, 0), "FR": (1, 1), "R": (0, 1), "BR": (-1, 1),
                "B": (-1, 0), "BL": (-1, -1), "L": (0, -1), "FL": (1, -1)}
    vectors = np.asarray([commands[s.rsplit("_", 1)[1]] for s in joined.scene], float)
    got_cos = np.sum(vectors * joined[["g0", "g1"]].to_numpy(), axis=1) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(joined[["g0", "g1"]].to_numpy(), axis=1))
    result["cosine_readout_check"] = dict(rows=len(joined),
                                            max_abs_error=float(np.max(np.abs(got_cos-joined.cos.to_numpy()))))
    (stage / "comparison.json").write_text(json.dumps(result, indent=2))
    return result


def reproduce_stationary(out: Path):
    stage = out / "aaai_stationary_reproduction"
    data = stage / "out"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("stat_hf.csv", "nocritic_stat_hf_v2.csv", "stat_scene_v2.csv"):
        shutil.copy2(HERE / "out" / name, data / name)
    shutil.copy2(HERE / "build_final_stationary.py", stage / "build_final_stationary.py")
    p = subprocess.run([sys.executable, str(stage / "build_final_stationary.py")],
                       cwd=stage, capture_output=True, text=True)
    (stage / "builder_stdout.txt").write_text(p.stdout)
    (stage / "builder_stderr.txt").write_text(p.stderr)
    if p.returncode:
        raise RuntimeError(f"stationary builder failed: {p.stderr[-1000:]}")
    got = pd.read_csv(data / "final_stationary_v2.csv").sort_values("model").reset_index(drop=True)
    ref = pd.read_csv(HERE / "out" / "final_stationary_v2.csv").sort_values("model").reset_index(drop=True)
    assert got.model.equals(ref.model)
    diffs = {c: float(np.nanmax(np.abs(got[c]-ref[c]))) for c in ("hf_v1", "hf", "reloc")}
    result = dict(models=len(ref), max_abs_differences=diffs,
                  scope="stationary v2 builder using shipped AAAI measurements")
    (stage / "comparison.json").write_text(json.dumps(result, indent=2))
    return result


def check_indices():
    assert frame_index(1, 16, 481, 30) == 480
    assert frame_index(13, 16, 493, 30) == 492
    assert frame_index(29, 16, 509, 30) == 508
    assert frame_index(33, 16, 513, 30) == 512
    assert frame_index(1, 25, 753, 30) == 750
    assert frame_index(13, 16, 493, 6) == 108
    assert frame_index(13, 16, 493, 15) == 252
    try:
        frame_index(13, 16, 492, 30)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range frame silently accepted")


def original_static_inl(path: str, i0: int, i1: int) -> int:
    """The computation in quality/fleet_static_inl.py, on this fleet's frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    for i in (i0, i1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"undecodable frame {i}: {path}")
        frames.append(frame)
    cap.release()
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kd = [orb.detectAndCompute(cv2.cvtColor(cv2.resize(f, (640, 352)),
                                               cv2.COLOR_BGR2GRAY), None) for f in frames]
    (k0, d0), (k1, d1) = kd
    if d0 is None or d1 is None:
        return 0
    ms = bf.match(d0, d1)
    if len(ms) < 8:
        return 0
    src = np.float32([k0[m.queryIdx].pt for m in ms])
    dst = np.float32([k1[m.trainIdx].pt for m in ms])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def original_hf(scene: str, model: str, end: int):
    """Exact index and Laplacian definition in quality/fleet_hf.py."""
    ctx = fc.ctx_of(model)
    fps = fc.meta(scene, model)[1]
    base_idx = list(range(ctx + int(round(fps)), ctx + int(round(fps)) + 4))
    end_idx = list(range(max(ctx, end - 14), end + 1, 2))
    fr = fc.frames_at(scene, model, base_idx + end_idx)
    def lap_var(rgb):
        g = cv2.cvtColor(cv2.resize(rgb, (832, 448)), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    base = float(np.mean([lap_var(f) for f in fr[:len(base_idx)]]))
    final = float(np.mean([lap_var(f) for f in fr[len(base_idx):]]))
    return round(base, 1), round(final, 1), round(final - base, 1)


def old_metric(name: str, horizon: int):
    dirs = [HERE / "out30s_bx" / f"h{horizon}",
            HERE / "out30s_bx" / f"minwm_ode_h{horizon}",
            HERE / "out30s_u6qf_ours" / f"h{horizon}",
            HERE / "out30s_u6qf_ours" / f"h{horizon}_a",
            HERE / "out30s_u6qf_ours" / f"h{horizon}_b",
            HERE / "out30s_u6qf_ours" / "flagship" / f"h{horizon}"]
    fs = [p / name for p in dirs if (p / name).exists()]
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(p) for p in fs], ignore_index=True)
    d["model"] = d.model.replace({"ours_base_v2": "ours_base_v2_BROKEN"})
    assert not d.duplicated(["scene", "model"]).any(), name
    return d.set_index(["scene", "model"])


def audit_sample(out: Path, manifest_df: pd.DataFrame):
    old_orb = old_metric("fleet30s_orb_h6.csv", 6)
    old_hf = old_metric("fleet_hf.csv", 6)
    rows = []
    for scene in ("u31_F", "m42_N"):
        for r in manifest_df[(manifest_df.scene == scene) & manifest_df.local_video].itertuples():
            end_aaai = r.context_frames + int(round(6 * r.fps))
            if end_aaai >= r.container_frames:
                continue
            static = original_static_inl(r.path, r.context_frames, end_aaai)
            base, final, d = original_hf(scene, r.model, end_aaai)
            key = (scene, r.model)
            rows.append(dict(scene=scene, model=r.model, original_endpoint=end_aaai,
                             old_static_inl=old_orb.loc[key, "static_inl"] if key in old_orb.index else np.nan,
                             producer_static_inl=static,
                             old_d_blur=old_hf.loc[key, "d_blur"] if key in old_hf.index else np.nan,
                             producer_base_blur=base, producer_end_blur=final, producer_d_blur=d,
                             static_delta=static - old_orb.loc[key, "static_inl"] if key in old_orb.index else np.nan,
                             hf_delta=d - old_hf.loc[key, "d_blur"] if key in old_hf.index else np.nan))
            print("audit", scene, r.model, flush=True)
    pd.DataFrame(rows).to_csv(out / "salvage_sample_6s.csv", index=False)


def decode_counts(out: Path):
    path = out / "video_manifest.csv"
    d = pd.read_csv(path)
    local = d[d.local_video].copy()
    def count(r):
        p = Path(r.path)
        # Count decoded frames without materialising RGB images. OpenCV's
        # FFmpeg backend intermittently exhausts colour-conversion contexts on
        # Isambard when several H.264 clips are audited together; ffprobe with
        # one decoder thread checks the same encoded stream without that
        # failure mode.
        cmd = ["ffprobe", "-threads", "1", "-v", "error", "-count_frames",
               "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
               "-of", "csv=p=0", str(p)]
        x = subprocess.run(cmd, capture_output=True, text=True)
        val = int(x.stdout.strip()) if x.returncode == 0 and x.stdout.strip().isdigit() else None
        error = x.stderr.strip()[:200]
        st = p.stat()
        return dict(scene=r.scene, model=r.model, decoded_frames=val, bytes=st.st_size,
                    mtime_ns=st.st_mtime_ns, error=error)
    rows = []
    # The panel setup runs on a 64-CPU holder and each ffprobe decoder is
    # explicitly single-threaded, so 32 independent streams use the node
    # without creating nested decoder pools.
    workers = 32 if fc.PANEL32_MODE else 8
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(count, r) for r in local.itertuples()]
        for i, fut in enumerate(as_completed(jobs), 1):
            rows.append(fut.result())
            if i % 100 == 0:
                pd.DataFrame(rows).to_csv(out / "decoded_counts.csv", index=False)
                print("decoded", i, "/", len(jobs), flush=True)
    pd.DataFrame(rows).to_csv(out / "decoded_counts.csv", index=False)
    c = pd.DataFrame(rows)[["scene", "model", "decoded_frames", "bytes", "mtime_ns"]]
    d = d.drop(columns=["decoded_frames"], errors="ignore").merge(c, on=["scene", "model"], how="left")
    d.to_csv(path, index=False)
    assert (d.loc[d.local_video, "decoded_frames"] == d.loc[d.local_video, "container_frames"]).all()
    return len(rows)


def enrich_manifest(out: Path):
    path = out / "video_manifest.csv"
    d = pd.read_csv(path)
    metadata = []
    remote_root = "/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/out"
    for r in d.itertuples():
        sidecar = Path(str(r.path) + ".json")
        x = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        inferred = (20 if r.model == "minwm_ode" else 0)
        ckpt = x.get("ckpt", x.get("checkpoint"))
        ck_source = "sidecar_ckpt" if ckpt else ""
        if not ckpt:
            label = {"ours_kl4rung": "rollkl10k@400 (ODE KL; handoff)",
                     "ours_mse4rung": "roll10k@400 (ODE MSE; handoff)",
                     "minwm_ode": "minWM ODE (checkpoint ID unavailable)",
                     "dreamx": "DreamX (checkpoint ID unavailable)",
                     "matrixgame2": "Matrix-Game 2 (checkpoint ID unavailable)"}.get(r.model)
            ckpt = label or x.get("model", "unavailable")
            ck_source = "handoff_or_model_label" if label else "sidecar_model_label" if "model" in x else "unavailable"
        remote = ""
        if r.model.startswith("ours_") and r.model not in ("ours_kl4rung", "ours_mse4rung"):
            arm = r.model[5:]
            prefix = "base_v2" if arm == "base_v2_BROKEN" else arm
            remote = f"{remote_root}/{arm}/{prefix}_{r.uid}_{r.direction}.mp4"
        metadata.append(dict(scene=r.scene, model=r.model,
                             sidecar_path=str(sidecar) if sidecar.exists() else "",
                             checkpoint_identity=ckpt, checkpoint_identity_source=ck_source,
                             seed_start_frame=x.get("seed_start_frame", inferred),
                             seed_start_source="sidecar" if "seed_start_frame" in x else "alignment_sample_or_handoff",
                             sidecar_generated_frames=x.get("generated_frames", ""),
                             remote_path_claimed_unverified=remote))
    extra = pd.DataFrame(metadata)
    d = d.drop(columns=[c for c in extra.columns if c in d.columns and c not in ("scene", "model")], errors="ignore")
    d = d.merge(extra, on=["scene", "model"], how="left", validate="one_to_one")
    d.to_csv(path, index=False)


def alignment(out: Path):
    """Check actual encoded context images against the shared seed video."""
    d = pd.read_csv(out / "video_manifest.csv")
    rows = []
    for uid in ("u31", "m42"):
        seed = str(fc.SEED_CLIP(uid))
        s = cv2.VideoCapture(seed)
        seed_frames = []
        for i in range(33):
            ok, f = s.read()
            if not ok:
                raise ValueError(f"seed {uid} missing frame {i}")
            seed_frames.append(f)
        s.release()
        for r in d[(d.uid == uid) & (d.direction == "F") & d.local_video].itertuples():
            cap = cv2.VideoCapture(r.path)
            for ci in sorted(set([0, max(0, r.context_frames - 1)])):
                cap.set(cv2.CAP_PROP_POS_FRAMES, ci)
                ok, f = cap.read()
                if not ok:
                    continue
                # Alignment is descriptive: resizing can obscure crop and codec differences.
                f = cv2.resize(f, (416, 240))
                diffs = [float(np.abs(f.astype(np.int16) - cv2.resize(x, (416, 240)).astype(np.int16)).mean())
                         for x in seed_frames]
                j = int(np.argmin(diffs))
                source_start = {1: 32, 13: 20, 29: 4, 33: 0}[int(r.context_frames)]
                rows.append(dict(uid=uid, model=r.model, context_index=ci,
                                 best_seed_index=j, mean_abs_pixel_error=round(diffs[j], 3),
                                 assumed_seed_index=source_start + ci))
            cap.release()
    pd.DataFrame(rows).to_csv(out / "context_alignment_sample.csv", index=False)


def context_quality(out: Path):
    orb = cv2.ORB_create(3000)
    rows = []
    thumbs = []
    for uid in fc.UIDS:
        cap = cv2.VideoCapture(fc.SEED_CLIP(uid))
        frames = {}
        for i in range(33):
            ok, f = cap.read()
            if not ok:
                raise ValueError(f"seed {uid} missing {i}")
            if i in (0, 20, 32):
                frames[i] = f
        cap.release()
        for i, f in frames.items():
            g = cv2.cvtColor(cv2.resize(f, (640, 352)), cv2.COLOR_BGR2GRAY)
            rows.append(dict(uid=uid, seed_index=i, orb_keypoints=len(orb.detect(g, None)),
                             gray_std=round(float(g.std()), 2), lap_var=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2)))
        t = cv2.resize(frames[32], (208, 120))
        cv2.putText(t, uid, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        thumbs.append(t)
    pd.DataFrame(rows).to_csv(out / "context_feature_diagnostics.csv", index=False)
    sheet = np.concatenate([np.concatenate(thumbs[i:i+8], axis=1) for i in range(0, 32, 8)])
    cv2.imwrite(str(out / "context_contact_sheet.png"), sheet)


def control_contact_sheet(out: Path):
    cases = [("u31_F", "minwm_ode", "ORB 596 > 500: near-static positive"),
             ("u31_F", "ours_no_aux", "ORB 500: boundary hard negative"),
             ("u31_F", "ours_recovery_base", "ORB 495: near-boundary negative"),
             ("u31_F", "lingbot", "sharpness-loss diagnostic")]
    rows = []
    for scene, model, label in cases:
        path = fc._path(scene, model)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ctx = fc.ctx_of(model)
        cells = []
        for seconds in (0, 1, 3, 6):
            i = ctx + int(round(seconds*fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                raise ValueError(f"contact frame {i} unavailable: {scene} {model}")
            f = cv2.resize(f, (320, 180))
            cv2.putText(f, f"{model}  t={seconds}s", (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            cells.append(f)
        cap.release()
        row = np.concatenate(cells, axis=1)
        header = np.full((25, row.shape[1], 3), 25, np.uint8)
        cv2.putText(header, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        rows.append(np.concatenate([header, row], axis=0))
    cv2.imwrite(str(out / "control_hf_sample_contact_sheet.png"), np.concatenate(rows))


def geometry_contact_sheet(out: Path):
    cases = [("m42_F", "ours_base_v2_BROKEN"),
             ("m42_F", "matrixgame2"),
             ("u31_F", "ours_recovery_base")]
    rows = []
    for scene, model in cases:
        candidates = []
        for p in (out / "gpu_samples").glob("geometry*.json"):
            d = json.loads(p.read_text())
            if d.get("scene") == scene and d.get("model") == model and d.get("window_start_s", 0) == 0:
                candidates.append(d)
        if len(candidates) != 1:
            raise ValueError(f"need one current Qwen sample for {scene} {model}: {len(candidates)}")
        d = candidates[0]
        p_uncanny = d["probs"]["p_uncanny"]
        label = f"Qwen p_uncanny={p_uncanny:.4f}: {'positive' if d['flag'] else 'negative'}"
        uid = scene.rsplit("_", 1)[0]
        cap = cv2.VideoCapture(fc.SEED_CLIP(uid))
        # fc.SEED_CLIP is the canonical source video rather than a
        # candidate-local conditioning prefix, so every model's boundary is
        # the same absolute source frame.
        real_index = 32
        cap.set(cv2.CAP_PROP_POS_FRAMES, real_index)
        ok, real = cap.read()
        cap.release()
        if not ok:
            raise ValueError(uid)
        cells = []
        for name, f in [("real", real)]:
            t = cv2.resize(f, (320, 180))
            cv2.putText(t, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cells.append(t)
        cap = cv2.VideoCapture(fc._path(scene, model))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ctx = fc.ctx_of(model)
        for s in (0, 2, 4, 6):
            i = ctx + int(round(s*fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                raise ValueError(f"{scene} {model} {i}")
            t = cv2.resize(f, (320, 180))
            cv2.putText(t, f"t={s}s", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cells.append(t)
        cap.release()
        line = np.concatenate(cells, axis=1)
        h = np.full((25, line.shape[1], 3), 25, np.uint8)
        cv2.putText(h, f"{model}: {label}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255,255,255), 1)
        rows.append(np.concatenate([h, line]))
    cv2.imwrite(str(out / "geometry_sample_contact_sheet.png"), np.concatenate(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stage", choices=["manifest", "reproduce", "audit", "decode", "enrich", "align", "contexts", "contacts", "all"], default="all")
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    check_indices()
    if a.stage in ("manifest", "all"):
        d = manifest(a.out)
        print("manifest", len(d), "local", int(d.local_video.sum()), flush=True)
    if a.stage in ("reproduce", "all"):
        r = reproduce_builder(a.out)
        print("AAAI builder", r["rows"], "rows; unequal flags",
              {k: v["unequal"] for k,v in r["column_diffs"].items() if v["unequal"]}, flush=True)
        s = reproduce_stationary(a.out)
        print("AAAI stationary builder", s["models"], "models", s["max_abs_differences"], flush=True)
    if a.stage in ("audit", "all"):
        d = pd.read_csv(a.out / "video_manifest.csv")
        audit_sample(a.out, d)
    if a.stage in ("decode", "all"):
        print("decoded total", decode_counts(a.out), flush=True)
    if a.stage in ("enrich", "all"):
        enrich_manifest(a.out)
    if a.stage in ("align", "all"):
        alignment(a.out)
    if a.stage in ("contexts", "all"):
        context_quality(a.out)
    if a.stage in ("contacts", "all"):
        control_contact_sheet(a.out)
        geometry_contact_sheet(a.out)


if __name__ == "__main__":
    main()
