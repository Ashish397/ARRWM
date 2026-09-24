"""Dual-reference long-horizon style and geometry evaluation.

The real conditioning span is retained as a global reference.  Every window
after the first also receives one second of generated lookback.  Style emits
separate seed, rolling, and seam DINOv2 distances.  Geometry emits separate
absolute-corruption and rolling-continuity probabilities; it never averages
the real and generated references into one opaque score.

The pilot mode additionally scores deliberately swapped references and warped
targets.  These controls must pass before a fleet run is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


WINDOW_STARTS = (0, 6, 12, 18, 24)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_range(first: int, last: int, count: int) -> list[int]:
    if last < first:
        raise ValueError((first, last))
    return np.linspace(first, last, count).round().astype(int).tolist()


def frame_plan(ctx: int, fps: float, n: int, start: int) -> dict[str, list[int]]:
    """Indices for a six-second target and its one-second generated lookback."""
    target_first = ctx + int(round(start * fps))
    target_last = ctx + int(round((start + 6) * fps)) - 1
    if target_last >= n:
        raise IndexError((ctx, fps, n, start, target_last))
    seed = sample_range(0, max(0, ctx - 1), min(16, max(1, ctx)))
    early_last = min(target_last, target_first + int(round(fps)) - 1)
    end_first = max(target_first, target_last - int(round(fps)) + 1)
    out = {
        "seed_style": seed,
        "seed_geometry": sample_range(0, max(0, ctx - 1), min(4, max(1, ctx))),
        "target_geometry": sample_range(target_first, target_last, 12),
        "target_early": sample_range(target_first, early_last, 16),
        "target_end": sample_range(end_first, target_last, 16),
    }
    if start > 0:
        prior_last = target_first - 1
        prior_first = max(ctx, prior_last - int(round(fps)) + 1)
        out["rolling_style"] = sample_range(prior_first, prior_last, 16)
        out["rolling_geometry"] = sample_range(prior_first, prior_last, 8)
    else:
        # The first target is grounded directly in the real conditioning span.
        out["rolling_style"] = out["seed_style"]
        out["rolling_geometry"] = out["seed_geometry"]
    return out


def read_indices(path: str, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = sorted(set(int(i) for i in indices))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    frames, k = {}, 0
    for i in range(wanted[-1] + 1):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"missing frame {i}: {path}")
        if i == wanted[k]:
            frames[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            k += 1
            if k == len(wanted):
                break
    cap.release()
    if len(frames) != len(wanted):
        raise ValueError(f"decoded {len(frames)}/{len(wanted)} requested frames")
    return frames


def frames_for(frames: dict[int, np.ndarray], indices: list[int]) -> list[np.ndarray]:
    return [frames[i] for i in indices]


def geometric_warp(images: list[np.ndarray]) -> list[np.ndarray]:
    """Strong, deterministic non-rigid corruption used only as a pilot control."""
    out = []
    for k, im in enumerate(images):
        h, w = im.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        amp = 0.12 * w
        phase = 2 * np.pi * k / max(1, len(images) - 1)
        map_x = xx + amp * np.sin(2 * np.pi * yy / max(h, 1) + phase)
        map_y = yy + 0.08 * h * np.sin(2 * np.pi * xx / max(w, 1) - phase)
        warped = cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT101)
        # Fold and duplicate the central scene structure.  This is intentionally
        # conspicuous: it is an instrument control, never an evaluation sample.
        left, right = int(0.22 * w), int(0.78 * w)
        core = warped[:, left:right]
        folded = np.concatenate([core[:, :core.shape[1] // 2],
                                 core[:, :core.shape[1] // 2][:, ::-1]], axis=1)
        warped[:, left:left + folded.shape[1]] = folded
        out.append(warped)
    return out


class DinoStyle:
    def __init__(self):
        import timm
        import torch
        self.torch = torch
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                       pretrained=True, num_classes=0,
                                       img_size=224).cuda().eval()

    def embed(self, images: list[np.ndarray]):
        import torch.nn.functional as F
        x = self.torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).cuda()
        with self.torch.no_grad():
            z = F.normalize(self.model(x), dim=-1).mean(0)
            return F.normalize(z, dim=0)

    @staticmethod
    def distance(a, b) -> float:
        return float(1 - (a @ b))

    def score(self, seed, prior, early, end) -> dict[str, float]:
        seed_e, prior_e = self.embed(seed), self.embed(prior)
        early_e, end_e = self.embed(early), self.embed(end)
        return {
            "seed_dino_drift": self.distance(seed_e, end_e),
            "rolling_dino_drift": self.distance(prior_e, end_e),
            "seam_dino_drift": self.distance(prior_e, early_e),
            "within_window_dino_drift": self.distance(early_e, end_e),
        }


def label_image(rgb: np.ndarray, label: str):
    from PIL import Image
    fr = cv2.resize(rgb, (640, 352))
    header = np.full((28, fr.shape[1], 3), 30, np.uint8)
    cv2.putText(header, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2)
    return Image.fromarray(np.concatenate([header, fr], axis=0))


class QwenGeometry:
    TARGET_ONLY = """The images labelled TARGET WINDOW are consecutive samples
from a six-second AI-generated driving video. Is there a SIGNIFICANT
reality-breaking geometric corruption in these images: impossible or melted
structure, surfaces dissolving into abstract patterns, duplicated topology, or
a large geometrically corrupted region? Camera travel, turning, occlusion,
ordinary motion, exposure change, blur, and compression are not failures.
Ignore small local artifacts. Answer Yes or No."""

    ABSOLUTE = """The SOURCE REAL images show the real conditioning video. The
TARGET WINDOW images are a later six-second continuation in temporal order.
The camera is allowed to travel, turn, and reveal an entirely different view,
so a changed location or layout is not itself a failure. Is there a SIGNIFICANT
reality-breaking geometric corruption inside the TARGET WINDOW: impossible or
melted structure, surfaces dissolving into abstract patterns, duplicated
topology, or a large geometrically corrupted region? Ignore ordinary motion,
occlusion, blur, compression, and small local artifacts. Answer Yes or No."""

    CONTINUITY = """The ROLLING REFERENCE images cover the generated second
immediately before the target interval. The first TARGET WINDOW frame follows
the last ROLLING REFERENCE frame directly: there is NO omitted time gap between
them. Camera motion, turning, occlusion, and newly revealed content are
allowed, but a sudden switch to an unrelated scene is not physically possible
between these consecutive frames. Is there a SIGNIFICANT impossible
discontinuity at the boundary or within the target, including an abrupt scene
replacement, stable surfaces or objects melting or duplicating, topology
changing impossibly, or large abstract corrupted structure? Ignore ordinary
smooth viewpoint change, exposure change, blur, and small local artifacts.
Answer Yes or No."""

    def __init__(self):
        import torch
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16,
            device_map="cuda").eval()
        tok = self.proc.tokenizer
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]

    def p_yes(self, images, question: str) -> float:
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question +
                        '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'})
        prompt = self.proc.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False) + '{"answer": "'
        inputs = self.proc(text=[prompt], images=images,
                           return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
            ids = self.torch.tensor([self.yes_id, self.no_id], device=logits.device)
            return float(self.torch.softmax(logits[ids], 0)[0])

    def score(self, seed, prior, target, seed_diagnostic: bool = False) -> dict[str, float]:
        seed_images = [label_image(x, "SOURCE REAL") for x in seed]
        prior_images = [label_image(x, "ROLLING REFERENCE") for x in prior]
        target_images = [label_image(x, "TARGET WINDOW") for x in target]
        scores = {
            "p_geometry_absolute": self.p_yes(target_images, self.TARGET_ONLY),
            "p_geometry_rolling_break": self.p_yes(prior_images + target_images,
                                                    self.CONTINUITY),
        }
        scores["p_geometry_seed_conditioned"] = (
            self.p_yes(seed_images + target_images, self.ABSOLUTE)
            if seed_diagnostic else np.nan
        )
        return scores


def select_rows(manifest: pd.DataFrame, models: str | None, scenes: str | None,
                starts: str | None, pilot: bool) -> tuple[pd.DataFrame, list[int]]:
    d = manifest[manifest.decoded_frames.notna()].copy()
    if models:
        d = d[d.model.isin(models.split(","))]
    if scenes:
        d = d[d.scene.isin(scenes.split(","))]
    selected_starts = [int(x) for x in starts.split(",")] if starts else list(WINDOW_STARTS)
    if pilot:
        preferred = ["ours_recovery_base", "lingbot", "dreamx", "matrixgame2",
                     "minwm", "ours_kl4rung", "ours_mse4rung", "ours_no_commit"]
        chosen = []
        for i, model in enumerate(preferred):
            z = d[d.model == model].sort_values("scene")
            if len(z):
                # Spread controls over distinct contexts and commands so the
                # next-row swapped reference is genuinely unrelated.
                chosen.append(z.iloc[(len(z) // 2 + i * 31) % len(z)])
        d = pd.DataFrame(chosen)
        selected_starts = [24]
    return d, selected_starts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--metric", choices=["style", "geometry"], required=True)
    p.add_argument("--models")
    p.add_argument("--scenes")
    p.add_argument("--starts", help="comma-separated window starts; default 0,6,12,18,24")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--path-prefix-from")
    p.add_argument("--path-prefix-to")
    p.add_argument("--shard-index", type=int)
    p.add_argument("--shard-count", type=int, default=1)
    a = p.parse_args()

    manifest = pd.read_csv(a.manifest)
    if a.path_prefix_from and a.path_prefix_to:
        manifest["path"] = manifest.path.str.replace(
            a.path_prefix_from, a.path_prefix_to, regex=False)
    rows, starts = select_rows(manifest, a.models, a.scenes, a.starts, a.pilot)
    if a.shard_index is not None:
        if not 0 <= a.shard_index < a.shard_count:
            raise ValueError("invalid shard")
        rows = rows.iloc[a.shard_index::a.shard_count]
    if rows.empty:
        raise ValueError("no videos selected")
    missing = rows[~rows.path.map(os.path.exists)]
    if len(missing):
        raise FileNotFoundError(missing[["scene", "model", "path"]].to_string(index=False))

    instrument = DinoStyle() if a.metric == "style" else QwenGeometry()
    prior_results = pd.read_csv(a.out) if a.out.exists() and a.out.stat().st_size else pd.DataFrame()
    results = prior_results.to_dict("records") if len(prior_results) else []
    done = (set(zip(prior_results.scene, prior_results.model,
                    prior_results.window_start_s, prior_results.condition))
            if len(prior_results) else set())
    # Pilot controls use the next selected video as a deliberately wrong
    # generated reference.  Baseline fleet mode never substitutes references.
    selected = list(rows.itertuples())
    for ri, r in enumerate(selected):
        wrong = selected[(ri + 1) % len(selected)] if a.pilot and len(selected) > 1 else None
        for start in starts:
            expected_conditions = ({"baseline", "swapped_prior", "swapped_seed"}
                                   if a.pilot and a.metric == "style" else
                                   {"baseline", "swapped_prior", "warped_target"}
                                   if a.pilot else {"baseline"})
            if all((r.scene, r.model, start, condition) in done
                   for condition in expected_conditions):
                continue
            plan = frame_plan(int(r.context_frames), float(r.fps),
                              int(r.decoded_frames), start)
            all_indices = sum(plan.values(), [])
            fr = read_indices(r.path, all_indices)
            seed = frames_for(fr, plan["seed_style" if a.metric == "style" else "seed_geometry"])
            prior = frames_for(fr, plan["rolling_style" if a.metric == "style" else "rolling_geometry"])
            early = frames_for(fr, plan["target_early"])
            end = frames_for(fr, plan["target_end"])
            target = frames_for(fr, plan["target_geometry"])

            common = dict(scene=r.scene, model=r.model, window_start_s=start,
                          window_end_s=start + 6, condition="baseline",
                          seed_indices=json.dumps(plan["seed_style" if a.metric == "style" else "seed_geometry"]),
                          rolling_indices=json.dumps(plan["rolling_style" if a.metric == "style" else "rolling_geometry"]),
                          target_indices=json.dumps(plan["target_end" if a.metric == "style" else "target_geometry"]),
                          video_sha256=sha256(r.path), reference_policy="real_seed_plus_1s_generated_lookback_v1")
            if a.metric == "style":
                results.append({**common, **instrument.score(seed, prior, early, end)})
            else:
                results.append({**common, **instrument.score(
                    seed, prior, target, seed_diagnostic=a.pilot)})

            if a.pilot and wrong is not None:
                wp = frame_plan(int(wrong.context_frames), float(wrong.fps),
                                int(wrong.decoded_frames), start)
                key = "rolling_style" if a.metric == "style" else "rolling_geometry"
                wrong_fr = read_indices(wrong.path, wp[key])
                wrong_prior = frames_for(wrong_fr, wp[key])
                if a.metric == "style":
                    wrong_seed_fr = read_indices(wrong.path, wp["seed_style"])
                    wrong_seed = frames_for(wrong_seed_fr, wp["seed_style"])
                    results.append({**common, "condition": "swapped_prior",
                                    **instrument.score(seed, wrong_prior, early, end)})
                    results.append({**common, "condition": "swapped_seed",
                                    **instrument.score(wrong_seed, prior, early, end)})
                else:
                    results.append({**common, "condition": "swapped_prior",
                                    **instrument.score(seed, wrong_prior, target,
                                                       seed_diagnostic=a.pilot)})
                    results.append({**common, "condition": "warped_target",
                                    **instrument.score(seed, prior, geometric_warp(target),
                                                       seed_diagnostic=a.pilot)})
            tmp = a.out.with_suffix(a.out.suffix + ".tmp")
            pd.DataFrame(results).to_csv(tmp, index=False)
            os.replace(tmp, a.out)
            print(a.metric, r.scene, r.model, start, "rows", len(results), flush=True)

    print(a.out, len(results))


if __name__ == "__main__":
    main()
