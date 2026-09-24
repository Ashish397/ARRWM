"""Resumable ICLR style/geometry/conjuration producer on fixed six-second windows.

The six-second anchor uses the deployed AAAI image sampling and thresholds.
Later windows retain the same frame density and are reported separately.
One model instance is kept resident per GPU process. Each clip is keyed by its
SHA256, with atomic per-video JSON output so a holder can be restarted safely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code_release" / "evaluation" / "quality"))
import fleet30s_common as fc  # noqa: E402
import vlm_external as V  # noqa: E402

WINDOWS = (0, 6, 9, 12, 18, 24)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_selected(path, indices):
    """Read selected native frames in order, asserting that each exists."""
    wanted = sorted(set(int(i) for i in indices))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    out = {}
    k = 0
    for i in range(wanted[-1] + 1):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"missing frame {i} in {path}")
        if i == wanted[k]:
            out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            k += 1
            if k == len(wanted):
                break
    cap.release()
    assert len(out) == len(wanted)
    return out


def ref_indices(model):
    """Absolute real-video references ending at shared boundary frame 32.

    Generated clips store different native context spans, but ``read_selected``
    below reads the original 65-frame real seed.  Therefore candidate-local
    context indices must be shifted by ``33 - ctx`` before indexing that seed.
    This gives real frames 20, 24, 28, and 32 for every model with at least
    13 context frames—including minWM DMD's 29-frame prefix—and repeats real
    frame 32 for one-frame systems.
    """
    ctx = int(fc.ctx_of(model))
    if ctx <= 1:
        return [32, 32, 32, 32]
    local_first = max(0, ctx - 13)
    source_offset = 33 - ctx
    return np.linspace(source_offset + local_first, 32, 4).round().astype(int).tolist()


def valid_cache(target, r, metric, digest, instrument_profile="deployed"):
    if not target.exists():
        return False
    cached = json.loads(target.read_text())
    rows = cached.get("rows", [])
    if (cached.get("video_sha256") != digest or
            cached.get("scene") != r.scene or
            cached.get("model") != r.model or
            cached.get("metric") != metric or
            len(rows) != len(WINDOWS) or
            {row.get("window_start_s") for row in rows} != set(WINDOWS) or
            any(row.get("video_sha256") != digest for row in rows) or
            cached.get("instrument_profile", "deployed") != instrument_profile):
        return False
    if metric != "geometry":
        return True
    expected_refs = ref_indices(r.model)
    expected_real_hash = sha256(fc.SEED_CLIP(r.scene.rsplit("_", 1)[0]))
    return all(
        row.get("real_indices") == expected_refs
        and row.get("real_sha256") == expected_real_hash
        for row in rows
    )


def geometry_indices(ctx, fps, n, start):
    first = ctx + int(round(start * fps))
    end = ctx + int(round((start + 6) * fps)) - (0 if start == 0 else 1)
    if end >= n:
        raise IndexError((start, end, n))
    return np.linspace(first, end, 16).round().astype(int).tolist()


class Geometry:
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
        self.question = dict(V.PROBES)["uncanny"]

    def score(self, images):
        content = [{"type": "image", "image": im} for im in images] + [{
            "type": "text", "text": V.INTRO.format(nref=1) + "\n\n" + self.question +
            '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        prompt = self.proc.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False) + '{"answer": "'
        inputs = self.proc(text=[prompt], images=images, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
            pair = logits[self.torch.tensor([self.yes_id, self.no_id], device=logits.device)]
            return float(self.torch.softmax(pair, 0)[0])


def run_geometry(r, instrument, digest):
    ctx, fps, n = int(r.context_frames), float(r.fps), int(r.decoded_frames)
    uid = r.scene.rsplit("_", 1)[0]
    seed = fc.SEED_CLIP(uid)
    refs = ref_indices(r.model)
    real = read_selected(seed, refs)
    anchor = [V.label_img(cv2.resize(real[i], (640, 352)), "REFERENCE") for i in refs]
    picks = {s: geometry_indices(ctx, fps, n, s) for s in WINDOWS}
    frames = read_selected(r.path, [i for ids in picks.values() for i in ids])
    seed_hash = sha256(seed)
    rows = []
    for start, ids in picks.items():
        images = anchor + [V.label_img(cv2.resize(frames[i], (640, 352)), "GENERATED") for i in ids]
        p_raw = instrument.score(images)
        p = round(p_raw, 4)  # deployed vlm_external.py writes four decimals before flagging
        rows.append(dict(scene=r.scene, model=r.model, window_start_s=start,
                         window_end_s=start+6, p_uncanny_raw=p_raw, p_uncanny=p,
                         geometry_flag=int(p > 0.5),
                         generated_indices=ids, real_indices=refs,
                         real_reference_policy="candidate_conditioning_v2",
                         real_sha256=seed_hash, video_sha256=digest,
                         producer="code_release/evaluation/quality/vlm_external.py",
                         threshold=0.5))
    return rows


class Conjuration:
    def __init__(self, profile="deployed"):
        import popin_backends as B
        import popin_detect as P
        if profile == "persistent-salient-v2":
            # In the deployed Action-Forcing instrument, KEEP contains people,
            # bicycles, bags, chairs, and other objects only as evidence that a
            # vehicle/cargo candidate existed previously.  That is too narrow
            # for a general "conjuration" claim: in particular, a persistent
            # person can visibly materialise while never reaching analyse().
            #
            # This audit profile promotes every tracked salient class to a
            # candidate.  It does *not* automatically call every candidate a
            # failure: all persistent candidates are exported for blinded
            # temporal adjudication, including those rejected by branches A/B.
            P.CONJURABLE = set(P.KEEP)
        self.dense, crop = B.build("rtdetr")
        P.set_detector(crop)
        self.detect = P
        self.profile = profile

    def score(self, r, digest):
        ctx, fps, n = int(r.context_frames), float(r.fps), int(r.decoded_frames)
        end = ctx + int(round(30 * fps)) - 1
        if end >= n:
            raise IndexError((end, n))
        cap = cv2.VideoCapture(r.path)
        frames = []
        for i in range(end+1):
            ok, bgr = cap.read()
            if not ok:
                cap.release()
                raise ValueError(f"missing frame {i}: {r.path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        video = np.stack(frames)
        detections = self.dense(video)
        assert len(detections) == len(video)
        self.detect.set_fps(fps)
        rows = []
        for start in WINDOWS:
            clip_start = 0 if start == 0 else ctx + int(round((start - 1) * fps))
            analysis_ctx = ctx if start == 0 else int(round(fps))
            clip_end = ctx + int(round((start + 6) * fps)) - (0 if start == 0 else 1)
            part = video[clip_start:clip_end+1]
            ev = self.detect.analyse(
                part, {"n": len(part), "w": part.shape[2], "h": part.shape[1],
                       "dets": detections[clip_start:clip_end+1]}, analysis_ctx)
            for x in ev:
                x["birth_s"] = round((clip_start + x["birth"] - ctx) / fps, 3)
                if isinstance(x.get("box"), np.ndarray):
                    x["box"] = x["box"].tolist()
            positive = [x for x in ev if x["score"] > 0]
            rows.append(dict(scene=r.scene, model=r.model, window_start_s=start,
                             window_end_s=start+6, conjuration_flag=int(bool(positive)),
                             candidate_events=len(ev),
                             events=ev, positive_events=len(positive),
                             top_score=ev[0]["score"] if ev else None,
                             clip_start_index=clip_start, clip_end_index=clip_end,
                             prior_history_frames=analysis_ctx,
                             video_sha256=digest,
                             producer="code_release/evaluation/quality/popin_detect.py",
                             backend="rtdetr", instrument_profile=self.profile))
        return rows


def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--metric", choices=["geometry", "conjuration"], required=True)
    p.add_argument(
        "--conjuration-profile",
        choices=["deployed", "persistent-salient-v2"],
        default="deployed",
        help=("deployed reproduces the validated vehicle/cargo detector; "
              "persistent-salient-v2 exports every persistent salient-class "
              "birth candidate for temporal human adjudication"),
    )
    p.add_argument(
        "--destination-name",
        help="output subdirectory (defaults to the metric name)",
    )
    p.add_argument("--models")
    p.add_argument("--scenes")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.shard_index < a.shard_count:
        raise ValueError("invalid shard")
    m = pd.read_csv(a.out / "video_manifest.csv")
    m = m[m.decoded_frames.notna()]
    if a.models:
        m = m[m.model.isin(a.models.split(","))]
    if a.scenes:
        m = m[m.scene.isin(a.scenes.split(","))]
    m = m.iloc[a.start_index:a.end_index]
    m = m.iloc[a.shard_index::a.shard_count]
    if a.metric != "conjuration" and a.conjuration_profile != "deployed":
        raise ValueError("--conjuration-profile applies only to conjuration")
    dest = a.out / (a.destination_name or a.metric)
    dest.mkdir(exist_ok=True)
    engine = (Geometry() if a.metric == "geometry" else
              Conjuration(a.conjuration_profile))
    errors = []
    for k, r in enumerate(m.itertuples(), 1):
        target = dest / f"{r.scene}__{r.model}.json"
        try:
            digest = sha256(r.path)
            if valid_cache(target, r, a.metric, digest, a.conjuration_profile):
                print(f"[{k}/{len(m)}] cached {r.scene} {r.model}", flush=True)
                continue
            rows = run_geometry(r, engine, digest) if a.metric == "geometry" else engine.score(r, digest)
            if len(rows) != len(WINDOWS):
                raise AssertionError("incomplete windows")
            obj = dict(scene=r.scene, model=r.model, metric=a.metric,
                       instrument_profile=a.conjuration_profile,
                       video_sha256=digest, rows=rows)
            tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(obj, default=json_default))
            os.replace(tmp, target)
            print(f"[{k}/{len(m)}] {r.scene} {r.model}", flush=True)
        except Exception as ex:
            errors.append(dict(scene=r.scene, model=r.model, error=repr(ex)))
            pd.DataFrame(errors).to_csv(a.out / f"{a.metric}_errors_shard{a.shard_index}.csv", index=False)
            print(f"ERROR {r.scene} {r.model}: {ex}", flush=True)
    if errors:
        raise RuntimeError(
            f"{a.metric} producer failed on {len(errors)} videos; "
            f"see {a.out / f'{a.metric}_errors_shard{a.shard_index}.csv'}"
        )


if __name__ == "__main__":
    main()
