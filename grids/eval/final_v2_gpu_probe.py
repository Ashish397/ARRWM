"""Small, explicitly partial ICLR probe using the deployed AAAI GPU instruments.

Run one (scene, model, metric) at a time. Results are evidence for feasibility,
not fleet estimates. Input SHA256 is part of the cache key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code_release" / "evaluation" / "quality"))
import fleet30s_common as fc  # noqa: E402
import vlm_external as V  # noqa: E402


def real_indices(ctx, model=None):
    """Four references spanning the final 13 real conditioning frames."""
    if fc.PANEL32_MODE:
        if model is None:
            raise ValueError("panel32 real references require the model")
        available = fc.source_indices(model, 13)
        if len(available) == 1:
            return available * 4
        return [available[i] for i in np.linspace(0, len(available)-1, 4).round().astype(int)]
    if ctx <= 1:
        return [0, 0, 0, 0]
    first = max(0, ctx - 13)
    return np.linspace(first, ctx - 1, 4).round().astype(int).tolist()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix(path, end):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in range(end + 1):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"frame {i} missing: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), fps


def run_geometry(scene, model, out, start_s):
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    uid = scene.rsplit("_", 1)[0]
    path = fc._path(scene, model)
    seed = fc.SEED_CLIP(uid)
    n, fps = fc.meta(scene, model)
    ctx = fc.ctx_of(model)
    first = ctx + int(round(start_s * fps))
    end = ctx + int(round((start_s + 6) * fps)) - (0 if start_s == 0 else 1)
    if end >= n:
        raise IndexError(f"AAAI 6 s sample requires index {end}, n={n}")
    vid, _ = prefix(path, end)
    ref_ids = real_indices(ctx, model)
    real, _ = prefix(seed, max(ref_ids))
    ref = [V.label_img(cv2.resize(real[i], (640, 352)), "REFERENCE") for i in ref_ids]
    gen_idx = np.linspace(first, end, 16).round().astype(int)
    ims = ref + [V.label_img(cv2.resize(vid[i], (640, 352)), "GENERATED") for i in gen_idx]
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    @torch.no_grad()
    def p_yes(question):
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": V.INTRO.format(nref=1) + "\n\n" + question +
             '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        prompt = proc.apply_chat_template([{"role": "user", "content": content}],
                                          add_generation_prompt=True, tokenize=False)
        prompt += '{"answer": "'
        inputs = proc(text=[prompt], images=ims, return_tensors="pt").to("cuda")
        logits = qwen(**inputs).logits[0, -1]
        p = torch.softmax(logits[torch.tensor([yes_id, no_id], device=logits.device)], 0)
        return float(p[0])
    probs = {f"p_{key}": round(p_yes(q), 4) for key, q in V.PROBES}
    return dict(metric="geometry_deployed_qwen3_vl", probs=probs,
                flag=probs["p_uncanny"] > 0.5, generated_indices=gen_idx.tolist(),
                window_start_s=start_s, window_end_s=start_s+6,
                real_indices=ref_ids, real_reference_policy="candidate_conditioning_v2",
                video_sha256=sha256(path),
                real_sha256=sha256(seed), producer=str(Path(V.__file__).resolve()))


def run_conj(scene, model, out, start_s):
    import popin_backends as B
    import popin_detect as P
    path = fc._path(scene, model)
    n, fps = fc.meta(scene, model)
    ctx = fc.ctx_of(model)
    end = ctx + int(round((start_s + 6) * fps)) - (0 if start_s == 0 else 1)
    if end >= n:
        raise IndexError(f"AAAI 6 s sample requires index {end}, n={n}")
    vid, _ = prefix(path, end)
    if start_s:
        history = int(round(fps))
        clip_start = ctx + int(round((start_s - 1) * fps))
        vid = vid[clip_start:]
        analyse_ctx = history
    else:
        clip_start = 0
        analyse_ctx = ctx
    dense, crop = B.build("rtdetr")
    P.set_detector(crop)
    P.set_fps(fps)
    ev = P.analyse(vid, {"n": len(vid), "w": vid.shape[2], "h": vid.shape[1],
                         "dets": dense(vid)}, analyse_ctx)
    for x in ev:
        x["birth_s"] = round((clip_start + x["birth"] - ctx) / fps, 3)
        if isinstance(x.get("box"), np.ndarray):
            x["box"] = x["box"].tolist()
    return dict(metric="conjuration_deployed_rtdetr", flag=any(x["score"] > 0 for x in ev),
                events=ev, decoded_window_frames=len(vid), input_clip_start_index=clip_start,
                prior_history_frames=analyse_ctx, window_start_s=start_s, window_end_s=start_s+6,
                video_sha256=sha256(path),
                producer=str(Path(P.__file__).resolve()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", choices=["geometry", "conj"], required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--start-s", type=int, default=0,
                   help="six-second window start; 0 retains the AAAI anchor")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    path = fc._path(a.scene, a.model)
    fingerprints = sha256(path)[:16]
    if a.metric == "geometry":
        fingerprints += "__" + sha256(fc.SEED_CLIP(a.scene.rsplit("_", 1)[0]))[:16]
        fingerprints += "__refs" + "_".join(map(str, real_indices(fc.ctx_of(a.model), a.model)))
    key = f"{a.metric}__s{a.start_s}__{a.scene}__{a.model}__{fingerprints}"
    dest = a.out / f"{key}.json"
    if dest.exists():
        print(dest, "cached")
        return
    result = (run_geometry if a.metric == "geometry" else run_conj)(a.scene, a.model, a.out, a.start_s)
    result.update(scene=a.scene, model=a.model)
    dest.write_text(json.dumps(result, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    print(dest)


if __name__ == "__main__":
    main()
