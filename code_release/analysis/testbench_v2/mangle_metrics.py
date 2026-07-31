"""Stage 2.5 — LOW-FREQUENCY COLLAPSE (mangle/mash/warp) detectors.

The stage-2 drift/IQA features are haze instruments: on the human notes
they score ~chance for mangle (best: rel_raft_warp_err 0.62). These two
detectors were validated in V1 on exactly this axis (r08_B / r08_BL / r01_R
mangle ground truths) and are ported onto the tb2 fleet loader so they run
on ablation and external videos alike:

  - pal4vst: ICCV23 Swin-L+UperNet unified artifact segmenter (TorchScript).
    Per frame: drop top 15% (haze band), split L/R, resize each to 512x512,
    score = artifact-pixel fraction. Video: mean + max over sampled frames.
  - qwen_melt: Qwen2.5-VL-7B logit-scored P(Yes) to the V1 melt question on
    upper crops (top 60%, L/R halves) of sampled generated frames; video
    score = mean of the top-4 crop scores (melt is judged at its worst).

Output: TB2_MANGLE_OUT csv (vid, pal4vst_mean, pal4vst_max, qwen_melt_pyes).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES = 8            # generated frames sampled per video
TOPCUT = 0.15
_ROOT = os.environ.get(
    "AF_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PAL_TS = os.environ.get(
    "TB2_PAL4VST_TS",
    os.path.join(_ROOT, "third_party", "PAL4VST", "deployment", "pal4vst",
                 "swin-large_upernet_unified_512x512", "end2end.pt"))

MELT_Q = ("Are any buildings, walls or structures in this image melted, "
          "warped, smeared or geometrically impossible, like a corrupted "
          "AI-generated image? Answer only Yes or No.")


class Pal4vst:
    def __init__(self):
        self.model = torch.jit.load(PAL_TS).to(DEV).eval()
        self.mean = torch.tensor([123.675, 116.28, 103.53],
                                 device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([58.395, 57.12, 57.375],
                                device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def frame_frac(self, img):
        t = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV).float()
        t = t[..., int(t.shape[-2] * TOPCUT):, :]
        W0 = t.shape[-1]
        fracs = []
        for tile in (t[..., :, :W0 // 2], t[..., :, W0 // 2:]):
            x = F.interpolate(tile, size=(512, 512), mode="bilinear",
                              align_corners=False)
            x = (x - self.mean) / self.std
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            mask = out.argmax(1) if out.dim() == 4 and out.shape[1] > 1 \
                else (out.squeeze(1) > 0.5).long()
            fracs.append(float(mask.float().mean()))
        return float(np.mean(fracs))


class QwenMelt:
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        mid = os.environ.get("TB2_MELT_JUDGE", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForImageTextToText.from_pretrained(
            mid, dtype=torch.bfloat16, device_map=DEV).eval()
        tok = self.proc.tokenizer
        self.yes = [tok.encode(t, add_special_tokens=False)[0]
                    for t in ("Yes", " Yes", "yes")]
        self.no = [tok.encode(t, add_special_tokens=False)[0]
                   for t in ("No", " No", "no")]

    @torch.no_grad()
    def p_yes(self, img):
        from PIL import Image
        im = Image.fromarray(img)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": im}, {"type": "text", "text": MELT_Q}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[im],
                           return_tensors="pt").to(self.model.device)
        lg = self.model(**inputs).logits[0, -1].float()
        return float(torch.sigmoid(torch.logsumexp(lg[self.yes], 0)
                                   - torch.logsumexp(lg[self.no], 0)))

    def video_score(self, frames):
        """upper crops (top 60%, L/R halves); mean of top-4 crop scores."""
        vals = []
        for f in frames:
            H, W = f.shape[:2]
            h = int(H * 0.6)
            for crop in (f[:h, :W // 2], f[:h, W // 2:]):
                vals.append(self.p_yes(crop))
        return float(np.mean(sorted(vals)[-4:]))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.environ.get("TB2_MANGLE_OUT", os.path.join(here, "out",
                                                        "mangle_features.csv"))
    shard = int(os.environ.get("TB2_SHARD", "0"))
    nshard = int(os.environ.get("TB2_NSHARD", "1"))
    if nshard > 1:
        out = out.replace(".csv", f".shard{shard}.csv")

    refs = fleet.refs_from_env()
    refs = [r for i, r in enumerate(refs) if i % nshard == shard]

    pal = Pal4vst()
    qwen = QwenMelt()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for i, ref in enumerate(refs):
        try:
            frames, times, _ = fleet.load_video(ref.path)
            ts = fleet.gen_fraction_times(times, N_FRAMES)
            gen = [fleet.frame_at(frames, times, t) for t in ts]
            fracs = [pal.frame_frac(f) for f in gen]
            rows.append(dict(
                vid=ref.vid, model=ref.model, scene=ref.scene,
                direction=ref.direction,
                pal4vst_mean=float(np.mean(fracs)),
                pal4vst_max=float(np.max(fracs)),
                qwen_melt_pyes=qwen.video_score(gen)))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ref.vid}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(refs)}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
