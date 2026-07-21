"""Metric-suite pass A: PAL4VST (warp) + Qwen P(Yes) (surface) + DINO spawn
(center-novelty + jump) for every phase-A eval video of the given runs.
One frame-read per video. VideoLISA runs separately (env conflict).

Env: SA_RUNS colon list, SA_OUT csv. ~20 s/video.
"""
import os
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn.functional as F
from PIL import Image

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
RUNS = os.environ.get("SA_RUNS", "pca8_8node").split(":")
OUT = os.environ.get("SA_OUT", f"{ARR}/analysis/eval_final/suiteA.csv")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]

_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
COMPARATORS = ("minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra")


def vid_path(run, wi, d):
    # minwm disk labels are yaw-sign-flipped (SIFT-verified); swap to TRUE direction
    if run == "minwm":
        return f"{ARR}/logs/eval_final/A_minwm/minwm_r{wi:02d}_{_MWSW.get(d, d)}.mp4"
    if run in COMPARATORS:
        return f"{ARR}/logs/eval_final/A_{run}/{run}_r{wi:02d}_{d}.mp4"
    return f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_r{wi:02d}_{d}_raw.mp4"

# ---- PAL4VST ----
pal = torch.jit.load(f"{ARR}/third_party/PAL4VST/deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt").to(DEV).eval()
PM = torch.tensor([123.675, 116.28, 103.53], device=DEV).view(1, 3, 1, 1)
PS = torch.tensor([58.395, 57.12, 57.375], device=DEV).view(1, 3, 1, 1)


@torch.no_grad()
def pal_frame(img):
    t = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV).float()
    t = t[..., int(t.shape[-2] * 0.15):, :]
    H0, W0 = t.shape[-2:]
    fr = []
    for sl in (slice(0, W0 // 2), slice(W0 // 2, W0)):
        x = F.interpolate(t[..., :, sl], size=(512, 512), mode="bilinear", align_corners=False)
        out = pal((x - PM) / PS)
        if isinstance(out, (list, tuple)):
            out = out[0]
        mask = out.argmax(1) if (out.dim() == 4 and out.shape[1] > 1) else (out.squeeze(1) > 0.5).long()
        fr.append(float(mask.float().mean().item()))
    return float(np.mean(fr))

# ---- Qwen P(Yes) ----
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device_map=DEV)
qproc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
YES = [qproc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")]
NO = [qproc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")]
MELT_Q = ("Are any buildings, walls or structures in this image melted, warped, smeared "
          "or geometrically impossible, like a corrupted AI-generated image? Answer only Yes or No.")


@torch.no_grad()
def qwen_frame(img):
    im = Image.fromarray(img)
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": MELT_Q}]}]
    text = qproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = qproc(text=[text], images=[im], return_tensors="pt").to(DEV)
    logits = qwen(**inputs).logits[0, -1]
    return float(torch.sigmoid(torch.logsumexp(logits[YES], 0) - torch.logsumexp(logits[NO], 0)).item())

# ---- DINO spawn ----
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
DM = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DS = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
PATCH, STRIDE = 320, 160


def unfold(t):
    return F.unfold(t, PATCH, stride=STRIDE).transpose(1, 2).reshape(-1, 3, PATCH, PATCH)


@torch.no_grad()
def dfeat(ps):
    v = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
    e = dino((v - DM) / DS)
    return (e / e.norm(dim=-1, keepdim=True)).float()


def aug(x):
    outs = [x]
    for k in (9, 21, 41):
        w = torch.ones(3, 1, k, k, device=DEV) / (k * k)
        outs.append(F.conv2d(x, w, padding=k // 2, groups=3))
    outs += [(x * 0.7).clamp(0, 1), (x * 1.3).clamp(0, 1)]
    return outs


@torch.no_grad()
def spawn_scores(frames):
    t = lambda f: torch.from_numpy(f).permute(2, 0, 1)[None].to(DEV).float() / 255.0
    bank = torch.cat([dfeat(unfold(v)) for i in (0, 3, 6, 9) for v in aug(t(frames[i]))], 0)
    traj, ctr = [], []
    for i in range(12, len(frames) - 1, 6):
        d = (1 - dfeat(unfold(t(frames[i]))) @ bank.T).min(1).values
        traj.append(float(np.percentile(d.cpu().numpy(), 95)))
        H, W = frames[i].shape[:2]
        cf = frames[i][H // 4:3 * H // 4, W // 3:2 * W // 3]
        ce = dfeat(F.interpolate(t(cf), size=224, mode="bilinear", align_corners=False))
        ctr.append(float((1 - ce @ bank.T).min().item()))
    jumps = np.diff(traj) if len(traj) > 1 else [0.0]
    return float(max(jumps)), float(max(ctr))


def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.run, prev.window, prev["dir"]))
    mode = "a" if done else "w"
    f = open(OUT, mode)
    if mode == "w":
        f.write("run,window,dir,pal_mean,qwen_top4,spawn_jump,center_nov\n")
    for run in RUNS:
        for wi in range(32):
            for d in DIRS:
                if (run, f"r{wi:02d}", d) in done:
                    continue
                path = vid_path(run, wi, d)
                if not os.path.exists(path):
                    continue
                try:
                    r = imageio.get_reader(path)
                    frames = [np.asarray(x) for x in r]
                    r.close()
                    gen = frames[13::8]
                    palv = float(np.mean([pal_frame(x) for x in gen]))
                    qv = sorted(qwen_frame(x) for x in frames[13::12])[-4:]
                    qv = float(np.mean(qv))
                    sj, cn = spawn_scores(frames)
                    f.write(f"{run},r{wi:02d},{d},{palv:.4f},{qv:.4f},{sj:.4f},{cn:.4f}\n")
                    f.flush()
                except Exception as e:
                    print(f"[skip] {run} r{wi:02d} {d}: {str(e)[:80]}", flush=True)
            print(f"[suiteA] {run} r{wi:02d} done", flush=True)
    f.close()
    print("[suiteA] DONE")


if __name__ == "__main__":
    main()
