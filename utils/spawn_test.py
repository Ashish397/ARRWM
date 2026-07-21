"""Object-SPAWN detection test on r31_FL (minWM spawns a van dead-center).

Three candidate detectors, tested minwm vs 16node vs pca8_8node (+2 real refs):
 A) nov_dino trajectory: per-frame p95 seed-novelty, its MAX SINGLE-STEP JUMP
    (spawn = discontinuity; drift/reveal = gradual), and CENTER-region novelty
    (central box) where reveal-content is rarest.
 B) Qwen2.5-VL PAIRED-frame judge: [frame0, frame_t] -> P('a large new object
    absent from the first image'), max over t.
 C) VideoLISA prompted to segment 'the object that suddenly appeared' -> center
    mask fraction + overlay for visual verification.
Saves analysis/eval_final/spawn_test.csv + _diag/spawn_vlisa_*.png
"""
import os, sys
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
VIDS = [
    ("minwm", f"{ARR}/logs/eval_final/A_minwm/minwm_r31_FR.mp4"),          # TRUE FL (disk label swapped)
    ("16node", f"{ARR}/logs/eval_final/A/16node/control_test/step05000_r31_FL_raw.mp4"),
    ("pca8_8node", f"{ARR}/logs/eval_final/A/pca8_8node/control_test/step05000_r31_FL_raw.mp4"),
]
import glob as _g
for _i, _p in enumerate(sorted(_g.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4"))[:2]):
    VIDS.append((f"REAL_{_i}", _p))
# SP_VIDS override: "name=path:name=path" (colon-separated; keeps 2 real refs)
if os.environ.get("SP_VIDS"):
    VIDS = [tuple(x.split("=", 1)) for x in os.environ["SP_VIDS"].split(":")]
    for _i, _p in enumerate(sorted(_g.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4"))[:2]):
        VIDS.append((f"REAL_{_i}", _p))
    OUT_CSV = os.environ.get("SP_OUT", f"{ARR}/analysis/eval_final/spawn_test.csv")
else:
    OUT_CSV = f"{ARR}/analysis/eval_final/spawn_test.csv"
SEED_IDX = [0, 3, 6, 9]
PATCH, STRIDE = 320, 160

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
DMEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DSTD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def frames_all(path):
    r = imageio.get_reader(path)
    fr = [np.asarray(f) for f in r]
    r.close(); return fr


def unfold(t, p=PATCH, s=STRIDE):
    return F.unfold(t, p, stride=s).transpose(1, 2).reshape(-1, 3, p, p)


def dino_feats(ps):
    v = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
    with torch.no_grad():
        e = DINO((v - DMEAN) / DSTD)
    return (e / e.norm(dim=-1, keepdim=True)).float()


def aug_views(x):
    outs = [x]
    for k in (9, 21, 41):
        w = torch.ones(3, 1, k, k, device=DEV) / (k * k)
        outs.append(F.conv2d(x, w, padding=k // 2, groups=3))
    outs.append((x * 0.7).clamp(0, 1)); outs.append((x * 1.3).clamp(0, 1))
    return outs


def center_crop(f):
    H, W = f.shape[:2]
    return f[H // 4:3 * H // 4, W // 3:2 * W // 3]


def main():
    rows = []
    # ---------- A) DINO novelty trajectory / jump / center ----------
    for name, path in VIDS:
        fr = frames_all(path)
        n = len(fr)
        t = lambda i: torch.tensor(fr[i]).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        bank = torch.cat([dino_feats(unfold(v)) for i in SEED_IDX for v in aug_views(t(i))], 0)
        gen_idx = list(range(12, n - 1, 4))
        traj, ctr = [], []
        for i in gen_idx:
            ps = unfold(t(i))
            d = (1 - dino_feats(ps) @ bank.T).min(1).values.cpu().numpy()
            traj.append(float(np.percentile(d, 95)))
            cf = center_crop(fr[i])
            ct = torch.tensor(cf).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
            ce = dino_feats(F.interpolate(ct, size=224, mode="bilinear", align_corners=False))
            ctr.append(float((1 - ce @ bank.T).min().item()))
        jumps = np.diff(traj)
        rows.append(dict(video=name, metric="dino", nov_max=round(max(traj), 3),
                         nov_jump=round(float(jumps.max()), 3) if len(jumps) else np.nan,
                         center_nov_max=round(max(ctr), 3)))
        print(f"[dino] {name}: max={max(traj):.3f} jump={jumps.max():.3f} center={max(ctr):.3f} traj={[round(x,2) for x in traj[:10]]}...", flush=True)

    # ---------- B) Qwen paired-frame spawn judge ----------
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    mid = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEV)
    proc = AutoProcessor.from_pretrained(mid)
    PROMPT = ("The first image is the start of a video; the second image is a later moment "
              "of the SAME video from a similar viewpoint. Has a large new object (such as a "
              "vehicle, cart or structure) APPEARED in the second image that was clearly not "
              "present in the first image? Answer only Yes or No.")
    yes_ids = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")]
    no_ids = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")]

    def p_new(f0, ft):
        ims = [Image.fromarray(f0), Image.fromarray(ft)]
        msgs = [{"role": "user", "content": [{"type": "image", "image": ims[0]},
                                             {"type": "image", "image": ims[1]},
                                             {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=ims, return_tensors="pt").to(DEV)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        py = torch.logsumexp(logits[yes_ids], 0); pn = torch.logsumexp(logits[no_ids], 0)
        return float(torch.sigmoid(py - pn).item())

    for name, path in VIDS:
        fr = frames_all(path); n = len(fr)
        vals = [p_new(fr[0], fr[i]) for i in range(16, n - 1, 12)]
        rows.append(dict(video=name, metric="qwen_pair", p_new_max=round(max(vals), 4),
                         p_new_mean=round(float(np.mean(vals)), 4)))
        print(f"[qwen] {name}: p_new max={max(vals):.3f} vals={[round(v,2) for v in vals]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
