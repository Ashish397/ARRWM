"""Detect WHERE and WHEN the rendering style of a video shifts.

Two complementary detectors, both per-frame:
  1. VGG16 Gram-matrix style features (relu1_2/relu2_2/relu3_3) -- the
     classic neural-style-transfer style representation. Texture/palette
     shifts move the Gram vector even when content is similar.
  2. CLIP (laion ViT-B/32) zero-shot style classification against a small
     prompt bank (photoreal / blocky-game / cartoon-3D / flat-untextured),
     plus the raw CLIP image embedding.

Changepoints: for each frame t, cosine distance between the mean feature
over the preceding vs following window (w frames). Peaks of that score
(scipy find_peaks) = style shifts. Reported for Gram and CLIP separately.

Outputs (to SS_OUT):
  <stem>_style_timeline.png   3-panel figure
  <stem>_style_scores.csv     per-frame scores
  <stem>_style_feats.npz      all features (replot without recompute)

Usage: python utils/style_shift_detect.py <video.mp4> [more.mp4 ...]
Env: SS_OUT (default analysis/style_shift), SS_STRIDE (default 2),
     SS_WIN (changepoint half-window in sampled frames, default 16).
"""
import os, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
OUT = os.environ.get("SS_OUT", f"{ARR}/analysis/style_shift")
STRIDE = int(os.environ.get("SS_STRIDE", "2"))
WIN = int(os.environ.get("SS_WIN", "16"))
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

STYLE_PROMPTS = {
    "photoreal": "a photorealistic photo of a real street taken by a real camera",
    "blocky game": "a screenshot of a blocky video game with large brick and plank textures, like Minecraft",
    "cartoon 3D": "a 3D rendered cartoon game environment with clean stylized textures",
    "flat/untextured": "a flat gray untextured surface filling the whole frame",
}


def read_frames(path):
    # PyAV, single-threaded decode: cv2's ffmpeg backend intermittently fails
    # to build its swscale conversion context on the login node.
    import av
    cont = av.open(path)
    vs = cont.streams.video[0]
    vs.thread_type = "NONE"
    fps = float(vs.average_rate or 25.0)
    frames, idxs = [], []
    for i, fr in enumerate(cont.decode(vs)):
        if i % STRIDE == 0:
            frames.append(fr.to_ndarray(format="rgb24"))
            idxs.append(i)
    cont.close()
    return frames, np.array(idxs) / fps, fps


def vgg_gram_feats(frames, batch=16):
    from torchvision.models import vgg16, VGG16_Weights
    net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
    taps = {3: "relu1_2", 8: "relu2_2", 15: "relu3_3"}
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    for s in range(0, len(frames), batch):
        x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255 for f in frames[s:s + batch]])
        x = TF.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        grams = []
        for li, layer in enumerate(net):
            x = layer(x)
            if li in taps:
                B, C, H, W = x.shape
                fm = x.reshape(B, C, H * W)
                g = torch.bmm(fm, fm.transpose(1, 2)) / (C * H * W)
                iu = torch.triu_indices(C, C)
                grams.append(torch.log1p(g[:, iu[0], iu[1]].abs()) * torch.sign(g[:, iu[0], iu[1]]))
            if li >= max(taps):
                break
        feats.append(torch.cat(grams, 1))
        print(f"[vgg] {min(s + batch, len(frames))}/{len(frames)}", flush=True)
    F = torch.cat(feats)
    return TF.normalize(F, dim=1).numpy()


def clip_feats(frames, batch=32):
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    txt = TF.normalize(model.encode_text(tok(list(STYLE_PROMPTS.values()))), dim=1)
    from PIL import Image
    embs = []
    for s in range(0, len(frames), batch):
        x = torch.stack([pre(Image.fromarray(f)) for f in frames[s:s + batch]])
        embs.append(TF.normalize(model.encode_image(x), dim=1))
        print(f"[clip] {min(s + batch, len(frames))}/{len(frames)}", flush=True)
    E = torch.cat(embs)
    probs = (100.0 * E @ txt.T).softmax(-1).numpy()
    return E.numpy(), probs


def changepoint_score(F, w=WIN):
    """Cosine distance between mean feature of the w frames before vs after t."""
    n = len(F)
    sc = np.zeros(n)
    for t in range(1, n):
        a = F[max(0, t - w):t].mean(0)
        b = F[t:min(n, t + w)].mean(0)
        sc[t] = 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return sc


def detect(video):
    stem = os.path.splitext(os.path.basename(video))[0]
    frames, tsec, fps = read_frames(video)
    print(f"[{stem}] {len(frames)} sampled frames ({tsec[-1]:.1f}s @ {fps:.0f}fps, stride {STRIDE})", flush=True)

    G = vgg_gram_feats(frames)
    E, P = clip_feats(frames)

    sc_g = changepoint_score(G)
    sc_c = changepoint_score(E)
    # distance of every frame's style to the first / last second of the clip
    ref0 = G[:max(3, int(fps / STRIDE))].mean(0)
    ref1 = G[-max(3, int(fps / STRIDE)):].mean(0)
    d0 = 1 - G @ ref0 / (np.linalg.norm(ref0) + 1e-9)
    d1 = 1 - G @ ref1 / (np.linalg.norm(ref1) + 1e-9)

    def peaks(sc):
        # prominence-only: an absolute height floor misses the first shift when
        # the clip spends most of its length far from the seed style
        pk, _ = find_peaks(sc, prominence=1.2 * sc.std(), distance=max(3, WIN // 2))
        return pk

    pk_g, pk_c = peaks(sc_g), peaks(sc_c)

    np.savez_compressed(f"{OUT}/{stem}_style_feats.npz",
                        gram=G.astype(np.float16), clip=E.astype(np.float16),
                        clip_probs=P, tsec=tsec, sc_gram=sc_g, sc_clip=sc_c,
                        d_first=d0, d_last=d1, pk_gram=pk_g, pk_clip=pk_c,
                        prompts=np.array(list(STYLE_PROMPTS.values())))
    import csv
    with open(f"{OUT}/{stem}_style_scores.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["tsec", "sc_gram", "sc_clip", "d_first", "d_last"] + list(STYLE_PROMPTS))
        for i in range(len(tsec)):
            wcsv.writerow([f"{tsec[i]:.2f}", f"{sc_g[i]:.5f}", f"{sc_c[i]:.5f}",
                           f"{d0[i]:.5f}", f"{d1[i]:.5f}"] + [f"{p:.4f}" for p in P[i]])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax = axes[0]
    for k, name in enumerate(STYLE_PROMPTS):
        ax.plot(tsec, P[:, k], lw=1.8, label=name)
    ax.set_ylabel("CLIP zero-shot style prob")
    ax.set_title(f"{stem} — CLIP zero-shot style classification per frame")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)

    ax = axes[1]
    ax.plot(tsec, sc_g, lw=1.6, color="#d62728", label="VGG Gram (texture/style)")
    ax.plot(tsec, sc_c, lw=1.6, color="#1f77b4", label="CLIP embedding (semantic)")
    for p in pk_g:
        ax.axvline(tsec[p], color="#d62728", ls="--", alpha=0.6)
        ax.text(tsec[p], ax.get_ylim()[1] * 0.95, f"{tsec[p]:.1f}s", color="#d62728",
                fontsize=8, ha="center", va="top")
    for p in pk_c:
        ax.axvline(tsec[p], color="#1f77b4", ls=":", alpha=0.6)
    ax.set_ylabel("style-change score\n(pre vs post window cos dist)")
    ax.set_title("Changepoint score — dashed red = Gram shifts, dotted blue = CLIP shifts")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(tsec, d0, lw=1.6, color="#2ca02c", label="style distance to FIRST second")
    ax.plot(tsec, d1, lw=1.6, color="#9467bd", label="style distance to LAST second")
    ax.set_ylabel("Gram cos distance"); ax.set_xlabel("time (s)")
    ax.set_title("Drift away from the seed style / toward the final ('favourite') style")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{stem}_style_timeline.png", dpi=130)
    plt.close(fig)

    print(f"\n[{stem}] Gram-style shifts at: " + ", ".join(f"{tsec[p]:.1f}s" for p in pk_g))
    print(f"[{stem}] CLIP shifts at:       " + ", ".join(f"{tsec[p]:.1f}s" for p in pk_c))
    dom = P.argmax(1); names = list(STYLE_PROMPTS)
    segs, s0 = [], 0
    for i in range(1, len(dom) + 1):
        if i == len(dom) or dom[i] != dom[s0]:
            segs.append((tsec[s0], tsec[min(i, len(dom) - 1)], names[dom[s0]])); s0 = i
    print(f"[{stem}] dominant CLIP style segments:")
    for a, b, nm in segs:
        if b - a >= 0.4:
            print(f"    {a:5.1f}s – {b:5.1f}s : {nm}")
    print(f"[{stem}] saved {OUT}/{stem}_style_timeline.png", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        detect(v)
