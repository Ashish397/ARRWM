"""Style-shift scan on blind100 (GPU: VGG16 Gram + CLIP).

Per video: compare a REAL-CONTEXT anchor window (first min(16,ctx) real frames)
against the END window (last 16 generated frames):
  ss_gram_dist : VGG16 Gram-matrix style distance (content-insensitive)
  ss_clip_dist : CLIP embedding cosine distance (semantic/style, content-confounded)
Higher = more style drift. Uses each model's context length so short-context externals
(yume/matrixgame/worldplay) anchor on real frames, not generated ones.

Writes out/blind_style_shift.csv.
"""
import os
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import blind100_common as bc
from analysis.testbench_v2.style_shift import VGGStyle, gram_distance, read_frames

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_style_shift.csv")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    vgg = VGGStyle().to(DEV)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_model = clip_model.to(DEV).eval()
    cmean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    cstd = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def clip_embed(frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - cmean) / cstd).to(DEV)
        with torch.no_grad():
            e = clip_model.encode_image(x)
        return F.normalize(e, dim=-1).mean(0)

    def vgg_grams(frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0).to(DEV)
        with torch.no_grad():
            out = [vgg(x[i:i + 8]) for i in range(0, len(x), 8)]
        return [torch.cat([o[l] for o in out]) for l in range(len(out[0]))]

    rows = []
    for r in bc.refs():
        frames = read_frames(r["path"])          # T,H,W,3 RGB
        k = max(1, min(16, r["ctx"]))
        start, end = frames[:k], frames[-16:]
        gram = gram_distance(vgg_grams(start), vgg_grams(end))
        clip = float(1 - (clip_embed(start) @ clip_embed(end)))
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"],
                         scene=r["scene"], ss_gram_dist=gram, ss_clip_dist=clip,
                         n_ctx=k, n_frames=len(frames)))
        print(f"[style] {r['blind_id']} {r['vid']:20s} gram={gram:.4f} clip={clip:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[style] wrote {OUT} ({len(rows)} videos)")


if __name__ == "__main__":
    main()
