"""Rerun the deployed style-shift instrument (DINOv2 ViT-S/14 drift) on the 7 ablation variants
across all 256 fleet scenes, via fleet_common (de-tiles ours_*). anchor = first min(16,ctx) real
frames; end = last 16 generated frames within the 6s horizon; drift = 1 - cos(mean L2 embeddings).
Writes out/ablation_style.csv."""
import os
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "ablation_style.csv")
ABL = ["ours_pca8", "ours_pca4", "ours_pca2", "ours_16node", "ours_4node", "ours_noatok", "ours_noadaln"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).cuda().eval()

    def embed(frames):
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255.0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - MEAN) / STD).cuda()
        with torch.no_grad():
            e = dino(x)
        return F.normalize(e, dim=-1).mean(0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    scenes = sorted(set(s for s, _ in fc.fleet_index()))
    rows = []
    for k, scene in enumerate(scenes):
        for m in ABL:
            try:
                n, fps = fc.meta(scene, m); ctx = fc.ctx_of(m)
                end = min(n - 1, ctx + int(round(6.0 * fps)))          # 6s horizon cap
                a_idx = list(range(0, max(1, min(16, ctx))))
                e_idx = list(range(max(ctx, end - 15), end + 1))       # last 16 frames up to 6s
                fr = fc.frames_at(scene, m, a_idx + e_idx)
                if len(fr) < len(a_idx) + 1:
                    continue
                anchor, endw = fr[:len(a_idx)], fr[len(a_idx):]
                drift = float(1 - (embed(anchor) @ embed(endw)))
                rows.append(dict(scene=scene, model=m, dino_drift=round(drift, 4)))
            except Exception as e:
                print("skip", m, scene, str(e)[:50], flush=True); continue
        if (k + 1) % 32 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"{k+1}/{len(scenes)} scenes", flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT, index=False)
    print("wrote", OUT, len(d))
    print("\nStyle-shift rate (dino_drift>0.72) per variant:")
    d["v"] = d.model.str.replace("ours_", "")
    for v in ["16node", "pca8", "pca4", "pca2", "4node", "noatok", "noadaln"]:
        s = d[d.v == v].dino_drift
        print(f"  {v:9s} n={len(s):3d}  {(s>0.72).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
