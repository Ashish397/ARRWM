"""Cross-model style-shift (DINOv2 ViT-S/14 drift), 6s-HORIZON-CAPPED, on all 13 fleet models
x 256 scenes via fleet_common. Supersedes fleet_dino.csv (whose end window was the last 16 frames
of the full video, which overshoots 6s for the long external clips). anchor = first min(16,ctx)
real frames; end = last 16 generated frames up to the 6s horizon; drift = 1 - cos(mean L2 embeds).
Writes out/fleet_style_6s.csv (resumable)."""
import os
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fleet_style_6s.csv")
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
    idx = list(fc.fleet_index())
    done = set(); rows = []
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT); done = set(zip(prev.scene, prev.model)); rows = prev.to_dict("records")
    for k, (scene, model) in enumerate(idx):
        if (scene, model) in done:
            continue
        try:
            n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
            end = min(n - 1, ctx + int(round(6.0 * fps)))          # 6s horizon cap
            a_idx = list(range(0, max(1, min(16, ctx))))
            e_idx = list(range(max(ctx, end - 15), end + 1))       # last 16 frames up to 6s
            fr = fc.frames_at(scene, model, a_idx + e_idx)
            if len(fr) < len(a_idx) + 1:
                continue
            anchor, endw = fr[:len(a_idx)], fr[len(a_idx):]
            drift = float(1 - (embed(anchor) @ embed(endw)))
            rows.append(dict(scene=scene, model=model, dino_drift=round(drift, 4)))
        except Exception as e:
            print("skip", model, scene, str(e)[:50], flush=True); continue
        if (k + 1) % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[{k+1}/{len(idx)}]", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
