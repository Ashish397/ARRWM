"""DINOv2 cosine-drift (style/scene candidate) on the full 256x13 fleet.
Anchor = first min(16,ctx) real-context frames vs end = last 16 generated frames;
drift = 1 - cosine of mean DINOv2 embeddings. Writes out/fleet_dino.csv."""
import os
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import fleet_common as fc

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fleet_dino.csv")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=224).to(DEV).eval()

    def embed(frames):
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255.0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - MEAN) / STD).to(DEV)
        with torch.no_grad():
            e = dino(x)
        return F.normalize(e, dim=-1).mean(0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    idx = fc.fleet_index()
    done = set(); rows = []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); done = set(zip(prev.scene, prev.model)); rows = prev.to_dict("records")
    for k, (scene, model) in enumerate(idx):
        if (scene, model) in done:
            continue
        try:
            n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
            a_idx = list(range(0, max(1, min(16, ctx))))
            e_idx = list(range(max(0, n - 16), n))
            fr = fc.frames_at(scene, model, a_idx + e_idx)
            if not fr:
                continue
            anchor, end = fr[:len(a_idx)], fr[len(a_idx):]
            drift = float(1 - (embed(anchor) @ embed(end)))
            rows.append(dict(scene=scene, model=model, dino_drift=drift))
        except Exception as e:
            print(f"[fdino] {scene} {model} FAIL {str(e)[:70]}", flush=True); continue
        if (k + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[fdino] {k+1}/{len(idx)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[fdino] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
