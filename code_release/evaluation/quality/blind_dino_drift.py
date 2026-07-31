"""DINOv2 cosine-drift relocation competitor on blind100 (the ~0.85 method in the
panel). Per video: DINOv2 embedding of a real-context anchor window vs the end window;
drift = 1 - cosine. High drift = scene identity changed = relocated.
Writes out/blind_dino_drift.csv."""
import os
import numpy as np, torch, torch.nn.functional as F, pandas as pd, imageio
import blind100_common as bc

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_dino_drift.csv")
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
    rows = []
    for r in bc.refs():
        rd = imageio.get_reader(r["path"]); fr = [np.asarray(f) for f in rd]; rd.close()
        n = len(fr); ctx = r["ctx"]
        anchor = fr[:max(1, min(16, ctx))]
        end = fr[-16:]
        drift = float(1 - (embed(anchor) @ embed(end)))
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"],
                         scene=r["scene"], dino_drift=drift))
        print(f"[dino] {r['blind_id']} {r['vid']:20s} drift={drift:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[dino] wrote {OUT}")


if __name__ == "__main__":
    main()
