"""Style-shift instrument (DINOv2 ViT-S/14 embedding drift) on the stationary set.
anchor = first min(16,ctx) real context frames; end = last 16 generated frames within the 6s
horizon; drift = 1 - cos(mean L2-normalized embeddings). Writes out/stat_style.csv."""
import os
import cv2, numpy as np, torch, torch.nn.functional as F, pandas as pd

DIR = "/home/ashish/stationary_evaluation"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stat_style.csv")
MODELS = ["16node", "4node", "pca8", "pca4", "pca2", "noatok", "noadaln",
          "astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"]
CTX = {"16node": 12, "4node": 12, "pca8": 12, "pca4": 12, "pca2": 12, "noatok": 12, "noadaln": 12,
       "astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def frames_at(path, idxs):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 16
    want = {i: k for k, i in enumerate(idxs) if 0 <= i < n}; got = {}
    i = 0
    while len(got) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release(); return [got[i] for i in idxs if i in got], n, fps


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
    rows = []
    for scene in range(32):
        for m in MODELS:
            fp = os.path.join(DIR, f"{m}_r{scene:02d}.mp4")
            if not os.path.exists(fp):
                continue
            cap = cv2.VideoCapture(fp); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 16; cap.release()
            ctx = CTX[m]; na = min(16, ctx)
            end = min(n - 1, ctx + int(round(6.0 * fps)))
            a_idx = list(range(0, na))
            e_idx = list(np.linspace(max(ctx, end - 30), end, 16).round().astype(int))
            fr, _, _ = frames_at(fp, a_idx + e_idx)
            anchor, endw = fr[:len(a_idx)], fr[len(a_idx):]
            if not anchor or not endw:
                continue
            drift = float(1 - (embed(anchor) @ embed(endw)))
            rows.append(dict(model=m, scene=scene, dino_drift=round(drift, 4)))
            print(f"{m}_r{scene:02d} drift={drift:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
