"""Style/plausibility-shift instruments on the external baseline fleet vs our good variants.

For each scene: 6 external models (native fps, windows by wall-clock seconds,
resized to 832x448) + our pca8/16node tiles. Per video:
  ss_musiq_drift  MUSIQ(first 1s) - MUSIQ(last 0.9s)
  ss_gram_dist    VGG-Gram distance first-vs-last
  ss_msswd        MS-SWD color distance first-vs-last
  ss_csd_drift    CSD style-embedding drift first-vs-last (content-invariant)
  haze_lap_loss   Laplacian variance loss t=1s -> end
  haze_dc_rise    dark-channel rise t=1s -> end
Writes results_external_style.csv.
"""
import glob, json, os, sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "MS-SWD"))
DEV = "cuda"
BASE_DIR = "/home/ashish/ARRWM/grids/baselines"
MODELS = ["astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"]
OURS = ["pca8", "16node"]
OUT = os.path.join(HERE, "results_external_style.csv")
W, H = 832, 448


def read_windows(path):
    """Return (ctx, base, end) frame stacks resized to 832x448, windows in seconds."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (W, H)))
    cap.release()
    if len(frames) < int(2 * fps):
        return None
    frames = np.stack(frames)
    n1 = max(2, int(round(fps)))            # 1 second
    ctx = frames[:n1]
    base = frames[n1:n1 + max(2, int(round(fps * 0.25)))]
    end = frames[-max(2, int(round(fps * 0.9))):]
    return ctx, base, end


def main():
    scenes = sys.argv[1:] or list(json.load(open(os.path.join(HERE, "gt.json")))["grids"].keys())

    import pyiqa
    musiq = pyiqa.create_metric("musiq-spaq", device=DEV)
    from style_shift import VGGStyle, gram_distance
    vgg = VGGStyle().to(DEV)
    from style_shift2 import load_csd
    csd = load_csd()
    from MS_SWD import MS_SWD
    msswd = MS_SWD(num_scale=5, num_proj=128).to(DEV)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def to01(fr):
        return torch.from_numpy(np.ascontiguousarray(fr)).permute(0, 3, 1, 2).float().div(255.0)

    @torch.no_grad()
    def csd_embed(fr):
        x = to01(fr[:: max(1, len(fr) // 8)])
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - clip_mean) / clip_std).to(DEV)
        return F.normalize(csd(x).float(), dim=-1).mean(0)

    @torch.no_grad()
    def grams(fr):
        x = to01(fr[:: max(1, len(fr) // 8)]).to(DEV)
        outs = [vgg(x[i:i + 8]) for i in range(0, len(x), 8)]
        return [torch.cat([o[l] for o in outs]) for l in range(len(outs[0]))]

    def lap(fr):
        return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in fr[::2]]))

    def dch(fr):
        return float(np.mean([cv2.erode(f.min(-1), np.ones((15, 15), np.uint8)).mean() for f in fr[::2]]))

    rows = []
    for scene in scenes:
        vids = {}
        for m in MODELS:
            fp = os.path.join(BASE_DIR, f"A_{m}", f"{m}_{scene}.mp4")
            if os.path.exists(fp):
                vids[m] = fp
        for v in OURS:
            for tdir in ("tiles", "tiles_new"):
                fp = os.path.join(HERE, tdir, f"{scene}__{v}.mp4")
                if os.path.exists(fp):
                    vids[f"ours_{v}"] = fp
                    break
        for name, fp in vids.items():
            wins = read_windows(fp)
            if wins is None:
                print(f"SKIP {scene} {name} (too short)", flush=True)
                continue
            ctx, base, end = wins
            m = {}
            with torch.no_grad():
                m["ss_musiq_drift"] = float(musiq(to01(ctx[::2]).to(DEV)).mean() - musiq(to01(end[::2]).to(DEV)).mean())
            m["ss_gram_dist"] = gram_distance(grams(ctx), grams(end))
            with torch.no_grad():
                def small(fr):
                    idx = np.linspace(0, len(fr) - 1, 6).round().astype(int)
                    x = to01(fr[idx])
                    return F.interpolate(x, size=(224, 416), mode="bilinear", align_corners=False).to(DEV)
                m["ss_msswd"] = float(msswd(small(ctx), small(end)).mean())
            torch.cuda.empty_cache()
            e_c, e_e = csd_embed(ctx), csd_embed(end)
            m["ss_csd_drift"] = float(1 - F.cosine_similarity(e_c, e_e, dim=0))
            m["haze_lap_loss"] = lap(base) - lap(end)
            m["haze_dc_rise"] = dch(end) - dch(base)
            for k, v2 in m.items():
                rows.append({"scene": scene, "model": name, "metric": k, "value": round(v2, 5)})
            print(f"{scene} {name}: musiq={m['ss_musiq_drift']:+.1f} gram={m['ss_gram_dist']:.3f} msswd={m['ss_msswd']:.2f} csd={m['ss_csd_drift']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        old = old[~old.set_index(["scene", "model", "metric"]).index.isin(df.set_index(["scene", "model", "metric"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
