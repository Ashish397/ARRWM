"""FVD of each model's no-op clips vs the real clips (cd-fvd, i3d).
video_numpy expects a .npy PATH (B,T,H,W,C), so we save arrays first."""
import os, glob, tempfile
import numpy as np, cv2, imageio, pandas as pd
from cdfvd import fvd

DIR = "/home/ashish/stationary_evaluation"
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "out", "_fvd_tmp")
os.makedirs(TMP, exist_ok=True)
CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
NF, SIZE = 16, (128, 128)


def ctx_of(m): return CTX.get(m, 12)


def frames(path, ctx):
    r = imageio.get_reader(path); n = r.count_frames()
    idx = np.linspace(min(ctx, n - 2), n - 1, NF).round().astype(int)
    fr = [cv2.resize(np.asarray(r.get_data(int(i))), SIZE) for i in idx]
    r.close(); return np.stack(fr)


def stack_npy(m):
    scenes = sorted({os.path.basename(f).split("_r")[1][:-4] for f in glob.glob(f"{DIR}/real_r*.mp4")})
    vids = []
    for s in scenes:
        if m == "freeze":                       # degenerate: last real ctx frame repeated
            p = f"{DIR}/real_r{s}.mp4"
            if not os.path.exists(p):
                continue
            r = imageio.get_reader(p); f11 = cv2.resize(np.asarray(r.get_data(11)), SIZE); r.close()
            vids.append(np.repeat(f11[None], NF, 0))
            continue
        p = f"{DIR}/{m}_r{s}.mp4"
        if os.path.exists(p):
            vids.append(frames(p, ctx_of(m)))
    arr = np.stack(vids).astype(np.uint8)
    path = os.path.join(TMP, f"fvd_{m}.npy"); np.save(path, arr); return path


def main():
    ev = fvd.cdfvd("i3d", device="cuda")
    ev.compute_real_stats(ev.load_videos(stack_npy("real"), data_type="video_numpy",
                                         resolution=128, sequence_length=NF))
    models = ["freeze", "pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln",
              "minwm", "astra", "matrixgame", "worldcam", "worldplay", "yume"]
    rows = []
    for m in models:
        ev.empty_fake_stats()
        ev.compute_fake_stats(ev.load_videos(stack_npy(m), data_type="video_numpy",
                                             resolution=128, sequence_length=NF))
        v = float(ev.compute_fvd_from_stats())
        rows.append((m, round(v, 1))); print(f"  {m:12s} FVD={v:.1f}", flush=True)
    pd.DataFrame(rows, columns=["model", "fvd"]).sort_values("fvd").to_csv(
        os.path.join(HERE, "out", "stationary_fvd.csv"), index=False)
    print("wrote out/stationary_fvd.csv")


if __name__ == "__main__":
    main()
