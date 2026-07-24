"""Quick style/haze checker: computes only the 3 style-z components
(MUSIQ-drift, VGG-Gram, MS-SWD) + haze-specific stats (dark-channel drift,
brightness/median drift) for given grid videos. For blind-testing predictions
on unseen grids.

Usage: ./venv/bin/python style_quick.py r03_BR r08_BL ...   (grid names)
"""
import json, os, sys
import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "MS-SWD"))
sys.path.insert(0, HERE)
DEV = "cuda"
W = 16
VARIANTS = ["pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
GRID_SRC = "/home/ashish/ARRWM/grids/grids_A/A"
POS = {"pca8": (0, 0), "pca4": (832, 0), "pca2": (1664, 0), "16node": (2496, 0),
       "4node": (0, 480), "noatok": (832, 480), "noadaln": (1664, 480)}


def read_grid_tiles(grid_name):
    path = os.path.join(GRID_SRC, f"{grid_name}_grid.mp4")
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack(frames)
    return {v: frames[:, y + 32 : y + 480, x : x + 832] for v, (x, y) in POS.items()}


def to01(f):
    return torch.from_numpy(np.ascontiguousarray(f)).permute(0, 3, 1, 2).float().div(255.0)


def main():
    grids = sys.argv[1:]
    import pyiqa
    musiq = pyiqa.create_metric("musiq-spaq", device=DEV)
    from style_shift import VGGStyle, gram_distance
    vgg = VGGStyle().to(DEV)
    from MS_SWD import MS_SWD
    msswd = MS_SWD(num_scale=5, num_proj=128).to(DEV)
    sp = json.load(open(os.path.join(HERE, "style_stats.json")))

    def grams(fr):
        x = to01(fr).to(DEV)
        outs = [vgg(x[i:i + 8]) for i in range(0, len(x), 8)]
        return [torch.cat([o[l] for o in outs]) for l in range(len(outs[0]))]

    for g in grids:
        tiles = read_grid_tiles(g)
        print(f"\n=== {g} ===")
        for v in VARIANTS:
            fr = tiles[v]
            start, end = fr[:W], fr[-W:]
            with torch.no_grad():
                md = float(musiq(to01(start).to(DEV)).mean() - musiq(to01(end).to(DEV)).mean())
                gd = gram_distance(grams(start[::2]), grams(end[::2]))
                sw = float(msswd(to01(start[::2]).to(DEV), to01(end[::2]).to(DEV)).mean())
            z = ((md - sp["ss_musiq_drift"]["mean"]) / sp["ss_musiq_drift"]["std"]
                 + (gd - sp["ss_gram_dist"]["mean"]) / sp["ss_gram_dist"]["std"]
                 + (sw - sp["ss_msswd"]["mean"]) / sp["ss_msswd"]["std"])
            # haze-specific: dark-channel + brightness drift
            def dch(fr2):
                return np.mean([cv2.erode(f.min(-1), np.ones((15, 15), np.uint8)).mean() for f in fr2[::4]])
            haze_d = dch(end) - dch(start)
            bright_d = end.astype(np.float32).mean() - start.astype(np.float32).mean()
            flag = z > sp["_threshold"]
            hazy = haze_d > 8 and bright_d > -5  # brightening veil, not blackening
            print(f"  {v:8s} style_z={z:+.2f} {'FLAG' if flag else '    '}  haze_d={haze_d:+.1f} bright_d={bright_d:+.1f}  -> {'HAZE' if (flag and hazy) else ('SHIFT(dark/other)' if flag else 'clean')}")


if __name__ == "__main__":
    main()
