"""Build 6-model comparison grids: for each seed window+direction (Phase A) and
each window (Phase B), tile all 6 models' videos into one labeled 3x2 grid mp4 so
the same seed driven the same way is directly comparable across models.

Out: analysis/eval_final/grids/A/rNN_DIR_grid.mp4 , grids/B/rNN_static_grid.mp4
Env: MG_SHARD/MG_NSHARDS to split across processes.
"""
import os, glob
import numpy as np, imageio
from PIL import Image, ImageDraw
from multiprocessing import Pool

RUNS = os.environ.get("MG_RUNS", "pca8_8node,pca4,pca2,16node,4node,noatok").split(",")
ROWS = 2
COLS = (len(RUNS) + ROWS - 1) // ROWS   # 6 models -> 3x2; 7 -> 4x2 (one blank cell)
BASE = "logs/eval_final"
OUT = "analysis/eval_final/grids"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]


def read(path):
    if not os.path.exists(path):
        return None
    try:
        r = imageio.get_reader(path); fr = [f for f in r]; r.close()
        return np.stack(fr) if fr else None
    except Exception:
        return None


def label(frame, txt):
    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    d.text((5, 3), txt, fill=(0, 0, 0)); d.text((4, 2), txt, fill=(255, 255, 0))
    return np.asarray(im)


def resize(v, H, W):
    return np.stack([np.asarray(Image.fromarray(f).resize((W, H))) for f in v])


def make_grid(args):
    phase, tag, out = args
    if os.path.exists(out):
        return "exists"
    vids = [read(f"{BASE}/{phase}/{r}/control_test/step05000_{tag}_raw.mp4") for r in RUNS]
    present = [v for v in vids if v is not None]
    if not present:
        return f"skip {tag} (no inputs)"
    T = max(v.shape[0] for v in present); H, W = present[0].shape[1:3]
    panels = []
    for run, v in zip(RUNS, vids):
        if v is None:
            v = np.zeros((T, H, W, 3), np.uint8)
        else:
            if v.shape[1:3] != (H, W):
                v = resize(v, H, W)
            if v.shape[0] < T:
                v = np.concatenate([v, np.repeat(v[-1:], T - v.shape[0], 0)], 0)
            elif v.shape[0] > T:
                v = v[:T]
        panels.append(np.stack([label(v[t], run) for t in range(T)]))
    while len(panels) < ROWS * COLS:                     # pad grid with black cells
        panels.append(np.zeros((T, H, W, 3), np.uint8))
    rows = [np.concatenate(panels[r * COLS:(r + 1) * COLS], axis=2) for r in range(ROWS)]
    grid = np.concatenate(rows, axis=1)           # [T, ROWS*H, COLS*W, 3]
    w = imageio.get_writer(out, fps=16, quality=8, macro_block_size=1)
    for t in range(grid.shape[0]):
        w.append_data(grid[t])
    w.close()
    return f"ok {os.path.basename(out)}"


def main():
    os.makedirs(f"{OUT}/A", exist_ok=True); os.makedirs(f"{OUT}/B", exist_ok=True)
    tasks = []
    for wi in range(32):
        for d in DIRS:
            tag = f"r{wi:02d}_{d}"; tasks.append(("A", tag, f"{OUT}/A/{tag}_grid.mp4"))
    for wi in range(64):
        tag = f"r{wi:02d}_static"; tasks.append(("B", tag, f"{OUT}/B/{tag}_grid.mp4"))
    sh = int(os.environ.get("MG_SHARD", "0")); ns = int(os.environ.get("MG_NSHARDS", "1"))
    tasks = tasks[sh::ns]
    print(f"[grids] shard {sh}/{ns}: {len(tasks)} grids", flush=True)
    with Pool(int(os.environ.get("MG_WORKERS", "16"))) as p:
        for i, r in enumerate(p.imap_unordered(make_grid, tasks)):
            if i % 40 == 0:
                print(f"  {i}/{len(tasks)} {r}", flush=True)
    print("[grids] done", flush=True)


if __name__ == "__main__":
    main()
