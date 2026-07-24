"""Matched-scene failure comparison for the appendix.

Every model ran the SAME 32 held-out seed windows under the SAME 8 commands,
so for a chosen (window, direction) we render one filmstrip row per model,
stacked, to show how the identical scene+command degrades differently. Far
stronger than unmatched random failures.

Env: FC_WIN (window index), FC_DIR (F/B/R/L/...), FC_OUT.
"""
import os, glob
import numpy as np
import av
from PIL import Image, ImageDraw

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
WIN = int(os.environ.get("FC_WIN", "6"))
DIR = os.environ.get("FC_DIR", "B")
OUT = os.environ.get("FC_OUT", f"{ARR}/analysis/reels")
PICKS = [6, 30, 54, 78, 107]

# (label, path-template). ours first, then ablations, then baselines.
ROWS = [
    ("Ours (batch 32)", "logs/eval_final/A/pca8_8node/control_test/step05000_r{w:02d}_{d}_raw.mp4"),
    ("Ours (batch 64)", "logs/eval_final/A/16node/control_test/step05000_r{w:02d}_{d}_raw.mp4"),
    ("no AdaLN",        "logs/eval_final/A/noadaln/control_test/step05000_r{w:02d}_{d}_raw.mp4"),
    ("pca2",            "logs/eval_final/A/pca2/control_test/step05000_r{w:02d}_{d}_raw.mp4"),
    ("Matrix-Game",     "logs/eval_final/A_matrixgame/matrixgame_r{w:02d}_{d}.mp4"),
    ("WorldCam",        "logs/eval_final/A_worldcam/worldcam_r{w:02d}_{d}.mp4"),
    ("WorldPlay",       "logs/eval_final/A_worldplay/worldplay_r{w:02d}_{d}.mp4"),
    ("Yume",            "logs/eval_final/A_yume/yume_r{w:02d}_{d}.mp4"),
]


def strip(path, label, tw=200):
    if not os.path.exists(path):
        return None
    c = av.open(path)
    frames = [np.asarray(f.to_image()) for f in c.decode(c.streams.video[0])]
    n = len(frames)
    idxs = [min(i, n - 1) for i in PICKS]
    tiles = []
    for i in idxs:
        im = Image.fromarray(frames[i]).resize((tw, int(tw * frames[i].shape[0] / frames[i].shape[1])))
        tiles.append(np.asarray(im))
    h = tiles[0].shape[0]
    pad = 3
    row = np.full((h, sum(t.shape[1] for t in tiles) + pad * (len(tiles) - 1) + 150, 3), 255, np.uint8)
    x = 150
    for t in tiles:
        row[:, x:x + t.shape[1]] = t
        x += t.shape[1] + pad
    im = Image.fromarray(row)
    ImageDraw.Draw(im).text((6, h // 2 - 6), label, fill=(0, 0, 0))
    return np.asarray(im)


def main():
    rows = []
    for label, tmpl in ROWS:
        r = strip(f"{ARR}/{tmpl.format(w=WIN, d=DIR)}", label)
        if r is not None:
            rows.append(r)
    W = max(r.shape[1] for r in rows)
    canvas = []
    for r in rows:
        if r.shape[1] < W:
            pad = np.full((r.shape[0], W - r.shape[1], 3), 255, np.uint8)
            r = np.concatenate([r, pad], 1)
        canvas.append(r)
        canvas.append(np.full((4, W, 3), 200, np.uint8))
    out = np.concatenate(canvas, 0)
    fn = f"{OUT}/failure_compare_w{WIN:02d}_{DIR}.png"
    Image.fromarray(out).save(fn)
    print(f"[fc] {len(rows)} models -> {fn}")


if __name__ == "__main__":
    main()
