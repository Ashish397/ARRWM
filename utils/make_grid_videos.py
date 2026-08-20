"""Tiled VIDEO grids (mp4, not contact sheets): all variants playing
side-by-side, synchronized, labels burned in.

Env: GV_DIR (default L), GV_STEPS (csv, default 200,400,600,800,1000,1200),
GV_FPS (default 16), GV_SEED (default 0).
Output: analysis/eval_final/flow_viz/gridvid_<DIR>_step<step>.mp4
Missing variants render as a labeled black tile.
"""
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
TEACH = f"{ARR}/logs/eval_final/A/pca8_8node/control_test/step05000_r08_{{d}}_raw.mp4"
VARIANTS = [("teacher", None), ("rollmse", "roll"), ("rollkl", "rollkl"),
            ("rollkl9", "rollkl9"), ("rollklts", "rollklts"),
            ("rollklvz", "rollklvz"), ("rollklrep2", "rollklrep2"),
            ("rollklaw", "rollklaw")]
# GV_VARIANTS="name:tag,name:tag,..." overrides the roster (empty tag =
# teacher reference tile). GV_TAGPREF changes the step-mode probe prefix
# (default v100). GV_OUT overrides the long-mode output path.
if os.environ.get("GV_VARIANTS"):
    VARIANTS = [(n, t or None) for n, t in
                (e.split(":", 1) for e in os.environ["GV_VARIANTS"].split(","))]
TAGPREF = os.environ.get("GV_TAGPREF", "v100")
D = os.environ.get("GV_DIR", "L")
STEPS = [int(x) for x in os.environ.get(
    "GV_STEPS", "200,400,600,800,1000,1200").split(",")]
FPS = int(os.environ.get("GV_FPS", "16"))
SEED = os.environ.get("GV_SEED", "0")
TILE_W, TILE_H = 416, 240          # per-tile size (2x4 grid -> 1664x480)


def load(p):
    if not p or not os.path.exists(p):
        return None
    r = imageio.get_reader(p, format="ffmpeg")
    fr = [np.asarray(Image.fromarray(f).resize((TILE_W, TILE_H)))
          for f in r]
    r.close()
    return fr


def label(img, text):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 7 * len(text) + 8, 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    if os.environ.get("GV_LONG"):
        _pref = os.environ["GV_LONG"]
        if _pref in ("", "1"):
            _pref = "long36"
        # LONG-rollout mode: one grid over .motion_check/long36_<tag>/,
        # no step dimension. Teacher tile freezes on its last frame once
        # its (shorter) reference video ends.
        tiles = []
        for name, tag in VARIANTS:
            if tag is None:
                fr = load(TEACH.format(d=D)); nm = name
            else:
                fr = load(f"{FV}/.motion_check/{_pref}_{tag}/r08_{D}_s{SEED}.mp4")
                nm = f"{name} ({_pref})"
            tiles.append((nm, fr))
        n = max((len(fr) for _, fr in tiles if fr), default=0)
        out = os.environ.get("GV_OUT") or f"{FV}/gridvid_{D}_{_pref}.mp4"
        w = imageio.get_writer(out, fps=FPS, quality=7, macro_block_size=16)
        blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        for k in range(n):
            row_imgs = []
            for name, fr in tiles:
                img = fr[min(k, len(fr) - 1)] if fr else blank
                row_imgs.append(label(img.copy(),
                                      name if fr else f"{name} (missing)"))
            while len(row_imgs) > 4 and len(row_imgs) % 4:
                row_imgs.append(blank.copy())
            rows = [np.hstack(row_imgs[i:i + 4])
                    for i in range(0, len(row_imgs), 4)]
            w.append_data(np.vstack(rows))
        w.close()
        print(f"[gv] wrote {out} ({n} frames @ {FPS}fps)")
        return
    for step in STEPS:
        tiles = []
        for name, tag in VARIANTS:
            if tag is None:
                fr = load(TEACH.format(d=D))
                nm = name
            else:
                fr = load(f"{FV}/.motion_check/{TAGPREF}_{tag}_s{step:07d}/"
                          f"r08_{D}_s{SEED}.mp4")
                nm = f"{name}@{step}"
            tiles.append((nm, fr))
        n = max((len(fr) for _, fr in tiles if fr), default=0)
        if n == 0:
            print(f"[gv] step {step}: nothing to render"); continue
        # Non-default tag prefixes get their own namespace so a v10k grid
        # can never overwrite the v100 campaign's gridvid_L_step*.mp4 files.
        _ns = "" if TAGPREF == "v100" else f"{TAGPREF}_"
        out = f"{FV}/gridvid_{D}_{_ns}step{step:04d}.mp4"
        w = imageio.get_writer(out, fps=FPS, quality=7,
                               macro_block_size=16)
        blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        for k in range(n):
            row_imgs = []
            for name, fr in tiles:
                img = fr[min(k, len(fr) - 1)] if fr else blank
                row_imgs.append(label(img.copy(),
                                      name if fr else f"{name} (not yet)"))
            while len(row_imgs) > 4 and len(row_imgs) % 4:
                row_imgs.append(blank.copy())
            rows = [np.hstack(row_imgs[i:i + 4])
                    for i in range(0, len(row_imgs), 4)]
            w.append_data(np.vstack(rows))
        w.close()
        print(f"[gv] wrote {out} ({n} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
