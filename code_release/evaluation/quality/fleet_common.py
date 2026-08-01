"""Resolver for the full 256x13 action-forcing fleet, sourced locally:
  ours_<variant> : de-tiled from grids/grids_A/A/<scene>_grid.mp4 (2x4 grid, GRID LAYOUT)
  <external>     : grids/baselines/A_<model>/<model>_<scene>.mp4
Only the frame indices needed are decoded (grids are 960x3328, so full reads are heavy).
Scene/model list is taken from results_external_vlm.csv so coverage/naming match exactly.
"""
import os
import imageio, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Rollout videos are large and are not distributed with the code. Point
# AF_FLEET_DIR at the directory holding grids_A/ and baselines/.
FLEET_DIR = os.environ.get(
    "AF_FLEET_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), "grids")
)
GRID_DIR = os.path.join(FLEET_DIR, "grids_A", "A")
BASE_DIR = os.path.join(FLEET_DIR, "baselines")

# Scene/model index: the coverage every instrument iterates over.
LABELS_VLM = os.environ.get(
    "AF_FLEET_INDEX", os.path.join(HERE, "results_external_vlm.csv")
)

# Ablations rendered after the grid was built live as standalone per-scene files
# named <scene>__<variant>.mp4, already cropped to the tile field of view. Point
# AF_EXTRA_TILES at a directory of them (colon-separated for several) and they
# resolve exactly like the grid-sourced variants.
EXTRA_TILE_DIRS = [d for d in os.environ.get("AF_EXTRA_TILES", "").split(os.pathsep) if d]


def _extra_path(scene, variant):
    for d in EXTRA_TILE_DIRS:
        p = os.path.join(d, f"{scene}__{variant}.mp4")
        if os.path.exists(p):
            return p
    return None


def extra_variants():
    """Variant names available as standalone files, e.g. {"nocritic"}."""
    names = set()
    for d in EXTRA_TILE_DIRS:
        if os.path.isdir(d):
            names |= {f.split("__", 1)[1][:-4] for f in os.listdir(d) if "__" in f and f.endswith(".mp4")}
    return names


# tile origin (x,y) in the grid for each ours variant (from GRID LAYOUT / fleet_pixscan)
POS = {"pca8": (0, 0), "pca4": (832, 0), "pca2": (1664, 0), "16node": (2496, 0),
       "4node": (0, 480), "noatok": (832, 480), "noadaln": (1664, 480)}
OURS_CTX = 12
EXT_CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}


def fleet_index(include_extra=True):
    """List of (scene, model). Base coverage comes from the VLM label file so it
    matches the reported fleet exactly; standalone ablations are appended."""
    d = pd.read_csv(LABELS_VLM)
    rows = list(d[["scene", "model"]].itertuples(index=False, name=None))
    if include_extra:
        scenes = sorted({s for s, _ in rows})
        for v in sorted(extra_variants()):
            rows += [(s, f"ours_{v}") for s in scenes if _extra_path(s, v)]
    return rows


def ctx_of(model):
    return OURS_CTX if model.startswith("ours_") else EXT_CTX[model]


def _path(scene, model):
    if model.startswith("ours_"):
        variant = model[len("ours_"):]
        if variant not in POS:
            return _extra_path(scene, variant)
        return os.path.join(GRID_DIR, f"{scene}_grid.mp4")
    return os.path.join(BASE_DIR, f"A_{model}", f"{model}_{scene}.mp4")


def meta(scene, model):
    """(n_frames, fps) without decoding all frames."""
    r = imageio.get_reader(_path(scene, model))
    n = r.count_frames(); fps = r.get_meta_data().get("fps", 16) or 16
    r.close()
    return n, fps


def frames_at(scene, model, idxs):
    """Decode only frames idxs; de-tile if ours. Returns list of HxWx3 RGB uint8."""
    p = _path(scene, model)
    if not os.path.exists(p):
        return None
    r = imageio.get_reader(p)
    out = []
    if model.startswith("ours_") and model[len("ours_"):] in POS:
        x, y = POS[model[len("ours_"):]]
        for i in idxs:
            f = np.asarray(r.get_data(int(i)))
            out.append(f[y + 32:y + 480, x:x + 832])       # 448x832 tile
    else:
        for i in idxs:
            out.append(np.asarray(r.get_data(int(i))))
    r.close()
    return out
