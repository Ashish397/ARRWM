"""Scene relocation across the fleet: has the rollout left the place it started?

Relocation is measured by place identity rather than appearance. A real
reference frame from the conditioning span is matched against each model's frame
at the six-second horizon using ORB features and a RANSAC homography. The count
of geometrically consistent inliers stays high while the rollout is still in the
same place and collapses once it has wandered somewhere else. A rollout counts
as relocated when that count falls below the threshold calibrated on the
human-labelled subset.

CPU only. Writes incrementally and resumes, so an interrupted scan continues
rather than recomputing.

Data locations come from the environment: AF_FLEET_DIR holds the baseline
rollouts, AF_TILES_DIR the de-tiled rollouts of our variants, AF_SCENES limits
the scan to named scenes.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_DIR = os.environ.get(
    "AF_FLEET_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), "grids")
)
BASE = os.path.join(FLEET_DIR, "baselines")
# De-tiled rollouts of our variants, named <scene>__<variant>.mp4. AF_TILES_DIR
# may be a colon-separated list; the original layout kept them in two directories.
TILE_DIRS = [
    d for d in os.environ.get(
        "AF_TILES_DIR", os.pathsep.join([os.path.join(HERE, "tiles"),
                                         os.path.join(HERE, "tiles_new")])
    ).split(os.pathsep) if d
]
OUT = os.environ.get("AF_SCENE_RELOC_OUT", os.path.join(HERE, "fleet_scene_reloc.csv"))

# Conditioning frames each release emits before its generated span, so the
# six-second horizon is measured from the same point in every rollout.
BASELINE_CTX = {"astra": 4, "matrixgame": 1, "minwm": 13,
                "worldcam": 65, "worldplay": 1, "yume": 1}
OURS_CTX = {"pca8": 9, "16node": 9}

REF_FRAME = 8            # real frame from the conditioning span
HORIZON_SEC = 6.0
MATCH_SIZE = (640, 352)  # common size, so inlier counts compare across models
N_FEATURES = 3000
RANSAC_PX = 5.0
MIN_MATCHES = 8

# Inlier count below which a rollout counts as relocated, calibrated against the
# human labels. Kept in a file so the scan and the figures cannot disagree.
THRESHOLD = int(open(os.path.join(HERE, "scene_reloc_threshold.txt")).read())

_orb = cv2.ORB_create(N_FEATURES)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def frame_at(path, index):
    """Decode a single frame by index, or None if it cannot be read."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_meta(path):
    """(fps, frame_count); fps falls back to 16 when the container omits it."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, count


def inliers(a, b):
    """RANSAC-verified ORB correspondences between two frames.

    Returns 0 rather than raising when a frame is too featureless to match,
    which is the case for the wet-lens contexts the evaluation excludes.
    """
    ka, da = _orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = _orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < MIN_MATCHES or len(kb) < MIN_MATCHES:
        return 0
    matches = _matcher.match(da, db)
    if len(matches) < MIN_MATCHES:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in matches])
    dst = np.float32([kb[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    return int(mask.sum()) if mask is not None else 0


def tile_path(scene, variant):
    """De-tiled rollout for one of our variants, or None if not extracted."""
    for directory in TILE_DIRS:
        path = os.path.join(directory, f"{scene}__{variant}.mp4")
        if os.path.exists(path):
            return path
    return None


def main():
    scenes = sorted({
        os.path.basename(p).split("_", 1)[1][:-4]
        for p in glob.glob(os.path.join(BASE, "A_astra", "*.mp4"))
    })
    only = os.environ.get("AF_SCENES")          # comma-separated, for spot checks
    if only:
        wanted = set(only.split(","))
        scenes = [s for s in scenes if s in wanted]

    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT)
        done = set(zip(prev.scene, prev.model))

    with open(OUT, "a") as out:
        if not done:
            out.write("scene,model,inliers,relocated\n")
        for scene in scenes:
            reference = tile_path(scene, "pca8")
            if not reference:
                continue
            ref = frame_at(reference, REF_FRAME)
            if ref is None:
                continue
            ref = cv2.resize(ref, MATCH_SIZE)

            rollouts = {
                m: (os.path.join(BASE, f"A_{m}", f"{m}_{scene}.mp4"), ctx)
                for m, ctx in BASELINE_CTX.items()
            }
            rollouts.update({
                f"ours_{v}": (tile_path(scene, v), ctx) for v, ctx in OURS_CTX.items()
            })

            for name, (path, ctx) in rollouts.items():
                if (scene, name) in done or not path or not os.path.exists(path):
                    continue
                fps, count = video_meta(path)
                end = frame_at(path, min(count - 2, ctx + int(round(HORIZON_SEC * fps))))
                if end is None:
                    continue
                n = inliers(ref, cv2.resize(end, MATCH_SIZE))
                out.write(f"{scene},{name},{n},{int(n < THRESHOLD)}\n")
                out.flush()
            print(scene, flush=True)
    print("done")


if __name__ == "__main__":
    main()
