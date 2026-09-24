"""Native-frame continuity gate for long-horizon relocation proposals.

The 4 Hz proposal stage intentionally favours recall and can skip intermediate
views during fast turns or occlusions.  This gate decodes every native frame in
a two-second interval around each proposal and builds a forward correspondence
graph.  A frame is reachable when it has at least ``threshold`` verified ORB
inliers to any reachable frame in the preceding ``lookback_s`` interval.  If
the final frame remains reachable from the first, the apparent break is a
continuous trajectory and cannot be relocation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def resolve(path: str, local_prefix: str, remote_prefix: str) -> str:
    if Path(path).exists():
        return path
    candidate = remote_prefix + path[len(local_prefix):]
    if Path(candidate).exists():
        return candidate
    raise FileNotFoundError(path)


def read_range(path: str, first: int, last: int) -> list[np.ndarray]:
    # Keep each worker's FFmpeg decoder single-threaded.  Otherwise several
    # concurrent H.264 opens can exhaust the process/thread limit and fail on
    # otherwise valid frames.
    cap = cv2.VideoCapture(
        path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    try:
        for index in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError((path, index))
            frames.append(frame)
    finally:
        cap.release()
    return frames


def descriptor(orb: cv2.ORB, frame: np.ndarray):
    grey = cv2.cvtColor(cv2.resize(frame, (640, 352)), cv2.COLOR_BGR2GRAY)
    points, desc = orb.detectAndCompute(grey, None)
    return np.float32([point.pt for point in points]), desc


def inliers(left, right) -> int:
    points_l, desc_l = left
    points_r, desc_r = right
    if desc_l is None or desc_r is None or len(points_l) < 8 or len(points_r) < 8:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_l, desc_r)
    if len(matches) < 8:
        return 0
    src = np.float32([points_l[item.queryIdx] for item in matches])
    dst = np.float32([points_r[item.trainIdx] for item in matches])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def graph_score(frames: list[np.ndarray], fps: float, threshold: int,
                lookback_s: float) -> dict[str, float | int]:
    orb = cv2.ORB_create(3000)
    desc = [descriptor(orb, frame) for frame in frames]
    gap = max(1, int(round(lookback_s * fps)))
    reachable = [False] * len(desc)
    reachable[0] = True
    best_edges = [0] * len(desc)
    for i in range(1, len(desc)):
        candidates = [j for j in range(max(0, i - gap), i) if reachable[j]]
        scores = [inliers(desc[j], desc[i]) for j in candidates]
        best_edges[i] = max(scores, default=0)
        reachable[i] = best_edges[i] >= threshold
    return {
        "dense_connected": int(reachable[-1]),
        "dense_reachable_fraction": float(np.mean(reachable)),
        "dense_first_unreachable_frame": next(
            (i for i, value in enumerate(reachable) if not value), -1),
        "dense_min_best_edge": int(min(best_edges[1:])),
        "dense_median_best_edge": float(np.median(best_edges[1:])),
        "dense_final_best_edge": int(best_edges[-1]),
    }


def boundary_score(frames: list[np.ndarray], boundary: int) -> dict[str, float]:
    """Measure whether the proposed boundary is an abrupt native-frame cut.

    Scores are normalised by the other adjacent-frame changes in the same
    two-second neighbourhood.  This distinguishes a hard replacement from a
    fast but temporally resolved turn without imposing an absolute appearance
    scale across models.
    """
    small = [cv2.resize(frame, (160, 88)) for frame in frames]
    grey = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in small]
    hsv = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in small]
    mad, hist = [], []
    for left, right, left_hsv, right_hsv in zip(
            grey[:-1], grey[1:], hsv[:-1], hsv[1:]):
        mad.append(float(np.mean(cv2.absdiff(left, right))) / 255.0)
        h1 = cv2.calcHist([left_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        h2 = cv2.calcHist([right_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        hist.append(float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)))
    seam = max(0, min(len(mad) - 1, boundary - 1))
    # Exclude the proposed seam from its local baseline.
    base_mad = np.asarray(mad[:seam] + mad[seam + 1:], dtype=float)
    base_hist = np.asarray(hist[:seam] + hist[seam + 1:], dtype=float)
    eps = 1e-6
    return {
        "seam_mad": mad[seam],
        "seam_mad_ratio": mad[seam] / (float(np.median(base_mad)) + eps),
        "seam_hist": hist[seam],
        "seam_hist_ratio": hist[seam] / (float(np.median(base_hist)) + eps),
    }


def score(row: dict, lookup: pd.DataFrame, local_prefix: str,
          remote_prefix: str, threshold: int, lookback_s: float,
          synthetic_count: int, mad_threshold: float,
          hist_ratio_threshold: float, boundary_only: bool) -> list[dict]:
    cv2.setNumThreads(1)
    meta = lookup.loc[(row["scene"], row["model"])]
    path = resolve(str(meta.path), local_prefix, remote_prefix)
    fps, context, count = float(meta.fps), int(meta.context_frames), int(meta.decoded_frames)
    center = context + int(round(float(row["time_s"]) * fps))
    first, last = max(0, center - int(round(fps))), min(count - 1, center + int(round(fps)))
    real_frames = read_range(path, first, last)
    base = {**row, "case": "real"}
    boundary = center - first
    seam = boundary_score(real_frames, boundary)
    dense = ({
        "dense_connected": np.nan,
        "dense_reachable_fraction": np.nan,
        "dense_first_unreachable_frame": np.nan,
        "dense_min_best_edge": np.nan,
        "dense_median_best_edge": np.nan,
        "dense_final_best_edge": np.nan,
    } if boundary_only else graph_score(real_frames, fps, threshold, lookback_s))
    outputs = [{**base, **dense, **seam,
                "abrupt_cut": int(seam["seam_hist_ratio"] > hist_ratio_threshold)}]
    if synthetic_count:
        alternatives = lookup.reset_index()
        alternatives = alternatives[(alternatives.model.eq(row["model"])) &
                                    (~alternatives.scene.eq(row["scene"]))].sort_values("scene")
        choices = np.linspace(0, len(alternatives) - 1,
                              min(synthetic_count, len(alternatives))).round().astype(int)
        for synthetic_index, choice in enumerate(choices):
            other = alternatives.iloc[choice]
            other_path = resolve(str(other.path), local_prefix, remote_prefix)
            other_fps = float(other.fps)
            other_center = int(other.context_frames) + int(round(6.0 * other_fps))
            post_count = int(round(fps)) + 1
            other_indices_end = min(int(other.decoded_frames) - 1,
                                    other_center + post_count - 1)
            other_frames = read_range(other_path, other_center, other_indices_end)
            # Resample the alternate bank to the candidate's number of post frames.
            targets = np.linspace(0, len(other_frames) - 1, post_count).round().astype(int)
            splice = real_frames[:boundary] + [other_frames[i] for i in targets]
            seam = boundary_score(splice, boundary)
            dense = ({
                "dense_connected": np.nan,
                "dense_reachable_fraction": np.nan,
                "dense_first_unreachable_frame": np.nan,
                "dense_min_best_edge": np.nan,
                "dense_median_best_edge": np.nan,
                "dense_final_best_edge": np.nan,
            } if boundary_only else graph_score(splice, fps, threshold, lookback_s))
            outputs.append({
                **row, "case": "synthetic_cut", "synthetic_index": synthetic_index,
                "after_scene": other.scene, **dense, **seam,
                "abrupt_cut": int(seam["seam_hist_ratio"] > hist_ratio_threshold),
            })
    return outputs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threshold", type=int, default=20)
    p.add_argument("--lookback-s", type=float, default=0.25)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-count", type=int, default=1)
    p.add_argument("--mad-threshold", type=float, default=0.22)
    p.add_argument("--hist-ratio-threshold", type=float, default=3.0)
    p.add_argument(
        "--boundary-only", action="store_true",
        help="compute the deployed native-frame histogram gate without unused ORB graph diagnostics",
    )
    p.add_argument("--local-prefix", default="/home/ashish/ARRWM")
    p.add_argument("--remote-prefix", default="")
    a = p.parse_args()
    try:
        events = pd.read_csv(a.events)
    except pd.errors.EmptyDataError:
        events = pd.DataFrame(columns=["scene", "model", "time_s"])
    manifest = pd.read_csv(a.manifest)
    lookup = manifest.set_index(["scene", "model"])
    if events.empty:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "scene", "model", "time_s", "case", "dense_connected",
            "dense_reachable_fraction", "dense_first_unreachable_frame",
            "dense_min_best_edge", "dense_median_best_edge",
            "dense_final_best_edge", "seam_mad", "seam_mad_ratio",
            "seam_hist", "seam_hist_ratio", "abrupt_cut",
        ]).to_csv(a.out, index=False)
        print("no temporal relocation proposals")
        return
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        futures = [executor.submit(
            score, row, lookup, a.local_prefix, a.remote_prefix,
            a.threshold, a.lookback_s,
            a.synthetic_count if a.synthetic else 0,
            a.mad_threshold, a.hist_ratio_threshold, a.boundary_only)
            for row in events.to_dict("records")]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if i % 20 == 0 or i == len(futures):
                print(f"[{i}/{len(futures)}]", flush=True)
    frame = pd.DataFrame(rows).sort_values(["model", "scene", "time_s", "case"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.groupby("case").dense_connected.agg(["size", "sum"]))
    print(frame.groupby("case").abrupt_cut.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
