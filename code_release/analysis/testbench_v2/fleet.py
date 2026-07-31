"""Fleet discovery + time-indexed video loading for testbench V2.

Analysis conventions. Metrics computed under different window definitions
are not comparable:
  - The first ~1 second of every rollout is real context frames.
  - All drift metrics compare a t=1s BASE window (never frame 0) against the
    END of generation. Style instruments compare the CTX (real first second)
    against END.
  - Videos are decoded at their NATIVE frame rate (16-30 fps across the
    fleet) and resized to canonical 832x448; analysis windows are defined in
    wall-clock seconds. Frame-rate differences are not resampled away —
    every output row records native_fps, and temporal metrics must be
    checked for fps confounds before interpretation (scorecard.py does).
  - Sibling group = one context video: all model rollouts generated from the
    same context PLUS the real continuation of that context. Real refs are
    members of their scene's group via out/real_scene_map.json
    (map_real_refs.py builds it by matching context frames).
"""
import dataclasses
import os
import re
import glob

import numpy as np

CANON_W, CANON_H = 832, 448

# Wall-clock windows (seconds). Default (legacy) windows used a nominal 1.0s
# hand-off; the TRUE hand-off is frame 12 at 0.75s (context = frames 0..11).
# Set TB2_CORRECT_BOUNDARY=1 for the corrected windows: CTX is real context
# only (ends at frame 11 / 0.75s) and BASE / generation start AT frame 12.
if os.environ.get("TB2_CORRECT_BOUNDARY"):
    CTX_T0, CTX_T1 = 0.20, 0.75      # frames 4..11 (all real; excludes frame 12)
    BASE_T0, BASE_T1 = 0.75, 1.50    # frames 12..23 (early generation)
    END_SEC = 0.90
    _GEN_START = 0.75                 # first generated frame (12)
else:
    CTX_T0, CTX_T1 = 0.20, 1.00
    BASE_T0, BASE_T1 = 0.90, 1.60
    END_SEC = 0.90
    _GEN_START = None                # legacy: gen sampling begins at BASE_T1

DIRS = ("F", "B", "L", "R", "FL", "FR", "BL", "BR")
_NAME_RE = re.compile(r"^(?P<model>[A-Za-z0-9]+)_r(?P<scene>\d+)_(?P<dir>F|B|L|R|FL|FR|BL|BR)\.mp4$")
_ABL_RE = re.compile(r"^step\d+_r(?P<scene>\d+)_(?P<dir>F|B|L|R|FL|FR|BL|BR)_raw\.mp4$")

EVAL_ROOT = os.environ.get(
    "TB2_EVAL_ROOT", os.path.join(os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "logs/eval_final"))
REAL_REFS = os.environ.get(
    "TB2_REAL_REFS",
    os.path.join(os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "analysis/eval_final/real_refs"))
# internal ablation runs (the human-tier-labeled grid); dir name -> model name
ABLATION_MODELS = {"pca8_8node": "pca8", "pca4": "pca4", "pca2": "pca2",
                   "16node": "16node", "4node": "4node",
                   "noatok": "noatok", "noadaln": "noadaln"}
_DEFAULT_SCENE_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "out", "real_scene_map.json")


@dataclasses.dataclass(frozen=True)
class VideoRef:
    model: str
    scene: int          # -1 for real refs with no scene mapping
    direction: str      # "" for real refs
    path: str

    @property
    def vid(self):
        if self.scene < 0:
            return f"{self.model}:{os.path.basename(self.path)}"
        return f"{self.model}_r{self.scene:02d}_{self.direction}"


def discover_fleet(models=None, real_refs=True, ablation=True, scenes=None):
    """Discover the fleet:
      - external models: logs/eval_final/A_{model}/{model}_r{NN}_{DIR}.mp4
        (model from the directory — A_worldplay_arl holds worldplay_*.mp4)
      - internal ablation grid: logs/eval_final/A/{run}/control_test/
        step*_r{NN}_{DIR}_raw.mp4 (the human-tier-labeled models)
      - real refs: model='real'; scene from TB2_REAL_SCENE_MAP or the
        default out/real_scene_map.json (basename -> scene int), else -1.
    models/scenes: optional filters (model names / scene ints).
    """
    refs = []
    for d in sorted(glob.glob(os.path.join(EVAL_ROOT, "A_*"))):
        model = os.path.basename(d)[2:]
        if models and model not in models:
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            m = _NAME_RE.match(os.path.basename(p))
            if not m:
                continue
            refs.append(VideoRef(model, int(m.group("scene")),
                                 m.group("dir"), p))
    if ablation:
        for run, model in ABLATION_MODELS.items():
            if models and model not in models:
                continue
            for p in sorted(glob.glob(os.path.join(
                    EVAL_ROOT, "A", run, "control_test", "*_raw.mp4"))):
                m = _ABL_RE.match(os.path.basename(p))
                if not m:
                    continue
                refs.append(VideoRef(model, int(m.group("scene")),
                                     m.group("dir"), p))
    if real_refs and os.path.isdir(REAL_REFS) and (not models or "real" in models):
        scene_map = {}
        map_path = os.environ.get("TB2_REAL_SCENE_MAP", _DEFAULT_SCENE_MAP)
        if map_path and os.path.exists(map_path):
            import json
            scene_map = json.load(open(map_path))
        for p in sorted(glob.glob(os.path.join(REAL_REFS, "*.mp4"))):
            b = os.path.basename(p)
            refs.append(VideoRef("real", int(scene_map.get(b, -1)), "", p))
    if scenes is not None:
        scenes = set(int(s) for s in scenes)
        refs = [r for r in refs if r.scene in scenes]
    return refs


def env_filters():
    """(models, scenes) filters from TB2_MODELS / TB2_SCENES env vars."""
    models = [m for m in os.environ.get("TB2_MODELS", "").split(",") if m]
    scenes = [s for s in os.environ.get("TB2_SCENES", "").split(",") if s]
    return (models or None), ([int(s) for s in scenes] or None)


# generalization probe set for the low-frequency/geometry detectors:
# r08 B/BL (V1 mangle ground truth) + r01_R (V1 blind window) on all
# ablation models, a slice of the external fleet, and all real refs.
EXTRA_ABLATION = [(8, "B"), (8, "BL"), (1, "R")]
# A_yume was emptied ~2026-07; its flat rollouts now live in A_yume_oldcap
EXTRA_EXTERNAL_MODELS = ("matrixgame", "minwm", "worldcam", "astra",
                         "yume", "yume_oldcap", "worldplay")
EXTRA_EXTERNAL = [(8, "F"), (8, "BL"), (22, "F"), (22, "BL")]


def extra_refs():
    """The generalization probe videos (no human tiers)."""
    refs = discover_fleet()
    abl_models = set(ABLATION_MODELS.values())
    keep = []
    for r in refs:
        if r.model in abl_models and (r.scene, r.direction) in EXTRA_ABLATION:
            keep.append(r)
        elif (r.model in EXTRA_EXTERNAL_MODELS
              and (r.scene, r.direction) in EXTRA_EXTERNAL):
            keep.append(r)
        elif r.model == "real":
            keep.append(r)
    return keep


def gate_plus_refs():
    """Labeled videos + generalization probes, deduplicated."""
    seen, out = set(), []
    for r in labeled_refs() + extra_refs():
        if r.vid not in seen:
            seen.add(r.vid)
            out.append(r)
    return out


def refs_from_env():
    """Standard ref selection used by all metric runners."""
    if os.environ.get("TB2_EXTRA_ONLY"):
        labeled = {r.vid for r in labeled_refs()}
        refs = [r for r in extra_refs() if r.vid not in labeled]
    elif os.environ.get("TB2_GATE_PLUS"):
        refs = gate_plus_refs()
    elif os.environ.get("TB2_LABELS_ONLY"):
        refs = labeled_refs()
    else:
        models, scenes = env_filters()
        refs = discover_fleet(models, scenes=scenes)
    excl = {m for m in os.environ.get("TB2_EXCLUDE", "").split(",") if m}
    if excl:
        refs = [r for r in refs if r.model not in excl]
    return refs


def labeled_refs(labels_csv=None):
    """Only the videos that carry human tier labels, plus the real siblings
    of their scenes. Used for gate (b): metric-vs-human-tier agreement runs
    before any fleet-wide spend."""
    import csv
    labels_csv = labels_csv or os.environ.get(
        "TB2_LABELS", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "labels", "human_tiers.csv"))
    want, scenes = set(), set()
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            want.add((row["model"], int(row["scene"]), row["direction"]))
            scenes.add(int(row["scene"]))
    refs = [r for r in discover_fleet()
            if (r.model, r.scene, r.direction) in want
            or (r.model == "real" and r.scene in scenes)]
    missing = want - {(r.model, r.scene, r.direction) for r in refs}
    if missing:
        print(f"WARNING: {len(missing)} labeled videos not found on disk: "
              f"{sorted(missing)[:10]}")
    return refs


MAX_SEC = float(os.environ.get("TB2_MAX_SEC", "7.0"))


def load_video(path, size=(CANON_W, CANON_H), max_seconds=MAX_SEC):
    """Decode frames at the native frame rate, canonical resize, capped at
    max_seconds (default 7.0 = nominal 1s context + 6s generated content —
    our rollouts generate exactly 6.0s; longer external videos would
    otherwise be scored on up to 7.6s of generation, an unfair extra
    degradation window). Frame-rate differences are intentionally not
    resampled away; windows are indexed by wall-clock time downstream.

    Returns (frames uint8 [T,H,W,3] RGB, times float [T] seconds, native_fps).
    """
    import av
    import cv2
    container = av.open(path)
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate)
    frames, times = [], []
    for i, frame in enumerate(container.decode(video=0)):
        t = i / native_fps
        if max_seconds is not None and t > max_seconds:
            break
        img = frame.to_ndarray(format="rgb24")
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        frames.append(img)
        times.append(t)
    container.close()
    return np.stack(frames), np.asarray(times, dtype=np.float64), native_fps


def window_idx(times, t0, t1):
    """Indices of sampled frames with t0 <= t < t1 (>=1 frame guaranteed)."""
    idx = np.where((times >= t0) & (times < t1))[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))])
    return idx


def windows(times):
    """(ctx, base, end) index arrays for the standard analysis windows."""
    t_last = times[-1]
    return (window_idx(times, CTX_T0, CTX_T1),
            window_idx(times, BASE_T0, BASE_T1),
            window_idx(times, t_last - END_SEC, t_last + 1e-6))


def gen_fraction_times(times, n):
    """n wall-clock timestamps evenly spanning the generated part.

    Legacy: span begins at BASE_T1. Corrected (TB2_CORRECT_BOUNDARY): span
    begins at the true generation start (frame 12 / 0.75s).
    """
    t0 = _GEN_START if _GEN_START is not None else BASE_T1
    t1 = times[-1]
    return np.linspace(t0, max(t1, t0 + 0.1), n)


def frame_at(frames, times, t):
    return frames[int(np.argmin(np.abs(times - t)))]


def gen_adjacent_pairs(times, n):
    """n (i, i+1) NATIVE-frame index pairs spread over the generated span —
    for temporal metrics (warp error, frame consistency). These are
    fps-dependent by construction; scorecard.py checks the confound."""
    idx = sorted(set(int(np.argmin(np.abs(times - t)))
                     for t in gen_fraction_times(times, n)))
    return [(i, i + 1) for i in idx if i + 1 < len(times)]
