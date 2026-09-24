"""Resolver for the 30 s ICLR fleets (32 E1 windows x 9 commands), same API as fleet_common so the
fleet_* instruments run unchanged: fleet_index(), ctx_of(), meta(), frames_at().
Activated by env FLEET=30s (fleet_common re-exports these). Scenes are "<uid>_<DIR>" (window + command).
Layout (env-overridable):
  external : logs/eval_final/fleet30s/<model>/<model>_<uid>_<DIR>.mp4  (matrixgame2 files are matrixgame_*)
  ours ODE : experiments/e1/rec/long40/.motion_check/<arm>/<uid>_<DIR>_s0.mp4  (kl4rung_cf, mse4rung_cf)
  minWM ODE: logs/eval_final/e1_minwm/ode30s/minwm_ode_<uid>_<DIR>.mp4
  ours DMD : logs/eval_final/ours30s/<arm>/<arm>_<uid>_<DIR>.mp4 (when the u6qf fleets are pulled back)
Context frames (first generated frame index): lingbot 1, dreamx 1, YUME 1,
matrixgame 1,
minWM DMD 29 in the aligned fleet (source frames 4--32; eight latents), our ODE
33 (source frames 0--32; nine latents), minWM ODE 13 (source frames 20--32;
four latents), and our DMD checkpoints from their sidecars.
Env: ARR (repo root), OURS30S_DIR (DMD arm dir), OURS_ODE_DIR,
MINWM_ODE_DIR, FLEET30S_MODELS csv (default all present), FLEET30S_UIDS csv,
FLEET30S_DIRS csv, EVAL_HORIZON_S (6/15/30).
"""
import os, json, glob
from pathlib import Path
import imageio, numpy as np

ARR = os.environ.get("ARR", "/home/ashish/ARRWM")                 # repo root (env ARR for cluster copies)
OURS30S_DIR = os.environ.get("OURS30S_DIR", f"{ARR}/logs/eval_final/ours30s")   # u6qf: $R/_logs/ours30s/out
OURS_ODE_DIR = os.environ.get("OURS_ODE_DIR", f"{ARR}/experiments/e1/rec/long40/.motion_check")
MINWM_ODE_DIR = os.environ.get("MINWM_ODE_DIR", f"{ARR}/logs/eval_final/e1_minwm/ode30s")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL", "N"]
CTX = {"lingbot": 1, "dreamx": 1, "yume5b": 1,
       "minwm": 13, "minwm_seed29": 29,
       "matrixgame2": 1, "ours_kl4rung": 33, "ours_mse4rung": 33,
       "minwm_ode": 13}
MINWM_SEED29_DIR = os.environ.get(
    "MINWM_SEED29_DIR", f"{ARR}/logs/eval_final/fleet30s/minwm_seed29")
ALIGNED32_DIR = os.environ.get("FLEET30S_ALIGNED32_DIR", "")
YUME30S_DIR = os.environ.get(
    "YUME30S_DIR",
    f"{ALIGNED32_DIR}/yume5b" if ALIGNED32_DIR else f"{ARR}/logs/eval_final/fleet30s/yume5b",
)
DREAMX30S_DIR = os.environ.get(
    "DREAMX30S_DIR",
    f"{ALIGNED32_DIR}/dreamx" if ALIGNED32_DIR else f"{ARR}/logs/eval_final/fleet30s/dreamx",
)
MINWM_ODE_SWAP_YAW = os.environ.get("MINWM_ODE_SWAP_YAW", "0") == "1"
PANEL32_EVAL_CONFIG = os.environ.get("PANEL32_EVAL_CONFIG", "")
PANEL32_MODE = bool(PANEL32_EVAL_CONFIG)
MODEL_FAMILY = {}
FAMILY_SEATS = {}
MODEL_SOURCE_SPAN = {}
_PANEL_SOURCE_CLIPS = {}


def _minwm_ode_path(u, d):
    # Compatibility only for the archived pre-alignment fleet. The corrected
    # aligned32 ODE fleet uses the explicit common-yaw adapter and is read
    # without filename swapping.
    if MINWM_ODE_SWAP_YAW:
        d = {"L": "R", "R": "L", "FL": "FR", "FR": "FL",
             "BL": "BR", "BR": "BL"}.get(d, d)
    return f"{MINWM_ODE_DIR}/minwm_ode_{u}_{d if d != 'N' else 'NOOP'}.mp4"


PATH = {
    "lingbot":      lambda u, d: f"{ARR}/logs/eval_final/fleet30s/lingbot/lingbot_{u}_{d}.mp4",
    "dreamx":       lambda u, d: f"{DREAMX30S_DIR}/dreamx_{u}_{d}.mp4",
    "yume5b":       lambda u, d: f"{YUME30S_DIR}/yume5b_{u}_{d}.mp4",
    "minwm":        lambda u, d: f"{ARR}/logs/eval_final/fleet30s/minwm/minwm_{u}_{d if d != 'N' else 'NOOP'}.mp4",
    "minwm_seed29": lambda u, d: f"{MINWM_SEED29_DIR}/minwm_seed29_{u}_{d if d != 'N' else 'NOOP'}.mp4",
    "matrixgame2":  lambda u, d: f"{ARR}/logs/eval_final/fleet30s/matrixgame2/matrixgame_{u}_{d if d != 'N' else 'NOOP'}.mp4",
    "ours_kl4rung": lambda u, d: f"{OURS_ODE_DIR}/kl4rung_cf/{u}_{d}_s0.mp4",
    "ours_mse4rung": lambda u, d: f"{OURS_ODE_DIR}/mse4rung_cf/{u}_{d}_s0.mp4",
    "minwm_ode":    _minwm_ode_path,
}
if ALIGNED32_DIR:
    # Corrected external fleet: every model begins generation immediately
    # after real-video frame 32. minWM uses frames 4--32 (29 pixels).
    PATH.update({
        "lingbot": lambda u, d: f"{ALIGNED32_DIR}/lingbot/lingbot_{u}_{d}.mp4",
        "dreamx": lambda u, d: f"{DREAMX30S_DIR}/dreamx_{u}_{d}.mp4",
        "minwm": lambda u, d: f"{ALIGNED32_DIR}/minwm/minwm_aligned32_{u}_{d if d != 'N' else 'NOOP'}.mp4",
        "matrixgame2": lambda u, d: f"{ALIGNED32_DIR}/matrixgame2/matrixgame_{u}_{d if d != 'N' else 'NOOP'}.mp4",
    })
    CTX["minwm"] = 29
ARMS = ("no_commit", "no_aux", "no_gan", "base", "base_v2", "no_carn", "stat_mean_only", "stat_nonmean_only", "recovery_base", "base_v3_stationary200", "base_v2_stationary200", "no_aux_stationary200", "base_v3b_stationary200")
for _arm in ARMS:
    PATH[f"ours_{_arm}"] = (lambda a: (lambda u, d: f"{OURS30S_DIR}/{a}/{a}_{u}_{d}.mp4"))(_arm)
# base_v2 was marked BROKEN (2026-09-17): folder renamed, clip files inside keep the base_v2_ prefix
ARMS = ARMS + ("base_v2_BROKEN",)
PATH["ours_base_v2_BROKEN"] = lambda u, d: f"{OURS30S_DIR}/base_v2_BROKEN/base_v2_{u}_{d}.mp4"
UIDS = [w["uid"] for w in json.load(open(f"{ARR}/experiments/e1/scene_shortlist/e1_32_windows.json"))]
SEED65_DIR = os.environ.get("SEED65_DIR", f"{ARR}/analysis/eval_final/seed65_e1")
SEED_CLIP = lambda u: f"{SEED65_DIR}/seed65_{u}.mp4"      # real footage, 65 frames @16fps

# A mixed-dataset evaluation is supplied as data, not as another set of
# hard-coded path exceptions.  This leaves all archived E1 behavior intact.
if PANEL32_MODE:
    _cfg_path = Path(PANEL32_EVAL_CONFIG).resolve()
    _cfg = json.load(open(_cfg_path, encoding="utf-8"))
    if _cfg.get("schema_version") != 1:
        raise RuntimeError(f"unsupported PANEL32_EVAL_CONFIG schema: {_cfg_path}")
    UIDS = list(_cfg["context_ids"])
    if len(UIDS) != 32 or len(set(UIDS)) != 32:
        raise RuntimeError("panel32 eval config must name exactly 32 unique contexts")
    _PANEL_SOURCE_CLIPS = {str(k): str(v) for k, v in _cfg["source_clips"].items()}
    if set(_PANEL_SOURCE_CLIPS) != set(UIDS):
        raise RuntimeError("panel32 source-clip keys differ from context IDs")
    PATH = {}
    CTX = {}
    MODEL_FAMILY = {}
    MODEL_SOURCE_SPAN = {}
    for _model, _record in _cfg["models"].items():
        _template = str(_record["path_template"])
        _noop = str(_record.get("noop_disk_action", "N"))
        PATH[_model] = (
            lambda template=_template, noop=_noop:
            (lambda u, d: template.format(context_id=u, action=(noop if d == "N" else d)))
        )()
        CTX[_model] = int(_record["context_frames"])
        MODEL_FAMILY[_model] = str(_record["family"])
        MODEL_SOURCE_SPAN[_model] = tuple(map(int, _record["source_frames_inclusive"]))
    FAMILY_SEATS = {str(k): str(v) for k, v in _cfg["family_seats"].items()}
    if not set(FAMILY_SEATS.values()).issubset(PATH):
        raise RuntimeError("panel32 family seats reference unknown models")
    SEED_CLIP = lambda u: _PANEL_SOURCE_CLIPS[u]

def _sel(env, default):
    v = os.environ.get(env, "")
    return [x for x in v.split(",") if x] if v else default

def models():
    return _sel("FLEET30S_MODELS", [m for m in PATH if any(os.path.exists(PATH[m](u, d)) for u in UIDS[:3] for d in DIRS)])

def eval_uids():
    """Context IDs selected for this evaluation run (all 32 by default)."""
    return _sel("FLEET30S_UIDS", UIDS)

def fleet_index():
    out = []
    for u in eval_uids():
        for d in _sel("FLEET30S_DIRS", DIRS):
            for m in models():
                if os.path.exists(PATH[m](u, d)): out.append((f"{u}_{d}", m))
    return out

def _split(scene):
    u, d = scene.rsplit("_", 1); return u, d

def _path(scene, model):
    u, d = _split(scene); return PATH[model](u, d)

def ctx_of(model):
    if not PANEL32_MODE and model.startswith("ours_") and model[5:] in ARMS:
        js = glob.glob(f"{OURS30S_DIR}/{model[5:]}/*.json")
        if js: return int(json.load(open(js[0])).get("seed_frames", 33))
    return CTX.get(model, 1)

# EVAL_DECODE_BITEXACT=1: bit-exact swscale YUV->RGB so x86 (local) and aarch64 (u6qf) decodes are identical.
# Default off (the default decode differs by ~0.7 mean / 3 max LSB across CPU archs). Changes results vs default.
_RD = {"output_params": ["-sws_flags", "bitexact+accurate_rnd+full_chroma_int"]} if os.environ.get("EVAL_DECODE_BITEXACT") == "1" else {}

def _reader(p):
    return imageio.get_reader(p, "ffmpeg", **_RD) if _RD else imageio.get_reader(p)

def meta(scene, model):
    r = _reader(_path(scene, model)); n = r.count_frames(); fps = r.get_meta_data().get("fps", 16) or 16; r.close()
    return n, fps

def frames_at(scene, model, idxs):
    p = _path(scene, model)
    if not os.path.exists(p): return None
    r = _reader(p); out = [np.asarray(r.get_data(int(i))) for i in idxs]; r.close(); return out

def horizon_s():
    return float(os.environ.get("EVAL_HORIZON_S", "6"))


def source_indices(model, count):
    """Final native real frames, expressed in canonical source coordinates."""
    if not PANEL32_MODE:
        ctx = ctx_of(model)
        return list(range(max(0, ctx - count), ctx)) if ctx > 1 else [0]
    first, last = MODEL_SOURCE_SPAN[model]
    available = last - first + 1
    if available == 1:
        return [last]
    start = max(first, last - int(count) + 1)
    return list(range(start, last + 1))


def source_frames(uid, model, count):
    """Decode source references without passing through a model's VAE."""
    indices = source_indices(model, count)
    r = _reader(SEED_CLIP(uid))
    frames = [np.asarray(r.get_data(int(i))) for i in indices]
    r.close()
    return frames, indices
