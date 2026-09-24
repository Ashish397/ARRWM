"""Precision / quantisation policy layer for the interactive world model.

WS-B deliverable. Owns ONLY the dtype + quantisation policy applied to an
already-built generator; it never builds, loads or rolls the model itself.

Modes (``EngineConfig.precision``)
---------------------------------
``fp32``     plain float32 weights (matches training/eval weight precision)
``bf16``     plain bfloat16 weights (reference path, = ``--weights_dtype bf16``
             in ``utils/play_world_model.py``)
``fp8_wo``   torchao ``Float8WeightOnlyConfig`` on the DiT block linears
``fp8_dyn``  torchao ``Float8DynamicActivationFloat8WeightConfig`` (e4m3 act +
             e4m3 weight, per-tensor dynamic) on the DiT block linears
``fp4_wo``   torchao ``NVFP4InferenceConfig(mm_config=WEIGHT_ONLY)`` on the DiT
             block linears (Blackwell NVFP4, block size 16)

All quantised modes start from a bf16 cast and then replace *only* the
transformer-block projection linears; everything else (patch embedding, text /
time embeddings, time projection, per-block norms + modulation parameters, the
output head, the action projections and the VAE) stays bf16. See
``PRECISION_NOTES.md`` for the exact matched set and measured numbers.

Public surface
--------------
``apply_precision(model, cfg) -> model``   in-place policy application
``compile_model(model, mode) -> model``    torch.compile wiring (call AFTER
                                           apply_precision)
``precision_available() -> dict[str,bool]`` which modes this env can run
``describe(model) -> str``                 achieved dtype per layer class

Graceful degradation: if torchao is missing / too old / the GPU is not capable,
``apply_precision`` logs a warning and falls back to ``bf16`` rather than
raising, and ``precision_available()`` reports the mode as False.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

log = logging.getLogger("precision")

__all__ = [
    "PRECISION_MODES",
    "QUANT_MODES",
    "DEFAULT_QUANT_EXCLUDE",
    "DEFAULT_QUANT_INCLUDE",
    "COMPILE_MODES",
    "apply_precision",
    "compile_model",
    "precision_available",
    "describe",
    "torchao_info",
]

PRECISION_MODES: Sequence[str] = ("fp32", "bf16", "fp8_wo", "fp8_dyn", "fp4_wo")
QUANT_MODES: Sequence[str] = ("fp8_wo", "fp8_dyn", "fp4_wo")
COMPILE_MODES: Sequence[str] = ("off", "reduce-overhead", "max-autotune")

# Linears we DO quantise: the transformer block q/k/v/o (self- and
# cross-attention) and the two FFN linears. Patterns are fnmatch globs against
# the module's fully-qualified name relative to the CausalWanModel root.
DEFAULT_QUANT_INCLUDE: Sequence[str] = (
    "blocks.*.self_attn.q",
    "blocks.*.self_attn.k",
    "blocks.*.self_attn.v",
    "blocks.*.self_attn.o",
    "blocks.*.cross_attn.q",
    "blocks.*.cross_attn.k",
    "blocks.*.cross_attn.v",
    "blocks.*.cross_attn.o",
    "blocks.*.ffn.0",
    "blocks.*.ffn.2",
)

# Never quantised, even if a future include pattern would match them:
#   - patch_embedding      first projection (Conv3d; would not match anyway)
#   - head.*               final projection
#   - text_embedding.*     prompt projection (runs once per session, cached)
#   - time_embedding.*, time_projection.*   timestep -> AdaLN modulation
#   - *norm*, *modulation* numerically sensible to keep in bf16/fp32
#   - action_*             action heads (tiny, zero-init, sensitive)
#   - head_alt.*           training-only alt head if present
#   - *img_emb*            i2v path, unused here
DEFAULT_QUANT_EXCLUDE: Sequence[str] = (
    "patch_embedding*",
    "head",
    "head.*",
    "head_alt*",
    "text_embedding*",
    "time_embedding*",
    "time_projection*",
    "*norm*",
    "*modulation*",
    "action_*",
    "*.action_*",
    "*img_emb*",
    "*state_probe*",
)

_TORCHAO: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# torchao discovery
# ---------------------------------------------------------------------------
def torchao_info() -> Dict[str, Any]:
    """Import torchao once and report what quantisation configs are usable.

    Never raises. Returns a dict with keys ``available`` (bool), ``version``,
    ``error``, ``fp8`` / ``fp8_dyn`` / ``fp4`` (bool)."""
    if _TORCHAO:
        return _TORCHAO

    info: Dict[str, Any] = {
        "available": False, "version": None, "error": None,
        "fp8": False, "fp8_dyn": False, "fp4": False,
        "quantize_": None, "cfg_fp8_wo": None, "cfg_fp8_dyn": None,
        "cfg_fp4_wo": None,
    }
    try:
        import torchao  # noqa: F401
        from torchao.quantization import quantize_
        info["available"] = True
        info["version"] = getattr(torchao, "__version__", "unknown")
        info["quantize_"] = quantize_
    except Exception as exc:                              # pragma: no cover
        info["error"] = f"{type(exc).__name__}: {exc}"
        _TORCHAO.update(info)
        return _TORCHAO

    try:
        from torchao.quantization import Float8WeightOnlyConfig
        info["cfg_fp8_wo"] = Float8WeightOnlyConfig
        info["fp8"] = True
    except Exception as exc:
        log.debug("Float8WeightOnlyConfig unavailable: %s", exc)
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        info["cfg_fp8_dyn"] = Float8DynamicActivationFloat8WeightConfig
        info["fp8_dyn"] = True
    except Exception as exc:
        log.debug("Float8DynamicActivationFloat8WeightConfig unavailable: %s", exc)
    try:
        from torchao.prototype.mx_formats import NVFP4InferenceConfig, NVFP4MMConfig
        info["cfg_fp4_wo"] = (NVFP4InferenceConfig, NVFP4MMConfig)
        info["fp4"] = True
    except Exception as exc:
        log.debug("NVFP4InferenceConfig unavailable: %s", exc)

    _TORCHAO.update(info)
    return _TORCHAO


def _cuda_capability() -> Optional[tuple]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return None


def precision_available() -> Dict[str, bool]:
    """Which precision modes are usable in this environment, right now."""
    cap = _cuda_capability()
    have_cuda = cap is not None
    # FP8 tensor cores: sm_89 (Ada) and up. NVFP4: sm_100/sm_120 (Blackwell).
    sm = (cap[0] * 10 + cap[1]) if cap else 0
    ao = torchao_info()
    return {
        "fp32": True,
        "bf16": have_cuda and torch.cuda.is_bf16_supported() if have_cuda else True,
        "fp8_wo": bool(have_cuda and ao["fp8"] and sm >= 89),
        "fp8_dyn": bool(have_cuda and ao["fp8_dyn"] and sm >= 89),
        "fp4_wo": bool(have_cuda and ao["fp4"] and sm >= 100),
    }


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------
def _base_dit(model: nn.Module) -> nn.Module:
    """Unwrap WanDiffusionWrapper / PEFT wrappers down to the CausalWanModel.

    Accepts the wrapper, the wrapped model, or the raw DiT and returns whatever
    level actually owns ``.blocks``. Falls back to the input unchanged."""
    cur = model
    for _ in range(6):
        if hasattr(cur, "get_base_model"):
            try:
                cur = cur.get_base_model()
                continue
            except Exception:
                pass
        if hasattr(cur, "blocks"):
            return cur
        if hasattr(cur, "model") and isinstance(getattr(cur, "model"), nn.Module):
            cur = cur.model
            continue
        break
    return cur


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    val = getattr(cfg, key, None)
    return default if val is None else val


def _extra(cfg: Any) -> Dict[str, Any]:
    ex = getattr(cfg, "extra", None)
    return ex if isinstance(ex, dict) else {}


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _quant_filter(include: Sequence[str], exclude: Sequence[str]):
    def _fn(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if _matches(fqn, exclude):
            return False
        return _matches(fqn, include)
    return _fn


def quant_targets(model: nn.Module, cfg: Any = None) -> List[str]:
    """The list of fully-qualified linear names the current policy would
    quantise. Exposed for logging / notes; does not modify the model."""
    dit = _base_dit(model)
    ex = _extra(cfg) if cfg is not None else {}
    include = tuple(ex.get("quant_include", DEFAULT_QUANT_INCLUDE))
    exclude = tuple(ex.get("quant_exclude", DEFAULT_QUANT_EXCLUDE))
    fn = _quant_filter(include, exclude)
    return [n for n, m in dit.named_modules() if fn(m, n)]


_NORM_TYPES = (
    nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm,
)


def _norms_to_fp32(model: nn.Module) -> int:
    """Cast norm modules (by class name, so Wan's own LayerNorm/RMSNorm
    subclasses are covered) back to fp32. Only used when explicitly asked for
    via ``cfg.extra['keep_norms_fp32']``; the reference bf16 path in
    utils/play_world_model.py casts everything, so the default is off."""
    n = 0
    for mod in model.modules():
        cls = type(mod).__name__.lower()
        if isinstance(mod, _NORM_TYPES) or "norm" in cls:
            if any(p.dtype.is_floating_point for p in mod.parameters(recurse=False)):
                mod.float()
                n += 1
    return n


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------
def apply_precision(model: nn.Module, cfg: Any) -> nn.Module:
    """Apply ``cfg.precision`` to an already-built generator, in place.

    ``cfg`` is an ``EngineConfig`` or anything with ``.precision``,
    ``.compile_mode`` and ``.extra``. Returns the same module object (torchao
    swaps tensor subclasses in place, it does not rebuild the module tree).

    Never raises on an unusable quantisation mode: it logs a warning and leaves
    the model in bf16.
    """
    mode = str(_cfg_get(cfg, "precision", "bf16")).lower()
    ex = _extra(cfg)
    device = _cfg_get(cfg, "device", "cuda")

    if mode not in PRECISION_MODES:
        log.warning("Unknown precision %r; falling back to bf16.", mode)
        mode = "bf16"

    avail = precision_available()
    if mode in QUANT_MODES and not avail.get(mode, False):
        ao = torchao_info()
        log.warning(
            "precision=%s requested but unavailable (torchao=%s err=%s, sm=%s) "
            "-> falling back to bf16.",
            mode, ao.get("version"), ao.get("error"), _cuda_capability(),
        )
        mode = "bf16"

    # ---- base cast --------------------------------------------------------
    base_dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    model.to(device=device, dtype=base_dtype)
    model.eval()

    if bool(ex.get("keep_norms_fp32", False)) and base_dtype is not torch.float32:
        n = _norms_to_fp32(model)
        log.info("keep_norms_fp32: cast %d norm modules back to fp32.", n)

    setattr(model, "_precision_mode", mode)
    if mode in ("fp32", "bf16"):
        log.info("precision=%s applied (plain %s cast).", mode, base_dtype)
        setattr(model, "_precision_quantized", [])
        return model

    # ---- quantised modes --------------------------------------------------
    ao = torchao_info()
    quantize_ = ao["quantize_"]
    include = tuple(ex.get("quant_include", DEFAULT_QUANT_INCLUDE))
    exclude = tuple(ex.get("quant_exclude", DEFAULT_QUANT_EXCLUDE))

    dit = _base_dit(model)
    targets = quant_targets(model, cfg)
    if not targets:
        log.warning(
            "precision=%s matched 0 linears under %s (include=%s) -> bf16.",
            mode, type(dit).__name__, list(include)[:3],
        )
        setattr(model, "_precision_mode", "bf16")
        setattr(model, "_precision_quantized", [])
        return model

    try:
        if mode == "fp8_wo":
            config = ao["cfg_fp8_wo"](weight_dtype=torch.float8_e4m3fn)
        elif mode == "fp8_dyn":
            kw = {}
            gran = ex.get("fp8_granularity")
            if gran is not None:
                kw["granularity"] = gran
            config = ao["cfg_fp8_dyn"](
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                **kw,
            )
        else:  # fp4_wo
            NVFP4InferenceConfig, NVFP4MMConfig = ao["cfg_fp4_wo"]
            config = NVFP4InferenceConfig(
                mm_config=NVFP4MMConfig.WEIGHT_ONLY,
                use_triton_kernel=bool(ex.get("nvfp4_triton", False)),
                use_dynamic_per_tensor_scale=False,
            )
        quantize_(dit, config, filter_fn=_quant_filter(include, exclude))
    except Exception as exc:
        log.warning(
            "precision=%s failed during torchao quantize_ (%s: %s) -> "
            "leaving the model in bf16.", mode, type(exc).__name__, exc,
        )
        setattr(model, "_precision_mode", "bf16")
        setattr(model, "_precision_quantized", [])
        return model

    log.info("precision=%s applied to %d linears (%s ... %s).",
             mode, len(targets), targets[0], targets[-1])
    setattr(model, "_precision_quantized", targets)
    return model


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------
def compile_model(model: nn.Module, mode: str = "off",
                  *, dynamic: Optional[bool] = None) -> nn.Module:
    """torch.compile the generator at ``mode``. Call AFTER apply_precision.

    ``off`` is a no-op. Anything else compiles the DiT's per-block forwards
    (``blocks[i].forward``) rather than the whole model: the KV-cache index
    bookkeeping and the infinity-RoPE python paths in the outer forward are
    graph-break heavy, and per-block compilation keeps the hot 30x transformer
    stack in compiled regions. Set ``mode='full:<inner>'`` to compile the whole
    module instead.

    ``dynamic=None`` (the default) is deliberate. The block forward takes
    ``current_start`` as a plain python int that advances by 3 frames every
    chunk; with ``dynamic=False`` dynamo specialises on its value and
    recompiles the whole 30-block stack on *every chunk* of a rolling session,
    which is far worse than eager. ``None`` lets automatic-dynamic promote it
    to a symint after the second distinct value. Latent shapes are static
    either way.

    Never raises: a compile failure logs and returns the eager model.
    """
    mode = (mode or "off").strip()
    if mode in ("", "off", "none", "false"):
        return model
    if not hasattr(torch, "compile"):
        log.warning("torch.compile unavailable in torch %s.", torch.__version__)
        return model

    full = mode.startswith("full:")
    inner = mode.split(":", 1)[1] if full else mode
    # max-autotune-no-cudagraphs: same kernel autotuning as max-autotune but
    # WITHOUT cuda-graph capture. reduce-overhead / max-autotune both capture
    # graphs, which is incompatible with this engine's persistent, in-place
    # KV cache (cudagraph trees see a graph input that a previous replay has
    # overwritten and refuse). This mode keeps the autotuning and drops the
    # capture, so it is the only autotuned option the rolling engine can use.
    if inner not in ("reduce-overhead", "max-autotune",
                     "max-autotune-no-cudagraphs", "default"):
        log.warning("Unknown compile_mode %r; not compiling.", mode)
        return model

    try:
        if full:
            compiled = torch.compile(model, mode=inner, dynamic=dynamic)
            setattr(model, "_compile_mode", mode)
            return compiled
        dit = _base_dit(model)
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            compiled = torch.compile(model, mode=inner, dynamic=dynamic)
            setattr(model, "_compile_mode", mode)
            return compiled
        n = 0
        for blk in blocks:
            if getattr(blk, "_precision_compiled", False):
                continue
            blk.forward = torch.compile(blk.forward, mode=inner, dynamic=dynamic)
            blk._precision_compiled = True
            n += 1
        log.info("torch.compile(mode=%s) applied to %d transformer blocks.", inner, n)
        setattr(model, "_compile_mode", mode)
        return model
    except Exception as exc:
        log.warning("torch.compile(mode=%s) failed (%s: %s) -> eager.",
                    mode, type(exc).__name__, exc)
        return model


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _param_desc(p: torch.Tensor) -> str:
    """dtype label for a parameter, naming the tensor subclass when quantised."""
    cls = type(p if not isinstance(p, nn.Parameter) else p.data).__name__
    if cls not in ("Tensor", "Parameter"):
        inner = getattr(p, "dtype", None)
        return f"{cls}[{inner}]" if inner is not None else cls
    return str(p.dtype).replace("torch.", "")


def _storage_bytes(t: torch.Tensor) -> int:
    """Real bytes a parameter occupies, descending into tensor subclasses so a
    Float8Tensor / NVFP4Tensor is counted at its packed size + scales, not at
    the bf16 size its ``.dtype`` still advertises."""
    try:
        flatten = getattr(t, "__tensor_flatten__", None)
        if flatten is not None:
            names, _ = flatten()
            return sum(_storage_bytes(getattr(t, n)) for n in names)
    except Exception:
        pass
    try:
        return t.untyped_storage().nbytes()
    except Exception:
        return t.numel() * t.element_size()


def describe(model: nn.Module) -> str:
    """Per-layer-class achieved dtype summary, for logging after a rebuild."""
    dit = _base_dit(model)
    rows: Dict[tuple, int] = {}
    total_bytes = 0
    for name, mod in model.named_modules():
        own = list(mod.named_parameters(recurse=False))
        if not own:
            continue
        cls = type(mod).__name__
        dts = sorted({_param_desc(p) for _, p in own})
        quantised = any(type(p.data).__name__ not in ("Tensor", "Parameter")
                        for _, p in own)
        key = (cls, ",".join(dts), quantised)
        rows[key] = rows.get(key, 0) + 1
        for _, p in own:
            total_bytes += _storage_bytes(p.data if isinstance(p, nn.Parameter) else p)

    mode = getattr(model, "_precision_mode", "?")
    cmode = getattr(model, "_compile_mode", "off")
    nquant = len(getattr(model, "_precision_quantized", []) or [])
    lines = [
        f"precision={mode}  compile={cmode}  quantised_linears={nquant}  "
        f"root={type(dit).__name__}",
        f"{'module class':<34} {'count':>6}  dtype(s)",
        "-" * 78,
    ]
    for (cls, dts, q), n in sorted(rows.items(), key=lambda kv: (-kv[1], kv[0][0])):
        mark = " *" if q else "  "
        lines.append(f"{cls:<34} {n:>6}{mark} {dts}")
    lines.append("-" * 78)
    lines.append(f"parameter storage (packed, incl. quant scales): "
                 f"{total_bytes / 2**20:.1f} MiB")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# validation:  python interactive/precision.py --check
# ---------------------------------------------------------------------------
_STANDIN_CKPT_GLOB = (
    "/home/ashish/isambard_weights_august26/phase3_rolling/"
    "*/*/**/phase1_step0001000.pt"
)


def _find_standin_checkpoint() -> str:
    """Any complete phase1_step0001000.pt under the phase3_rolling tree.

    Prefers the plan's named sibling run; falls back to the largest complete
    file found. The target run's own file has not synced yet."""
    import glob
    import os
    root = "/home/ashish/isambard_weights_august26/phase3_rolling"
    cands = [p for p in glob.glob(os.path.join(root, "**", "phase1_step0001000.pt"),
                                  recursive=True)
             if os.path.getsize(p) > 10 * 2**30]
    if not cands:
        raise SystemExit(
            f"No complete phase1_step0001000.pt under {root} — "
            "cannot run --check without a stand-in checkpoint."
        )
    preferred = [p for p in cands if "no_aux_gbs16_3008_commitgan_rehab" in p]
    return sorted(preferred or cands)[0]


def _check_one(mode: str, args) -> dict:
    """Build the real generator fresh, apply `mode`, run dummy forwards."""
    import os
    import sys
    import time

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from utils.play_world_model import (                          # do not edit it
        WorldModelPlayer, _autodetect_wan_path,
    )
    from interactive.engine_api import (
        EngineConfig, LATENT_C, LATENT_H, LATENT_W, NUM_FRAME_PER_BLOCK,
    )

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    ckpt = args.ckpt or _find_standin_checkpoint()
    log.info("[%s] stand-in checkpoint: %s", mode, ckpt)

    player = WorldModelPlayer(
        config_path=args.config,
        ckpt_path=ckpt,
        device=device,
        wan_model_path=args.wan_model_path or _autodetect_wan_path(),
        use_ema=True,
        denoising_steps=None,
        kv_cache_chunks=args.kv_cache_chunks,
        infinity_rope=None,
        weights_dtype="bf16",
    )

    cfg = EngineConfig(precision=mode, compile_mode=args.compile_mode,
                       device="cuda", kv_cache_chunks=args.kv_cache_chunks)
    t_apply = time.time()
    apply_precision(player.generator, cfg)
    achieved = getattr(player.generator, "_precision_mode", mode)
    if args.compile_mode != "off":
        compile_model(player.generator, args.compile_mode)
    apply_s = time.time() - t_apply

    desc = describe(player.generator)

    # --- dummy conditioning: zero action + dummy prompt embedding ----------
    g = torch.Generator(device="cpu").manual_seed(0)
    prompt = (torch.randn(1, 512, 4096, generator=g) * 0.05).to(device, torch.bfloat16)
    seed_lat = torch.randn(
        [1, NUM_FRAME_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W], generator=g,
    ).to(device)

    player.reset(seed_lat, prompt, neutral_action=(0.0, 0.0))

    # --- one generator forward, correct shapes ----------------------------
    x = torch.randn([1, NUM_FRAME_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W],
                    generator=g).to(device, torch.bfloat16)
    cond = player._action_cond(player._make_action_fa(0.0, 0.0))
    ts = torch.full([1, NUM_FRAME_PER_BLOCK], float(player.denoising_step_list[0]),
                    device=device, dtype=torch.float32)

    def _fwd(start_frame: Optional[int] = None):
        sf = player.current_start_frame if start_frame is None else start_frame
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return player.generator(
                noisy_image_or_video=x,
                conditional_dict=cond,
                timestep=ts,
                kv_cache=player.kv_cache,
                crossattn_cache=player.crossattn_cache,
                current_start=sf * player.frame_seq_length,
            )

    def _graph_count() -> int:
        try:
            from torch._dynamo.utils import counters
            return int(counters["stats"].get("unique_graphs", 0))
        except Exception:
            return -1

    t_warm = time.time()
    for _ in range(args.warmup):
        out = _fwd()
    torch.cuda.synchronize()
    warmup_s = time.time() - t_warm
    graphs_after_warmup = _graph_count()

    lat_ms: List[float] = []
    for _ in range(args.iters):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        out = _fwd()
        ev1.record()
        torch.cuda.synchronize()
        lat_ms.append(ev0.elapsed_time(ev1))

    # --- rolling probe: does an advancing current_start force recompiles? ---
    roll_ms: List[float] = []
    base = player.current_start_frame
    for k in range(1, 5):
        t0 = time.time()
        _fwd(base + k * NUM_FRAME_PER_BLOCK)
        torch.cuda.synchronize()
        roll_ms.append((time.time() - t0) * 1e3)
    graphs_after_roll = _graph_count()

    pred = out[1] if isinstance(out, (tuple, list)) else out
    pred_f = pred.float()
    finite = bool(torch.isfinite(pred_f).all().item())
    stats = {
        "mode": mode,
        "achieved_mode": achieved,
        "fell_back": achieved != mode,
        "ok": finite,
        "finite": finite,
        "out_shape": list(pred.shape),
        "out_absmax": float(pred_f.abs().max().item()),
        "out_std": float(pred_f.std().item()),
        "latency_ms_median": float(sorted(lat_ms)[len(lat_ms) // 2]),
        "latency_ms_min": float(min(lat_ms)),
        "latency_ms_all": [round(v, 2) for v in lat_ms],
        "apply_seconds": round(apply_s, 2),
        "warmup_seconds": round(warmup_s, 2),
        "graphs_after_warmup": graphs_after_warmup,
        "graphs_after_roll": graphs_after_roll,
        "rolling_probe_ms": [round(v, 1) for v in roll_ms],
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "quantised_linears": len(getattr(player.generator, "_precision_quantized", [])),
        "compile_mode": args.compile_mode,
        "ckpt": ckpt,
        "describe": desc,
    }
    player.close()
    return stats


def _main() -> int:
    import argparse
    import json
    import os
    import subprocess
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="run every available mode, each in a fresh subprocess")
    p.add_argument("--check-one", default=None, metavar="MODE",
                   help="internal: run a single mode in this process")
    p.add_argument("--modes", default=None,
                   help="comma-separated subset of modes to check")
    p.add_argument("--ckpt", default=None, help="override the stand-in checkpoint")
    p.add_argument("--config", default="configs/action_forcing_phase3_dmd.yaml")
    # NOTE: this is the PARENT of Wan2.1-T2V-1.3B (utils/wan_wrapper appends the
    # model-name dir). None => utils.play_world_model._autodetect_wan_path().
    p.add_argument("--wan_model_path", default=None)
    p.add_argument("--kv_cache_chunks", type=int, default=7)
    p.add_argument("--compile_mode", default="off", choices=list(COMPILE_MODES))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    avail = precision_available()
    ao = torchao_info()

    if args.check_one:
        res = _check_one(args.check_one, args)
        print("\n" + res["describe"])
        print("\n__RESULT__" + json.dumps({k: v for k, v in res.items()
                                           if k != "describe"}))
        return 0 if res["ok"] else 1

    if not args.check:
        print(f"torch {torch.__version__}  cuda_cap={_cuda_capability()}")
        print(f"torchao {ao.get('version')}  err={ao.get('error')}")
        print("precision_available():")
        for k in PRECISION_MODES:
            print(f"  {k:<8} {avail.get(k)}")
        return 0

    modes = ([m.strip() for m in args.modes.split(",")] if args.modes
             else [m for m in PRECISION_MODES if avail.get(m)])
    print(f"torch {torch.__version__}  cuda_cap={_cuda_capability()}  "
          f"torchao {ao.get('version')}")
    print(f"available: {avail}")
    print(f"checking : {modes}\n")

    results = []
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for m in modes:
        print(f"===== {m} =====", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--check-one", m,
               "--config", args.config,
               "--kv_cache_chunks", str(args.kv_cache_chunks),
               "--compile_mode", args.compile_mode,
               "--warmup", str(args.warmup), "--iters", str(args.iters)]
        if args.ckpt:
            cmd += ["--ckpt", args.ckpt]
        if args.wan_model_path:
            cmd += ["--wan_model_path", args.wan_model_path]
        env = dict(os.environ, PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", ""))
        proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                payload = json.loads(line[len("__RESULT__"):])
            else:
                print(line)
        if payload is None:
            print(proc.stderr[-4000:])
            payload = {"mode": m, "ok": False, "error": "no result",
                       "returncode": proc.returncode}
        else:
            print(proc.stderr[-1500:] if proc.returncode else "", end="")
        results.append(payload)
        print()

    print("\n" + "=" * 92)
    hdr = (f"{'mode':<9} {'ok':<4} {'achieved':<9} {'quantL':>6} "
           f"{'lat_ms(med)':>12} {'lat_ms(min)':>12} {'peakVRAM_GB':>12} {'absmax':>9}")
    print(hdr)
    print("-" * 92)
    for r in results:
        if not r.get("ok"):
            print(f"{r.get('mode'):<9} {'FAIL':<4} {r.get('error', '')}")
            continue
        print(f"{r['mode']:<9} {'ok':<4} {r['achieved_mode']:<9} "
              f"{r['quantised_linears']:>6} {r['latency_ms_median']:>12.2f} "
              f"{r['latency_ms_min']:>12.2f} {r['peak_vram_gb']:>12.2f} "
              f"{r['out_absmax']:>9.2f}")
    print("=" * 92)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
