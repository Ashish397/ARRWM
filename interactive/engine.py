#!/usr/bin/env python3
"""WS-A: engine core for the interactive world model.

Implements :class:`interactive.engine_api.EngineBase` on top of the lean
inference stack validated in ``utils/play_world_model.py`` (generator +
action heads + VAE + one-shot CPU T5 encode).  That script stays the
reference implementation and is imported, never edited.

What this module adds over the old play script
----------------------------------------------
1. **Trained denoising ladder.**  ``[1000, 625, 357.142857, 208.333333]``
   from the rolling-definitive launchers, NOT the phase-1 yaml default
   ``[1000, 625, 312.5, 178.6]``. ``cfg.denoising_steps=N`` takes the top-N
   prefix through four; for N > 4 the default ladder subdivides the gaps while
   retaining every trained anchor. ``cfg.extra['denoising_step_list']``
   overrides the pool.
2. **KV buffer 45 frames / attention span 21 frames / sink 0** — serving
   allocates 21 seed + 21 rollout + 3 headroom frames of buffer but caps the
   attention span at the trained ``local_attn_size=21``.  1561 tokens per
   frame (1560 spatial + 1 action token).
3. **7-chunk ground-truth serving prefill** (21 latent frames), filling the
   complete attention span and conditioned on the seed frames' OWN actions
   (not neutral zeros). Training itself used 3 chunks / 9 latent frames.
4. **pca_raw actions on dims [0, 1]** (pca_0 = throttle, pca_1 = steer);
   the retired ``[z2, z7]`` ss_vae lineage is gone.
5. **CARN commit correction.**  At step 1000 the trained recurrence applies
   the reverse operator G on every commit (one Euler step, moment
   preservation, alpha=0.125 blend under a 1.5 % relative-L2 trust cap).
   Enabled automatically when the checkpoint carries ``reverse_noiser``.
6. **Raw ``generator`` weights by default.**  ``generator_ema`` is a
   trainable-params-only fp32 shadow and cannot load strictly; it is only
   overlaid non-strictly when explicitly requested.
7. **6-chunk latched action horizon** with VAE decode pipelined onto a
   second CUDA stream so ``decode(chunk N)`` overlaps ``gen(chunk N+1)``.

Selftest::

    conda run -n flash python interactive/engine.py --selftest \
        --ckpt <phase1_step0001000.pt> --seed_zarr ~/20240224003808.zarr \
        --wan_model_path /home/ashish/Wan2.1/Wan2.1-T2V-1.3B/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Pin inductor's on-disk cache to a stable path BEFORE torch is imported, so
# compiled kernels and the FX graph cache survive across launches. Without
# this it lands in a per-boot /tmp directory and every session pays the full
# multi-minute compile again. Respect an explicit override.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      str(_REPO_ROOT / "interactive" / ".inductor_cache"))

import torch

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interactive.engine_api import (  # noqa: E402
    Action,
    EngineBase,
    EngineConfig,
    HorizonStats,
    LATENT_C,
    LATENT_H,
    LATENT_W,
    NUM_FRAME_PER_BLOCK,
    PIXEL_FRAMES_PER_CHUNK,
    PHYSICAL_NULL_STEER,
    PHYSICAL_NULL_THROTTLE,
    SEED_PREFILL_CHUNKS,
    TRAINED_DENOISING_STEP_LIST,
)

log = logging.getLogger("interactive.engine")


# ---------------------------------------------------------------------------
# Model construction — thin wrapper over the validated reference builder.
# ---------------------------------------------------------------------------
def _normalize_wan_path(path: str) -> str:
    """utils.wan_wrapper expects the PARENT of ``Wan2.1-T2V-1.3B/``.

    ``EngineConfig.wan_model_path`` (and the plan doc) quote the path with
    the model directory included, so strip a trailing model-name component
    and accept either form.
    """
    q = Path(path.rstrip("/"))
    if q.name.startswith("Wan2.1-T2V"):
        q = q.parent
    return str(q) + "/"


#: Keys the interactive engine actually consumes. Everything else in a
#: training checkpoint (optimizer, fake_score, discriminator, EMA shadow,
#: critics) is dead weight at inference time.
LEAN_CKPT_KEYS = ("generator", "action_projection", "action_token_projection",
                  "reverse_noiser", "forward_noiser", "step", "config_name")


def lean_ckpt_path(ckpt_path: str) -> Path:
    return Path(str(ckpt_path) + ".lean.pt")


def _torch_load(path, *, mmap: bool = True):
    """torch.load with mmap when the file supports it.

    mmap keeps the (17 GB) checkpoint out of RSS and pages in only the
    tensors actually touched, which matters because the full file is opened
    more than once during startup. Older archive formats reject it, so fall
    back rather than fail.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, TypeError, ValueError) as exc:
        if mmap:
            log.debug("mmap load unavailable for %s (%s); plain load.", path, exc)
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


def save_lean_checkpoint(ckpt_path: str, out_path: Optional[str] = None,
                         dtype: torch.dtype = torch.bfloat16) -> str:
    """Write a bf16, inference-only copy of ``ckpt_path``.

    Keeps just the tensors the engine loads and casts floating-point ones to
    bf16 -- which is exactly what the engine builds by default, so for a bf16
    session this is lossless. Integer/bool tensors are left alone.
    """
    out = Path(out_path) if out_path else lean_ckpt_path(ckpt_path)
    t0 = time.perf_counter()
    raw = _torch_load(ckpt_path)
    if not isinstance(raw, dict):
        raise SystemExit(f"unexpected checkpoint format: {type(raw)}")

    def _cast(v):
        if torch.is_tensor(v):
            return v.to(dtype).contiguous() if v.is_floating_point() else v.contiguous()
        if isinstance(v, dict):
            return {k: _cast(x) for k, x in v.items()}
        return v

    lean = {}
    for k in LEAN_CKPT_KEYS:
        if k in raw:
            lean[k] = _cast(raw[k])
    lean["_lean_from"] = str(ckpt_path)
    lean["_lean_dtype"] = str(dtype)
    missing = [k for k in ("generator", "action_projection",
                           "action_token_projection") if k not in lean]
    if missing:
        raise SystemExit(f"source checkpoint is missing required keys: {missing}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(lean, tmp)
    tmp.replace(out)
    src_gb = Path(ckpt_path).stat().st_size / 2**30
    dst_gb = out.stat().st_size / 2**30
    log.info("Lean checkpoint: %s (%.1f GB) -> %s (%.1f GB) in %.1fs [kept %s]",
             Path(ckpt_path).name, src_gb, out, dst_gb,
             time.perf_counter() - t0, sorted(set(lean) & set(LEAN_CKPT_KEYS)))
    return str(out)


def resolve_ckpt(cfg: EngineConfig) -> str:
    """Prefer a sibling ``.lean.pt`` when one exists and is usable."""
    src = cfg.ckpt_path
    # Logged because a stray use_ema=True silently costs both the EMA
    # semantics and the lean-checkpoint speedup.
    log.info("Resolved use_ema=%s, precision=%s", cfg.use_ema, cfg.precision)
    if not src or bool(cfg.extra.get("no_lean_ckpt", False)):
        return src
    lean = lean_ckpt_path(src)
    if not lean.is_file():
        return src
    if cfg.use_ema:
        log.info("Lean checkpoint present but use_ema=True needs the full file "
                 "(no EMA shadow in a lean copy); using %s", src)
        return src
    if cfg.precision == "fp32":
        log.warning("Lean checkpoint is bf16 but precision=fp32 was asked for; "
                    "using the full checkpoint to honour it.")
        return src
    log.info("Using lean checkpoint %s (%.1f GB)", lean.name,
             lean.stat().st_size / 2**30)
    return str(lean)


def build_player(cfg: EngineConfig):
    """Build the lean generator + action heads + VAE.

    Reuses ``utils.play_world_model.WorldModelPlayer.__init__`` verbatim
    (that is the validated build path: CausalWanModel + action_projection +
    action_token_projection + VAE, no teacher / fake_score / GAN / CARN).
    The module-global Wan-weights override is handled exactly as the play
    script does (``utils.wan_wrapper._default_wan_model_path``), which the
    constructor itself performs given ``wan_model_path``.
    """
    from utils.play_world_model import WorldModelPlayer

    weights_dtype = "fp32" if cfg.precision == "fp32" else "bf16"
    if cfg.precision not in ("fp32", "bf16"):
        log.warning(
            "precision=%s not handled by engine.py; building bf16 weights and "
            "deferring to interactive/precision.py if present.", cfg.precision,
        )

    # use_ema=False here ALWAYS: production evals load the raw `generator`
    # key (strict, 825 tensors).  `generator_ema` is a trainable-params-only
    # fp32 shadow with no buffers/frozen params and cannot load strictly, so
    # when it is requested we overlay it non-strictly on top of the raw
    # weights below.
    ckpt_for_build = resolve_ckpt(cfg)
    player = WorldModelPlayer(
        config_path=cfg.config_path,
        ckpt_path=ckpt_for_build,
        device=torch.device(cfg.device),
        wan_model_path=_normalize_wan_path(cfg.wan_model_path),
        use_ema=False,
        denoising_steps=None,              # ladder is set by the engine below
        kv_cache_chunks=int(cfg.kv_cache_chunks),
        infinity_rope=None,                # read from config (true)
        weights_dtype=weights_dtype,
    )
    player.weights_source = "generator (raw, strict)"

    if cfg.use_ema:
        raw = _torch_load(cfg.ckpt_path)
        ema = raw.get("generator_ema")
        if ema is None:
            log.warning("use_ema requested but checkpoint has no 'generator_ema'; "
                        "keeping raw 'generator' weights.")
        else:
            ema = {k.replace("_fsdp_wrapped_module.", "").replace("module.", ""): v
                   for k, v in ema.items()}
            missing, unexpected = player.generator.model.load_state_dict(ema, strict=False)
            log.warning(
                "WEIGHTS: overlaid 'generator_ema' NON-STRICTLY on top of raw "
                "'generator' (%d ema tensors; %d keys left at raw values, "
                "%d unexpected). Production evals use raw 'generator'.",
                len(ema), len(missing), len(unexpected),
            )
            player.weights_source = "generator + generator_ema overlay (non-strict)"
        del raw
    log.info("WEIGHTS LOADED: %s", player.weights_source)

    # WS-B hook: optional precision/compile layer.  Guarded — engine.py must
    # not depend on a file it does not own.
    try:
        from interactive.precision import apply_precision  # type: ignore
    except ImportError:
        if cfg.precision not in ("fp32", "bf16") or cfg.compile_mode != "off":
            log.warning("interactive/precision.py absent; precision=%s compile=%s ignored.",
                        cfg.precision, cfg.compile_mode)
    else:
        log.info("Applying interactive.precision.apply_precision(precision=%s)", cfg.precision)
        apply_precision(player.generator, cfg)
        if cfg.compile_mode != "off":
            try:
                from interactive.precision import compile_model  # type: ignore
            except ImportError:
                log.warning("precision.compile_model absent; compile_mode=%s ignored.",
                            cfg.compile_mode)
            else:
                log.warning(
                    "compile_mode=%s: WARMUP TAKES MINUTES and the first "
                    "horizons will be slow. utils/infinity_rope.py calls "
                    ".item()/.tolist(), which the inductor backend cannot "
                    "trace, so it graph-breaks noisily and falls back to "
                    "eager for those regions. Measured on this engine, "
                    "compile is NOT worth it (-1.6%% e2e at 4 steps, +3.6%% "
                    "at 1 step); compile_mode='off' is the recommended "
                    "default.", cfg.compile_mode)
                log.info("Compiling generator (mode=%s)", cfg.compile_mode)
                compile_model(player.generator, cfg.compile_mode)

    return player


def load_reverse_noiser(ckpt_path: str, *, device, dtype):
    """Build the frozen CARN reverse operator G from the checkpoint.

    Full phase-1 checkpoints of this run carry a ``reverse_noiser`` key (the
    ForwardNoiser trained with ``forward_noiser_reverse=true``, i.e. CARN
    level n -> n-1).  Hyper-parameters are inferred from the state dict so a
    non-default hidden_dim/num_blocks cannot be silently absorbed.

    Returns ``None`` when the key is absent.
    """
    # mmap: this is a SECOND full open of the checkpoint (the player already
    # read it once for the generator), and we want exactly one small key out
    # of it -- paging in just that key instead of 17 GB.
    raw = _torch_load(ckpt_path)
    sd = raw.get("reverse_noiser") if isinstance(raw, dict) else None
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict) or "out_proj.weight" not in sd:
        del raw
        return None
    sd = {k.replace("_fsdp_wrapped_module.", "").replace("module.", ""): v
          for k, v in sd.items()}

    from model.forward_noiser import ForwardNoiser
    hidden_dim = int(sd["out_proj.weight"].shape[1])
    latent_ch = int(sd["out_proj.weight"].shape[0])
    n_blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    max_step = 16
    for k, v in sd.items():
        if k.endswith("emb.weight") and v.dim() == 2:
            max_step = int(v.shape[0]) - 1
            break
    g = ForwardNoiser(latent_channels=latent_ch, hidden_dim=hidden_dim,
                      num_blocks=n_blocks, max_carn_step=max_step)
    g.load_state_dict(sd, strict=True)
    g = g.to(device=device, dtype=dtype).eval().requires_grad_(False)
    log.info("CARN reverse_noiser loaded from checkpoint "
             "(latent_ch=%d hidden=%d blocks=%d max_carn_step=%d).",
             latent_ch, hidden_dim, n_blocks, max_step)
    del raw
    return g


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class WorldModelEngine(EngineBase):
    """6-chunk latched-action horizon engine with pipelined VAE decode."""

    def __init__(self, cfg: EngineConfig):
        super().__init__(cfg)
        self.device = torch.device(cfg.device)
        torch.set_grad_enabled(False)

        self.p = build_player(cfg)
        self.dtype = self.p.dtype
        self.npb = int(self.p.num_frame_per_block)
        if self.npb != NUM_FRAME_PER_BLOCK:
            log.warning("config num_frame_per_block=%d != API constant %d",
                        self.npb, NUM_FRAME_PER_BLOCK)

        # --- multi-chunk generation ---------------------------------------
        self.block_chunks = int(cfg.extra.get("block_chunks",
                                              getattr(cfg, "block_chunks", 1)) or 1)
        if self.block_chunks not in (1, 2, 4):
            log.warning("block_chunks=%d is not one of (1, 2, 4); clamping to 1.",
                        self.block_chunks)
            self.block_chunks = 1
        self.generation_mode = "sequential" if self.block_chunks == 1 else "joint"
        self.block_frames = self.block_chunks * self.npb
        self._set_ladder(cfg.denoising_steps)

        # --- KV geometry (training truth) ---------------------------------
        # Attention SPAN is the trained local_attn_size (21 frames = 7
        # chunks); the BUFFER is larger (21 seed + 21 rollout + 3 headroom =
        # 45 frames) so the FIFO can evict one chunk per commit without ever
        # clipping the live span.  frame_seq_length = 1561.
        self.frame_seq_length = int(self.p.frame_seq_length)
        self.attn_span_frames = int(cfg.kv_cache_chunks) * self.npb
        self.kv_buffer_frames = int(cfg.extra.get(
            "kv_buffer_frames",
            # Allocate for the maximum selectable seed even when this ride
            # starts shorter, so changing seed length only needs a reset and
            # never a heavyweight model rebuild.
            SEED_PREFILL_CHUNKS * self.npb + self.attn_span_frames + self.npb,
        ))
        self.p.local_attn_size_frames = self.attn_span_frames
        self.p.kv_cache_tokens = self.kv_buffer_frames * self.frame_seq_length

        # --- CARN commit correction ---------------------------------------
        # Training's recurrence applies G on EVERY commit at absolute step
        # 1000: one Euler step, moment preservation, alpha=0.125 blend under
        # a 1.5 % relative-L2 trust cap.  'auto' = on iff the checkpoint has
        # a reverse_noiser.
        mode = str(cfg.extra.get("carn_commit", "auto"))
        self.carn_alpha = float(cfg.extra.get("carn_commit_alpha", 0.125))
        self.carn_max_rel_shift = float(cfg.extra.get("carn_commit_max_relative_shift", 0.015))
        self.carn_min_level = int(cfg.extra.get("carn_commit_min_level", 1))
        self.carn_max_level = int(cfg.extra.get("carn_commit_max_level", 16))
        self.reverse_noiser = None
        if mode in ("auto", "on"):
            self.reverse_noiser = load_reverse_noiser(
                cfg.ckpt_path, device=self.device, dtype=self.dtype)
            if self.reverse_noiser is None:
                log.warning(
                    "CARN commit correction DISABLED: no 'reverse_noiser' in the "
                    "checkpoint. This matches the pre-step-150 / no-commit "
                    "regime; expect a bounded ~1.5%%/chunk deviation from the "
                    "trained recurrence.")
            else:
                log.info("CARN commit correction ON (alpha=%.4f, cap=%.4f, "
                         "levels %d..%d).", self.carn_alpha,
                         self.carn_max_rel_shift, self.carn_min_level,
                         self.carn_max_level)
        else:
            log.info("CARN commit correction disabled by config (carn_commit=%s).", mode)

        # Eval-only seam affine (run_eval60.sh serving extra, lambda=0.5).
        # Training's seam correction is identity in this run -> default OFF.
        self.seam_affine_lambda = float(cfg.extra.get("carn_seam_affine_lambda", 0.0))
        self._seam_target = None
        self._last_carn = None      # (level, effective alpha) telemetry

        self.round_timesteps = bool(cfg.extra.get("round_timesteps", False))

        # --- decode stream ownership --------------------------------------
        # The Wan VAE's streaming decode is STATEFUL: `cached_decode` keeps a
        # feat_cache of ~30 tensors and reassigns (frees) them every call.
        # The caching allocator returns a freed block to the pool of the
        # stream it was ALLOCATED on, so a cache tensor allocated on stream A
        # and freed while stream B still reads it is a use-after-free.  The
        # invariant that fixes it: for the whole session, EVERY VAE op runs
        # on exactly one stream (`self._vae_stream`) — seed decode at reset
        # included.  Only the two tensors that cross (latents in, frames out)
        # are event-gated and record_stream'd.
        self.decode_same_stream = bool(cfg.extra.get("decode_same_stream", False))
        self._decode_stream = (
            None if self.decode_same_stream else torch.cuda.Stream(device=self.device)
        )
        log.info("Decode mode: %s",
                 "SAME-STREAM (no gen/decode overlap)" if self.decode_same_stream
                 else "OVERLAPPED (VAE pinned to a dedicated side stream)")

        # --- latent->pixel decoder ----------------------------------------
        # "wan" keeps the original in-line path through `p.vae` (the
        # training/eval reference) so nothing about the existing numbers
        # moves, and so we don't pay for a second copy of the Wan VAE.
        # Anything else comes from interactive/decoders.py, which obeys the
        # same streaming contract (reset() once per ride, then one
        # decode_chunk per chunk) and is built on the VAE stream so its
        # activations belong to the stream that later frees them.
        self.decoder = None
        dec_name = str(getattr(cfg, "decoder", "wan") or "wan").lower()
        if dec_name != "wan":
            from interactive.decoders import build_decoder
            with self._vae_scope():
                self.decoder = build_decoder(
                    dec_name, device=self.device, dtype=self.dtype)
            log.info("Decoder: %s (%.1fM params)", dec_name,
                     self.decoder.parameter_count() / 1e6)
        else:
            log.info("Decoder: wan (reference VAE)")

        self._stats: Optional[HorizonStats] = None
        self._horizon_index = 0
        self._prompt_embeds: Optional[torch.Tensor] = None
        self._seed_latents: Optional[torch.Tensor] = None
        self.latents_log: List[torch.Tensor] = []
        # Mid-horizon action pickup: a callable the UI installs so the engine
        # can re-read the action mailbox at each chunk boundary.  Without it
        # the action is latched once per horizon (up to horizon_chunks chunks
        # of already-decided video before a keypress is even consulted).
        self._action_provider: Optional[Callable[[], Optional[Action]]] = None
        self._live_action: Optional[Action] = None
        self._action_changed_at_chunk: int = -1

        log.info(
            "Engine ready | ladder=%s | attn_span=%d frames (%d chunks) | "
            "kv_buffer=%d frames x %d tokens | horizon=%d chunks | "
            "block=%d chunks mode=%s (%d frames, ~%d context frames in span) | "
            "precision=%s",
            [round(float(x), 3) for x in self._ladder.tolist()],
            self.attn_span_frames, int(cfg.kv_cache_chunks),
            self.kv_buffer_frames, self.frame_seq_length,
            int(cfg.horizon_chunks), self.block_chunks, self.generation_mode,
            self.block_frames,
            max(self.attn_span_frames - self.block_frames, 0), cfg.precision,
        )
        if self.block_chunks > 1:
            log.info(
                "Legacy joint-block mode: %d frames denoised jointly per forward. Every "
                "block frame attends to ALL other block frames AND to an "
                "IDENTICAL context set (the attention path builds one K/V "
                "slice per forward and runs causal=False, so there is no "
                "per-query window asymmetry). Context in span drops to ~%d "
                "frames; action pickup coarsens to one action per block "
                "(%d pixel frames).",
                self.block_frames, max(self.attn_span_frames - self.block_frames, 0),
                self.block_chunks * PIXEL_FRAMES_PER_CHUNK)

    # -- denoising ladder ---------------------------------------------------
    @staticmethod
    def _subdivide_ladder(pool: List[float], n: int) -> List[float]:
        """Keep every trained rung and subdivide the gaps between them.

        ``n - K`` extra rungs are spread over the ``K - 1`` gaps as evenly as
        possible; leftovers go to the WIDEST gaps first, since those are the
        jumps the sampler is least able to make accurately.  New rungs are
        linear interpolations between their neighbours (linear in t/1000 is
        the same ordering as linear in t, the 1/1000 being a constant scale).

        The result is strictly descending, starts at the first trained rung
        and ENDS ON the last trained one -- never below it, so the sampler
        still finishes where training finished.
        """
        K = len(pool)
        n_gaps = K - 1
        extra = max(0, n - K)
        base, rem = divmod(extra, n_gaps)
        counts = [base] * n_gaps
        widest = sorted(range(n_gaps), key=lambda i: -(pool[i] - pool[i + 1]))
        for j in range(rem):
            counts[widest[j]] += 1

        out: List[float] = []
        for i in range(n_gaps):
            a, b = pool[i], pool[i + 1]
            out.append(a)
            m = counts[i]
            for k in range(1, m + 1):
                out.append(a + (b - a) * k / (m + 1))
        out.append(pool[-1])
        return out

    def _set_ladder(self, n_steps: Optional[int]) -> None:
        """Resolve the denoising ladder for ``n_steps`` rungs.

        The trained pool is ``[1000, 625, 357.142857, 208.333333]`` (resolved
        from sbatch/carncommit_long.sbatch, NOT the yaml's phase-1 list).
        Override it with ``EngineConfig.extra['denoising_step_list']``.

        ``extra['ladder_mode']`` selects what happens past the trained pool:

        interp (default)  keep all trained rungs, subdivide the gaps.  Every
                          timestep the model was trained on is still visited,
                          and the new ones sit between them -- so nothing is
                          far out of distribution.
        trained           top-N prefix while N <= K; past that fall back to
                          the sigma grid (the historical behaviour).
        grid              always build the model's shift-5 sigma grid
                          t = 1000*s*sigma / (1 + (s-1)*sigma).

        For N <= K, interp and trained are identical (there is nothing to
        subdivide), so the default change only affects N > 4.
        """
        pool = self.cfg.extra.get("denoising_step_list", TRAINED_DENOISING_STEP_LIST)
        pool_l = sorted((float(x) for x in pool), reverse=True)
        K = len(pool_l)
        n = K if n_steps is None else int(n_steps)
        if n < 1:
            log.warning("denoising_steps=%s < 1; using %d.", n_steps, K)
            n = K
        mode = str(self.cfg.extra.get("ladder_mode", "interp")).lower()
        if mode not in ("interp", "trained", "grid"):
            log.warning("unknown ladder_mode=%r; using 'interp'.", mode)
            mode = "interp"

        def _grid(nn: int) -> List[float]:
            shift = float(self.cfg.extra.get("timestep_shift", 5.0))
            sig = torch.linspace(1.0, 0.0, nn + 1)[:-1]
            return (1000.0 * shift * sig / (1.0 + (shift - 1.0) * sig)).tolist()

        if mode == "grid":
            vals = _grid(n)
        elif n <= K:
            vals = pool_l[:n]
        elif mode == "interp":
            vals = self._subdivide_ladder(pool_l, n)
        else:                                   # trained, past the pool
            vals = _grid(n)
            log.warning("denoising_steps=%d > trained pool %d and "
                        "ladder_mode='trained' -> sigma grid.", n, K)

        self._ladder = torch.tensor(vals, dtype=torch.float32,
                                    device=self.device).contiguous()
        # A non-descending ladder would silently re-noise mid-traverse.
        d = self._ladder[1:] - self._ladder[:-1]
        if self._ladder.numel() > 1 and bool((d >= 0).any()):
            log.error("ladder is not strictly descending: %s",
                      [round(v, 3) for v in vals])
        log.info("Ladder: mode=%s steps=%d -> %s", mode, n,
                 [round(v, 2) for v in vals])
        # Keep the reference player's attribute in sync so any borrowed code
        # path (and the HUD) sees the same ladder.
        self.p.denoising_step_list = self._ladder

    # -- actions ------------------------------------------------------------
    def _action_vector(self, action: Action) -> torch.Tensor:
        """Action -> per-frame conditioning tensor [1, npb, raw_action_dim].

        Lineage: ``pca_raw`` with ``action_dims=[0, 1]`` — pca_0 = throttle,
        pca_1 = steer.  Training feeds tanh-squashed values in (-1, 1)
        (utils/zarr_dataset.py: ``tanh(P / _PCA_RAW_SCALES)``), so a
        normalized Action maps straight through with no extra scaling.

        The alignment audit CONFIRMED this mapping: the model consumes
        [1, 3, 2] per chunk, the same value on all 3 frames (actions are
        chunk-grain), already tanh-squashed.  The projections do no clipping,
        so inputs are clamped defensively here.  Practical dynamic range:
        the squashed training distribution has std ~0.3, so |value| -> 1 is
        deep saturation.  This is deliberately the ONLY place the mapping
        lives: any revision is a one-line patch here.
        """
        return self._action_vector_raw(action.throttle, action.steer,
                                       frames=self.block_frames)

    def _action_vector_raw(self, throttle: float, steer: float,
                           frames: Optional[int] = None) -> torch.Tensor:
        """[1, frames, raw_action_dim]; `frames` defaults to one chunk.

        In block mode the SAME action is replicated across every frame of the
        block, exactly as it is replicated across the 3 frames of a chunk --
        actions are chunk-grain in training, and a block is a whole number of
        chunks. ``p._action_cond`` derives its frame count from this tensor
        (ActionModulationProjection overrides its own ``num_frames`` kwarg
        with ``action_features.shape[1]``), so a wider block needs no other
        change on the conditioning side.
        """
        d = int(self.p.raw_action_dim)
        nf = int(frames if frames is not None else self.npb)
        vec = torch.zeros(d, dtype=self.dtype, device=self.device)
        vec[0] = float(np.clip(throttle, -1.0, 1.0))   # pca_0 = throttle
        if d > 1:
            vec[1] = float(np.clip(steer, -1.0, 1.0))  # pca_1 = steer
        return vec.view(1, 1, -1).expand(1, nf, d).contiguous()

    def _cond(self, action_fa: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.p._action_cond(action_fa)

    # -- compile warmup ------------------------------------------------------
    @torch.no_grad()
    def warmup(self, progress: Optional[Callable[[str, int, int], None]] = None,
               horizons: int = 1) -> float:
        """Pre-compile every shape the live loop will hit, then reset clean.

        With ``compile_mode != off`` inductor compiles on FIRST HIT of each
        shape. Left to happen during live play that stalls generation for
        minutes inside the opening horizons, which looks exactly like "it
        isn't generating while rendering" -- the pipeline is fine, it is
        simply blocked in the compiler. Doing it here, up front and
        announced, turns an invisible freeze into a visible one-time cost.

        Rolls throwaway horizons, discards their latents, and resets the ride
        so the session starts from a clean seed. Returns seconds spent.

        ``progress(phase, done, total)`` is called as it advances so the UI
        can show something other than a frozen window.
        """
        if str(self.cfg.compile_mode or "off") == "off":
            return 0.0
        n_chunks = int(self.cfg.horizon_chunks)
        bc = max(int(self.block_chunks), 1)
        # Chunk counts per forward block; the last one is short when the
        # horizon is not a whole number of blocks.
        block_sizes = [min(bc, n_chunks - i) for i in range(0, n_chunks, bc)]
        total = max(1, int(horizons)) * n_chunks
        t0 = time.perf_counter()
        log.warning("Compile warmup: running %d throwaway horizon(s) to compile "
                    "kernels before going live. This is the one-time cost; with "
                    "TORCHINDUCTOR_CACHE_DIR=%s later launches at the same "
                    "settings reuse the FX graph cache and start much faster.",
                    horizons, os.environ.get("TORCHINDUCTOR_CACHE_DIR", "<default>"))
        if progress:
            progress("seed", 0, total)
        # reset() compiles the seed-prefill / commit shapes.
        self.reset()
        done = 0
        for h in range(max(1, int(horizons))):
            for _ in self.generate_horizon(Action()):
                done += 1
                if progress:
                    progress("chunk", min(done, total), total)
        # Discard everything the warmup produced; the ride must start clean.
        self.latents_log = []
        self.reset()
        dt = time.perf_counter() - t0
        log.warning("Compile warmup done in %.1fs (%d chunks). Live play now "
                    "streams per chunk at steady-state speed.", dt, done)
        if progress:
            progress("done", total, total)
        return dt

    # -- prompt embedding cache ---------------------------------------------
    def _load_seed_caption_embedding(self) -> Optional[torch.Tensor]:
        """Load the exact ride-level embedding used during training.

        Training reads ``caption_encoded`` from the ride's ``*_encoded.json``.
        Serving used to replace it with one generic sentence, and also kept
        that sentence when the seed picker changed rides.  Resolve the seed's
        training caption from its zarr metadata so inference conditioning now
        matches training.  An explicit ``cfg.extra['prompt']`` still wins.
        """
        if self.cfg.extra.get("prompt"):
            return None
        seed = Path(str(self.cfg.seed_zarr or ""))
        attrs_path = seed / ".zattrs"
        if seed.suffix != ".zarr" or not attrs_path.is_file():
            return None
        try:
            with attrs_path.open("r", encoding="utf-8") as fh:
                attrs = json.load(fh)
        except Exception as exc:
            log.warning("Could not read seed zarr metadata for caption lookup: %s", exc)
            return None

        roots: List[Path] = []
        configured = self.cfg.extra.get("caption_root") or os.environ.get(
            "ARRWM_CAPTION_ROOT", "")
        if configured:
            roots.append(Path(str(configured)))
        roots.extend([
            Path.home() / "frodobots" / "frodobots_captions" / "train",
            Path("/projects/u6ex/fbots/frodobots_captions/train"),
            Path("/projects/u5dk/as1748/frodobots_captions/train"),
        ])

        ride_dir = str(attrs.get("ride_dir_2k", ""))
        parts = Path(ride_dir).parts
        rel: Optional[Path] = None
        for i, part in enumerate(parts):
            if part.startswith("output_rides_"):
                rel = Path(*parts[i:])
                break
        ride_ts = str(attrs.get("ride_ts") or seed.stem)

        caption_path: Optional[Path] = None
        for root in roots:
            if not root.is_dir():
                continue
            matches = sorted((root / rel).glob("*_encoded.json")) if rel else []
            if not matches:
                matches = sorted(root.glob(
                    f"output_rides_*/ride_*_{ride_ts}/*_encoded.json"))
            if matches:
                caption_path = matches[0]
                break
        if caption_path is None:
            log.warning("No encoded training caption found for seed %s; using text fallback.",
                        seed.name)
            return None

        try:
            with caption_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            raw = payload.get("caption_encoded")
            if raw is None:
                raise ValueError("caption_encoded key missing")
            emb = torch.tensor(raw, dtype=torch.float32)
            if emb.shape != (512, 4096):
                raise ValueError(f"expected (512, 4096), got {tuple(emb.shape)}")
            log.info("Prompt embedding from training caption %s -> %s",
                     caption_path, tuple(emb.shape))
            return emb.unsqueeze(0)
        except Exception as exc:
            log.warning("Could not load training caption %s (%s); using text fallback.",
                        caption_path, exc)
            return None

    def _encode_prompt_cached(self, prompt: str) -> torch.Tensor:
        """Encode the prompt, memoised on disk.

        The prompt is a fixed string for a whole session (and across sessions,
        since it has a default), but encoding it runs the ~11 GB fp32 umt5-xxl
        encoder on CPU -- roughly a MINUTE of every launch, for a tensor that
        never changes.  Cache it keyed by the prompt text and the encoder
        weights it came from, so a changed prompt or a changed Wan checkpoint
        misses the cache instead of silently reusing the wrong embedding.
        """
        import hashlib

        enc_id = _normalize_wan_path(self.cfg.wan_model_path)
        key = hashlib.sha256(
            f"umt5-xxl|{enc_id}|{prompt}".encode("utf-8")).hexdigest()[:32]
        cache_dir = _REPO_ROOT / "interactive" / "prompt_cache"
        path = cache_dir / f"{key}.pt"

        if path.is_file():
            try:
                emb = torch.load(path, map_location="cpu", weights_only=False)
                log.info("Prompt embedding from cache (%s) -> %s",
                         path.name, tuple(emb.shape))
                return emb
            except Exception as exc:
                log.warning("prompt cache %s unreadable (%s); re-encoding.",
                            path.name, exc)

        t0 = time.perf_counter()
        emb = self.p.encode_prompt(prompt, encode_device="cpu")
        log.info("Prompt encoded on CPU in %.1fs", time.perf_counter() - t0)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".pt.tmp")
            torch.save(emb.detach().cpu(), tmp)
            tmp.replace(path)                    # atomic: no torn cache files
            log.info("Prompt embedding cached -> %s", path)
        except Exception as exc:
            log.warning("could not cache prompt embedding: %s", exc)
        return emb

    def set_action_provider(
            self, fn: Optional[Callable[[], Optional[Action]]]) -> None:
        """Install (or clear) the mid-horizon action mailbox reader.

        ``fn`` is called once per chunk boundary and must be cheap and
        non-blocking -- it runs on the generation thread between chunks.
        """
        self._action_provider = fn

    # -- session state ------------------------------------------------------
    @property
    def ride_id(self) -> str:
        """Id of the ride currently seeding the session (zarr stem)."""
        return Path(self.cfg.seed_zarr).stem if self.cfg.seed_zarr else "(noise)"

    @torch.no_grad()
    def reset_with_seed(self, zarr_path: str) -> np.ndarray:
        """Restart the ride from a DIFFERENT seed zarr, without a rebuild.

        Everything session-scoped is already rebuilt by ``reset()`` -- KV and
        cross-attn caches, decoder temporal state, RNG, latents log -- so the
        only extra work is dropping the cached seed latents so they are
        re-read from the new ride.  The model, VAE/decoder and CUDA stream
        are untouched, which is the whole point: swapping rides costs one
        reset, not a multi-second engine rebuild.

        Seed-action resolution is deliberately left alone: it keys off
        ``cfg.seed_zarr`` (ride stem), so pointing that at the new ride makes
        it resolve the new ride's motion, including the zeros-fallback
        warning when motion_root is unreachable.
        """
        path = str(zarr_path or "")
        if path and not Path(path).exists():
            raise FileNotFoundError(f"seed zarr not found: {path}")
        self.cfg.seed_zarr = path
        self._seed_latents = None          # force re-read in reset()
        self._prompt_embeds = None         # each ride has its own training caption
        log.info("Reseeding session from ride %s", self.ride_id)
        return self.reset()

    @torch.no_grad()
    def reset(self) -> np.ndarray:
        """Rebuild session state and prefill the selected serving context.

        Training used 3 clean chunks (``dmd_context_clean_frames=9``), but the
        causal attention span is 7 chunks. The interactive default fills that
        complete span; shorter values trade context for faster seed playback.
        """
        from utils.infinity_rope import infinity_rope_active
        from utils.play_world_model import _set_attention_window

        p = self.p
        if self._prompt_embeds is None:
            self._prompt_embeds = self._load_seed_caption_embedding()
            if self._prompt_embeds is None:
                prompt = self.cfg.extra.get(
                    "prompt", "a first-person view driving forward along a city sidewalk")
                self._prompt_embeds = self._encode_prompt_cached(prompt)
        if self._seed_latents is None:
            self._seed_latents = self._load_seed()

        p.prompt_embeds = self._prompt_embeds.to(self.device, self.dtype)
        p.current_start_frame = 0
        self.latents_log = []
        self._horizon_index = 0
        self._stats = None
        # Reproducible chunk noise for the bench harness (eval convention).
        torch.manual_seed(int(self.cfg.seed) + 1000003)

        # RoPE context for the whole session.
        if p._rope_ctx is not None:
            p._rope_ctx.__exit__(None, None, None)
        p._rope_ctx = infinity_rope_active(p.infinity_rope, p._base_dit)
        p._rope_ctx.__enter__()

        p.generator.seq_len = max(int(p.generator.seq_len),
                                  self.block_frames * self.frame_seq_length)
        # Attention SPAN = trained local_attn_size (21 frames), NOT the buffer.
        _set_attention_window(
            p._base_dit,
            local_attn_size_frames=self.attn_span_frames,
            max_tokens=self.attn_span_frames * self.frame_seq_length,
        )
        # sink_size = 0: the yaml comment says 3 but the trained VALUE is 0 —
        # a pure sliding 21-frame window with no pinned sink.
        # (Set on the self-attention modules only — those hold the live value
        # infinity_rope/causal_model read; the top-level CausalWanModel
        # attribute is a deprecated config alias.)
        sink = int(self.cfg.extra.get("sink_size", 0))
        n_sink = 0
        for name, module in p._base_dit.named_modules():
            if name.endswith("self_attn") and hasattr(module, "sink_size"):
                module.sink_size = sink
                n_sink += 1
        log.info("sink_size=%d applied to %d self-attention modules.", sink, n_sink)

        from utils.play_world_model import (
            _initialize_crossattn_cache, _initialize_kv_cache,
        )
        num_blocks = len(p._base_dit.blocks)
        p.kv_cache = _initialize_kv_cache(
            num_blocks=num_blocks, batch_size=1,
            kv_cache_tokens=self.kv_buffer_frames * self.frame_seq_length,
            dtype=self.dtype, device=self.device,
        )
        p.crossattn_cache = _initialize_crossattn_cache(
            num_blocks=num_blocks, batch_size=1, dtype=self.dtype, device=self.device,
        )

        # Streaming VAE decode contract: clear_cache() ONCE here, then
        # decode_to_pixel(..., use_cache=True) per chunk.  Never loop plain
        # decode_to_pixel.  Run it on the VAE stream so the cache tensors it
        # allocates belong to the same stream every later decode frees them
        # from (see the _vae_stream invariant in __init__).
        with self._vae_scope():
            if self.decoder is not None:
                self.decoder.reset()
            else:
                p.vae.model.clear_cache()

        seed_frames: List[np.ndarray] = []
        want_chunks = int(np.clip(
            getattr(self.cfg, "seed_prefill_chunks", SEED_PREFILL_CHUNKS),
            1, SEED_PREFILL_CHUNKS))
        # A user video may simply be too short for the requested depth. Use
        # every WHOLE chunk it does supply rather than refusing to start; a
        # zarr ride is length-checked by the picker and normally hits this
        # branch only if hand-pointed at a stub.
        avail_chunks = int(self._seed_latents.shape[1]) // self.npb
        seed_chunks = min(want_chunks, avail_chunks)
        if seed_chunks < 1:
            raise SystemExit(
                f"seed supplies {self._seed_latents.shape[1]} latent frames, "
                f"need at least {self.npb} for one chunk")
        if seed_chunks < want_chunks:
            log.warning("seed supports %d/%d seed chunks (%d latent frames); "
                        "prefilling %d chunks = %d of the %d-frame span.",
                        seed_chunks, want_chunks, self._seed_latents.shape[1],
                        seed_chunks, seed_chunks * self.npb, self.attn_span_frames)
        self.cfg.seed_prefill_chunks = seed_chunks
        need = seed_chunks * self.npb
        seed_all = self._seed_latents[:, :need].to(self.device, self.dtype)

        # Training conditions the seed prefill on the seed frames' OWN GT
        # actions (chunk-grain tanh(pca_raw/scales), dims [0, 1]), not on
        # neutral zeros.
        seed_actions = self._load_seed_actions(need)

        # Seam-affine target stats come from the REAL prefill latents.
        if self.seam_affine_lambda > 0.0:
            sf = seed_all.float()
            self._seam_target = (sf.mean(dim=(0, 1, 3, 4)), sf.std(dim=(0, 1, 3, 4)))
            log.info("Eval-style CARN seam affine ON at commit (lambda=%.3f).",
                     self.seam_affine_lambda)

        for c in range(seed_chunks):
            chunk = seed_all[:, c * self.npb:(c + 1) * self.npb]
            thr, ste = seed_actions[c]
            cond = self._cond(self._action_vector_raw(thr, ste))
            self._commit_chunk(chunk, cond, correct=False)
            self.latents_log.append(chunk.float().cpu())
            seed_frames.append(self._decode_to_np(chunk))
        # Reset is not a hot path: fully drain both streams so the session
        # starts from a clean, race-free state.
        torch.cuda.synchronize(self.device)
        log.info("reset: prefilled %d GT chunks (%d latent frames = %d%% of the "
                 "%d-frame attention span) into the KV cache (buffer %d frames).",
                 seed_chunks, need, round(100 * need / max(self.attn_span_frames, 1)),
                 self.attn_span_frames, self.kv_buffer_frames)
        return np.concatenate(seed_frames, axis=0)

    def _load_seed_actions(self, need_frames: int) -> List[tuple]:
        """Per-seed-chunk GT actions (throttle, steer), chunk-grain.

        Resolution order:
          (a) ``cfg.extra['seed_actions']`` — a .pt holding [frames, >=2] or
              [chunks, >=2] ALREADY-squashed pca_raw values;
          (b) motion from the seed ride via ``utils.zarr_dataset``.  Serving
              resolves a local mirror as well as the cluster training path;
          (c) zeros, with a loud warning that seed conditioning is misaligned.
        """
        n_chunks = need_frames // self.npb
        path = self.cfg.extra.get("seed_actions")

        # A user VIDEO has no motion track and no ride id, so the zarr-motion
        # lookup below cannot succeed and would only emit a misleading "seed
        # ride" warning. Go straight to the physical no-op -- which is the
        # right prior anyway: it is the pca_raw coordinate of an all-zero
        # optical-flow field, i.e. "the camera was not commanded to move",
        # rather than numeric (0, 0) which is a small real motion.
        from interactive.seed_picker import is_video as _is_video
        if not path and _is_video(str(self.cfg.seed_zarr or "")):
            log.warning(
                "VIDEO SEED: the clip's true actions are unknowable, so all %d "
                "seed chunks are conditioned on the PHYSICAL NULL action "
                "(throttle=%.6f, steer=%.6f) -- the action-space coordinate of "
                "zero optical flow. The model therefore believes the seed was "
                "recorded standing still; if your clip is moving, expect a "
                "motion mismatch at the seam. Pass --seed_actions to override.",
                n_chunks, PHYSICAL_NULL_THROTTLE, PHYSICAL_NULL_STEER)
            return [(PHYSICAL_NULL_THROTTLE, PHYSICAL_NULL_STEER)] * n_chunks

        if path:
            t = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(t, dict):
                t = t.get("z_actions", t.get("actions", t))
            t = torch.as_tensor(t).float().reshape(-1, int(t.shape[-1]))
            if t.shape[0] >= need_frames:
                t = t[:need_frames:self.npb]        # chunk-grain sample
            t = t[:n_chunks]
            if t.shape[0] < n_chunks:
                raise SystemExit(f"seed_actions has {t.shape[0]} rows, need {n_chunks}")
            log.info("Seed prefill actions from %s: %s", path,
                     [(round(float(r[0]), 3), round(float(r[1]), 3)) for r in t])
            return [(float(r[0]), float(r[1])) for r in t]

        configured = self.cfg.extra.get("motion_root") or os.environ.get(
            "ARRWM_MOTION_ROOT", "")
        roots = ([Path(str(configured))] if configured else []) + [
            Path(str(self.p.cfg.get("motion_root", "") or "")),
            Path.home() / "frodobots" / "frodobots_motion",
            Path("/projects/u6ex/fbots/frodobots_motion"),
            Path("/projects/u5dk/as1748/frodobots_motion"),
        ]
        failures = []
        motion_root = ""
        for root in roots:
            if not str(root) or not root.is_dir():
                continue
            motion_root = str(root)
            try:
                return self._seed_actions_from_motion(motion_root, n_chunks)
            except Exception as exc:
                failures.append(f"{root}: {exc}")
        if failures:
            log.warning("GT seed action lookup failed: %s", "; ".join(failures))

        log.warning(
            "SEED CONDITIONING MISALIGNED: no GT actions available for the seed "
            "ride (no usable motion_root and no extra['seed_actions'] "
            "given), falling back to the physical no-op. The seed chunks should use their "
            "own GT actions; expect a context mismatch at the seam when they do "
            "not. Training itself used 3 seed chunks. "
            "Supply --motion_root or --seed_actions to fix.")
        return [(PHYSICAL_NULL_THROTTLE, PHYSICAL_NULL_STEER)] * n_chunks

    def _seed_actions_from_motion(self, motion_root: str, n_chunks: int) -> List[tuple]:
        """GT pca_raw actions for the seed ride, via the training encoder.

        Reuses ``utils.zarr_dataset`` so the squash convention is shared:
        raw top-N PCA projection of the ride's per-frame motion, then
        ``tanh(P / _PCA_RAW_SCALES)``.  Only dims [0, 1] are consumed.
        """
        import numpy as _np
        from utils.zarr_dataset import (
            _PCA_RAW_SCALES,
            _encode_motion_pca_raw,
            _load_aligned_motion_for_zarr,
        )

        seed = Path(self.cfg.seed_zarr)
        attrs_path = seed / ".zattrs"
        if not attrs_path.is_file():
            raise FileNotFoundError(f"seed zarr metadata not found: {attrs_path}")
        with attrs_path.open("r", encoding="utf-8") as fh:
            attrs = json.load(fh)

        # The raw-PCA basis used by training is embedded in the ss-VAE
        # checkpoint.  It is not the unrelated exploratory pca.npz under
        # action_query/checkpoints/pca_motion.
        ss_path = Path(str(self.p.cfg.get("ss_vae_checkpoint", "") or ""))
        if not ss_path.is_absolute():
            ss_path = _REPO_ROOT / ss_path
        if not ss_path.is_file():
            raise FileNotFoundError(f"ss_vae checkpoint not found: {ss_path}")
        blob = torch.load(ss_path, map_location="cpu", weights_only=False)
        if "pca_mean" not in blob or "pca_comp" not in blob:
            raise KeyError(f"PCA basis missing from {ss_path}")

        off = int(self.cfg.extra.get("seed_frame_index", 0))
        need = off + n_chunks * self.npb
        motion_per_latent = _load_aligned_motion_for_zarr(
            attrs, need, Path(motion_root))
        chunk_motion = motion_per_latent[off:need:self.npb]
        if chunk_motion.shape[0] != n_chunks:
            raise ValueError(f"resolved {chunk_motion.shape[0]} action chunks, need {n_chunks}")
        raw = _encode_motion_pca_raw(
            chunk_motion, _np.asarray(blob["pca_mean"], dtype=_np.float64),
            _np.asarray(blob["pca_comp"], dtype=_np.float64), n_out=8)
        sq = _np.tanh(raw / _PCA_RAW_SCALES[:raw.shape[-1]])
        log.info("Seed prefill actions from %s (GT, training-aligned/squashed): %s",
                 motion_root,
                 [(round(float(r[0]), 3), round(float(r[1]), 3)) for r in sq])
        return [(float(r[0]), float(r[1])) for r in sq]

    def _load_seed(self) -> torch.Tensor:
        from utils.play_world_model import load_seed_from_zarr
        seed_chunks = int(np.clip(
            getattr(self.cfg, "seed_prefill_chunks", SEED_PREFILL_CHUNKS),
            1, SEED_PREFILL_CHUNKS))
        need = seed_chunks * self.npb
        src = self.cfg.seed_zarr
        if not src:
            log.warning("No seed_zarr — seeding with random noise (OOD start).")
            return torch.randn(1, need, LATENT_C, LATENT_H, LATENT_W)
        # A seed may be a ride zarr OR any local video: both resolve to the
        # same normalised latent space, so everything downstream (reset,
        # reset_with_seed, the picker) is agnostic about which it got.
        from interactive.seed_picker import (
            DEFAULT_CROP_BAND, DEFAULT_FISHEYE, is_video, load_seed_from_video)
        if is_video(src):
            # The video path negotiates its own depth: it returns as many
            # WHOLE chunks as the clip supports, up to seed_chunks. reset()
            # then adopts whatever came back.
            seed = load_seed_from_video(
                src, wan_model_path=self.cfg.wan_model_path,
                device=str(self.device),
                start=self.cfg.extra.get("seed_video_start", 0),
                start_s=self.cfg.extra.get("seed_video_start_s"),
                seed_chunks=seed_chunks,
                fisheye=float(self.cfg.extra.get("seed_video_fisheye",
                                                 DEFAULT_FISHEYE)),
                crop=str(self.cfg.extra.get("seed_video_crop",
                                            DEFAULT_CROP_BAND)),
                vae=getattr(self.p, "vae", None))
            log.info("Seed from video %s -> %s (%d chunks of %d requested)",
                     src, tuple(seed.shape),
                     seed.shape[1] // self.npb, seed_chunks)
            return seed
        off = int(self.cfg.extra.get("seed_frame_index", 0))
        seed = load_seed_from_zarr(src, off + need)[:, off:off + need]
        log.info("Seed from zarr %s -> %s", src, tuple(seed.shape))
        return seed

    # -- per-chunk generator core ------------------------------------------
    @torch.no_grad()
    def _denoise_chunk(self, x_noise: torch.Tensor, cond: Dict[str, torch.Tensor],
                       *, rung_from: int = 0, rung_to: Optional[int] = None,
                       return_noised: bool = False) -> torch.Tensor:
        """Traverse the denoising ladder for one chunk; return pred_x0.

        Matches ``eval_causal_AR.generate_ar``'s ``cache_refresh="append"``
        branch: forward at each rung, re-noise pred_x0 to the NEXT rung's
        timestep with ``scheduler.add_noise``, repeat, and always walk to the
        last rung (training's random exit-rung machinery is a grad-flow
        device with no inference meaning).  Timesteps are passed as FLOATS
        (eval convention); training rounds to int — toggle with
        ``extra['round_timesteps']=True``.  The KV cache is read (and
        transiently written) here; the durable commit is :meth:`_commit_chunk`.
        """
        p = self.p
        x = x_noise
        nf = int(x.shape[1])          # npb, or block_frames in block mode
        ts = self._ladder
        n_all = int(ts.shape[0])
        n = n_all if rung_to is None else int(rung_to)
        pred_x0 = None
        f0 = int(p.current_start_frame)
        t_rows = [self._rung_timesteps(d, nf) for d in range(n_all)]
        for d in range(int(rung_from), n):
            tt = t_rows[d].view(1, nf)
            with torch.amp.autocast("cuda", dtype=self.dtype):
                out = p.generator(
                    noisy_image_or_video=x,
                    conditional_dict=cond,
                    timestep=tt,
                    kv_cache=p.kv_cache,
                    crossattn_cache=p.crossattn_cache,
                    current_start=p.current_start_frame * p.frame_seq_length,
                )
            pred_x0 = out[1]
            if d < n - 1 or (return_noised and d == n - 1 and n < n_all):
                flat = pred_x0.flatten(0, 1).float()
                # Per-FRAME next timestep: identical for every frame unless
                # block_stagger is on, in which case each chunk of the block
                # walks its own (offset) ladder.
                flat_t = t_rows[d + 1].to(flat.device).float()
                x = (
                    p.scheduler.add_noise(flat, self._noise_like(flat, f0, d + 1), flat_t)
                    .view(1, nf, LATENT_C, LATENT_H, LATENT_W)
                    .to(self.dtype)
                )
        assert pred_x0 is not None
        # ``return_noised`` hands back the state re-noised to the NEXT rung's
        # timestep instead of the clean prediction, so a caller can stop the
        # joint pass part-way and resume the remaining rungs elsewhere.
        return x if return_noised else pred_x0

    # -- comparison instrumentation ----------------------------------------
    def _rung_timesteps(self, d: int, nf: int) -> torch.Tensor:
        """Per-frame timestep row for ladder rung ``d`` over ``nf`` frames.

        Normally a constant row (the trained contract). With
        ``extra['block_stagger'] = s`` in (0, 1] this becomes a
        diffusion-forcing-style staggered row: chunk j of the block sits
        ``s * j/B`` of the way BACK toward the previous (noisier) rung, so at
        every rung the earlier chunks of the block are cleaner than the later
        ones. That approximates the sequential information pattern the model
        was trained on while still costing one forward per rung. At rung 0
        there is no noisier rung to back off toward, so the row is flat.
        """
        ts = self._ladder
        t_val = float(ts[d].item())
        if self.round_timesteps:
            t_val = float(int(round(t_val)))
        row = torch.full((nf,), t_val, device=self.device, dtype=torch.float32)
        s = float(self.cfg.extra.get("block_stagger", 0.0) or 0.0)
        if s <= 0.0 or nf <= self.npb:
            return row
        prev_t = float(ts[d - 1].item()) if d > 0 else t_val
        n_blk = nf // self.npb
        if n_blk > 1 and prev_t > t_val:
            for j in range(n_blk):
                frac = s * (j / float(n_blk))
                tj = t_val + frac * (prev_t - t_val)
                if self.round_timesteps:
                    tj = float(int(round(tj)))
                row[j * self.npb:(j + 1) * self.npb] = tj
        return row

    def _init_noise(self, nf: int, frame0: int) -> torch.Tensor:
        """The block's starting noise [1, nf, C, H, W], frame-locked if asked."""
        if not bool(self.cfg.extra.get("frame_locked_noise", False)):
            return torch.randn([1, nf, LATENT_C, LATENT_H, LATENT_W],
                               dtype=torch.float32, device=self.device).to(self.dtype)
        flat = torch.empty((nf, LATENT_C, LATENT_H, LATENT_W),
                           dtype=torch.float32, device=self.device)
        return (self._noise_like(flat, frame0, 0)
                .view(1, nf, LATENT_C, LATENT_H, LATENT_W).to(self.dtype))

    def _noise_like(self, flat: torch.Tensor, frame0: int, rung: int) -> torch.Tensor:
        """Noise for ``flat`` ([nf, C, H, W]), optionally locked to frame index.

        With ``extra['frame_locked_noise']`` every absolute latent frame gets
        the SAME noise no matter how the rollout was chunked. That is what
        makes a block=1 rollout and a block=4 rollout comparable at all: the
        two consume the RNG stream in different numbers of draws, so without
        locking, most of the measured divergence is just different noise
        rather than the architectural change under test. Off by default --
        this is an instrument, not a production path.
        """
        if not bool(self.cfg.extra.get("frame_locked_noise", False)):
            return torch.randn_like(flat)
        parts = []
        for i in range(flat.shape[0]):
            g = torch.Generator(device=self.device)
            g.manual_seed((int(self.cfg.seed) * 1_000_003
                           + (frame0 + i) * 1_009 + rung) % (2**63 - 1))
            parts.append(torch.randn(flat.shape[1:], generator=g,
                                     device=self.device, dtype=flat.dtype))
        return torch.stack(parts, 0)

    @torch.no_grad()
    def _correct_chunk(self, pred_x0: torch.Tensor) -> torch.Tensor:
        """Apply the trained CARN commit correction to a generated chunk.

        This is the single tensor the engine then BOTH emits and commits —
        training commits the finish-denoised ladder endpoint after the CARN
        correction and eval emits that same committed tensor.

        Contract (model/carn_commit.py + ActionForcingDMD._dedrift_with_
        reverse_noiser, as configured by the rolling-definitive launcher):

          level      = clamp(frame_start/npb - (num_seed_chunks-1), 0, 16)
          corrected  = cur + G(cur, level)                # one Euler step
          corrected  = moment-preserved (per-channel DC + mean-abs contrast)
          commit     = blend(raw, corrected, alpha=0.125, cap=1.5 % rel-L2)

        NOTE: ``eval_causal_AR`` applies G at alpha0=1.0 with no blend/cap;
        the training recurrence is the faithful target, so the blend and the
        trust cap are used here.
        """
        from model.carn_commit import absolute_carn_level, blend_carn_commit

        out = pred_x0
        if self.reverse_noiser is not None:
            # BLOCK MODE APPROXIMATION: the level is a function of
            # frame_start, and current_start_frame is the FIRST frame of the
            # block, so a block of B chunks is corrected entirely at the level
            # its first chunk would have had. Chunks 2..B are therefore
            # corrected one to three levels "younger" than the trained
            # recurrence would have corrected them. The level saturates at
            # carn_max_level (16) so the error only exists early in a ride,
            # and the correction is alpha=0.125-blended under a 1.5%
            # relative-L2 trust cap either way -- but it IS a deviation from
            # the per-chunk recurrence and is the main reason block mode is
            # not bit-comparable to block_chunks=1.
            level = absolute_carn_level(
                frame_start=int(self.p.current_start_frame),
                frames_per_block=self.npb,
                num_seed_chunks=int(np.clip(
                    getattr(self.cfg, "seed_prefill_chunks", SEED_PREFILL_CHUNKS),
                    1, SEED_PREFILL_CHUNKS)),
                max_level=self.carn_max_level,
            )
            if level >= self.carn_min_level:
                lvl = torch.full((out.shape[0],), level, dtype=torch.long, device=out.device)
                delta = self.reverse_noiser(out.to(self.dtype), lvl, residual=False)
                cur = out + delta.to(out.dtype)
                # reverse_noiser_preserve_moments=true: exact per-channel DC
                # plus centred mean-absolute contrast (texture-only change).
                dims = [1, 3, 4]
                mu_in = out.mean(dim=dims, keepdim=True)
                mu_out = cur.mean(dim=dims, keepdim=True)
                dev_in, dev_out = out - mu_in, cur - mu_out
                a_in = dev_in.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                a_out = dev_out.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                cur = dev_out * (a_in / (a_out + 1.0e-6)) + mu_in
                out, eff = blend_carn_commit(
                    out, cur, alpha=self.carn_alpha,
                    max_relative_shift=self.carn_max_rel_shift,
                )
                self._last_carn = (level, float(eff.max()))

        if self._seam_target is not None:
            from utils.eval_causal_AR import _apply_carn_seam_affine
            out = _apply_carn_seam_affine(
                out, target_mean=self._seam_target[0],
                target_std=self._seam_target[1], strength=self.seam_affine_lambda,
            )
        return out

    @torch.no_grad()
    def _commit_chunk(self, chunk: torch.Tensor, cond: Dict[str, torch.Tensor],
                      *, correct: bool = True) -> torch.Tensor:
        """Commit one chunk VERBATIM into the KV cache and advance.

        ``context_noise = 0``, so the commit is a single clean forward of the
        chunk at t=0 with the SAME action conditioning — no re-noising.  The
        returned tensor is what the caller must also decode/emit, so emitted
        video and cache memory never disagree.  ``correct=False`` for GT seed
        chunks (nothing to de-drift).
        """
        p = self.p
        nf = int(chunk.shape[1])      # npb, or block_frames in block mode
        if correct:
            chunk = self._correct_chunk(chunk)
        refresh_t = torch.full([1, nf], float(p.context_noise),
                               device=self.device, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=self.dtype):
            p.generator(
                noisy_image_or_video=chunk,
                conditional_dict=cond,
                timestep=refresh_t,
                kv_cache=p.kv_cache,
                crossattn_cache=p.crossattn_cache,
                current_start=p.current_start_frame * self.frame_seq_length,
            )
        p.current_start_frame += nf
        return chunk

    # -- decode -------------------------------------------------------------
    @torch.no_grad()
    def _decode_gpu(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        """Cached streaming decode -> uint8 RGB [T, H, W, 3] on GPU."""
        if self.decoder is not None:
            # decoders.py already returns uint8 [T, H, W, 3] and applies
            # half-res itself, so this path is complete here.
            return self.decoder.decode_chunk(
                latent_chunk, half_res=self.cfg.decode_half_res)
        px = self.p.vae.decode_to_pixel(latent_chunk.to(self.dtype), use_cache=True)
        vid = (0.5 * (px[0].float() + 1.0)).clamp(0, 1)          # [T, C, H, W]
        if self.cfg.decode_half_res:
            vid = torch.nn.functional.interpolate(
                vid, scale_factor=0.5, mode="area")
        return (vid.permute(0, 2, 3, 1) * 255).to(torch.uint8)   # [T, H, W, 3]

    @contextlib.contextmanager
    def _vae_scope(self):
        """Enter the session's single VAE stream, ordered after the caller.

        Yields the stream the VAE work is enqueued on.  In same-stream mode
        this is just the current stream and the context is a no-op.
        """
        cur = torch.cuda.current_stream(self.device)
        if self._decode_stream is None:
            yield cur
            return
        ev = torch.cuda.Event()
        ev.record(cur)
        with torch.cuda.stream(self._decode_stream):
            self._decode_stream.wait_event(ev)
            yield self._decode_stream

    @torch.no_grad()
    def _decode_to_np(self, latent_chunk: torch.Tensor) -> np.ndarray:
        """Blocking decode of one chunk on the VAE stream (reset path)."""
        with self._vae_scope() as s:
            frames = self._decode_gpu(latent_chunk)
            ev = torch.cuda.Event()
            ev.record(s)
        latent_chunk.record_stream(s)
        return self._drain((frames, ev), torch.cuda.current_stream(self.device))

    # -- horizon ------------------------------------------------------------
    @torch.no_grad()
    def generate_horizon(self, action: Action) -> Iterator[np.ndarray]:
        """Latch `action`, roll cfg.horizon_chunks chunks, yield decoded blocks.

        Unless ``extra['decode_same_stream']`` is set, VAE decode runs on the
        dedicated VAE stream so decode(block N) overlaps the denoise of
        block N+1.  Blocks are yielded in order either way.

        LEGACY JOINT MODE (block_chunks > 1)
        =====================================
        A block of B chunks is ``B*3`` latent frames denoised as ONE forward
        sequence per ladder rung, with the action replicated across all of
        them. Reading the inference path end to end
        (``CausalWanSelfAttention.forward`` / ``utils.infinity_rope``):

        * There is NO causal or per-query mask on the KV-cache path. The K/V
          slice is computed ONCE per forward from ``local_end_index``
          (``rotated_temp_k[window_start:local_end_index]``, window_start =
          ``max(0, local_end_index - max_attention_size)``) and handed to
          ``attention(q, k, v)`` with ``causal=False`` and
          ``window_size=(-1, -1)``.
        * Therefore EVERY query frame in the block attends to EXACTLY the
          same key set: the whole block (full bidirectional attention among
          the new frames) PLUS an identical set of committed context frames.
          There is no per-query sliding window and no position asymmetry --
          frame 12 of the block sees precisely what frame 1 sees. The
          "uniform information access" property holds by construction, for
          block_chunks=1 as well; block mode does not change the rule, only
          how many frames share the window.
        * The window is ``max_attention_size = local_attn_size * 1560`` =
          32760 tokens against ``frame_seq_length`` 1561 (1560 spatial + 1
          action token), i.e. 20.98 frames -- a pre-existing off-by-one that
          applies identically at every block width.
        * CONTEXT DEPTH IS THE PRICE. The window is shared between the block
          and its context, so context frames = span - block_frames:
              B=1  ->  3 new + ~18 context frames
              B=2  ->  6 new + ~15 context
              B=4  -> 12 new +  ~9 context   (exactly the seed prefill size)
          B=4 still leaves a full seed's worth of context, but it is less
          than half of what the model was trained to condition on.

        Other legacy joint-mode consequences:
          * ONE commit forward at t=0 writes all B*3 frames into the KV cache
            in a single FIFO insert (the roll math is expressed in tokens and
            is generic over the write width).
          * The CARN commit correction is applied blockwise at the level of
            the block's FIRST chunk -- see ``_correct_chunk``.
          * Action pickup coarsens to one action per BLOCK: with B=4 that is
            12 latent frames = 48 pixel frames = 3.0 s at 16 fps.
          * A horizon that is not a whole number of blocks ends in a short
            block; the forward path is generic over width, so nothing special
            is needed for it.
        """
        p = self.p
        if p.kv_cache is None:
            raise RuntimeError("generate_horizon() before reset()")
        n_chunks = int(self.cfg.horizon_chunks)
        bc = max(int(self.block_chunks), 1)
        # Chunk counts per forward block; the last one is short when the
        # horizon is not a whole number of blocks.
        block_sizes = [min(bc, n_chunks - i) for i in range(0, n_chunks, bc)]
        # Action latching granularity.  'horizon' is the original behaviour:
        # one action decides the whole horizon, so a keypress can be up to
        # horizon_chunks chunks (~4.5 s at 6 chunks) stale before it is even
        # consulted.  'chunk' re-reads the mailbox at every chunk boundary,
        # which costs nothing -- chunks commit sequentially into the KV cache,
        # so there is no rollback to do -- and cuts that to one chunk.
        latch = str(self.cfg.extra.get("action_latch", "chunk")).lower()
        provider = self._action_provider if latch == "chunk" else None
        cur_action = action
        cond = self._cond(self._action_vector_raw(
            cur_action.throttle, cur_action.steer,
            frames=(block_sizes[0] * self.npb if block_sizes else self.npb)))
        self._live_action = cur_action
        main = torch.cuda.current_stream(self.device)

        torch.cuda.reset_peak_memory_stats(self.device)
        t_start = time.perf_counter()
        first_frame_latency = 0.0
        gen_events: List[tuple] = []
        dec_events: List[tuple] = []
        pending = None  # (uint8 gpu tensor, done-event)

        for i, blk_chunks in enumerate(block_sizes):
            blk_frames = blk_chunks * self.npb
            # Mid-horizon action pickup.  Re-reading here is safe because the
            # KV cache is append-only: chunks already committed stay committed,
            # and the new action simply conditions the next one.  In block
            # mode this is one pickup per BLOCK, not per chunk.
            if provider is not None:
                try:
                    fresh = provider()
                except Exception as exc:            # never kill a horizon for this
                    log.warning("action provider failed: %s", exc)
                    fresh = None
                if fresh is not None and (
                        fresh.throttle != cur_action.throttle
                        or fresh.steer != cur_action.steer):
                    cur_action = fresh
                    cond = self._cond(self._action_vector_raw(
                        cur_action.throttle, cur_action.steer, frames=blk_frames))
                    self._live_action = cur_action
                    self._action_changed_at_chunk = i

            ev_g0 = torch.cuda.Event(enable_timing=True)
            ev_g1 = torch.cuda.Event(enable_timing=True)
            ev_g0.record(main)
            # A short trailing block needs conditioning of its own width.
            if blk_frames != int(cond["_action_tokens"].shape[1]):
                cond = self._cond(self._action_vector_raw(
                    cur_action.throttle, cur_action.steer, frames=blk_frames))
            x = self._init_noise(blk_frames, int(p.current_start_frame))
            # HYBRID LADDER (extra['block_hybrid_tail'] = k > 0): run the
            # first n-k (high-noise, coarse) rungs JOINTLY over the whole
            # block -- that is where most of the compute is and where joint
            # attention is cheapest to justify -- then finish the last k
            # rungs per chunk, sequentially, each chunk seeing the previous
            # one already committed. The aim is most of the block speedup
            # with the trained sequential information pattern restored for
            # the detail-forming rungs.
            hyb = int(self.cfg.extra.get("block_hybrid_tail", 0) or 0)
            n_rungs = int(self._ladder.shape[0])
            if hyb > 0 and blk_frames > self.npb and n_rungs > hyb:
                split = n_rungs - hyb
                x_mid = self._denoise_chunk(x, cond, rung_to=split,
                                            return_noised=True)
                piece_cond = self._cond(self._action_vector_raw(
                    cur_action.throttle, cur_action.steer, frames=self.npb))
                outs = []
                for j in range(blk_chunks):
                    piece = x_mid[:, j * self.npb:(j + 1) * self.npb]
                    done = self._denoise_chunk(piece, piece_cond,
                                               rung_from=split)
                    outs.append(self._commit_chunk(done, piece_cond))
                pred_x0 = torch.cat(outs, dim=1)
                ev_g1.record(main)
                gen_events.append((ev_g0, ev_g1))
                self.latents_log.append(pred_x0.float().cpu())
                ev_d0 = torch.cuda.Event(enable_timing=True)
                ev_d1 = torch.cuda.Event(enable_timing=True)
                with self._vae_scope() as s_:
                    ev_d0.record(s_)
                    frames_gpu = self._decode_gpu(pred_x0)
                    ev_d1.record(s_)
                pred_x0.record_stream(s_)
                dec_events.append((ev_d0, ev_d1))
                if pending is not None:
                    out = self._drain(pending, main)
                    if not first_frame_latency:
                        first_frame_latency = time.perf_counter() - t_start
                    yield out
                pending = (frames_gpu, ev_d1)
                continue
            pred_x0 = self._denoise_chunk(x, cond)
            # Commit returns the tensor actually written to the KV cache
            # (post CARN correction); decode and emit exactly that.
            #
            # COMMIT MODE (extra['block_commit']):
            #   'joint'      (default) -- ONE t=0 forward writing all
            #                blk_frames into the KV cache in a single FIFO
            #                insert. This is the fully-joint arm: one
            #                generation, one cache write.
            #   'sequential' -- the block is generated jointly but committed
            #                as npb-frame pieces, so the RECURRENT memory is
            #                built exactly the way the trained per-chunk
            #                recurrence builds it (and the CARN correction
            #                lands at each chunk's own level instead of the
            #                block's first). Costs blk_chunks cheap t=0
            #                forwards instead of one wide one.
            if (str(self.cfg.extra.get("block_commit", "joint")).lower() == "sequential"
                    and blk_frames > self.npb):
                pieces = []
                for j in range(blk_chunks):
                    piece = pred_x0[:, j * self.npb:(j + 1) * self.npb]
                    piece_cond = self._cond(self._action_vector_raw(
                        cur_action.throttle, cur_action.steer, frames=self.npb))
                    pieces.append(self._commit_chunk(piece, piece_cond))
                pred_x0 = torch.cat(pieces, dim=1)
            else:
                pred_x0 = self._commit_chunk(pred_x0, cond)
            ev_g1.record(main)
            gen_events.append((ev_g0, ev_g1))
            self.latents_log.append(pred_x0.float().cpu())

            # Hand the chunk to the VAE stream.  `_vae_scope` records the
            # main->VAE ordering event; `record_stream` keeps pred_x0's
            # block alive until the VAE stream is done reading it.
            ev_d0 = torch.cuda.Event(enable_timing=True)
            ev_d1 = torch.cuda.Event(enable_timing=True)
            with self._vae_scope() as s:
                ev_d0.record(s)
                frames_gpu = self._decode_gpu(pred_x0)
                ev_d1.record(s)
            pred_x0.record_stream(s)
            dec_events.append((ev_d0, ev_d1))

            # Yield the PREVIOUS chunk now: its decode has had a full
            # denoise's worth of wall time to overlap.
            if pending is not None:
                out = self._drain(pending, main)
                if not first_frame_latency:
                    first_frame_latency = time.perf_counter() - t_start
                yield out
            pending = (frames_gpu, ev_d1)

        if pending is not None:
            out = self._drain(pending, main)
            if not first_frame_latency:
                first_frame_latency = time.perf_counter() - t_start
            yield out

        wall = time.perf_counter() - t_start
        torch.cuda.synchronize(self.device)
        per_chunk_gen_ms = [a.elapsed_time(b) for a, b in gen_events]
        gen_s = sum(per_chunk_gen_ms) / 1000.0
        dec_s = sum(a.elapsed_time(b) for a, b in dec_events) / 1000.0
        n_frames = n_chunks * PIXEL_FRAMES_PER_CHUNK   # unchanged by blocking
        self._stats = HorizonStats(
            horizon_index=self._horizon_index,
            action=action,
            denoising_steps=int(self._ladder.shape[0]),
            precision=self.cfg.precision,
            gen_seconds=gen_s,
            decode_seconds=dec_s,
            first_frame_latency_s=first_frame_latency,
            gen_fps=(n_frames / gen_s) if gen_s > 0 else 0.0,
            end_to_end_fps=(n_frames / wall) if wall > 0 else 0.0,
            peak_vram_gb=torch.cuda.max_memory_allocated(self.device) / 1e9,
            per_chunk_gen_ms=per_chunk_gen_ms,
        )
        self._horizon_index += 1

    def _drain(self, pending, main) -> np.ndarray:
        """Copy one decoded chunk back to the host.

        Order matters: make the consuming stream wait on the VAE stream's
        completion event FIRST, then tag the block with record_stream so the
        allocator cannot hand the VAE stream's memory to a main-stream
        allocation while the D2H copy is still in flight.
        """
        frames_gpu, ev_done = pending
        main.wait_event(ev_done)
        frames_gpu.record_stream(main)
        out = frames_gpu.cpu().numpy()
        return out

    # -- API surface --------------------------------------------------------
    @property
    def last_stats(self) -> Optional[HorizonStats]:
        return self._stats

    def apply_live_settings(self, *, denoising_steps: Optional[int] = None,
                            horizon_chunks: Optional[int] = None,
                            decode_half_res: Optional[bool] = None,
                            ladder_mode: Optional[str] = None,
                            seed_prefill_chunks: Optional[int] = None) -> None:
        if ladder_mode is not None:
            self.cfg.extra["ladder_mode"] = str(ladder_mode)
            if denoising_steps is None:      # re-resolve at the current count
                self._set_ladder(int(self.cfg.denoising_steps))
        if denoising_steps is not None:
            n = int(denoising_steps)
            self.cfg.denoising_steps = n
            self._set_ladder(n)
        if horizon_chunks is not None:
            self.cfg.horizon_chunks = max(1, int(horizon_chunks))
        if decode_half_res is not None:
            self.cfg.decode_half_res = bool(decode_half_res)
        if seed_prefill_chunks is not None:
            chunks = int(np.clip(seed_prefill_chunks, 1, SEED_PREFILL_CHUNKS))
            if chunks != int(getattr(
                    self.cfg, "seed_prefill_chunks", SEED_PREFILL_CHUNKS)):
                self.cfg.seed_prefill_chunks = chunks
                # A shorter cached slice cannot later satisfy a longer seed.
                self._seed_latents = None

    def close(self) -> None:
        try:
            self.p.close()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------
def _fmt(s: HorizonStats) -> str:
    return (
        f"horizon {s.horizon_index}: action(throttle={s.action.throttle:+.2f}, "
        f"steer={s.action.steer:+.2f}) steps={s.denoising_steps} "
        f"gen={s.gen_seconds:.2f}s decode={s.decode_seconds:.2f}s "
        f"gen_fps={s.gen_fps:.2f} e2e_fps={s.end_to_end_fps:.2f} "
        f"first_frame={s.first_frame_latency_s:.2f}s "
        f"peak_vram={s.peak_vram_gb:.2f}GB "
        f"per_chunk_ms={[round(x, 1) for x in s.per_chunk_gen_ms]}"
    )


def run_selftest(args) -> int:
    from utils.play_world_model import FrameSink

    cfg = EngineConfig(
        ckpt_path=args.ckpt,
        config_path=args.config,
        wan_model_path=args.wan_model_path,
        seed_zarr=(args.seed_video or args.seed_zarr or ""),
        use_ema=args.use_ema,
        precision=args.precision,
        denoising_steps=args.denoising_steps,
        kv_cache_chunks=args.kv_cache_chunks,
        horizon_chunks=args.horizon_chunks,
        block_chunks=args.block_chunks,
        **({"seed_prefill_chunks": args.seed_prefill_chunks}
           if args.seed_prefill_chunks is not None else {}),
        device=args.device,
        decoder=args.decoder,
        compile_mode=args.compile_mode,
        extra={"carn_commit": args.carn_commit,
               "decode_same_stream": bool(args.decode_same_stream),
               "ladder_mode": args.ladder_mode,
               "seed_video_start": args.seed_video_start,
               "seed_video_start_s": args.seed_video_start_s,
               "seed_video_crop": args.seed_video_crop,
               **({"seed_video_fisheye": args.seed_video_fisheye}
                  if args.seed_video_fisheye is not None else {}),
               "seed_frame_index": int(args.seed_frame_index),
               "no_lean_ckpt": bool(args.no_lean_ckpt),
               "frame_locked_noise": bool(args.frame_locked_noise),
               "carn_seam_affine_lambda": float(args.carn_seam_affine_lambda),
               **({"prompt": args.prompt} if args.prompt else {}),
               **({"caption_root": args.caption_root} if args.caption_root else {}),
               **({"motion_root": args.motion_root} if args.motion_root else {}),
               **({"seed_actions": args.seed_actions} if args.seed_actions else {})},
    )
    eng = WorldModelEngine(cfg)
    print(f"[selftest] weights: {eng.p.weights_source}")
    print(f"[selftest] seed={cfg.seed_zarr}")
    print(f"[selftest] block_chunks={eng.block_chunks} "
          f"mode={eng.generation_mode} ({eng.block_frames} latent frames/forward, "
          f"~{max(eng.attn_span_frames - eng.block_frames, 0)} context frames in span)")
    print(f"[selftest] carn_commit: "
          f"{'on' if eng.reverse_noiser is not None else 'off'} "
          f"(alpha={eng.carn_alpha}, cap={eng.carn_max_rel_shift})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"selftest_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    sink = FrameSink(str(out_mp4), 16.0)

    seed_frames = eng.reset()
    print(f"[selftest] seed frames: {seed_frames.shape}  "
          f"prefilled {eng.cfg.seed_prefill_chunks} chunks "
          f"({eng.cfg.seed_prefill_chunks * eng.npb} latent frames "
          f"of the {eng.attn_span_frames}-frame span)")
    sink.add(seed_frames)

    base = [
        ("straight",   Action(throttle=0.5)),
        ("left",       Action(throttle=0.4, steer=-0.6)),
        ("throttle",   Action(throttle=0.9)),
        ("right",      Action(throttle=0.4, steer=0.6)),
        ("hard-left",  Action(throttle=0.6, steer=-0.9)),
        ("hard-right", Action(throttle=0.6, steer=0.9)),
        ("coast",      Action()),
        ("reverse",    Action(throttle=-0.4)),
    ]
    n_h = max(1, int(args.horizons))
    if args.selftest_action:
        fixed = dict(base)[args.selftest_action]
        script = [(args.selftest_action, fixed)] * n_h
    else:
        script = [base[i % len(base)] for i in range(n_h)]
    nan_hits = 0
    for name, act in script:
        t0 = time.perf_counter()
        n = 0
        blocks = []
        for chunk in eng.generate_horizon(act):
            sink.add(chunk)
            n += chunk.shape[0]
            blocks.append(int(chunk.shape[0]))
        st = eng.last_stats
        assert st is not None
        print(f"[selftest] {name:9s} {n} frames in {time.perf_counter() - t0:.2f}s"
              f"  (decoded blocks: {blocks})")
        print(f"[selftest]   {_fmt(st)}")
        if eng._last_carn is not None:
            print(f"[selftest]   carn: level={eng._last_carn[0]} "
                  f"effective_alpha={eng._last_carn[1]:.4f}")
        last = eng.latents_log[-1]
        if not torch.isfinite(last).all():
            nan_hits += 1
            print(f"[selftest]   !! non-finite latents in horizon {st.horizon_index}")

    sink.write()
    print(f"[selftest] wrote {len(sink.frames)} frames -> {out_mp4}")
    eng.close()
    if nan_hits:
        print(f"[selftest] FAILED: {nan_hits} horizons produced non-finite latents")
        return 1
    print("[selftest] OK")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [engine] %(levelname)s | %(message)s",
    )
    ap = argparse.ArgumentParser(description="Interactive world-model engine (WS-A).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--save_lean_ckpt", nargs="?", const=True, default=None,
                    metavar="OUT",
                    help="write a bf16 inference-only copy of --ckpt (generator "
                         "+ action heads + CARN) and exit. Default output is "
                         "<ckpt>.lean.pt, which later runs prefer automatically.")
    ap.add_argument("--no_lean_ckpt", action="store_true",
                    help="ignore a sibling .lean.pt and load the full checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/action_forcing_phase3_dmd.yaml")
    ap.add_argument("--wan_model_path", default="/home/ashish/Wan2.1/Wan2.1-T2V-1.3B/")
    ap.add_argument("--seed_zarr", default=str(Path.home() / "20240224003808.zarr"))
    ap.add_argument("--seed_video", default=None,
                    help="seed from a local video instead of a ride zarr (OOD)")
    ap.add_argument("--seed_video_start", default="0",
                    help="offset into --seed_video: bare int = FRAMES @16fps, "
                         "float or 's' suffix = SECONDS, 'f' suffix = frames")
    ap.add_argument("--seed_video_start_s", type=float, default=None,
                    help="offset into --seed_video in seconds (unambiguous)")
    ap.add_argument("--seed_video_fisheye", type=float, default=None,
                    help="barrel pre-warp strength for user footage "
                         "(OFF by default; try 0.15-0.22 to opt in). The "
                         "robot cam is a fisheye; phone cameras are not.")
    ap.add_argument("--seed_video_crop", choices=["center", "top", "bottom"],
                    default="bottom",
                    help="which VERTICAL band survives when the crop to the "
                         "camera's 16:9 aspect trims height (PORTRAIT video). "
                         "Default bottom: the training cam is low-mounted, so "
                         "the ground ahead matches it better than sky. "
                         "Horizontal trims stay centred regardless.")
    ap.add_argument("--seed_prefill_chunks", type=int, default=None,
                    help="GT chunks to prefill (1..7; default 7 = the full "
                         "trained attention span). A short video uses as many "
                         "whole chunks as it supports.")
    ap.add_argument("--seed_frame_index", type=int, default=0,
                    help="latent-frame offset within --seed_zarr")
    # Production evals load the RAW `generator` key; generator_ema is a
    # trainable-params-only shadow that cannot load strictly.
    ap.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed_actions", default=None,
                    help="Optional .pt of GT squashed pca_raw actions for the seed prefill.")
    ap.add_argument("--prompt", default=None,
                    help="Explicit text prompt (overrides automatic per-ride training caption).")
    ap.add_argument("--caption_root", default=None,
                    help="Root containing output_rides_*/ride_*/*_encoded.json; "
                         "defaults to ARRWM_CAPTION_ROOT or the local FrodoBots caption store.")
    ap.add_argument("--motion_root", default=None,
                    help="Root containing output_rides_*/ride_*/motion.npy; "
                         "defaults to ARRWM_MOTION_ROOT or the local FrodoBots motion store.")
    ap.add_argument("--carn_commit", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--horizons", type=int, default=3,
                    help="Number of scripted horizons to roll in the selftest.")
    ap.add_argument("--selftest_action", choices=[
                        "straight", "left", "throttle", "right", "hard-left",
                        "hard-right", "coast", "reverse"], default=None,
                    help="repeat one action for every selftest horizon instead of cycling")
    ap.add_argument("--decode_same_stream", action="store_true",
                    help="Decode on the main stream (no gen/decode overlap).")
    ap.add_argument("--frame_locked_noise", action="store_true",
                    help="derive every noise draw from absolute frame+rung so "
                         "different generation modes are directly comparable")
    ap.add_argument("--carn_seam_affine_lambda", type=float, default=0.0,
                    help="optional per-channel seed-stat anchor at commit (0..1)")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--denoising_steps", type=int, default=4)
    ap.add_argument("--ladder_mode", default="interp",
                    choices=["interp", "trained", "grid"],
                    help="how to build ladders past the 4 trained rungs")
    ap.add_argument("--compile_mode", default="off",
                    help="off | max-autotune-no-cudagraphs | max-autotune | "
                         "reduce-overhead (the last two capture cuda graphs "
                         "and are incompatible with the persistent KV cache)")
    ap.add_argument("--kv_cache_chunks", type=int, default=7)
    ap.add_argument("--horizon_chunks", type=int, default=6)
    ap.add_argument("--block_chunks", type=int, default=EngineConfig.block_chunks,
                    choices=[1, 2, 4],
                    help="fully-joint research block width; production default 1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--decoder", default="wan",
                    help="latent->pixel decoder: wan (reference) | lightvaew2_1 "
                         "| taew2_1 | lighttaew2_1 (see interactive/decoders.py)")
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "interactive" / "selftest_out"))
    args = ap.parse_args()

    if args.save_lean_ckpt is not None:
        out = None if args.save_lean_ckpt is True else str(args.save_lean_ckpt)
        save_lean_checkpoint(args.ckpt, out)
        return 0

    if not args.selftest:
        ap.error("engine.py is a library; pass --selftest to run the built-in check.")
    return run_selftest(args)


if __name__ == "__main__":
    raise SystemExit(main())
