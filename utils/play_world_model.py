#!/usr/bin/env python3
"""Interactive "play" harness for the action-forcing world model.

Drive a trained Action-Forcing DMD student (e.g. the phase-3 one-step
``v8e8`` run — ``sbatch/train_action_forcing_phase3_dmd_one_step_..._v8e8.sbatch``)
live on a single local GPU. You feed per-chunk **action** inputs (steering
``z2`` + throttle ``z7``) and the model autoregressively rolls the world
forward, streaming the decoded video to a window (or to disk in headless
mode).

What ``v8e8`` saves
-------------------
The trainer (``trainer/causal_action_forcing_train.py`` →
``trainer/causal_rolling_staircase_train.py:save``) writes
``<log_dir>/phase1_step<NNNNNNN>.pt`` containing (among training-only keys):

  * ``generator``                 — full-rank ``CausalWanModel`` state dict (825 tensors)
  * ``generator_ema``             — EMA shadow (only after ``ema_start_step``)
  * ``action_projection``         — Stream-A AdaLN modulation MLP
  * ``action_token_projection``   — Stream-B per-frame action-token MLP

Only those four matter for inference; the teacher / fake_score / GAN /
forward-noiser / optimizer states are ignored here. This script rebuilds
ONLY the generator + the two action heads + VAE (+ a one-shot text
encoder), so it stays light enough to run on a 32 GB 5090. It deliberately
does NOT construct ``ActionForcingDMD`` / ``ODERegression`` (which would
also build the frozen teacher, fake_score, action_critic, CoTracker and
ss_vae — none of which are needed to roll the student forward).

Generation path
---------------
Byte-for-byte the same streaming rollout that ``utils/eval_causal_AR.py``
uses in its ``cache_refresh="append"`` mode (KV-cache prefill of a real
seed chunk → per-chunk few-step denoise → clean cache-refresh forward →
advance), except the per-chunk action vector comes from *you* live
instead of from a zarr ride. Wrapped in ``infinity_rope_active`` so the
rolling KV cache + sink behave exactly as they did at train time.

Actions
-------
The model consumes a 2-D action per frame: ``[z2, z7]`` — dims ``2`` and
``7`` of the 8-D ss_vae motion latent, each ``tanh``-squashed into
``(-1, 1)`` (see ``utils/zarr_dataset._tanh_squash`` and the
``action_dims: [2, 7]`` config knob). Physically (``zarr_dataset.py:87``):

  * ``z2`` = turn / sides  (steering)   — 0 ≈ straight,  ± ≈ turn
  * ``z7`` = forward / back (throttle)  — 0 ≈ idle,      ± ≈ drive

We feed these squashed scalars directly; no ss_vae needed for live play.

Usage
-----
Headless REPL (always works over SSH)::

    conda activate flash
    python utils/play_world_model.py \
        --ckpt   /path/to/phase1_step0000200.pt \
        --config configs/action_forcing_phase3_dmd.yaml \
        --seed_zarr ~/20240224003808.zarr \
        --mode repl

Live window (needs a display)::

    python utils/play_world_model.py --ckpt ... --config ... \
        --seed_zarr ~/20240224003808.zarr --mode window

Self-test (build + load + roll a few scripted chunks + write mp4, no input)::

    python utils/play_world_model.py --ckpt ... --config ... --selftest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [play_wm] %(levelname)s | %(message)s",
)
log = logging.getLogger("play_world_model")

# ---------------------------------------------------------------------------
# Geometry constants (must match the trained model — see eval_causal_AR.py).
# ---------------------------------------------------------------------------
FRAME_SPATIAL_TOKENS = 1560     # tokens / latent frame at 60x104 spatial.
ACTION_TOKENS_PER_FRAME = 1     # Stream-B: one action token / frame -> 1561.
LATENT_C, LATENT_H, LATENT_W = 16, 60, 104


# ---------------------------------------------------------------------------
# KV / cross-attn cache allocators (copied from eval_causal_AR.py so this
# file carries no heavy eval imports).
# ---------------------------------------------------------------------------
def _initialize_kv_cache(*, num_blocks, batch_size, kv_cache_tokens, dtype, device):
    cache = []
    for _ in range(num_blocks):
        cache.append({
            "k": torch.zeros([batch_size, kv_cache_tokens, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_tokens, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        })
    return cache


def _initialize_crossattn_cache(*, num_blocks, batch_size, dtype, device):
    cache = []
    for _ in range(num_blocks):
        cache.append({
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        })
    return cache


def _load_config_with_extends(path: str):
    """Load a yaml config, recursively resolving a top-level ``_extends:``
    directive (vendored from trainer/causal_action_forcing_train.py so the
    phase-3 overlay correctly inherits model_kwargs from phase-1)."""
    from omegaconf import OmegaConf
    stack: list = []

    def _load(p: str):
        ap = os.path.abspath(p)
        if ap in stack:
            raise RuntimeError(f"Circular _extends: {' -> '.join(stack + [ap])}")
        stack.append(ap)
        try:
            cfg = OmegaConf.load(ap)
            ext = None
            if isinstance(cfg, type(OmegaConf.create({}))) and "_extends" in cfg:
                ext = cfg["_extends"]
                del cfg["_extends"]
            if ext is None:
                return cfg
            ext_path = os.path.normpath(os.path.join(os.path.dirname(ap), str(ext)))
            return OmegaConf.merge(_load(ext_path), cfg)
        finally:
            stack.pop()

    return _load(path)


def _set_attention_window(base_dit, *, local_attn_size_frames, max_tokens):
    base_dit.local_attn_size = local_attn_size_frames
    base_dit.max_attention_size = max_tokens
    for _, module in base_dit.named_modules():
        if hasattr(module, "max_attention_size"):
            try:
                module.max_attention_size = max_tokens
            except Exception:
                pass
        if hasattr(module, "local_attn_size"):
            try:
                module.local_attn_size = local_attn_size_frames
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class WorldModelPlayer:
    """Holds the lean generator + action heads + VAE and streams a rollout."""

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        *,
        device: torch.device,
        wan_model_path: str,
        use_ema: bool = False,
        denoising_steps: Optional[int] = None,
        kv_cache_chunks: int = 8,
        infinity_rope: Optional[bool] = None,
        weights_dtype: str = "bf16",
    ):
        from omegaconf import OmegaConf

        self.device = device
        self.dtype = torch.bfloat16
        # Stored generator weight dtype. bf16 (default) halves the DiT
        # footprint (~5.4 GB fp32 -> ~2.7 GB) so the tool fits alongside
        # other GPU jobs; the forward autocasts to bf16 regardless. Use
        # fp32 to match the training/eval weight precision exactly.
        self.weights_dtype = torch.float32 if weights_dtype == "fp32" else torch.bfloat16

        # --- Point the Wan loaders at the LOCAL weights before any build. ---
        import utils.wan_wrapper as ww
        if not wan_model_path.endswith("/"):
            wan_model_path = wan_model_path + "/"
        ww._default_wan_model_path = wan_model_path
        log.info("Wan base weights: %s", wan_model_path)

        cfg = _load_config_with_extends(config_path)
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg

        self.num_frame_per_block = int(cfg.get("num_frame_per_block", 3))
        self.raw_action_dim = int(cfg.get("raw_action_dim", cfg.get("action_dim", 2)))
        self.num_training_frames = int(cfg.get("num_training_frames", 21))
        self.context_noise = float(cfg.get("context_noise", 0.0))

        # local_attn_size: prefer config model_kwargs, but the rolling
        # KV window for live (unbounded) play is sized from kv_cache_chunks.
        mk = OmegaConf.to_container(cfg.get("model_kwargs", {}), resolve=True) or {}
        self.cfg_local_attn = int(mk.get("local_attn_size", self.num_training_frames))
        self.local_attn_size_frames = kv_cache_chunks * self.num_frame_per_block
        self.frame_seq_length = FRAME_SPATIAL_TOKENS + ACTION_TOKENS_PER_FRAME
        self.kv_cache_tokens = self.local_attn_size_frames * self.frame_seq_length

        if infinity_rope is None:
            infinity_rope = bool(cfg.get("infinity_rope", True))
        self.infinity_rope = infinity_rope

        self._build_generator(cfg, mk)
        self._load_checkpoint(ckpt_path, use_ema=use_ema)
        self._setup_denoising(denoising_steps)

        # VAE for live decode (frozen).
        from utils.wan_wrapper import WanVAEWrapper
        self.vae = WanVAEWrapper().to(self.device, self.dtype).eval()

        # Rollout state (set by reset()).
        self.kv_cache = None
        self.crossattn_cache = None
        self.current_start_frame = 0
        self.prompt_embeds: Optional[torch.Tensor] = None
        self._rope_ctx = None
        self.latents_log: List[torch.Tensor] = []   # generated latent chunks (cpu, f32)

        log.info(
            "Player ready | npb=%d  raw_action_dim=%d  denoise=%s  "
            "kv_window=%d frames (%d chunks)  infinity_rope=%s",
            self.num_frame_per_block, self.raw_action_dim,
            [round(float(x), 1) for x in self.denoising_step_list.tolist()],
            self.local_attn_size_frames, kv_cache_chunks, self.infinity_rope,
        )

    # ---- build (mirrors model/base.py:_initialize_models, generator-only) --
    def _build_generator(self, cfg, model_kwargs):
        from utils.wan_wrapper import WanDiffusionWrapper
        from model.action_model_patch import apply_action_patches
        from model.action_modulation import (
            ActionModulationProjection, ActionTokenProjection,
        )

        mk = dict(model_kwargs)
        mk.pop("model_name", None)  # let the wrapper default to Wan2.1-T2V-1.3B
        log.info("Building generator (CausalWanModel, is_causal=True)...")
        self.generator = WanDiffusionWrapper(**mk, is_causal=True)
        self.generator.configure_impose_stat(
            str(cfg.get("impose_stat_mode", "") or "")
        )

        if not bool(cfg.get("action_patch_enabled", True)):
            raise SystemExit(
                "action_patch_enabled is false in the config — this harness "
                "only supports action-conditioned checkpoints."
            )
        acm = str(cfg.get("action_conditioning_mode", "both"))
        if acm != "both":
            raise SystemExit(
                f"action_conditioning_mode must be 'both' (got {acm!r})."
            )

        apply_action_patches(self.generator)
        model_dim = getattr(self.generator.model, "dim", 2048)
        activation = cfg.get("action_modulation_activation", None)

        self.action_projection = ActionModulationProjection(
            action_dim=self.raw_action_dim, activation=activation,
            hidden_dim=model_dim, num_frames=1, zero_init=True,
        ).to(device=self.device, dtype=self.dtype)
        self.action_token_projection = ActionTokenProjection(
            action_dim=self.raw_action_dim, activation=activation,
            hidden_dim=model_dim, zero_init=True,
        ).to(device=self.device, dtype=self.dtype)

        # Reserve one action token / frame so the DiT runs at the trained
        # 1561-tokens/frame layout (NOT 1560). See model/base.py:118-137.
        self.generator.model.action_tokens_per_frame = ACTION_TOKENS_PER_FRAME
        self.generator.adjust_seq_len_for_action_tokens(
            num_frames=self.num_training_frames,
            action_per_frame=ACTION_TOKENS_PER_FRAME,
        )
        if int(getattr(self.generator.model, "action_tokens_per_frame", 0)) != 1:
            raise SystemExit("Stream-B wiring failed: action_tokens_per_frame != 1")

        self.generator.to(self.device, self.weights_dtype).eval()
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
        if getattr(self.scheduler, "sigmas", None) is not None:
            self.scheduler.sigmas = self.scheduler.sigmas.to(self.device)

    # ---- checkpoint overlay (mirrors eval_causal_AR.load_checkpoint) -------
    def _load_checkpoint(self, ckpt_path: str, *, use_ema: bool):
        log.info("Loading checkpoint %s (use_ema=%s)", ckpt_path, use_ema)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise SystemExit(f"Unexpected checkpoint format: {type(raw)}")

        gen_key = "generator_ema" if (use_ema and "generator_ema" in raw) else "generator"
        if use_ema and gen_key != "generator_ema":
            log.warning("--use_ema requested but no 'generator_ema' in ckpt; using 'generator'.")
        if gen_key not in raw:
            raise SystemExit(f"Checkpoint missing '{gen_key}' key; keys={list(raw)[:12]}")

        gen_sd = raw[gen_key]
        # FSDP/DDP prefix cleanup, defensive (trainer unwraps .module before
        # save, so this is normally a no-op).
        def _clean(k: str) -> str:
            return k.replace("_fsdp_wrapped_module.", "").replace("module.", "")
        gen_sd = {_clean(k): v for k, v in gen_sd.items()}

        missing, unexpected = self.generator.model.load_state_dict(gen_sd, strict=False)
        if missing:
            log.warning("generator: %d missing keys (e.g. %s)", len(missing), missing[:4])
        if unexpected:
            log.warning("generator: %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:4])
        if missing or unexpected:
            # Strict parity is expected for a matching config; warn loudly but
            # continue (lets you probe a near-matching checkpoint).
            log.warning(
                "generator state_dict did not match strictly — verify --config "
                "matches the checkpoint's training config."
            )
        else:
            log.info("generator weights loaded (strict match, %d tensors).", len(gen_sd))

        if "action_projection" in raw:
            self.action_projection.load_state_dict(raw["action_projection"])
            log.info("action_projection loaded.")
        else:
            log.warning("No 'action_projection' in checkpoint — AdaLN actions are zero-init!")
        if "action_token_projection" in raw:
            self.action_token_projection.load_state_dict(raw["action_token_projection"])
            log.info("action_token_projection loaded.")
        else:
            log.warning("No 'action_token_projection' — action tokens are zero-init!")

        self.embedded_step = int(raw.get("step", -1))
        log.info("Checkpoint embedded step = %s", self.embedded_step)
        del raw

    # ---- denoising schedule (mirrors eval set_denoising_steps top-n) -------
    def _setup_denoising(self, n: Optional[int]):
        trained = torch.tensor(
            list(self.cfg.get("denoising_step_list", [1000, 625, 312.5, 178.6])),
            dtype=torch.float32, device=self.device,
        )
        trained, _ = torch.sort(trained, descending=True)
        K = int(trained.shape[0])
        if n is None or n == K:
            ts = trained
        elif 1 <= n < K:
            ts = trained[:n].contiguous()
        else:
            ts = torch.linspace(1000.0, 50.0, steps=n, device=self.device)
            log.warning("denoise n=%d > trained pool %d -> OOD linspace.", n, K)
        self.denoising_step_list = ts

    # ---- prompt conditioning ----------------------------------------------
    def encode_prompt(self, text: str, *, encode_device: str = "cpu") -> torch.Tensor:
        """Encode a caption with the umt5-xxl text encoder.

        ``encode_device='cpu'`` (default) runs the ~11 GB fp32 T5 entirely
        on CPU and only moves the small embedding to the GPU — so this tool
        coexists with other GPU jobs. Use ``encode_device='cuda'`` only when
        the card is otherwise idle. The encoder is freed immediately after.
        """
        if encode_device == "cuda":
            from utils.wan_wrapper import WanTextEncoder
            log.info("Encoding prompt on GPU: %r", text)
            te = WanTextEncoder().to(self.device).eval()
            with torch.no_grad():
                emb = te(text_prompts=[text])["prompt_embeds"].detach()
            del te
            torch.cuda.empty_cache()
            return emb.to(self.device)

        # CPU path — replicate WanTextEncoder internals without the .cuda().
        log.info("Encoding prompt on CPU (T5 fp32): %r", text)
        from pathlib import Path as _P
        import utils.wan_wrapper as ww
        from wan.modules.t5 import umt5_xxl
        from wan.modules.tokenizers import HuggingfaceTokenizer

        root = _P(ww._default_wan_model_path) / "Wan2.1-T2V-1.3B"
        enc = umt5_xxl(encoder_only=True, return_tokenizer=False,
                       dtype=torch.float32, device=torch.device("cpu")).eval().requires_grad_(False)
        enc.load_state_dict(torch.load(
            str(root / "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", weights_only=False))
        tok = HuggingfaceTokenizer(name=str(root / "google" / "umt5-xxl"),
                                   seq_len=512, clean="whitespace")
        ids, mask = tok([text], return_mask=True, add_special_tokens=True)
        seq_lens = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            context = enc(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        del enc, tok
        return context.detach().to(self.device)

    # ---- conditional dict for one chunk (mirrors _build_action_cond_chunk) -
    def _action_cond(self, action_fa: torch.Tensor) -> Dict[str, torch.Tensor]:
        cond: Dict[str, torch.Tensor] = {
            "prompt_embeds": self.prompt_embeds.to(dtype=self.dtype, device=self.device),
        }
        cond["_action_modulation"] = self.action_projection(
            action_fa, num_frames=self.num_frame_per_block,
        )
        cond["_action_tokens"] = self.action_token_projection(action_fa)
        return cond

    def _make_action_fa(self, z2: float, z7: float) -> torch.Tensor:
        """[1, npb, raw_action_dim] — same squashed [z2, z7] on every frame."""
        vec = torch.zeros(self.raw_action_dim, dtype=self.dtype, device=self.device)
        vec[0] = float(z2)
        if self.raw_action_dim > 1:
            vec[1] = float(z7)
        return vec.view(1, 1, -1).expand(1, self.num_frame_per_block, -1).contiguous()

    @property
    def _base_dit(self):
        base = self.generator.model
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        return base

    # ---- reset: install rope, alloc caches, prefill the seed chunk ---------
    @torch.no_grad()
    def reset(self, seed_latents: torch.Tensor, prompt_embeds: torch.Tensor,
              neutral_action: Tuple[float, float] = (0.0, 0.0)):
        """Prime the KV cache with one real seed chunk.

        ``seed_latents`` : [1, >=npb, C, H, W] (the chain-equivalent
        CONTEXT_FRAMES). Only the first ``npb`` frames are used as the seed.
        """
        from utils.infinity_rope import infinity_rope_active

        if seed_latents.shape[1] < self.num_frame_per_block:
            raise SystemExit(
                f"seed needs >= {self.num_frame_per_block} latent frames; "
                f"got {seed_latents.shape[1]}"
            )

        self.prompt_embeds = prompt_embeds.to(self.device, self.dtype)
        self.latents_log = []
        self.current_start_frame = 0

        # Enter the rope context for the whole session (held until close()).
        if self._rope_ctx is not None:
            self._rope_ctx.__exit__(None, None, None)
        self._rope_ctx = infinity_rope_active(self.infinity_rope, self._base_dit)
        self._rope_ctx.__enter__()

        self.generator.seq_len = max(
            int(self.generator.seq_len),
            self.num_frame_per_block * self.frame_seq_length,
        )
        _set_attention_window(
            self._base_dit,
            local_attn_size_frames=self.local_attn_size_frames,
            max_tokens=self.kv_cache_tokens,
        )

        num_blocks = len(self._base_dit.blocks)
        self.kv_cache = _initialize_kv_cache(
            num_blocks=num_blocks, batch_size=1,
            kv_cache_tokens=self.kv_cache_tokens, dtype=self.dtype, device=self.device,
        )
        self.crossattn_cache = _initialize_crossattn_cache(
            num_blocks=num_blocks, batch_size=1, dtype=self.dtype, device=self.device,
        )

        # VAE streaming-decode cache: clear ONCE up front, then cached_decode
        # per chunk (see MEMORY: Wan VAE chunked-decode quirk).
        self.vae.model.clear_cache()

        seed = seed_latents[:, :self.num_frame_per_block].to(self.device, self.dtype)
        z2, z7 = neutral_action
        cond = self._action_cond(self._make_action_fa(z2, z7))
        refresh_t = torch.full(
            [1, self.num_frame_per_block], self.context_noise,
            device=self.device, dtype=torch.float32,
        )
        with torch.amp.autocast("cuda", dtype=self.dtype):
            self.generator(
                noisy_image_or_video=seed,
                conditional_dict=cond,
                timestep=refresh_t,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=self.current_start_frame * self.frame_seq_length,
            )
        self.current_start_frame += self.num_frame_per_block

        # Decode the seed chunk so the viewer starts on real ground truth.
        seed_frames = self._decode_chunk(seed)
        self.latents_log.append(seed.float().cpu())
        log.info("reset: seeded %d latent frames into KV cache.", self.num_frame_per_block)
        return seed_frames

    # ---- one streaming step (mirrors generate_ar append path) --------------
    @torch.no_grad()
    def step(self, z2: float, z7: float) -> np.ndarray:
        """Generate ONE chunk under action (z2, z7); return decoded frames
        [T, H, W, 3] uint8 (RGB)."""
        action_fa = self._make_action_fa(z2, z7)
        cond = self._action_cond(action_fa)

        x = torch.randn(
            [1, self.num_frame_per_block, LATENT_C, LATENT_H, LATENT_W],
            dtype=torch.float32, device=self.device,
        ).to(self.dtype)

        ts = self.denoising_step_list
        pred_x0 = None
        for d in range(int(ts.shape[0])):
            tt = torch.full(
                [1, self.num_frame_per_block], float(ts[d].item()),
                device=self.device, dtype=torch.float32,
            )
            with torch.amp.autocast("cuda", dtype=self.dtype):
                out = self.generator(
                    noisy_image_or_video=x,
                    conditional_dict=cond,
                    timestep=tt,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=self.current_start_frame * self.frame_seq_length,
                )
            pred_x0 = out[1]
            if d < int(ts.shape[0]) - 1:
                next_t = float(ts[d + 1].item())
                flat = pred_x0.flatten(0, 1).float()
                flat_t = torch.full((flat.shape[0],), next_t, device=self.device, dtype=torch.float32)
                x = (
                    self.scheduler.add_noise(flat, torch.randn_like(flat), flat_t)
                    .view(1, self.num_frame_per_block, LATENT_C, LATENT_H, LATENT_W)
                    .to(self.dtype)
                )
        assert pred_x0 is not None

        # Clean cache-refresh forward so the next chunk attends to clean KVs.
        refresh_t = torch.full(
            [1, self.num_frame_per_block], self.context_noise,
            device=self.device, dtype=torch.float32,
        )
        with torch.amp.autocast("cuda", dtype=self.dtype):
            self.generator(
                noisy_image_or_video=pred_x0,
                conditional_dict=cond,
                timestep=refresh_t,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=self.current_start_frame * self.frame_seq_length,
            )
        self.current_start_frame += self.num_frame_per_block

        self.latents_log.append(pred_x0.float().cpu())
        return self._decode_chunk(pred_x0)

    # ---- decode one chunk via the warm streaming cache --------------------
    @torch.no_grad()
    def _decode_chunk(self, latent_chunk: torch.Tensor) -> np.ndarray:
        # latent: [1, T, C, H, W]. Feed at the VAE's dtype (cached_decode
        # internally permutes to [1, C, T, H, W]).
        px = self.vae.decode_to_pixel(latent_chunk.to(self.dtype), use_cache=True)
        vid = (0.5 * (px[0].float() + 1.0)).clamp(0, 1)           # [T, C, H, W]
        vid_np = (vid.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        return vid_np                                            # [T, H, W, 3]

    def close(self):
        if self._rope_ctx is not None:
            self._rope_ctx.__exit__(None, None, None)
            self._rope_ctx = None


# ---------------------------------------------------------------------------
# Seed + prompt loaders
# ---------------------------------------------------------------------------
def load_seed_from_zarr(path: str, n_frames: int) -> torch.Tensor:
    import zarr as zarr_lib
    g = zarr_lib.open_group(path, mode="r")
    lat = g["latents"][0:n_frames]            # _LATENT_HEAD_DROP == 0
    t = torch.from_numpy(np.asarray(lat).astype(np.float32))   # [T, C, H, W]
    return t.unsqueeze(0)                                       # [1, T, C, H, W]


def load_seed_from_pt(path: str, n_frames: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("latents", "full_latents", "gen_latents"):
            if k in obj:
                obj = obj[k]
                break
    t = obj.float()
    if t.dim() == 4:
        t = t.unsqueeze(0)
    return t[:, :n_frames]


# ---------------------------------------------------------------------------
# Control loops
# ---------------------------------------------------------------------------
class FrameSink:
    """Collects RGB frames and writes an mp4 on demand (ffmpeg via stdpipe)."""

    def __init__(self, out_path: str, fps: float):
        self.out_path = out_path
        self.fps = fps
        self.frames: List[np.ndarray] = []

    def add(self, frames_np: np.ndarray):
        for f in frames_np:
            self.frames.append(f)

    def write(self):
        if not self.frames:
            log.warning("No frames to write.")
            return
        import subprocess
        arr = np.stack(self.frames, 0)
        h, w = arr.shape[1], arr.shape[2]
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", self.out_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = proc.communicate(input=arr.tobytes(), timeout=180)
            if proc.returncode != 0:
                log.warning("ffmpeg failed: %s", err.decode(errors="replace"))
            else:
                log.info("Wrote %d frames -> %s", len(self.frames), self.out_path)
        except FileNotFoundError:
            # No ffmpeg — fall back to imageio.
            import imageio
            imageio.mimsave(self.out_path.replace(".mp4", ".gif"), self.frames, fps=self.fps)
            log.info("ffmpeg missing; wrote GIF instead.")


def _save_preview_png(frame: np.ndarray, path: str):
    try:
        import imageio
        imageio.imwrite(path, frame)
    except Exception as exc:  # pragma: no cover
        log.warning("preview png failed: %s", exc)


def run_repl(player: WorldModelPlayer, sink: FrameSink, preview_png: str,
             seed_frames: np.ndarray):
    """Headless stdin control loop — one command = generate chunk(s)."""
    sink.add(seed_frames)
    _save_preview_png(seed_frames[-1], preview_png)
    z2 = z7 = 0.0
    help_txt = (
        "\n=== world-model REPL ===\n"
        "  <z2> <z7> [n]   set steering z2 / throttle z7 in (-1,1), gen n chunks (default 1)\n"
        "  w / s           throttle +0.25 / -0.25 then gen 1\n"
        "  a / d           steer  -0.25 / +0.25 then gen 1\n"
        "  <enter>         repeat last action, gen 1\n"
        "  g <n>           gen n chunks with current action\n"
        "  0               neutral action (0,0)\n"
        "  save            write mp4 now\n"
        "  q / quit        write mp4 and exit\n"
        f"  (preview frame -> {preview_png})\n"
    )
    print(help_txt)
    print(f"[state] z2={z2:+.2f} z7={z7:+.2f}  current action set. Type a command.")

    def gen(n: int):
        nonlocal z2, z7
        for _ in range(max(1, n)):
            t0 = time.time()
            frames = player.step(z2, z7)
            sink.add(frames)
            _save_preview_png(frames[-1], preview_png)
            log.info("chunk @z2=%+.2f z7=%+.2f -> %d frames in %.2fs (total frames=%d)",
                     z2, z7, len(frames), time.time() - t0, len(sink.frames))

    while True:
        try:
            line = input(f"z2={z2:+.2f} z7={z7:+.2f} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            gen(1)
            continue
        low = line.lower()
        if low in ("q", "quit", "exit"):
            break
        if low in ("h", "help", "?"):
            print(help_txt)
            continue
        if low == "save":
            sink.write()
            continue
        if low == "0":
            z2 = z7 = 0.0
            gen(1)
            continue
        if low[0] in "wasd" and (len(low) == 1):
            if low == "w":
                z7 = min(1.0, z7 + 0.25)
            elif low == "s":
                z7 = max(-1.0, z7 - 0.25)
            elif low == "a":
                z2 = max(-1.0, z2 - 0.25)
            elif low == "d":
                z2 = min(1.0, z2 + 0.25)
            gen(1)
            continue
        if low.startswith("g"):
            try:
                n = int(low[1:].strip() or "1")
            except ValueError:
                n = 1
            gen(n)
            continue
        # numeric "<z2> <z7> [n]"
        parts = line.replace(",", " ").split()
        try:
            z2 = max(-1.0, min(1.0, float(parts[0])))
            if len(parts) > 1:
                z7 = max(-1.0, min(1.0, float(parts[1])))
            n = int(parts[2]) if len(parts) > 2 else 1
        except ValueError:
            print("  ? unrecognized — type 'help'")
            continue
        gen(n)

    sink.write()
    player.close()


def run_window(player: WorldModelPlayer, sink: FrameSink, fps: float,
               seed_frames: np.ndarray, upscale: int = 4):
    """Live cv2 window: hold W/A/S/D (or arrows) to drive; ESC/q to quit."""
    import cv2

    sink.add(seed_frames)
    win = "world-model (WASD drive | SPACE neutral | q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w = seed_frames.shape[1], seed_frames.shape[2]
    cv2.resizeWindow(win, w * upscale, h * upscale)

    z2 = z7 = 0.0
    decay = 0.85          # action eases back toward neutral when no key held.
    step_mag = 0.35

    def show(frame_rgb, info):
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(bgr, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, bgr)

    show(seed_frames[-1], "SEED")
    cv2.waitKey(1)
    log.info("Live window open. Focus it and drive with W/A/S/D. q/ESC to quit.")

    running = True
    while running:
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("w"):
            z7 = min(1.0, z7 + step_mag)
        elif key == ord("s"):
            z7 = max(-1.0, z7 - step_mag)
        elif key == ord("a"):
            z2 = max(-1.0, z2 - step_mag)
        elif key == ord("d"):
            z2 = min(1.0, z2 + step_mag)
        elif key == ord(" "):
            z2 = z7 = 0.0
        else:
            # no steering key this tick -> ease toward neutral
            z2 *= decay
            z7 *= decay
            if abs(z2) < 1e-3:
                z2 = 0.0
            if abs(z7) < 1e-3:
                z7 = 0.0

        frames = player.step(z2, z7)
        sink.add(frames)
        info = f"z2={z2:+.2f} z7={z7:+.2f}  f={len(sink.frames)}"
        for f in frames:
            show(f, info)
            if (cv2.waitKey(max(1, int(1000 / fps))) & 0xFF) in (27, ord("q")):
                running = False
                break

    cv2.destroyAllWindows()
    sink.write()
    player.close()


def run_selftest(player: WorldModelPlayer, sink: FrameSink, seed_frames: np.ndarray):
    """Scripted rollout: straight, left, right, forward — no interaction."""
    sink.add(seed_frames)
    script = (
        [(0.0, 0.5)] * 4 +     # drive forward straight
        [(-0.6, 0.4)] * 4 +    # bear left
        [(0.6, 0.4)] * 4 +     # bear right
        [(0.0, 0.0)] * 2       # coast
    )
    log.info("Self-test: rolling %d scripted chunks...", len(script))
    t0 = time.time()
    for i, (z2, z7) in enumerate(script):
        frames = player.step(z2, z7)
        sink.add(frames)
        log.info("  [%2d/%2d] z2=%+.2f z7=%+.2f -> %d frames",
                 i + 1, len(script), z2, z7, len(frames))
    log.info("Self-test rollout done in %.1fs (%d total frames).",
             time.time() - t0, len(sink.frames))
    sink.write()
    player.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _autodetect_wan_path() -> str:
    for cand in ("/home/ashish/Wan2.1", str(Path.home() / "Wan2.1")):
        if (Path(cand) / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth").exists():
            return cand
    return "/home/ashish/Wan2.1"


def main():
    p = argparse.ArgumentParser(
        description="Interactively drive the action-forcing world model on a local GPU.")
    p.add_argument("--ckpt", required=True, help="phase1_step*.pt checkpoint from a v8e8-style run.")
    p.add_argument("--config", default="configs/action_forcing_phase3_dmd.yaml",
                   help="Training config (geometry must match the checkpoint).")
    p.add_argument("--wan_model_path", default=None,
                   help="Dir holding Wan2.1-T2V-1.3B/ (auto-detected if omitted).")
    p.add_argument("--use_ema", action="store_true", help="Use 'generator_ema' weights if present.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--denoising_steps", type=int, default=None,
                   help="Override #denoise steps (default: trained pool size).")
    p.add_argument("--kv_cache_chunks", type=int, default=8,
                   help="Rolling KV window in chunks (8*3=24 frames ~ v8e8 local_attn=24). "
                        "Lower it (e.g. 2) to cut KV-cache memory on a busy GPU.")
    p.add_argument("--weights_dtype", choices=["bf16", "fp32"], default="bf16",
                   help="DiT weight precision. bf16 (default) is memory-light; fp32 matches training exactly.")
    p.add_argument("--infinity_rope", action=argparse.BooleanOptionalAction, default=None,
                   help="Override config's infinity_rope (default: read from config).")
    # seed
    p.add_argument("--seed_zarr", default=None, help="Zarr ride to pull the seed latent chunk from.")
    p.add_argument("--seed_latents", default=None, help="A .pt of latents [.,T,C,H,W] for the seed.")
    p.add_argument("--seed_frame_index", type=int, default=0,
                   help="(zarr) latent index to start the seed at.")
    # prompt
    p.add_argument("--prompt", default="a first-person view driving forward along a city sidewalk",
                   help="Caption to T5-encode for conditioning.")
    p.add_argument("--prompt_embeds", default=None, help="A .pt with precomputed prompt embeds [1,L,dim].")
    p.add_argument("--prompt_device", choices=["cpu", "cuda"], default="cpu",
                   help="Where to run the T5 prompt encoder (cpu keeps the GPU free for the DiT).")
    # output / mode
    p.add_argument("--mode", choices=["repl", "window", "selftest"], default=None,
                   help="Default: 'window' if a display is present else 'repl'.")
    p.add_argument("--out", default=None, help="Output mp4 path (default eval/play_<ts>/rollout.mp4).")
    p.add_argument("--fps", type=float, default=16.0, help="Playback / output fps.")
    p.add_argument("--selftest", action="store_true", help="Shortcut for --mode selftest.")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    wan_path = args.wan_model_path or _autodetect_wan_path()

    # output dir
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "eval" / f"play_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(out_dir / "rollout.mp4")
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    preview_png = str(Path(args.out).with_name("latest_frame.png"))

    player = WorldModelPlayer(
        config_path=args.config, ckpt_path=args.ckpt, device=device,
        wan_model_path=wan_path, use_ema=args.use_ema,
        denoising_steps=args.denoising_steps, kv_cache_chunks=args.kv_cache_chunks,
        infinity_rope=args.infinity_rope, weights_dtype=args.weights_dtype,
    )

    # prompt embeds
    if args.prompt_embeds:
        emb = torch.load(args.prompt_embeds, map_location="cpu", weights_only=False)
        if isinstance(emb, dict):
            emb = emb.get("prompt_embeds", emb)
        emb = emb.float()
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        prompt_embeds = emb.to(device)
        log.info("Loaded prompt embeds %s", tuple(prompt_embeds.shape))
    else:
        prompt_embeds = player.encode_prompt(args.prompt, encode_device=args.prompt_device)

    # seed latents
    npb = player.num_frame_per_block
    if args.seed_zarr:
        seed = load_seed_from_zarr(args.seed_zarr, args.seed_frame_index + npb)
        seed = seed[:, args.seed_frame_index:args.seed_frame_index + npb]
        log.info("Seed from zarr %s -> %s", args.seed_zarr, tuple(seed.shape))
    elif args.seed_latents:
        seed = load_seed_from_pt(args.seed_latents, npb)
        log.info("Seed from .pt %s -> %s", args.seed_latents, tuple(seed.shape))
    else:
        log.warning("No --seed_zarr / --seed_latents — seeding with random noise (OOD start).")
        seed = torch.randn(1, npb, LATENT_C, LATENT_H, LATENT_W)

    seed_frames = player.reset(seed, prompt_embeds)

    mode = args.mode
    if args.selftest:
        mode = "selftest"
    if mode is None:
        mode = "window" if os.environ.get("DISPLAY") else "repl"
        log.info("Mode auto-selected: %s (DISPLAY=%s)", mode, os.environ.get("DISPLAY"))

    sink = FrameSink(args.out, args.fps)
    if mode == "selftest":
        run_selftest(player, sink, seed_frames)
    elif mode == "window":
        try:
            run_window(player, sink, args.fps, seed_frames)
        except Exception as exc:
            log.warning("window mode failed (%s); falling back to repl.", exc)
            run_repl(player, sink, preview_png, seed_frames)
    else:
        run_repl(player, sink, preview_png, seed_frames)


if __name__ == "__main__":
    main()
