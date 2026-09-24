"""Shared API contract for the interactive world-model player.

This file is the ONLY shared surface between the engine (engine.py), the UI
(play.py / mock_engine.py), the precision layer (precision.py) and the bench
harness (bench.py). Workstream agents: code against this; do not widen it
without noting the change in docs/INTERACTIVE_WM_PLAN_3008.md.

Geometry facts (resolved from the rolling-definitive training launchers —
sbatch/carncommit_long.sbatch + sbatch/run_phase3_rolling_definitive_4node.sbatch):

  * chunk = num_frame_per_block = 3 latent frames = 12 pixel frames @ 16 fps
  * trained denoising ladder: [1000, 625, 357.142857, 208.333333],
    timestep_shift = 5.0  (NOT the phase-1 yaml default [1000,625,312.5,178.6])
  * KV attention window: local_attn_size = 21 latent frames (7 chunks),
    infinity_rope = true, context_noise = 0
  * seed prefill at training time: 3 GT chunks (9 latent frames)
  * interactive serving prefill: 7 GT chunks (21 latent frames), so the
    trained seven-chunk attention span starts completely populated
  * actions: pca_raw lineage, action_dims = [0, 1]
    (pca_0 = throttle, pca_1 = steer); z2/z7 is RETIRED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

# Trained defaults — single source of truth for every workstream.
TRAINED_DENOISING_STEP_LIST: Tuple[float, ...] = (1000.0, 625.0, 357.142857, 208.333333)
TIMESTEP_SHIFT: float = 5.0
NUM_FRAME_PER_BLOCK: int = 3            # latent frames per chunk
PIXEL_FRAMES_PER_CHUNK: int = 12        # 3 latent * 4x temporal VAE
# Streaming seed decode emits 9 frames for its first chunk, then the regular
# 12 for every subsequent chunk (see DECODER_NOTES.md).
SEED_FIRST_PIXEL_FRAMES: int = 9
LOCAL_ATTN_FRAMES: int = 21             # trained KV window (7 chunks)
TRAINED_SEED_PREFILL_CHUNKS: int = 3    # checkpoint training context
SEED_PREFILL_CHUNKS: int = 7            # interactive serving default
CHUNKS_PER_ACTION: int = 6              # one action owns 6 chunks (user spec)
PLAYBACK_FPS: float = 16.0
CHUNK_SECONDS: float = PIXEL_FRAMES_PER_CHUNK / PLAYBACK_FPS
LATENT_C, LATENT_H, LATENT_W = 16, 60, 104
PIXEL_H, PIXEL_W = 480, 832

# Inverting the paper's frozen PCA transform for an all-zero optical-flow
# field does not land at numeric (0, 0).  These are the normalized pca_raw
# coordinates of physical stillness and therefore the action-space no-op.
PHYSICAL_NULL_THROTTLE: float = -0.023394260555505753
PHYSICAL_NULL_STEER: float = -0.0013313499512150884


@dataclass
class Action:
    """One interactive action, held constant for a whole horizon.

    throttle/steer are NORMALIZED values in [-1, 1] in the model's
    pca_raw-scaled space (pca_0 = throttle, pca_1 = steer). The engine maps
    them to the per-frame action tensor; exact scaling internals live in the
    engine only.
    """
    throttle: float = PHYSICAL_NULL_THROTTLE
    steer: float = PHYSICAL_NULL_STEER


@dataclass
class EngineConfig:
    ckpt_path: str = ""
    config_path: str = "configs/action_forcing_phase3_dmd.yaml"
    # Parent dir: the loader appends the model-name subdir (Wan2.1-T2V-1.3B).
    wan_model_path: str = "/home/ashish/Wan2.1/"
    seed_zarr: str = ""                     # seed ride (latents)
    use_ema: bool = False                   # raw generator = production-eval convention

    # Precision / speed knobs (rebuild required when changed):
    precision: str = "bf16"                 # fp32 | bf16 | fp8_wo | fp8_dyn | fp4_wo
    # Latent->pixel decoder (rebuild required). "wan" = the training/eval
    # reference (Wan2.1 VAE, cached streaming decode). Alternatives trade
    # quality for speed and live in interactive/decoders.py:
    #   taew2_1 | lightvaew2_1 | lighttaew2_1
    decoder: str = "taew2_1"
    compile_mode: str = "off"               # off | reduce-overhead | max-autotune
    # Live-tunable (take effect at the next horizon):
    # Four steps traverse every timestep rung seen during training.
    denoising_steps: int = 4                # top-n of the trained ladder; >4 = interp
    kv_cache_chunks: int = 7                # 7 = trained local_attn 21 frames
    # Number of real GT chunks used to seed a ride. Selectable from 1..7;
    # seven fills the complete trained attention span, while shorter values
    # trade context for a quicker transition into generated content.
    seed_prefill_chunks: int = SEED_PREFILL_CHUNKS
    # Width of a fully-joint generation block. The trained production path is
    # sequential (1); widths 2/4 remain opt-in research comparisons only.
    # Rebuild required.
    block_chunks: int = 1
    horizon_chunks: int = CHUNKS_PER_ACTION
    decode_half_res: bool = False
    playback_fps: float = PLAYBACK_FPS
    max_buffered_horizons: int = 2          # backpressure bound

    device: str = "cuda"
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class HorizonStats:
    """Per-horizon telemetry — the bench harness and HUD both consume this."""
    horizon_index: int = 0
    action: Action = field(default_factory=Action)
    denoising_steps: int = 4
    precision: str = "bf16"
    gen_seconds: float = 0.0                # generator time, all chunks
    decode_seconds: float = 0.0             # VAE decode time, all chunks
    first_frame_latency_s: float = 0.0      # horizon start -> first frame out
    gen_fps: float = 0.0                    # pixel frames / gen wall second
    end_to_end_fps: float = 0.0
    peak_vram_gb: float = 0.0
    per_chunk_gen_ms: List[float] = field(default_factory=list)


class EngineBase:
    """Abstract engine. engine.py implements it for real; mock_engine.py fakes it.

    Threading contract: the caller owns pacing. `generate_horizon` is a
    blocking generator that yields decoded chunks as soon as each is ready;
    the UI/bench wraps it in its own thread and drains frames at playback fps.
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    def reset(self) -> np.ndarray:
        """(Re)build session state, prefill seed context; return the decoded
        seed frames [T, H, W, 3] uint8 RGB."""
        raise NotImplementedError

    def generate_horizon(self, action: Action) -> Iterator[np.ndarray]:
        """Generate cfg.horizon_chunks chunks under one latched action.
        Yields one decoded chunk at a time: [12, H, W, 3] uint8 RGB.
        After exhaustion, `last_stats` is populated."""
        raise NotImplementedError

    @property
    def last_stats(self) -> Optional[HorizonStats]:
        raise NotImplementedError

    def apply_live_settings(self, *, denoising_steps: Optional[int] = None,
                            horizon_chunks: Optional[int] = None,
                            decode_half_res: Optional[bool] = None,
                            seed_prefill_chunks: Optional[int] = None) -> None:
        """Settings that apply from the next horizon without a rebuild."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
