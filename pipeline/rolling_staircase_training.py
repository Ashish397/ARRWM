# pyright: reportGeneralTypeIssues=false
"""
Rolling-staircase training pipeline for phase-1 action-aware DMD.

Geometry (convention: t=0 = clean true frame, t=1000 = pure noise):
  4-slot live window, slot 0 = closest to present (cleanest),
  slot 3 = furthest future (noisiest). Steady-state staircase:
  [t_slot0, t_slot1, t_slot2, t_slot3] = [250, 500, 750, 1000].

At every rolling step, ONE forward is done over the 4 live slots as a single
12-frame chunk. Because the Wan KV-cache path applies unmasked attention across
all current q/k tokens (see wan/modules/causal_model.py `_forward_inference`),
this forward naturally implements "bidir self-attention within the live window
+ causal to the KV-cached past" without any extra masking.

After the rolling forward, slot 0's x0 is committed to the persistent clean KV
cache via a separate t=0 forward pass, while the noisy keys from the live-slot
forward are prevented from polluting the persistent cache by setting
`skip_cache_update=True` on the underlying causal model.

Gradient scope per rolling step:
  - Slot 0 (committed output) is always a grad slot.
  - One "aux" slot e in {1, 2, 3} is picked uniformly at random (synced across
    DDP ranks); that slot also carries gradient.
  - The other two slots run under `torch.no_grad` inside the same shared
    forward (grad is selected by tensor slicing after the forward).

The pipeline yields a list of `RollingStepOutput` records per rolling step
(plus per-pass records during staircase-fill warm-up); the trainer consumes
these records to compute DMD + GAN losses on the designated grad slots.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)

from model.action_modulation import ActionModulationProjection, ActionTokenProjection
from utils.scheduler import SchedulerInterface
from utils.wan_wrapper import WanDiffusionWrapper
from utils.debug_option import DEBUG


# ---------------------------------------------------------------------------
# Step-output record
# ---------------------------------------------------------------------------
@dataclass
class RollingStepOutput:
    """
    A single grad-slot payload produced by the pipeline during warmup or
    steady-state rolling. The trainer consumes these to compute losses.

    Fields:
      pred_x0:         [B, num_frame_per_block, C, H, W] — student x0 prediction
                       at this slot, grad-enabled.
      slot_timestep:   int — the noise level the slot was at when this forward
                       was run (one of the staircase rungs, or warmup pass level).
      slot_idx:        int — which slot in the 4-slot window (0..3).
      global_frame_start: int — absolute frame index (into the ride) of this
                       slot's first frame. Used for real_score RoPE alignment.
      action_frame:    [B, num_frame_per_block, action_dim] — the (scaled) action
                       used to condition this slot's frames. Detached.
      prompt_embeds:   [B, L, C_txt] — text embedding for this ride.
      phase:           str — "warmup" or "steady".
      state_preds_live: [B, num_slots, state_head_out_dim] or None — state-probe
                       predictions collected during THIS rolling step's single
                       shared cached forward. Populated only on the FIRST
                       grad-slot record of a step so the model can read the
                       full live-window probe output once per step without
                       duplicating the tensor across records. Shares the same
                       autograd graph as `pred_x0`, so backprop through the
                       probe loss flows into the DiT.
      per_slot_actions_live: [B, num_slots, action_dim] or None — the scaled
                       commanded action per slot (mean-pooled across the
                       slot's 3 frames), aligned with `state_preds_live`.
                       Also populated only on the first record.

    Batched-S_n DMD context (Option 4; all-slot scoring). Populated ONLY on
    the first record of a rolling step when the pipeline is configured for
    4-slot gradients + S_n batched scoring; otherwise they are None and the
    DMD model falls back to its legacy single-slot path.

    For slot n in 0..3 the teacher's 4-chunk input window S_n is:
        [ GT_{n-3}, GT_{n-2}, prev_{n-1}, live_n ]
    where
        prev_{-1}   = kv_anchor_chunk            (last committed KV chunk)
        prev_{n-1}  = pred_x0_all_slots[:, (n-1)*npb : n*npb].detach()
                                                 (clean live context for n>=1)
        live_n      = pred_x0_all_slots[:, n*npb : (n+1)*npb]
                                                 (noisy target, grad-enabled)
    and GT positions are contiguous chunks of gt_latents around the ride's
    action_base_frame_idx.

      pred_x0_all_slots:    [B, NUM_SLOTS * npb, C, H, W] — all 4 slots' x0
                            with the full autograd graph. Slots used as
                            clean context for later S_n are detached by the
                            DMD loss; slots used as noisy targets keep
                            their gradient.
      kv_anchor_chunk:      [B, npb, C, H, W] — the last committed chunk
                            from the previous rolling step (detached). For
                            the first steady-state step this is the
                            transition's committed slot-0 output.
      gt_context_chunks:    [B, 5 * npb, C, H, W] — 5 GT chunks at ride
                            frame positions
                              [action_base + (k-3)*npb : action_base + (k-2)*npb]
                            for k in 0..4, i.e. timesteps -3..+1 relative
                            to slot 0's own ride position. Padded by the
                            last available ride frame at the end-of-ride
                            boundary.
      gt_context_action_frames: [B, 5 * npb, action_dim] — GT actions for the
                            same 5 chunks, NO per-slot decay (these are
                            "ground truth" action signals, not commanded-
                            with-decay actions).
      per_slot_action_frames: [B, NUM_SLOTS * npb, action_dim] — the
                            live-window's per-slot commanded actions
                            (with decay already applied). Same as the
                            `per_frame_actions` the generator saw in its
                            cached forward.
      per_slot_timesteps_list: List[int] (len NUM_SLOTS) — the noise level
                            each slot was at during the live forward.
      fake_context_chunks:  [B, 3 * npb, C, H, W] — the last 3 STUDENT-
                            COMMITTED chunks in chronological order
                            (oldest first). Used by the fake_score's S_n
                            context (asymmetric with real_score which
                            sees GT context). For S_0 the fake window is
                            [commit_{-3}, commit_{-2}, commit_{-1}, live_0];
                            for S_n>=1 the oldest commit is replaced by
                            the previous live slot's pred_x0.detach() —
                            this substitution is done inside the DMD
                            loss. Seeded at warmup time from the 3
                            primed GT chunks (processed through the
                            student at t=0) so there is no cold-start.
      fake_context_actions: [B, 3 * npb, action_dim] — commanded actions
                            that produced each of the above commits.
                            During warmup seeding these are the GT
                            actions (no per-slot decay) that primed the
                            cache; during steady-state rolling they
                            become the slot-0 commanded action
                            (``decay[0] * a_{ride}``) used at each
                            commit forward.
    """

    pred_x0: torch.Tensor
    slot_timestep: int
    slot_idx: int
    global_frame_start: int
    action_frame: torch.Tensor
    prompt_embeds: torch.Tensor
    phase: str = "steady"
    state_preds_live: Optional[torch.Tensor] = None
    per_slot_actions_live: Optional[torch.Tensor] = None

    # --- Commit-stream payload (vis / eval) ---
    # ``pred_x0_committed`` is the detached x0 from the FINAL denoising pass
    # of the rolling step for this slot. For ``passes_per_step == 1`` this
    # is identical (up to detach) to ``pred_x0``. For ``passes_per_step > 1``
    # ``pred_x0`` is the GRAD-pass output (randomly selected pass, required
    # for the DMD autograd path), whereas ``pred_x0_committed`` is what the
    # KV cache / commit-history actually absorbs. Consumers that render or
    # report the student's output (visualization, eval capture) must prefer
    # ``pred_x0_committed`` so the rendered stream matches the cache state
    # the student sees on the next step. Losses that need gradients (DMD,
    # action/state aux) keep using ``pred_x0`` / ``pred_x0_all_slots``.
    pred_x0_committed: Optional[torch.Tensor] = None

    # --- Batched-S_n DMD context (Option 4) ---
    pred_x0_all_slots: Optional[torch.Tensor] = None
    kv_anchor_chunk: Optional[torch.Tensor] = None
    gt_context_chunks: Optional[torch.Tensor] = None
    gt_context_action_frames: Optional[torch.Tensor] = None
    per_slot_action_frames: Optional[torch.Tensor] = None
    per_slot_timesteps_list: Optional[List[int]] = None
    # --- Fake-score asymmetric context (student-rolled, no GT access) ---
    fake_context_chunks: Optional[torch.Tensor] = None
    fake_context_actions: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Live-window state
# ---------------------------------------------------------------------------
@dataclass
class LiveWindowState:
    """In-flight noisy latents for the 4 live slots. Detached from the graph
    between rolling steps (we always re-noise student x0 with a fresh noise
    sample for the next input)."""

    # [B, num_slots*num_frame_per_block, C, H, W] in slot-0..slot-3 order.
    noisy_latents: torch.Tensor
    # Timestep per slot (int, same convention as scheduler). Length num_slots.
    slot_timesteps: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RollingStaircaseTrainingPipeline:
    """
    Rolling 4-slot staircase denoiser for action-aware phase-1 training.

    Args:
      denoising_step_list:    list of 4 int timesteps (standard convention,
                              t=1000 noisiest, t=0 cleanest). Must be length 4.
                              Will be sorted ascending inside the pipeline to
                              give the slot ladder [t_slot0, ..., t_slot3]; e.g.
                              config `[1000, 750, 500, 250]` -> ladder
                              `[250, 500, 750, 1000]`.
      scheduler:              the flow-matching scheduler (provides add_noise).
      generator:              student generator (ODE-init merged, action-patched).
      num_frame_per_block:    3 (LongLive parity).
      num_slots:              4 (this pipeline is hard-coded to the 4-slot
                              staircase; the ladder length must match).
      action_decay_per_slot:  scaling for commanded action per slot. Applied
                              identically during warmup passes and steady-state
                              rolling — the per-slot decay is a fixed ladder
                              [1.0, 0.75, 0.5, 0.25]. The action INDEX at each
                              slot slides with the pass/step counter: at pass
                              (or step) k, slot s receives the per-frame
                              actions `gt_actions[base + (k+s)*npb : base +
                              (k+s+1)*npb]` scaled by `decay[s]`.
      prime_kv_frames:        number of clean GT frames pushed into KV cache
                              before the first rolling window (default 9 = 3
                              blocks @ num_frame_per_block=3).
      kv_frames_total:        physical buffer size (in frames) of the KV cache.
                              Must be >= kv_committed_max_frames + live_frames
                              (= 9 + 12 = 21 by default) so the live forward's
                              temp_k has room for committed + all live frames
                              without overflowing the buffer.
      kv_committed_max_frames:
                              hard cap (in frames) on the COMMITTED region of
                              the KV cache. Enforced by the pipeline via an
                              explicit trim after every slot-0 commit forward;
                              this is fully decoupled from `local_attn_size`
                              and `kv_frames_total`. Default 9 = 3 chunks.
      local_attn_size:        int (passed through to CausalWanModel; -1 =
                              global). This sets `max_attention_size =
                              local_attn_size * frame_seq_length` on every
                              attention layer — i.e. how many of the most
                              recent temp-k tokens a query may attend to.
                              With kv_committed_max_frames=9 and live_frames=12,
                              set local_attn_size=21 so every live query
                              attends to all 9 committed past frames + all 12
                              live frames (fully symmetric bidirectional
                              attention inside the live window, causal-to-past
                              for the committed region).
      action_projection:      shared ActionModulationProjection module that
                              converts raw actions to adaLN-zero modulation.
    """

    # Default backwards-compat constant. The actual slot count used at
    # runtime is `self.num_live_slots` (see __init__). Do not read
    # `NUM_SLOTS` in new code — read `self.num_live_slots`.
    NUM_SLOTS = 4

    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        *,
        num_frame_per_block: int = 3,
        num_slots: int = 4,
        passes_per_step: int = 1,
        action_decay_per_slot: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
        prime_kv_frames: int = 9,
        kv_frames_total: int = 21,
        kv_committed_max_frames: int = 9,
        local_attn_size: int = 21,
        context_noise: int = 0,
        action_projection: Optional[ActionModulationProjection] = None,
        action_token_projection: Optional[ActionTokenProjection] = None,
        real_score_num_gt_chunks: int = 2,
        disable_renoise: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # --- Live-window topology ---
        # `num_live_slots` is the number of slots in the live staircase.
        # Historically this pipeline was hard-coded to 4; 2 is now supported
        # so a caller can trade a narrower attention window for more passes.
        # `passes_per_step` is the number of internal denoising passes the
        # generator runs per rolling step. With P passes, each committed
        # chunk sees NS*P total denoising ops over its lifetime, matching
        # the ODE student's step budget (normally 4).
        if int(num_slots) not in (1, 2, 4):
            raise ValueError(
                f"num_slots must be 1, 2, or 4; got {num_slots}. Values "
                f"outside this set would require generalizing the transition-"
                f"slide logic and the rollout helpers further."
            )
        if int(passes_per_step) not in (1, 2, 4):
            raise ValueError(
                f"passes_per_step must be 1, 2, or 4; got {passes_per_step}."
            )
        if int(num_slots) * int(passes_per_step) != 4:
            raise ValueError(
                f"num_slots * passes_per_step must equal 4 (the ODE student's "
                f"total denoising budget per committed chunk); got "
                f"num_slots={num_slots}, passes_per_step={passes_per_step}."
            )
        self.num_live_slots: int = int(num_slots)
        self.passes_per_step: int = int(passes_per_step)
        # Keep .NUM_SLOTS in sync so legacy reads that still reference
        # `self.NUM_SLOTS` pick up the runtime value.
        self.NUM_SLOTS = self.num_live_slots

        # Staircase ladder: ascending timesteps assigned slot 0 -> slot NS-1,
        # interpreted as the INPUT noise level for PASS 0 of a rolling step.
        # The user-supplied `denoising_step_list` is expected to match this
        # slot-0..slot-(NS-1) ordering.
        if len(denoising_step_list) != self.num_live_slots:
            raise ValueError(
                f"denoising_step_list must have length {self.num_live_slots}; got "
                f"{len(denoising_step_list)}: {denoising_step_list}"
            )
        ladder = sorted(int(t) for t in denoising_step_list)
        self.ladder_timesteps: List[int] = ladder  # [t_slot0, ..., t_slot(NS-1)]
        # Total ODE step budget per committed chunk = NS * P. Per-sub-step
        # noise increment is `max_t // total_steps`; with max_t=1000 and
        # NS*P=4 this is the same 250-per-step schedule the ODE student
        # was distilled at.
        max_t = self.ladder_timesteps[-1]
        total_denoise_steps = self.num_live_slots * self.passes_per_step
        step = max_t // total_denoise_steps
        # Pass-p input noise level for slot s (see formula in design notes):
        #   noise[p, s] = step * (s * P + P - p)
        # With P=1 this reduces to noise[0, s] = step * (s + 1) = ladder[s].
        self.pass_input_timesteps: List[List[int]] = [
            [step * (s * self.passes_per_step + self.passes_per_step - p)
             for s in range(self.num_live_slots)]
            for p in range(self.passes_per_step)
        ]
        # Cross-check: the user-supplied denoising_step_list (which becomes
        # the pass-0 input ladder) must match the derived pass_input_timesteps[0].
        # Otherwise the ODE schedule the student was trained on is inconsistent
        # with what the pipeline will actually run.
        if self.pass_input_timesteps[0] != self.ladder_timesteps:
            raise ValueError(
                f"denoising_step_list {self.ladder_timesteps} does not match the "
                f"pass-0 input ladder derived from (num_live_slots={self.num_live_slots}, "
                f"passes_per_step={self.passes_per_step}, max_t={max_t}): "
                f"{self.pass_input_timesteps[0]}. For 4x1 use [250,500,750,1000]; "
                f"for 2x2 use [500,1000]."
            )
        # Warmup-pass lockstep levels go t=max_t -> 0 in `total_denoise_steps`
        # increments, uniform across slots (the staircase gets FILLED here;
        # the per-slot offset only kicks in after transition_to_steady_state).
        self.warmup_pass_input_timesteps: List[int] = [
            max_t - step * k for k in range(total_denoise_steps)
        ]
        self.warmup_pass_output_timesteps: List[int] = [
            max(0, t - step) for t in self.warmup_pass_input_timesteps
        ]

        self.scheduler = scheduler
        self.generator = generator
        self.num_frame_per_block = int(num_frame_per_block)
        # action_decay_per_slot must match num_live_slots. Defaults:
        #   NS=4 -> (1.0, 0.75, 0.5, 0.25)
        #   NS=2 -> (1.0, 0.5)
        action_decay_per_slot = tuple(float(x) for x in action_decay_per_slot)
        if len(action_decay_per_slot) != self.num_live_slots:
            raise ValueError(
                f"action_decay_per_slot has length {len(action_decay_per_slot)}; "
                f"expected num_live_slots={self.num_live_slots}."
            )
        self.action_decay_per_slot = action_decay_per_slot
        self.prime_kv_frames = int(prime_kv_frames)
        self.kv_frames_total = int(kv_frames_total)
        self.kv_committed_max_frames = int(kv_committed_max_frames)
        self.local_attn_size = int(local_attn_size)
        self.context_noise = int(context_noise)
        self.action_projection = action_projection
        self.action_token_projection = action_token_projection
        self.disable_renoise: bool = bool(disable_renoise)
        # Real-score GT context width (per-slot GT chunks). Defaults to 2
        # (S_n^real = [GT_{n-3}, GT_{n-2}, prev_{n-1}, live_n]). Increasing
        # this to 3 or 4 widens the left context with older GT chunks —
        # requires the GT slice window to grow from 5*npb to (k+3)*npb
        # and the real-score's seq_len to grow from 4*npb to (k+2)*npb.
        # The fake-score context is UNCHANGED (still 3 chunks: 2 commits +
        # 1 live-detached or similar), so this is a real-only widening.
        if int(real_score_num_gt_chunks) < 2:
            raise ValueError(
                f"real_score_num_gt_chunks must be >= 2 (to preserve "
                f"prev_{{n-1}} + at least 2 GT chunks per slot); got "
                f"{real_score_num_gt_chunks}."
            )
        self.real_score_num_gt_chunks = int(real_score_num_gt_chunks)

        # --- Stream-B action-token invariants ---
        # The generator's inner DiT was trained with `action_tokens_per_frame=1`
        # (and the wrapper's `seq_len` extended to match). This pipeline
        # always populates `_action_tokens` in the conditional dict, and
        # the wrapper's forward only looks at that key when it's non-None.
        # We fail loud if the upstream wiring is inconsistent:
        #   (a) pipeline was not given an action_token_projection, OR
        #   (b) the inner DiT is not expecting action tokens per frame.
        # Either one silently reduces the generator to AdaLN-only, which
        # is out-of-distribution for the ODE-distilled student.
        inner = self._inner_model()
        a_per_f = int(getattr(inner, "action_tokens_per_frame", 0))
        if a_per_f != 1:
            raise RuntimeError(
                "RollingStaircaseTrainingPipeline requires the generator's "
                "inner DiT to have `action_tokens_per_frame == 1` (Stream "
                f"B). Got {a_per_f}. This pipeline is for the ODE-"
                "distilled, action-aware student — run through "
                "`BaseModel` with `action_patch_enabled=True` and "
                "`action_conditioning_mode='both'` so that "
                "`adjust_seq_len_for_action_tokens` is wired at init."
            )
        if self.action_token_projection is None:
            raise RuntimeError(
                "action_token_projection is None but "
                f"action_tokens_per_frame={a_per_f}. The pipeline would "
                "leave `_action_tokens` unset and the DiT would attend to "
                "zero-valued action-token slots — out-of-distribution for "
                "the ODE student. Pass the trained `ActionTokenProjection` "
                "via the `action_token_projection=` kwarg."
            )

        # Sanity check: buffer must fit committed + live scratch.
        _live_frames = self.num_live_slots * self.num_frame_per_block
        _required_buffer = self.kv_committed_max_frames + _live_frames
        if self.kv_frames_total < _required_buffer:
            raise ValueError(
                f"kv_frames_total ({self.kv_frames_total}) must be >= "
                f"kv_committed_max_frames + live_frames ({self.kv_committed_max_frames} "
                f"+ {_live_frames} = {_required_buffer}) so temp_k can hold "
                f"the committed region plus all live slots during the live forward."
            )
        # Sanity: attention reach should cover committed + live so every live
        # frame sees every other live frame and the full committed region.
        if self.local_attn_size != -1 and self.local_attn_size < _required_buffer:
            raise ValueError(
                f"local_attn_size ({self.local_attn_size}) must be either -1 "
                f"(global) or >= kv_committed_max_frames + live_frames "
                f"({_required_buffer}) so every live-frame query attends to "
                f"all committed past frames AND every other live frame."
            )

        # Wan specifics.
        self.num_transformer_blocks = 30
        self._spatial_frame_seq_length = 1560

        self.kv_cache1: Optional[list] = None
        self.crossattn_cache: Optional[list] = None

        # Propagate local_attn_size to the inner DiT up front so cache sizing
        # is consistent. Dynamic per-step scheduling is not used here.
        self._inner_model().local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[RollingStaircase] ladder (slot 0->3): {self.ladder_timesteps} | "
                f"warmup input ts: {self.warmup_pass_input_timesteps} -> output: "
                f"{self.warmup_pass_output_timesteps} | num_frame_per_block="
                f"{self.num_frame_per_block} | kv_frames_total={self.kv_frames_total}"
            )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------
    def _inner_model(self):
        """Return the inner Wan DiT, unwrapping DDP / PEFT wrappers if any.

        Scalar attributes like `local_attn_size`, `skip_cache_update`,
        `num_frame_per_block` live on the inner CausalWanModel and must be
        written there — if we write them on a DDP wrapper they never reach
        the attention blocks that read them."""
        model = self.generator.model
        # Unwrap DDP.
        if hasattr(model, "module"):
            try:
                from torch.nn.parallel import DistributedDataParallel as _DDP  # type: ignore
                if isinstance(model, _DDP):
                    model = model.module
            except Exception:
                pass
        # Unwrap PEFT (get_base_model) if the model is a PeftModel.
        if hasattr(model, "get_base_model"):
            try:
                model = model.get_base_model()
            except Exception:
                pass
        return model

    @property
    def frame_seq_length(self) -> int:
        extra = int(getattr(self._inner_model(), "action_tokens_per_frame", 0))
        return self._spatial_frame_seq_length + extra

    @property
    def live_frames(self) -> int:
        return self.num_live_slots * self.num_frame_per_block

    # -----------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------
    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        tokens = self.kv_frames_total * self.frame_seq_length
        kv_cache1: list = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros([batch_size, tokens, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, tokens, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        crossattn_cache: list = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def clear_kv_cache(self) -> None:
        if getattr(self, "kv_cache1", None) is not None:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()
        if getattr(self, "crossattn_cache", None) is not None:
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def _set_all_modules_max_attention_size(self, value: int) -> None:
        if int(value) == -1:
            target_size = 32760
        else:
            target_size = int(value) * self.frame_seq_length
        inner = self._inner_model()
        if hasattr(inner, "max_attention_size"):
            setattr(inner, "max_attention_size", target_size)
        for _, module in inner.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass

    def _set_skip_cache_update(self, value: bool) -> None:
        """Toggle the "do not commit this forward's k/v into the KV cache"
        flag on the underlying CausalWanModel. Used for the noisy live-slot
        forward so slots 1-3 do not pollute the persistent clean cache."""
        setattr(self._inner_model(), "skip_cache_update", bool(value))

    @torch.no_grad()
    def _trim_committed_kv_cache(self) -> None:
        """Enforce a hard FIFO cap on the committed region of the KV cache at
        `self.kv_committed_max_frames * frame_seq_length` tokens.

        Called after every commit forward (priming, warmup-to-steady
        transition, and every steady-state slot-0 commit). Decouples the
        committed-region capacity from:
          - the attention reach (`local_attn_size`), which sets how far back
            each query can look during a forward, and
          - the physical buffer size (`kv_frames_total`), which must be
            large enough to scratch the committed region PLUS the live
            window during the live forward.

        After this call, for every block in `self.kv_cache1`:
          - `local_end_index <= kv_committed_max_frames * frame_seq_length`
          - the `[sink_tokens, local_end_index)` slice contains the most
            recent committed tokens (older ones shifted out).
          - the immediately-following slice `[sink_tokens + num_rolled,
            local_end_old)` is explicitly zeroed (defense against a future
            recompute path bumping `local_end` back up).
          - `global_end_index` is UNCHANGED (it remains the logical absolute
            write position used by RoPE alignment; the oldest logical
            positions just disappear from the cache).

        MAINTAINER NOTE — buffer tail is intentionally NOT zeroed:
          Positions `[local_end_new, kv_frames_total * frame_seq_length)`
          (i.e. everything past the trimmed committed region) are left with
          whatever bytes were previously there. This is safe under the
          current design because:

            (a) the next live forward runs with `skip_cache_update=True`
                and writes fresh k/v into a CLONE (`temp_k`) of the
                persistent buffer — the clone is fully overwritten at
                positions `[local_end_new, local_end_new + live_frames*fs)`
                before attention reads it (see `_forward_inference` in
                `wan/modules/causal_model.py`);
            (b) the subsequent slot-0 commit forward writes at the SAME
                range and then runs this trim again before any other
                read.

          If you ever refactor so that `temp_k` is NOT fully overwritten
          past `local_end_new` (e.g. a partial-append code path, or an
          attention mask that reads beyond `local_end_new`), you MUST
          either zero the full tail here or fix the consumer — otherwise
          stale k/v from a previous step will silently participate in
          attention. Skipping the memset is a ~4MB/block/step saving in
          bf16, which adds up across 30 transformer blocks and hundreds
          of rolling steps per ride.
        """
        if self.kv_cache1 is None:
            return

        fs = self.frame_seq_length
        max_committed_tokens = int(self.kv_committed_max_frames) * fs
        inner = self._inner_model()
        sink_size = int(getattr(inner.config, "sink_size", 0))
        sink_tokens = sink_size * fs

        for blk in self.kv_cache1:
            local_end = int(blk["local_end_index"].item())
            if local_end <= max_committed_tokens:
                continue
            num_evicted = local_end - max_committed_tokens
            # Protect the sink region (if any): shift only the post-sink content.
            num_rolled = local_end - num_evicted - sink_tokens
            if num_rolled < 0:
                # Pathological: sink region alone exceeds max_committed.
                # Nothing safe to trim; leave it and warn.
                continue
            if num_rolled > 0:
                blk["k"][:, sink_tokens : sink_tokens + num_rolled] = blk["k"][
                    :,
                    sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
                ].clone()
                blk["v"][:, sink_tokens : sink_tokens + num_rolled] = blk["v"][
                    :,
                    sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
                ].clone()
            # Zero the tail so stale tokens never leak into attention if
            # local_end is ever set higher in a recompute branch.
            blk["k"][:, sink_tokens + num_rolled : local_end].zero_()
            blk["v"][:, sink_tokens + num_rolled : local_end].zero_()
            blk["local_end_index"].fill_(max_committed_tokens)
            # global_end_index is deliberately untouched — it carries the
            # logical absolute frame count used for RoPE and for the
            # eviction-trigger arithmetic in _forward_inference.

    # -----------------------------------------------------------------
    # Action / conditional helpers
    # -----------------------------------------------------------------
    def _broadcast_int(self, value: int, device: torch.device) -> int:
        """Broadcast a rank-0 integer choice across DDP ranks."""
        tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(tensor, src=0)
        return int(tensor.item())

    def _sample_aux_slot(self, device: torch.device) -> int:
        """Sample aux slot in {1, ..., NS-1}, synced across ranks. Returns 0
        if NS == 1 (degenerate; not used in practice)."""
        if self.num_live_slots <= 1:
            return 0
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            e = int(torch.randint(1, self.num_live_slots, size=(1,)).item())
        else:
            e = 0
        return self._broadcast_int(e, device)

    def _sample_grad_pass(self, device: torch.device) -> int:
        """Sample a pass index in [0, passes_per_step) that will carry the
        generator's autograd graph this rolling step. Rank-synced so DDP
        ranks all pick the same pass. With passes_per_step=1 returns 0
        deterministically, making this a no-op for the legacy 4x1 path."""
        if self.passes_per_step <= 1:
            return 0
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            p = int(torch.randint(0, self.passes_per_step, size=(1,)).item())
        else:
            p = 0
        return self._broadcast_int(p, device)

    def _build_sliding_actions(
        self,
        gt_actions: torch.Tensor,
        base_frame_idx: int,
    ) -> torch.Tensor:
        """Build per-frame action conditioning for the 4-slot live window
        with PER-SLOT DECAY and SLIDING ACTION INDICES.

        At pass/step k (where `base_frame_idx = prime_end + k * num_frame_per_block`),
        slot s receives:

            action_frame[slot s] = decay[s] *
                gt_actions[:, base_frame_idx + s*npb : base_frame_idx + (s+1)*npb]

        Args:
          gt_actions:     [B, T_ride, action_dim] ride per-frame actions.
          base_frame_idx: absolute frame index at which slot 0's 3-frame chunk
                          begins for this pass/step. Equivalent to "step k"'s
                          slot-0 frame-start.

        Returns:
          per_frame:      [B, live_frames (= 12), action_dim] with per-slot
                          decay applied.

        If `base_frame_idx + NUM_SLOTS * npb` exceeds the ride length, the
        action for out-of-range slots is clamped to the last available frame
        (i.e., we pad with the last GT action). This matches the steady-state
        rolling termination condition: `rollout_ride` stops when slot 0 runs
        out of actions, so this padding only affects the noisier future slots.
        """
        if gt_actions.dim() != 3:
            raise ValueError(
                f"gt_actions must be [B, T, action_dim], got shape {tuple(gt_actions.shape)}"
            )
        batch_size, total_frames, action_dim = gt_actions.shape
        npb = self.num_frame_per_block
        per_frame = torch.zeros(
            batch_size,
            self.live_frames,
            action_dim,
            device=gt_actions.device,
            dtype=gt_actions.dtype,
        )
        for slot_idx in range(self.num_live_slots):
            # Each slot's chunk starts at `base_frame_idx + slot_idx * npb`.
            slot_f0 = base_frame_idx + slot_idx * npb
            slot_f1 = slot_f0 + npb
            if slot_f0 >= total_frames:
                # Fully past end-of-ride — clamp to last frame's action.
                last = gt_actions[:, -1:, :]  # [B, 1, action_dim]
                slot_actions = last.expand(-1, npb, -1)
            elif slot_f1 > total_frames:
                # Partially past end — take what's available, pad with last.
                avail = gt_actions[:, slot_f0:total_frames, :]
                pad_count = slot_f1 - total_frames
                last = gt_actions[:, -1:, :]
                pad = last.expand(-1, pad_count, -1)
                slot_actions = torch.cat([avail, pad], dim=1)
            else:
                slot_actions = gt_actions[:, slot_f0:slot_f1, :]
            out_f0 = slot_idx * npb
            out_f1 = out_f0 + npb
            per_frame[:, out_f0:out_f1, :] = slot_actions * self.action_decay_per_slot[slot_idx]
        return per_frame

    def _compute_live_modulation(
        self,
        per_frame_actions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the shared action_projection on the full live window
        ([B, 12, action_dim]) to produce [B, 12, 6, hidden_dim] modulation.
        Does NOT cache — caller decides whether to enable gradients."""
        if self.action_projection is None:
            raise RuntimeError(
                "action_projection is required but was not provided to the pipeline."
            )
        per_frame_actions = per_frame_actions.to(device=device, dtype=dtype)
        modulation = self.action_projection(per_frame_actions, num_frames=per_frame_actions.shape[1])
        return modulation.to(device=device, dtype=dtype)

    def _compute_live_action_tokens(
        self,
        per_frame_actions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the shared action_token_projection to produce per-frame action
        tokens ``[B, F, hidden_dim]`` that get appended to each frame's
        spatial token sequence inside the DiT. Stream B of the two-stream
        action conditioning the ODE student was trained on.

        Fails loud if action_token_projection is missing — the generator's
        inner DiT is configured with ``action_tokens_per_frame=1`` and
        would otherwise attend to zero-valued action token slots.
        """
        if self.action_token_projection is None:
            raise RuntimeError(
                "action_token_projection is required but was not provided to "
                "the pipeline (Stream B). Without it the DiT runs out-of-"
                "distribution: seq-len per frame is 1561 but only 1560 "
                "tokens carry signal."
            )
        per_frame_actions = per_frame_actions.to(device=device, dtype=dtype)
        tokens = self.action_token_projection(per_frame_actions)
        return tokens.to(device=device, dtype=dtype)

    def _prepare_conditional(
        self,
        prompt_embeds: torch.Tensor,
        action_modulation: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> dict:
        # Both action streams are always populated — this pipeline refuses
        # to run in single-stream mode (see __init__ validation).
        return {
            "prompt_embeds": prompt_embeds,
            "_action_modulation": action_modulation,
            "_action_tokens": action_tokens,
        }

    # -----------------------------------------------------------------
    # Timestep tensor helper
    # -----------------------------------------------------------------
    def _build_live_timestep(
        self,
        slot_timesteps: Iterable[int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [B, live_frames] timestep tensor assigning slot_timesteps[i] to
        frames [i*npb, (i+1)*npb)."""
        ts = torch.zeros(batch_size, self.live_frames, device=device, dtype=torch.int64)
        for slot_idx, t_slot in enumerate(slot_timesteps):
            f0 = slot_idx * self.num_frame_per_block
            f1 = f0 + self.num_frame_per_block
            ts[:, f0:f1] = int(t_slot)
        return ts

    def _renoise(
        self,
        clean_x0: torch.Tensor,
        target_t: int,
    ) -> torch.Tensor:
        """Re-noise a clean x0 block [B, F, C, H, W] to `target_t`.
        Returns a tensor with the same shape."""
        if int(target_t) <= 0:
            return clean_x0
        flat = clean_x0.flatten(0, 1)
        noise = torch.randn_like(flat)
        batch_frames = flat.shape[0]
        t = int(target_t) * torch.ones([batch_frames], device=flat.device, dtype=torch.long)
        noised = self.scheduler.add_noise(flat, noise, t)
        return noised.unflatten(0, clean_x0.shape[:2])

    # -----------------------------------------------------------------
    # Warmup: prime KV cache with clean GT frames
    # -----------------------------------------------------------------
    @torch.no_grad()
    def warmup_prime_kv_cache(
        self,
        gt_latents: torch.Tensor,
        gt_actions: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> int:
        """Feed the first `prime_kv_frames` GT frames into the KV cache using
        t=0 forwards with GT actions (no gradient).

        Args:
          gt_latents:   [B, >=prime_kv_frames, C, H, W] ride latents.
          gt_actions:   [B, >=prime_kv_frames, action_dim] ride actions.
          prompt_embeds:[B, L, C_txt].

        Returns:
          The number of frames committed (= prime_kv_frames, rounded down to
          a multiple of num_frame_per_block).
        """
        device = gt_latents.device
        dtype = gt_latents.dtype
        batch_size = gt_latents.shape[0]

        n_blocks = self.prime_kv_frames // self.num_frame_per_block
        frames_prime = n_blocks * self.num_frame_per_block
        if frames_prime <= 0 or gt_latents.shape[1] < frames_prime:
            return 0

        # Commit one block at a time so CausalWanModel's per-block cache update
        # semantics match training-time inference (3 frames per forward).
        current_start_frame = 0
        self._set_skip_cache_update(False)
        for block_index in range(n_blocks):
            f0 = block_index * self.num_frame_per_block
            f1 = f0 + self.num_frame_per_block
            x_block = gt_latents[:, f0:f1]
            a_block = gt_actions[:, f0:f1]  # [B, npb, action_dim]
            modulation = self._compute_live_modulation(a_block, device=device, dtype=dtype)
            action_tokens = self._compute_live_action_tokens(
                a_block, device=device, dtype=dtype,
            )
            cond = self._prepare_conditional(prompt_embeds, modulation, action_tokens)
            timestep = torch.zeros([batch_size, self.num_frame_per_block], device=device, dtype=torch.int64)
            self.generator(
                noisy_image_or_video=x_block,
                conditional_dict=cond,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            current_start_frame += self.num_frame_per_block

        return frames_prime

    # -----------------------------------------------------------------
    # Warmup: lockstep-denoise NS noise blocks all the way to clean
    # -----------------------------------------------------------------
    def warmup_fill_staircase(
        self,
        gt_actions: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        prime_end: int,
        enable_grad_on_final_clean: bool = False,
    ) -> Tuple[torch.Tensor, List[RollingStepOutput]]:
        """Do `NS * P` lockstep denoising passes that bring NS pure-noise
        blocks all the way to clean. Cache updates are suppressed for the
        entire warmup (we commit to the clean cache only in
        `transition_to_steady_state` via a separate slot-0 t=0 forward).

        In both (NS=4, P=1) and (NS=2, P=2) configurations the warmup runs
        `NS*P = total_denoise_steps` passes — the same ODE step budget.
        Only the width of the live window (= NS * npb frames per pass)
        differs.

        Action scheme (sliding, per-slot decay):
          At pass k (k in 0..3), slot s receives
            `decay[s] * gt_actions[:, prime_end + (k+s)*npb : prime_end + (k+s+1)*npb]`.
          i.e. each warmup pass is a "virtual rolling step" — slot 0 targets
          post-prime chunk k, slot 1 targets chunk k+1, etc., with per-slot
          decay applied.

        RoPE / cache position: the generator's `current_start` is FIXED at
        `prime_end * frame_seq_length` for every warmup pass. That is, the
        warmup live-window's RoPE positions are `[prime_end, prime_end+1, ...,
        prime_end + live_frames - 1]` (= 9..20 for defaults). The virtual
        time advance of the sliding-action scheme is carried entirely in
        which actions we pull from the ride, NOT in the RoPE stream — this
        keeps RoPE positions contiguous with the priming commits and avoids
        zero gaps in the KV cache buffer.

        If `enable_grad_on_final_clean` is True, the FINAL pass runs with
        grad on all 4 slots so the warmup emits a dense gradient signal.
        Default is False — warmup runs entirely under no_grad.

        Returns:
          clean_blocks:  [B, live_frames, C, H, W] — the 4 fully-denoised
                         blocks, detached. At pass=3 end, slot s carries the
                         content of post-prime chunk (3 + s) (in the user's
                         sliding-action semantics), but its RoPE position in
                         the generator stream is `prime_end + s*npb + f`.
          warmup_step_outputs: list of RollingStepOutput records with phase=
                         "warmup" (only populated if enable_grad_on_final_clean).
        """
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        batch_size = prompt_embeds.shape[0]
        npb = self.num_frame_per_block

        inner = self._inner_model()
        noise_shape = (
            batch_size,
            self.live_frames,
            int(getattr(inner, "in_dim", 16)),
            60,
            104,
        )
        live = torch.randn(noise_shape, device=device, dtype=dtype)

        warmup_outputs: List[RollingStepOutput] = []
        num_warmup_passes = self.num_live_slots * self.passes_per_step
        self._set_skip_cache_update(True)
        try:
            for pass_idx in range(num_warmup_passes):
                input_t = self.warmup_pass_input_timesteps[pass_idx]
                output_t = self.warmup_pass_output_timesteps[pass_idx]

                # Sliding action base for this pass: slot 0 targets chunk
                # `pass_idx` in the ride's action stream, slot s targets
                # chunk (pass_idx + s). This drives ACTION selection only;
                # the generator's RoPE start is fixed at prime_end (below).
                action_base_frame_idx = prime_end + pass_idx * npb

                per_frame_actions = self._build_sliding_actions(
                    gt_actions, base_frame_idx=action_base_frame_idx
                ).contiguous()
                modulation = self._compute_live_modulation(
                    per_frame_actions, device=device, dtype=dtype
                )
                action_tokens = self._compute_live_action_tokens(
                    per_frame_actions, device=device, dtype=dtype,
                )
                cond = self._prepare_conditional(
                    prompt_embeds, modulation, action_tokens,
                )

                timestep = torch.full(
                    (batch_size, self.live_frames),
                    int(input_t),
                    device=device,
                    dtype=torch.int64,
                )

                want_grad = enable_grad_on_final_clean and (pass_idx == num_warmup_passes - 1)
                cm = torch.enable_grad() if want_grad else torch.no_grad()
                with cm:
                    out = self.generator(
                        noisy_image_or_video=live,
                        conditional_dict=cond,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        # Fixed RoPE start across all warmup passes: the live
                        # window sits at logical frame positions [prime_end,
                        # prime_end + live_frames). Skip-cache-update is on,
                        # so nothing is written to the persistent cache.
                        current_start=prime_end * self.frame_seq_length,
                    )
                flow_pred, pred_x0 = out[0], out[1]
                state_preds_live: Optional[torch.Tensor] = (
                    out[2] if len(out) >= 3 else None
                )

                if want_grad:
                    per_slot_actions_live = per_frame_actions.reshape(
                        batch_size, self.num_live_slots, npb, -1
                    ).mean(dim=2).detach()
                    for slot_idx in range(self.num_live_slots):
                        f0 = slot_idx * npb
                        f1 = f0 + npb
                        is_first = (slot_idx == 0)
                        warmup_outputs.append(
                            RollingStepOutput(
                                pred_x0=pred_x0[:, f0:f1],
                                slot_timestep=int(input_t),
                                slot_idx=int(slot_idx),
                                # Global frame start follows the ACTION base
                                # (i.e., which logical chunk this slot carries
                                # in the sliding-action semantics), not the
                                # RoPE position. real_score/GAN use this for
                                # alignment with GT action/frame streams.
                                global_frame_start=action_base_frame_idx + f0,
                                action_frame=per_frame_actions[:, f0:f1].detach(),
                                prompt_embeds=prompt_embeds.detach(),
                                phase="warmup",
                                state_preds_live=state_preds_live if is_first else None,
                                per_slot_actions_live=per_slot_actions_live if is_first else None,
                                # Warmup records are only emitted on the FINAL
                                # pass (lockstep fill), so the "committed" view
                                # equals this pass's pred_x0 detached. Kept in
                                # sync so vis/eval can always reach for
                                # pred_x0_committed regardless of phase.
                                pred_x0_committed=pred_x0[:, f0:f1].detach().contiguous(),
                            )
                        )

                with torch.no_grad():
                    if int(output_t) > 0:
                        live = self._renoise(pred_x0.detach(), int(output_t))
                    else:
                        live = pred_x0.detach()
        finally:
            self._set_skip_cache_update(False)

        return live, warmup_outputs

    # -----------------------------------------------------------------
    # Transition from warmup-clean-stack -> steady-state staircase
    # -----------------------------------------------------------------
    @torch.no_grad()
    def transition_to_steady_state(
        self,
        clean_live: torch.Tensor,
        gt_actions: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        prime_end: int,
    ) -> Tuple[LiveWindowState, int, int, torch.Tensor]:
        """After `warmup_fill_staircase`, `clean_live` is [B, 12, C, H, W] with
        all 4 slots clean. Under the sliding-action scheme, the final warmup
        pass (pass 3) had slot s conditioned on action `a_{3+s}` (user's
        notation, indexing into the ride action stream starting at prime_end).

        Transition:
          1) Commit slot 0 to the KV cache. IMPORTANT: the commit's RoPE /
             cache position is `prime_end` (LOGICAL position, contiguous with
             the 9 primed frames already in the cache). This avoids leaving
             zero gaps in the cache's local buffer. The slot-0 action used
             for conditioning is still `decay[0] * a_3 = decay[0] *
             gt_actions[prime_end+3*npb : prime_end+4*npb]` — the ACTION
             index is independent of the RoPE / logical cache index.
          2) Trim the committed region to `kv_committed_max_frames` via FIFO
             (this drops the oldest prime frame and keeps slot-0 as the new
             latest-committed slot).
          3) Slide slots 1->0, 2->1, 3->2. Sample fresh pure noise as new
             slot 3. Re-noise sliced clean slots to [250, 500, 750].

        Returns:
          state:                  LiveWindowState for the first steady-state step.
          next_logical_start:     logical RoPE-stream frame index of slot 0's
                                  first frame in the next rolling forward
                                  (= prime_end + npb, i.e. contiguous with
                                  the just-committed frame).
          next_action_base:       absolute ride action-stream base index for
                                  the first steady-state step's slot 0
                                  (= prime_end + (NUM_SLOTS) * npb, i.e. the
                                  NEXT chunk after the one we just committed).
          initial_kv_anchor:      [B, npb, C, H, W] detached — the just-
                                  committed slot-0 chunk. Serves as the
                                  kv_anchor_chunk for the very first steady-
                                  state rolling_step call.
        """
        device = clean_live.device
        dtype = clean_live.dtype
        batch_size = clean_live.shape[0]
        npb = self.num_frame_per_block

        # --- Commit slot 0 (= virtual chunk (num_warmup_passes-1) in the
        # sliding-action semantics) ---
        slot0_clean = clean_live[:, :npb]

        # ACTION: pull a_{NW-1} where NW = NS * P = total warmup passes. This
        # is slot 0's action at the LAST warmup pass (slot 0 targeted chunk
        # (pass_idx) at each warmup pass; at pass_idx = NW-1 that is chunk
        # NW-1). Equivalently for (4,1) NW-1 = 3; for (2,2) NW-1 = 3.
        num_warmup_passes = self.num_live_slots * self.passes_per_step
        slot0_action_ride_base = prime_end + (num_warmup_passes - 1) * npb
        slot0_actions = gt_actions[
            :, slot0_action_ride_base : slot0_action_ride_base + npb, :
        ] * self.action_decay_per_slot[0]
        # End-of-ride padding guard (rollout_ride gates on this but be safe).
        if slot0_actions.shape[1] < npb:
            last = gt_actions[:, -1:, :]
            pad = last.expand(-1, npb - slot0_actions.shape[1], -1)
            slot0_actions = torch.cat([slot0_actions, pad], dim=1)
        slot0_actions_c = slot0_actions.contiguous()
        modulation = self._compute_live_modulation(
            slot0_actions_c, device=device, dtype=dtype
        )
        action_tokens = self._compute_live_action_tokens(
            slot0_actions_c, device=device, dtype=dtype,
        )
        cond = self._prepare_conditional(prompt_embeds, modulation, action_tokens)
        ts0 = torch.zeros([batch_size, npb], device=device, dtype=torch.int64)

        # RoPE / cache position: LOGICAL = prime_end, contiguous with priming.
        # This is INTENTIONALLY decoupled from the action-base above.
        self._set_skip_cache_update(False)
        self.generator(
            noisy_image_or_video=slot0_clean,
            conditional_dict=cond,
            timestep=ts0,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=prime_end * self.frame_seq_length,
        )
        # Cap committed cache at kv_committed_max_frames (FIFO).
        self._trim_committed_kv_cache()

        # --- Slide: clean slots 1..NS-1 into positions 0..NS-2; new slot NS-1
        # gets fresh noise. Renoise each slided clean chunk to the steady-
        # state ladder level at its new slot index (unless
        # ``self.disable_renoise`` is set, in which case carryover slots are
        # fed through at t=0). ---
        NS = self.num_live_slots
        max_t = self.ladder_timesteps[-1]
        slided_noisy: List[torch.Tensor] = []
        for new_slot in range(NS - 1):
            src_f0 = (new_slot + 1) * npb
            src_f1 = src_f0 + npb
            slided_clean = clean_live[:, src_f0:src_f1]
            if self.disable_renoise:
                slided_noisy.append(slided_clean.contiguous())
            else:
                slided_noisy.append(
                    self._renoise(slided_clean, self.ladder_timesteps[new_slot])
                )
        # Fresh pure noise for the new slot NS-1 (max ladder step).
        fresh_noise = torch.randn_like(clean_live[:, :npb])
        slided_noisy.append(fresh_noise)
        live = torch.cat(slided_noisy, dim=1)
        next_slot_timesteps = (
            [0] * (NS - 1) + [int(max_t)]
            if self.disable_renoise
            else list(self.ladder_timesteps)
        )
        state = LiveWindowState(
            noisy_latents=live.contiguous(),
            slot_timesteps=next_slot_timesteps,
        )
        next_logical_start = prime_end + npb
        # Action base for the first steady-state step: slot 0 targets chunk
        # NW (= NS*P, the next chunk after the one we just committed).
        next_action_base = prime_end + num_warmup_passes * npb
        # The initial kv_anchor for the first steady-state step IS this
        # just-committed slot-0 chunk; its action frame can be reconstructed
        # at DMD time from gt_actions using (action_base - npb) =
        # prime_end + 3*npb. We therefore don't thread the action frame
        # through — just the latent tensor.
        initial_kv_anchor = slot0_clean.detach().clone().contiguous()
        return state, next_logical_start, next_action_base, initial_kv_anchor

    # -----------------------------------------------------------------
    # Steady-state rolling step
    # -----------------------------------------------------------------
    def rolling_step(
        self,
        state: LiveWindowState,
        gt_actions: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        logical_start_frame: int,
        action_base_frame_idx: int,
        kv_anchor_chunk: Optional[torch.Tensor] = None,
        gt_latents: Optional[torch.Tensor] = None,
        fake_context_chunks: Optional[torch.Tensor] = None,
        fake_context_actions: Optional[torch.Tensor] = None,
        grad_slot_indices: Optional[List[int]] = None,
    ) -> Tuple[LiveWindowState, List[RollingStepOutput], int, int, torch.Tensor]:
        """One steady-state rolling staircase step.

        RoPE / cache position vs action index (DECOUPLED):
          - `logical_start_frame` is the RoPE-stream frame index for slot 0's
            first frame in THIS forward. It is contiguous with the KV cache's
            committed region (global_end_index // frame_seq_length), so the
            cache buffer stays packed and the trim helper can shift cleanly.
          - `action_base_frame_idx` is the ride's absolute frame index used
            to pull slot 0's action; it is typically
            `logical_start_frame + (warmup_passes - 1) * npb` during steady
            state (e.g. +9 for warmup_passes=4, npb=3).

        Action scheme (sliding, per-slot decay):
          slot s receives `decay[s] * gt_actions[:, action_base_frame_idx +
          s*npb : action_base_frame_idx + (s+1)*npb]`. So slot 0 carries the
          action for the chunk it is ABOUT TO COMMIT (in ride semantics), slot
          1 the action for the NEXT chunk (one step in the future), etc.

        Performs:
          1) Live-slot forward (skip_cache_update=True) — 4 slots, sliding
             per-slot action conditioning, bidir-within-live attention (free
             via KV-cache path), no pollution of the persistent cache. Grad
             is enabled by default only on slot 0 and a random aux slot `e`.
          2) Collect RollingStepOutput records for the grad slots.
          3) Commit slot 0's x0 to the clean KV cache via a separate t=0
             forward (no grad), conditioned on slot-0's action.
          4) Re-noise x0 of slots 1,2,3 to [250, 500, 750] respectively,
             slide into positions 0,1,2 for the NEXT step, and sample fresh
             pure noise for new slot 3 (t=1000).

        Args:
          state:                  current LiveWindowState.
          gt_actions:             [B, T_ride, action_dim] the full ride actions.
          prompt_embeds:          [B, L, C_txt] text conditioning.
          logical_start_frame:    RoPE-stream frame index (contiguous with cache).
          action_base_frame_idx:  ride-absolute frame index for action sliding.
          kv_anchor_chunk:        [B, npb, C, H, W] — the last-committed clean
                                  chunk (student's slot-0 pred_x0 from the
                                  previous step, or transition's committed
                                  slot-0 for the first steady step). Required
                                  when attaching S_n context to the records;
                                  if None no S_n context is attached and the
                                  DMD model falls back to legacy single-slot
                                  scoring.
          gt_latents:             [B, T_ride, C, H, W] full ride latents. Used
                                  to slice 5 GT context chunks at positions
                                  `action_base + (k-3)*npb` for k in 0..4 for
                                  the batched S_n teacher input. Required for
                                  S_n context attachment; if None the pipeline
                                  emits records without S_n context.
          fake_context_chunks:    [B, 3*npb, C, H, W] — last 3 student-
                                  committed chunks (oldest first), detached.
                                  Attached to the first record for the
                                  fake_score's asymmetric S_n input. Must be
                                  supplied by the caller (batcher); the
                                  pipeline does not maintain a persistent
                                  commit buffer because it is called with
                                  possibly-different slot state across
                                  invocations.
          fake_context_actions:   [B, 3*npb, action_dim] — commanded actions
                                  (decayed for post-warmup commits, GT for
                                  the prime-seeded commits) paired with
                                  ``fake_context_chunks``. Used to build the
                                  fake_score's AdaLN modulation.
          grad_slot_indices:      optionally override the set of slots with grad.
                                  Default [0, 1, 2, 3] — all 4 slots are grad-
                                  enabled. Slots used as clean context in later
                                  S_n windows are detached by the DMD loss.

        Returns:
          next_state:        LiveWindowState for the next step.
          step_outputs:      list of RollingStepOutput, one per grad slot.
          logical_advance:   frames to add to logical_start for next step (=npb).
          action_advance:    frames to add to action_base for next step (=npb).
          next_kv_anchor:    [B, npb, C, H, W] detached — slot-0's pred_x0 that
                             was just committed, for use as the NEXT step's
                             kv_anchor_chunk.
        """
        device = state.noisy_latents.device
        dtype = state.noisy_latents.dtype
        batch_size = state.noisy_latents.shape[0]
        npb = self.num_frame_per_block

        # Guard against RoPE temporal-table overflow. The live-window forward
        # indexes `freqs[0][logical_start_frame : logical_start_frame + F]`
        # with F = live_frames, and the commit forward indexes
        # `[logical_start_frame : logical_start_frame + npb]`. If we exceed
        # the precomputed table we'd silently short-slice and then crash
        # inside a .view(), which is a nightmare to debug on a long ride.
        # `rope_max_seq_len` is set to 10000 in CausalWanModel.__init__ (see
        # `wan/modules/causal_model.py`). This check fails loudly with a
        # ride-actionable message long before any silent corruption.
        rope_cap = int(getattr(self._inner_model(), "rope_max_seq_len", 1024))
        max_pos_needed = int(logical_start_frame) + int(self.live_frames)
        if max_pos_needed > rope_cap:
            raise RuntimeError(
                f"RoPE temporal-table overflow: logical_start_frame="
                f"{logical_start_frame} + live_frames={self.live_frames} = "
                f"{max_pos_needed} exceeds rope_max_seq_len={rope_cap}. "
                f"At npb={npb} this is roughly step "
                f"{(max_pos_needed - 9) // max(npb, 1)} of the ride. Either "
                f"cap ride length, or bump _ROPE_MAX_SEQ_LEN in "
                f"wan/modules/causal_model.py (and wan/modules/model.py for "
                f"the bidirectional scorers if you also extend their range). "
                f"Current default is 10000 (≈3330 steady steps at npb=3)."
            )

        if grad_slot_indices is None:
            # Option 4: all NS slots have generator gradients enabled on the
            # grad-pass. Combined with S_n batched-scoring DMD this gives
            # every slot a first-class DMD signal every rolling step. See
            # `DMD2B2BLAM_Staircase.generator_loss_on_slots` for the batched
            # real_score/fake_score call and the detach() on clean-context
            # live chunks that keeps each S_n's gradient localized to its
            # own target slot. The DMD loss can further subset which slots
            # are actually scored via `dmd_active_slot_policy`.
            grad_slot_indices = list(range(self.num_live_slots))
        grad_slot_indices = sorted(set(int(i) for i in grad_slot_indices))

        # --- Build sliding per-slot action modulation + conditional ---
        # Actions are SHARED across all internal passes within one rolling
        # step: slot s uses decay[s] * gt_actions[base + s*npb : base + (s+1)*npb]
        # for every pass. The INPUT NOISE level changes per pass; the ACTION
        # conditioning is fixed per rolling step.
        per_frame_actions = self._build_sliding_actions(
            gt_actions, base_frame_idx=action_base_frame_idx
        )
        modulation = self._compute_live_modulation(per_frame_actions, device=device, dtype=dtype)
        action_tokens_live = self._compute_live_action_tokens(
            per_frame_actions, device=device, dtype=dtype,
        )
        cond = self._prepare_conditional(
            prompt_embeds, modulation, action_tokens_live,
        )

        # --- Internal denoising passes (1 for 4x1 default, 2 for 2x2) ---
        # Within each rolling step we run `passes_per_step` generator forwards:
        #   pass 0 input ladder = self.pass_input_timesteps[0] (= state.slot_timesteps)
        #   pass p>0 input ladder = self.pass_input_timesteps[p], reached by
        #     renoising pass (p-1)'s clean pred_x0 down by one sub-step per
        #     slot (matches the ODE student's 4-step schedule: for 2x2 that
        #     means 1000->500->250 and 500->250->0 across the two passes).
        # Exactly ONE pass carries the autograd graph (`grad_pass_idx`); the
        # others run under torch.no_grad(). DMD's target = the grad-pass's
        # clean prediction; commit + slide always use the FINAL pass's
        # detached prediction (cleanest, regardless of which pass owns grad).
        grad_pass_idx = self._sample_grad_pass(device)

        pred_x0_grad: Optional[torch.Tensor] = None
        state_preds_grad: Optional[torch.Tensor] = None
        # When disable_renoise is set, the live window's actual noise levels
        # (state.slot_timesteps) diverge from the ladder-derived
        # pass_input_timesteps; align both the DiT timestep tensor and the
        # per-record ``grad_pass_slot_timesteps`` with the state in that case.
        if self.disable_renoise:
            grad_pass_slot_timesteps = list(state.slot_timesteps)
        else:
            grad_pass_slot_timesteps = list(self.pass_input_timesteps[grad_pass_idx])
        pred_x0_final_detached: Optional[torch.Tensor] = None

        live = state.noisy_latents
        self._set_skip_cache_update(True)
        try:
            for pass_p in range(self.passes_per_step):
                if self.disable_renoise and pass_p == 0:
                    pass_slot_ts = list(state.slot_timesteps)
                else:
                    pass_slot_ts = self.pass_input_timesteps[pass_p]
                timestep_p = self._build_live_timestep(
                    slot_timesteps=pass_slot_ts,
                    batch_size=batch_size,
                    device=device,
                )
                want_grad = (pass_p == grad_pass_idx)
                cm = torch.enable_grad() if want_grad else torch.no_grad()
                with cm:
                    out_p = self.generator(
                        noisy_image_or_video=live,
                        conditional_dict=cond,
                        timestep=timestep_p,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=logical_start_frame * self.frame_seq_length,
                    )
                pred_x0_p = out_p[1]
                if want_grad:
                    pred_x0_grad = pred_x0_p
                    state_preds_grad = out_p[2] if len(out_p) >= 3 else None
                # Inter-pass renoise (no_grad): bring each slot from clean
                # pred down to the next pass's input ladder level. On the
                # final pass we also save the detached clean prediction for
                # commit + slide below.
                if pass_p < self.passes_per_step - 1:
                    with torch.no_grad():
                        next_levels = self.pass_input_timesteps[pass_p + 1]
                        pred_detached = pred_x0_p.detach()
                        renoised_slots = [
                            self._renoise(
                                pred_detached[:, s * npb : (s + 1) * npb],
                                int(next_levels[s]),
                            )
                            for s in range(self.num_live_slots)
                        ]
                        live = torch.cat(renoised_slots, dim=1).contiguous()
                else:
                    pred_x0_final_detached = pred_x0_p.detach()
        finally:
            self._set_skip_cache_update(False)

        # pred_x0 for downstream bookkeeping = the grad-pass's output, which
        # is the one with the autograd graph; DMD will score THIS (as the
        # target) via pred_x0_all_slots_attach.
        assert pred_x0_grad is not None
        pred_x0 = pred_x0_grad
        state_preds_live: Optional[torch.Tensor] = state_preds_grad

        # Gather per-slot step outputs (grad slots keep the graph; the others
        # are detached inside `next_state` construction below). Note that
        # `global_frame_start` uses the ACTION base index so real_score / GAN
        # heads align with the ride's ground-truth frame/action stream.
        #
        # State-probe plumbing: `state_preds_live` / `per_slot_actions_live`
        # are attached ONLY to the first grad-slot record so the loss can
        # consume the full 4-slot probe output once per step without
        # duplicating the tensor across records. Per-slot actions are
        # mean-pooled across the 3-frame chunk (matches critic/probe
        # supervision convention).
        per_slot_actions_live = per_frame_actions.reshape(
            batch_size, self.num_live_slots, npb, -1
        ).mean(dim=2).detach()

        # --- Batched-S_n DMD scoring context (Option 4; attached to first
        # record only) ---
        #
        # For slot n the teacher's 4-chunk window is
        #   S_n = [ GT_{n-3}, GT_{n-2}, prev_{n-1}, live_n ]
        # where prev_{-1} is the last-committed KV anchor and prev_{n-1}
        # for n>=1 is the PREVIOUS live slot's pred_x0 (detached by the
        # DMD loss so each S_n's gradient is localized to its own target).
        #
        # We slice 5 GT chunks at ride positions
        #   [action_base + (k-3)*npb : action_base + (k-2)*npb]  for k in 0..4
        # covering timesteps -3..+1 relative to slot-0's own ride position.
        # End-of-ride padding clamps out-of-range slices to the last
        # available ride frame, matching `_build_sliding_actions`'s policy.
        sn_context_available = (
            gt_latents is not None and kv_anchor_chunk is not None
        )
        pred_x0_all_slots_attach: Optional[torch.Tensor] = None
        kv_anchor_attach: Optional[torch.Tensor] = None
        gt_ctx_chunks_attach: Optional[torch.Tensor] = None
        gt_ctx_actions_attach: Optional[torch.Tensor] = None
        per_slot_action_frames_attach: Optional[torch.Tensor] = None
        per_slot_timesteps_attach: Optional[List[int]] = None
        fake_ctx_chunks_attach: Optional[torch.Tensor] = None
        fake_ctx_actions_attach: Optional[torch.Tensor] = None
        if sn_context_available:
            total_frames_lat = int(gt_latents.shape[1])
            total_frames_act = int(gt_actions.shape[1])
            # For real_score's S_n^real = [GT_{n-k-1}, ..., GT_{n-2},
            # prev_{n-1}, live_n] we need GT chunks at ride positions
            # {n - real_k - 1, ..., n - 2} for n in [0, NS-1]. That spans
            # chunk indices [-real_k - 1, +1] (inclusive) so the slice width
            # is (real_k + 3) chunks. Anchor chunk (the last committed
            # student chunk = GT index -1 in absolute frames) is accessed
            # via kv_anchor_chunk, not this slice.
            real_k = int(self.real_score_num_gt_chunks)
            start_ctx = action_base_frame_idx - (real_k + 1) * npb
            end_ctx = action_base_frame_idx + 2 * npb
            if start_ctx < 0:
                raise RuntimeError(
                    f"[rolling_step] start_ctx={start_ctx} < 0. The pipeline's "
                    f"invariant is that rolling steps only run once the prime "
                    f"region + warmup has pushed `action_base_frame_idx` past "
                    f"`(real_k+1) * npb` = {(real_k + 1) * npb} (real_k="
                    f"{real_k}). Got action_base={action_base_frame_idx}. If "
                    f"real_k > 2 you must increase `prime_kv_frames` or delay "
                    f"the first rolling step accordingly."
                )

            def _slice_with_pad(src: torch.Tensor, s: int, e: int, total: int) -> torch.Tensor:
                # Slice src[:, s:e] along dim=1, right-padding with src[:, -1:]
                # when e exceeds `total`. Handles arbitrary trailing dims.
                if e <= total:
                    return src[:, s:e]
                last = src[:, -1:]  # [B, 1, ...]
                rep_shape = [1] * src.dim()
                if s >= total:
                    rep_shape[1] = e - s
                    return last.repeat(*rep_shape)
                avail = src[:, s:total]
                rep_shape[1] = e - total
                pad = last.repeat(*rep_shape)
                return torch.cat([avail, pad], dim=1)

            gt_ctx_chunks_attach = _slice_with_pad(
                gt_latents, start_ctx, end_ctx, total_frames_lat,
            ).to(device=device, dtype=dtype).contiguous()
            gt_ctx_actions_attach = _slice_with_pad(
                gt_actions, start_ctx, end_ctx, total_frames_act,
            ).to(device=device, dtype=dtype).contiguous()
            kv_anchor_attach = kv_anchor_chunk.detach().to(
                device=device, dtype=dtype
            ).contiguous()
            pred_x0_all_slots_attach = pred_x0  # keep graph; DMD loss will
                                                # detach per-slot as needed.
            per_slot_action_frames_attach = per_frame_actions.detach().contiguous()
            # Timesteps reported to DMD = the grad-pass's input ladder. This
            # is the actual noise level the scorers' target slots were
            # predicted at. (For 4x1 this equals state.slot_timesteps.)
            per_slot_timesteps_attach = list(grad_pass_slot_timesteps)

            # --- Fake-score asymmetric context (student-rolled, no GT) ---
            # Batcher is authoritative for the commit history. We accept
            # both tensors here and validate shape; the DMD loss reads
            # them off the first record.
            if fake_context_chunks is not None and fake_context_actions is not None:
                expected_frames = 3 * npb
                if fake_context_chunks.shape[1] != expected_frames:
                    raise RuntimeError(
                        f"[rolling_step] fake_context_chunks has "
                        f"{fake_context_chunks.shape[1]} frames; expected "
                        f"3*npb = {expected_frames}. The batcher must "
                        f"maintain exactly 3 prior commits (seeded from "
                        f"the 3 primed GT chunks so there is never a "
                        f"cold-start)."
                    )
                if fake_context_actions.shape[1] != expected_frames:
                    raise RuntimeError(
                        f"[rolling_step] fake_context_actions has "
                        f"{fake_context_actions.shape[1]} frames; expected "
                        f"3*npb = {expected_frames}."
                    )
                fake_ctx_chunks_attach = fake_context_chunks.detach().to(
                    device=device, dtype=dtype
                ).contiguous()
                fake_ctx_actions_attach = fake_context_actions.detach().to(
                    device=device, dtype=dtype
                ).contiguous()

        step_outputs: List[RollingStepOutput] = []
        # Final-pass detached x0 slice — what ACTUALLY gets committed /
        # slid into the next step. Vis + eval consume this so the rendered
        # stream matches the KV cache state, independent of which pass
        # happened to carry the autograd graph.
        assert pred_x0_final_detached is not None
        for i, slot_idx in enumerate(grad_slot_indices):
            f0 = slot_idx * npb
            f1 = f0 + npb
            is_first = (i == 0)
            step_outputs.append(
                RollingStepOutput(
                    pred_x0=pred_x0[:, f0:f1],
                    slot_timestep=int(grad_pass_slot_timesteps[slot_idx]),
                    slot_idx=int(slot_idx),
                    global_frame_start=int(action_base_frame_idx + f0),
                    action_frame=per_frame_actions[:, f0:f1].detach(),
                    prompt_embeds=prompt_embeds.detach(),
                    phase="steady",
                    state_preds_live=state_preds_live if is_first else None,
                    per_slot_actions_live=per_slot_actions_live if is_first else None,
                    pred_x0_committed=pred_x0_final_detached[:, f0:f1].contiguous(),
                    pred_x0_all_slots=pred_x0_all_slots_attach if is_first else None,
                    kv_anchor_chunk=kv_anchor_attach if is_first else None,
                    gt_context_chunks=gt_ctx_chunks_attach if is_first else None,
                    gt_context_action_frames=gt_ctx_actions_attach if is_first else None,
                    per_slot_action_frames=per_slot_action_frames_attach if is_first else None,
                    per_slot_timesteps_list=per_slot_timesteps_attach if is_first else None,
                    fake_context_chunks=fake_ctx_chunks_attach if is_first else None,
                    fake_context_actions=fake_ctx_actions_attach if is_first else None,
                )
            )

        # --- Commit slot 0's clean x0 to the KV cache (t=0, no grad) ---
        # ACTION: reuse slot-0 slice from the sliding-action tensor (decay[0] *
        # gt_actions[action_base : action_base + npb]).
        # ROPE / CACHE: current_start = logical_start_frame (contiguous).
        # Also this slot0_x0 is returned as the NEXT step's kv_anchor_chunk
        # so the downstream DMD batched-S_0 always conditions on the exact
        # latent chunk that the KV cache just absorbed. We ALWAYS commit from
        # the FINAL pass's detached prediction (cleanest/most-refined),
        # regardless of which pass got the autograd graph this step.
        assert pred_x0_final_detached is not None
        with torch.no_grad():
            slot0_x0 = pred_x0_final_detached[:, :npb]
            next_kv_anchor_chunk = slot0_x0.clone().contiguous()
            slot0_actions = per_frame_actions[:, :npb].detach()
            slot0_actions_c = slot0_actions.contiguous()
            slot0_modulation = self._compute_live_modulation(
                slot0_actions_c, device=device, dtype=dtype
            )
            slot0_action_tokens = self._compute_live_action_tokens(
                slot0_actions_c, device=device, dtype=dtype,
            )
            slot0_cond = self._prepare_conditional(
                prompt_embeds, slot0_modulation, slot0_action_tokens,
            )
            ts0 = torch.zeros([batch_size, npb], device=device, dtype=torch.int64)
            self._set_skip_cache_update(False)
            self.generator(
                noisy_image_or_video=slot0_x0,
                conditional_dict=slot0_cond,
                timestep=ts0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=logical_start_frame * self.frame_seq_length,
            )
            # Cap committed cache at kv_committed_max_frames (FIFO).
            self._trim_committed_kv_cache()

        # --- Build the next-state live window ---
        # Next-step slots (generalized over NS):
        #   new_slot_k = renoise(old_slot_{k+1} final-pass x0, ladder[k])  for k in 0..NS-2
        #   new_slot_{NS-1} = fresh pure noise (implicitly ladder[NS-1] = max_t)
        # The "final-pass x0" is the cleanest prediction for each old slot;
        # it is detached (already done above via `pred_x0_final_detached`).
        # With ``self.disable_renoise`` set, carryover slots pass their clean
        # x0 through at t=0 (no noise added), only the trailing slot is seeded
        # with fresh pure noise.
        max_t = self.ladder_timesteps[-1]
        with torch.no_grad():
            x0_detached = pred_x0_final_detached
            slided: List[torch.Tensor] = []
            for new_slot in range(self.num_live_slots - 1):
                src_f0 = (new_slot + 1) * npb
                src_f1 = src_f0 + npb
                if self.disable_renoise:
                    slided.append(x0_detached[:, src_f0:src_f1].contiguous())
                else:
                    slided.append(
                        self._renoise(
                            x0_detached[:, src_f0:src_f1],
                            int(self.ladder_timesteps[new_slot]),
                        )
                    )
            # Fresh noise for the new final slot (noise level = ladder[-1]).
            slided.append(torch.randn_like(x0_detached[:, :npb]))
            next_live = torch.cat(slided, dim=1)

        next_slot_timesteps = (
            [0] * (self.num_live_slots - 1) + [int(max_t)]
            if self.disable_renoise
            else list(self.ladder_timesteps)
        )
        next_state = LiveWindowState(
            noisy_latents=next_live.contiguous(),
            slot_timesteps=next_slot_timesteps,
        )
        return next_state, step_outputs, npb, npb, next_kv_anchor_chunk

    # -----------------------------------------------------------------
    # Full-ride rollout (generator side, used by the trainer)
    # -----------------------------------------------------------------
    def rollout_ride(
        self,
        gt_latents: torch.Tensor,
        gt_actions: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        max_rolling_steps: Optional[int] = None,
    ) -> Iterable[List[RollingStepOutput]]:
        """Yield a LIST of RollingStepOutput records per rolling step (warmup
        + rolling-steady-state until GT actions are exhausted).

        Each yielded list is all the grad-enabled records for a single
        rolling step — typically [slot_0_record, aux_slot_record] (the 2-grad
        slots). Because those records share the SAME underlying grad graph
        (one live forward produces all slots), the trainer MUST sum their
        losses and call `.backward()` once per yielded list; backward-ing
        each record separately would try to backprop through the same graph
        twice and raise.

        Args:
          gt_latents:   [B, T_ride, C, H, W] full ride latents.
          gt_actions:   [B, T_ride, action_dim] full ride actions.
          prompt_embeds:[B, L, C_txt].
          max_rolling_steps: optional safety cap on the number of rolling
                        steady-state steps (0 = no cap).
        """
        device = gt_latents.device
        dtype = gt_latents.dtype
        batch_size = gt_latents.shape[0]

        # --- Init caches ---
        self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)

        # --- Prime KV cache with clean GT frames ---
        n_primed = self.warmup_prime_kv_cache(
            gt_latents=gt_latents,
            gt_actions=gt_actions,
            prompt_embeds=prompt_embeds,
        )
        npb = self.num_frame_per_block

        # Need at least enough actions to (a) run NW = NS*P warmup passes with
        # sliding indices up to prime_end + (NW - 1 + NS - 1)*npb, and (b)
        # commit slot 0 at transition at prime_end + (NW-1)*npb. If the ride
        # is shorter, exit without training.
        NW = self.num_live_slots * self.passes_per_step
        min_actions_for_warmup = n_primed + (NW + self.num_live_slots - 1) * npb
        if gt_actions.shape[1] < min_actions_for_warmup:
            return

        # --- Warmup: 4 lockstep passes with sliding action indices ---
        clean_live, _warmup_outputs = self.warmup_fill_staircase(
            gt_actions=gt_actions,
            prompt_embeds=prompt_embeds,
            prime_end=n_primed,
            enable_grad_on_final_clean=False,
        )

        # --- Transition to steady-state staircase ---
        # After transition: the commit has been trimmed to kv_committed_max_frames
        # in the KV cache. Two counters for the first steady-state step:
        #   - logical_start_frame = n_primed + npb (contiguous with cache commits).
        #   - action_base_frame_idx = n_primed + NUM_SLOTS*npb (slot 0 of first
        #     steady step carries a_4 in the user's sliding-action semantics).
        (
            state,
            logical_start_frame,
            action_base_frame_idx,
            kv_anchor_chunk,
        ) = self.transition_to_steady_state(
            clean_live=clean_live,
            gt_actions=gt_actions,
            prompt_embeds=prompt_embeds,
            prime_end=n_primed,
        )

        # --- Seed the fake-score commit history with the 3 primed GT chunks ---
        # The fake_score's S_n context is student-rolled (no GT access) once
        # training is underway, but at the very first steady-state step we
        # have only the transition's commit (slot-0 anchor). Seeding with
        # the 3 primed GT chunks eliminates the cold-start (user preference:
        # "we always start everything off with three GT chunks") and matches
        # what the fake_score's left-context attention would have seen right
        # after priming.
        commit_latents: List[torch.Tensor] = []
        commit_actions: List[torch.Tensor] = []
        for b in range(3):
            f0 = b * npb
            f1 = f0 + npb
            commit_latents.append(
                gt_latents[:, f0:f1].detach().to(device=device, dtype=dtype).contiguous()
            )
            commit_actions.append(
                gt_actions[:, f0:f1].detach().to(device=device, dtype=dtype).contiguous()
            )

        # --- Steady-state rolling loop ---
        total_actions = gt_actions.shape[1]
        steps_done = 0
        while True:
            # Slot 0's ride-absolute action slice is [action_base_frame_idx,
            # action_base_frame_idx + npb). Stop when it runs off the end.
            if action_base_frame_idx + npb > total_actions:
                break
            if max_rolling_steps is not None and max_rolling_steps > 0 and steps_done >= max_rolling_steps:
                break

            # Build the fake-context tensors for this rolling step from the
            # 3-entry commit history (chronological, oldest first).
            fake_ctx_chunks = torch.cat(commit_latents, dim=1).contiguous()
            fake_ctx_actions = torch.cat(commit_actions, dim=1).contiguous()

            (
                state,
                step_outputs,
                logical_adv,
                action_adv,
                kv_anchor_chunk,
            ) = self.rolling_step(
                state=state,
                gt_actions=gt_actions,
                prompt_embeds=prompt_embeds,
                logical_start_frame=logical_start_frame,
                action_base_frame_idx=action_base_frame_idx,
                kv_anchor_chunk=kv_anchor_chunk,
                gt_latents=gt_latents,
                fake_context_chunks=fake_ctx_chunks,
                fake_context_actions=fake_ctx_actions,
                grad_slot_indices=None,  # default: all 4 slots (Option 4 DMD)
            )
            if step_outputs:
                yield step_outputs

            # Update commit history: the step just committed slot-0's x0
            # (= next_kv_anchor / kv_anchor_chunk now) with slot-0's decayed
            # action. Push it and drop the oldest.
            slot0_commit_action = self._build_sliding_actions(
                gt_actions, base_frame_idx=action_base_frame_idx,
            )[:, :npb].detach().contiguous()
            commit_latents.append(kv_anchor_chunk.detach().contiguous())
            commit_actions.append(slot0_commit_action)
            if len(commit_latents) > 3:
                commit_latents.pop(0)
                commit_actions.pop(0)

            logical_start_frame += logical_adv
            action_base_frame_idx += action_adv
            steps_done += 1

    # -----------------------------------------------------------------
    # Sequential-commit rollout (eval-only, additive warmup + per-step
    # slot-0 commit). Not used by the trainer — this is the simpler
    # "one chunk appears in the mp4 per rolling step" rollout the
    # user wants for visualisation: prime with GT chunks 0..P-1, then
    # progressively denoise a staircase and commit one fresh chunk per
    # step starting at chunk index P.
    # -----------------------------------------------------------------
    def _chunk_action_value(
        self,
        gt_actions: torch.Tensor,
        chunk_idx: int,
    ) -> torch.Tensor:
        """Pull the per-chunk action value as a ``[B, 1, action_dim]`` tensor.

        Actions are semantically chunk-wise (the underlying motion encoding
        produces one latent per chunk that is broadcast across the chunk's
        ``num_frame_per_block`` frames). We take frame 0 of the chunk as the
        canonical value; broadcasting to the 3 frames happens downstream in
        ``_build_per_slot_actions_chunkwise`` via ``expand``.

        Out-of-range chunk indices clamp to the last available frame.
        """
        npb = self.num_frame_per_block
        f0 = int(chunk_idx) * npb
        total = int(gt_actions.shape[1])
        if f0 >= total:
            f0 = max(0, total - 1)
        return gt_actions[:, f0:f0 + 1, :].contiguous()

    def _build_per_slot_actions_chunkwise(
        self,
        slot0_chunk_action: torch.Tensor,
        num_live_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the ``[B, num_live_slots * npb, action_dim]`` per-frame
        action tensor by broadcasting slot-0's chunk-wise action across all
        live slots with the configured per-slot decay.

        The returned tensor has one unique chunk-wise value per slot (the 3
        frames within each slot share the same value), matching how the
        training data's motion latents arrive at the projection modules.
        """
        B, _, D = slot0_chunk_action.shape
        npb = self.num_frame_per_block
        per_frame = torch.zeros(
            B, num_live_slots * npb, D, device=device, dtype=dtype,
        )
        a = slot0_chunk_action.to(device=device, dtype=dtype)
        for s in range(num_live_slots):
            decay = float(self.action_decay_per_slot[s])
            f0 = s * npb
            f1 = f0 + npb
            per_frame[:, f0:f1, :] = decay * a.expand(-1, npb, -1)
        return per_frame

    def _commit_slot0_clean(
        self,
        slot0_clean: torch.Tensor,
        slot0_chunk_action: torch.Tensor,
        prompt_embeds: torch.Tensor,
        logical_frame: int,
    ) -> None:
        """Push slot-0's clean x0 into the persistent KV cache via a t=0
        forward with cache-update enabled. ``slot0_chunk_action`` is the
        [B, 1, D] chunk action value (broadcast to npb frames here)."""
        device = slot0_clean.device
        dtype = slot0_clean.dtype
        B = slot0_clean.shape[0]
        npb = self.num_frame_per_block
        a_frames = slot0_chunk_action.to(device=device, dtype=dtype).expand(-1, npb, -1).contiguous()
        modulation = self._compute_live_modulation(a_frames, device=device, dtype=dtype)
        action_tokens = self._compute_live_action_tokens(a_frames, device=device, dtype=dtype)
        cond = self._prepare_conditional(prompt_embeds, modulation, action_tokens)
        ts0 = torch.zeros([B, npb], device=device, dtype=torch.int64)
        self._set_skip_cache_update(False)
        self.generator(
            noisy_image_or_video=slot0_clean,
            conditional_dict=cond,
            timestep=ts0,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=logical_frame * self.frame_seq_length,
        )
        self._trim_committed_kv_cache()

    @torch.inference_mode()
    def rollout_ride_sequential(
        self,
        gt_latents: torch.Tensor,
        gt_actions: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        max_commits: Optional[int] = None,
        denoising_ladder: Optional[List[float]] = None,
    ) -> Iterable[dict]:
        """Sequential-commit causal rollout — one fresh chunk per rolling step.

        Layout:
          1. Prime KV cache with ``prime_kv_frames`` GT latents (chunks 0..P-1,
             P = prime_kv_frames / npb, default 3) via t=0 cache-refresh.
          2. Additive warmup: passes 1..NS grow the live window one slot at a
             time. Pass k has k live slots at timesteps
             ``[ladder[NS-k], ..., ladder[NS-1]]``. Slot 0 in warmup is
             always chunk P (the first chunk to be committed). Slots 1..k-1
             carry decayed echoes of slot-0's chunk action (no future GT
             peeking — this mirrors real-time deployment).
          3. Pass NS brings slot 0 to t=0 and commits it → chunk P appears in
             the output. From there steady state runs 1 commit per rolling
             step: slide slots, add fresh noise at the trailing slot,
             denoise once, commit slot 0.

        Yields one dict per commit with keys:
          - ``latent``:      [B, npb, C, H, W] fp32 CPU (the committed clean chunk)
          - ``chunk_idx``:   int (ride chunk index of this commit)
          - ``action``:      [B, action_dim] float CPU (the chunk action
                             that drove this commit; taken from gt_actions)
          - ``slot_timestep``: int (always 0 — the commit is clean)
          - ``phase``:       str ("warmup" for the first commit that
                             finishes the additive fill, "steady" thereafter)
        """
        device = gt_latents.device
        dtype = gt_latents.dtype
        B = gt_latents.shape[0]
        npb = self.num_frame_per_block
        NS = self.num_live_slots
        P = self.passes_per_step
        total_denoise = NS * P            # = 4 (the ODE student's step budget)

        # --- Denoising ladder (descending, length == NS*P) -----------------
        # Matches the semantics of ``eval_causal_AR.set_denoising_steps``:
        # when the caller provides a ladder use it as-is; otherwise fall
        # back to ``linspace(1000, 50, NS*P)`` so the last pass is a
        # polish rung (t=50) rather than half-noise. The eval script passes
        # the ODE student's ``trained`` list when its length matches
        # ``NS*P``; this keeps the student on-distribution for the whole
        # rollout instead of stepping through unvisited rungs.
        if denoising_ladder is None:
            ladder_rungs = [
                float(x) for x in torch.linspace(1000.0, 50.0, steps=total_denoise).tolist()
            ]
        else:
            ladder_rungs = [float(t) for t in denoising_ladder]
            if len(ladder_rungs) != total_denoise:
                raise ValueError(
                    f"denoising_ladder must have length NS*P = {total_denoise}; "
                    f"got {len(ladder_rungs)}"
                )
            # Sort descending so ladder_rungs[i] is the input t at pass-index i.
            ladder_rungs = sorted(ladder_rungs, reverse=True)

        self._initialize_kv_cache(batch_size=B, dtype=dtype, device=device)
        self._initialize_crossattn_cache(batch_size=B, dtype=dtype, device=device)

        n_primed = self.warmup_prime_kv_cache(
            gt_latents=gt_latents,
            gt_actions=gt_actions,
            prompt_embeds=prompt_embeds,
        )
        if n_primed == 0:
            return
        prime_chunks = n_primed // npb
        next_commit_chunk = prime_chunks
        logical_frame = n_primed

        C = int(gt_latents.shape[2])
        H = int(gt_latents.shape[3])
        W = int(gt_latents.shape[4])

        total_ride_chunks = int(gt_actions.shape[1]) // npb

        log.info(
            "rollout_sequential: NS=%d P=%d total_denoise=%d ladder=%s "
            "prime_chunks=%d first_commit_chunk=%d",
            NS, P, total_denoise,
            [round(float(x), 2) for x in ladder_rungs],
            prime_chunks, next_commit_chunk,
        )

        # Live window state: each entry is (latent, ladder_idx) where
        # ``ladder_idx`` is the slot's position in ``ladder_rungs`` (0 =
        # freshest / highest-noise rung). Every forward pass advances every
        # live slot's ``ladder_idx`` by 1; a slot reaches ``total_denoise``
        # only when slot 0 has accumulated all NS*P passes and is ready to
        # commit.
        live_slots: List[Tuple[torch.Tensor, int]] = []

        def _slot0_action_3frames() -> torch.Tensor:
            """Return ``gt_actions[:, commit_chunk*npb : (commit_chunk+1)*npb]``
            — the 3-frame slice for the chunk slot 0 is about to commit.
            Matches ``eval_causal_AR.generate_ar``'s
            ``block_fa = noisy_fa_full[:, frame_lo:frame_hi]``."""
            f0 = int(next_commit_chunk) * npb
            f1 = f0 + npb
            total = int(gt_actions.shape[1])
            if f1 <= total:
                return gt_actions[:, f0:f1, :].contiguous()
            # End-of-ride: clamp to last available frame.
            avail = gt_actions[:, min(f0, total - 1):total, :]
            pad_count = f1 - total
            pad = gt_actions[:, -1:, :].expand(-1, pad_count, -1)
            return torch.cat([avail, pad], dim=1).contiguous()

        def _build_cond_for_width(num_slots_in_window: int) -> dict:
            """Build the prompt + action cond dict for a live window of the
            given width. Slot 0 carries the full 3-frame chunk-action slice
            (decay=1.0); slots 1..num_slots-1 carry ``decay[s] * slot0_action``
            (decayed echoes, matching the user's deployment semantics that
            future ride actions are unavailable at inference time)."""
            slot0_a = _slot0_action_3frames().to(device=device, dtype=dtype)
            assert slot0_a.shape[1] == npb, slot0_a.shape
            D = int(slot0_a.shape[-1])
            per_frame = torch.zeros(
                B, num_slots_in_window * npb, D, device=device, dtype=dtype,
            )
            for s in range(num_slots_in_window):
                decay = float(self.action_decay_per_slot[s])
                f0 = s * npb
                f1 = f0 + npb
                per_frame[:, f0:f1, :] = decay * slot0_a
            modulation = self._compute_live_modulation(
                per_frame, device=device, dtype=dtype,
            )
            action_tokens = self._compute_live_action_tokens(
                per_frame, device=device, dtype=dtype,
            )
            return self._prepare_conditional(prompt_embeds, modulation, action_tokens)

        def _forward_live(denoise_cond: dict) -> torch.Tensor:
            """Run one DiT forward over the current live window with
            skip_cache_update=True. Per-slot input timestep is read from
            each slot's ``ladder_idx``. Returns pred_x0."""
            k = len(live_slots)
            assert k >= 1
            live_cat = torch.cat([x for x, _ in live_slots], dim=1).contiguous()
            timestep = torch.zeros(B, k * npb, device=device, dtype=torch.int64)
            for s, (_, idx) in enumerate(live_slots):
                timestep[:, s * npb:(s + 1) * npb] = int(round(ladder_rungs[idx]))
            self._set_skip_cache_update(True)
            out = self.generator(
                noisy_image_or_video=live_cat,
                conditional_dict=denoise_cond,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=logical_frame * self.frame_seq_length,
            )
            return out[1]

        def _commit_and_cache_refresh(
            slot0_clean: torch.Tensor, commit_cond: dict,
        ) -> None:
            """Cache-refresh forward at t=0 on ``slot0_clean`` with cache
            update ENABLED. This writes clean K/V for slot 0's 3 frames at
            RoPE position ``logical_frame`` into the persistent cache —
            the same pattern eval_causal_AR uses after its denoise loop
            (``self.wrapper(pred_x0, cond, timestep=refresh_t_block=0, ...)``).
            """
            self._set_skip_cache_update(False)
            ts0 = torch.zeros(B, npb, device=device, dtype=torch.int64)
            self.generator(
                noisy_image_or_video=slot0_clean,
                conditional_dict=commit_cond,
                timestep=ts0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=logical_frame * self.frame_seq_length,
            )
            self._trim_committed_kv_cache()

        commits_emitted = 0

        def _run_P_passes(phase_tag: str):
            """Run ``P`` forward passes over the current live window with a
            single reusable ``denoise_cond`` (built once for this virtual
            step). Each pass advances every live slot's ``ladder_idx`` by 1.
            When slot 0's next ``ladder_idx`` reaches ``total_denoise`` the
            slot is clean — commit it via a t=0 cache-refresh forward and
            yield the record."""
            nonlocal logical_frame, next_commit_chunk, commits_emitted, live_slots
            # Build cond ONCE per virtual step (chunk action is fixed for
            # all P passes within the step AND for the commit forward). The
            # prompt_embeds slot is a reference, not recomputed; only the
            # action_modulation / action_tokens get re-projected, and only
            # because the chunk-wise action changes between chunks (ride
            # actions vary sharply — chunks 0..23 of Brighton span
            # [-0.31, +0.99] in z2). The commit cond is just the slot-0
            # slice of the denoise cond (slot 0 carries decay=1.0 * chunk
            # action, which is exactly what the commit forward needs).
            k_start = len(live_slots)
            denoise_cond = _build_cond_for_width(k_start)
            commit_cond = {
                "prompt_embeds": denoise_cond["prompt_embeds"],
                "_action_modulation": denoise_cond["_action_modulation"][:, :npb, :, :].contiguous(),
                "_action_tokens": denoise_cond["_action_tokens"][:, :npb, :].contiguous(),
            }
            for _p in range(P):
                pred_x0 = _forward_live(denoise_cond)
                rebuilt: List[Tuple[torch.Tensor, int]] = []
                commit_now: Optional[torch.Tensor] = None
                for s in range(len(live_slots)):
                    slot_pred = pred_x0[:, s * npb:(s + 1) * npb].detach()
                    _, idx = live_slots[s]
                    next_idx = int(idx) + 1
                    if next_idx >= total_denoise:
                        assert s == 0, (
                            f"unexpected non-slot-0 reaching clean at s={s} "
                            f"pass={_p} ladder_idx={next_idx}"
                        )
                        commit_now = slot_pred
                    else:
                        next_t = float(ladder_rungs[next_idx])
                        rebuilt.append(
                            (self._renoise(slot_pred, int(round(next_t))), next_idx),
                        )
                live_slots = rebuilt
                if commit_now is not None:
                    assert _p == P - 1, (
                        f"commit must fall on the last pass; got p={_p} of P={P}"
                    )
                    _commit_and_cache_refresh(commit_now, commit_cond)
                    yield {
                        "latent": commit_now.detach().to(
                            device="cpu", dtype=torch.float32,
                        ).clone(),
                        "chunk_idx": int(next_commit_chunk),
                        "action": _slot0_action_3frames().detach()[:, 0, :].to(
                            device="cpu", dtype=torch.float32,
                        ).clone(),
                        "slot_timestep": 0,
                        "phase": phase_tag,
                    }
                    commits_emitted += 1
                    logical_frame += npb
                    next_commit_chunk += 1

        # --- Additive warmup: NS virtual steps, P passes each ---
        for vk in range(1, NS + 1):
            new_slot = torch.randn(B, npb, C, H, W, device=device, dtype=dtype)
            live_slots.append((new_slot, 0))   # fresh noise at ladder_idx=0
            yield from _run_P_passes("warmup")

        # --- Steady state: one commit per rolling step, P passes each ---
        while True:
            if max_commits is not None and commits_emitted >= max_commits:
                break
            if next_commit_chunk >= total_ride_chunks:
                break
            new_slot = torch.randn(B, npb, C, H, W, device=device, dtype=dtype)
            live_slots.append((new_slot, 0))
            yield from _run_P_passes("steady")
