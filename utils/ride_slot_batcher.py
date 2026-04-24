"""Per-slot independent ride batcher for Phase 1 rolling-staircase training.

This is the rolling-step analogue of ``model/causal_teacher_streaming.py``'s
``LockstepRideBatcher``. Where the teacher trainer cuts rides into fixed
21-frame windows and advances all slots by one window per training step,
this batcher cuts rides by ROLLING STEPS (not windows) and advances all
slots by a fixed ``rolling_steps_per_iter`` per training iteration.

Each slot holds its own full pipeline state:

  - its own ``kv_cache1`` and ``crossattn_cache`` (so long-horizon KV
    context accumulates end-to-end within a ride, identical to the
    one-ride-per-rank trainer),
  - its own ``LiveWindowState`` (4-slot staircase),
  - its own ``logical_start_frame`` and ``action_base_frame_idx``
    counters,
  - a phase tracker so setup (priming + warmup + transition) runs once
    per ride and rolling runs forever after.

Cross-rank DDP lockstep is guaranteed by ``prepare_for_iter``: before
each training iter, any slot with fewer than ``required_rolling_steps``
remaining is declared exhausted and refilled. After ``prepare_for_iter``,
every slot on every rank has >= ``required_rolling_steps`` steps left,
so every rank will do exactly ``slots_per_rank * required_rolling_steps``
rolling-step forwards per iter. DDP allreduces pair up cleanly.

Memory cost per slot = one full pipeline cache ≈ 2.6 GB bf16 for the
default 30-block / 21-frame buffer / 12-head / 128-headdim CausalWan
config. ``slots_per_rank=4`` → ~11 GB/rank for caches on top of the
model copies; reduce if you're on 40 GB GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# Type aliases for readability.
RideMetadata = Dict[str, Any]
RideTensors = Dict[str, Any]


@dataclass
class SlotRideState:
    """Full per-slot state for one active ride.

    A slot is either:
      * EMPTY: no ride loaded; awaiting ``load_slot``.
      * ROLLING: ride loaded, setup done (priming + warmup + transition),
        in steady-state rolling. ``remaining_rolling_steps`` > 0.
      * EXHAUSTED: ride ran out of actions; awaiting refill.

    The ``phase`` string tracks this.
    """

    # --- Ride identity / metadata ---
    zarr_path: str = ""
    n_latent_frames: int = 0

    # --- Loaded ride tensors (kept on GPU for the ride's lifetime) ---
    gt_latents: Optional[torch.Tensor] = None      # [1, T, C, H, W]
    gt_actions: Optional[torch.Tensor] = None      # [1, T, action_dim]
    prompt_embeds: Optional[torch.Tensor] = None   # [1, L, C_txt]

    # --- Pipeline state (per-slot; swapped into pipeline by index) ---
    kv_cache1: Optional[list] = None
    crossattn_cache: Optional[list] = None
    live: Any = None   # LiveWindowState (optional import loop avoidance)
    logical_start: int = 0
    action_base: int = 0
    prime_end: int = 0
    # Detached latent chunk [1, npb, C, H, W] that was last committed to the
    # KV cache. Threaded through `rolling_step(kv_anchor_chunk=...)` so the
    # batched-S_n DMD teacher can condition S_0 on the ACTUAL committed
    # chunk (not on GT at the same position). For the first steady-state
    # step this is the `initial_kv_anchor` returned by
    # `transition_to_steady_state`.
    kv_anchor_chunk: Optional[torch.Tensor] = None
    # Rolling 3-entry buffer of the LAST THREE committed student chunks
    # (oldest first). Used to build the fake_score's S_n context which is
    # STUDENT-ROLLED — no GT access — matching the test-time distribution.
    # Seeded at setup time with the 3 primed GT chunks (detached, no decay
    # on actions) so there is never a cold-start. Each entry is a dict:
    #   {"latent": [1, npb, C, H, W], "action": [1, npb, action_dim]}
    commit_history: List[Dict[str, torch.Tensor]] = field(default_factory=list)

    # --- Phase ---
    phase: str = "empty"  # "empty" | "rolling" | "exhausted"

    # --- Bookkeeping ---
    rolling_steps_done: int = 0
    ride_id: str = ""


class PerSlotRideBatcher:
    """Multi-slot lockstep-friendly driver for RollingStaircaseTrainingPipeline.

    Usage (per training iter):
        batcher.prepare_for_iter(required_rolling_steps=M)
        for step_idx in range(M):
            for slot_idx in range(B):
                records = batcher.step_slot(slot_idx)
                # ... backward + no_sync dance
        # optimizer.step()

    The batcher assumes the trainer supplies ``ride_loader`` — a callable
    that returns a fresh ride's loaded tensors (latents, actions,
    prompt_embeds, zarr_path, n_latent_frames) on demand. This decouples
    data loading from the batcher's state machine.
    """

    def __init__(
        self,
        pipeline,
        slots_per_rank: int,
        device: torch.device,
        dtype: torch.dtype,
        ride_loader: Callable[[], Optional[RideTensors]],
    ) -> None:
        """Args:
          pipeline:         RollingStaircaseTrainingPipeline. The batcher
                            mutates ``pipeline.kv_cache1`` /
                            ``pipeline.crossattn_cache`` as it swaps
                            slot state in/out.
          slots_per_rank:   number of independent rides running on this rank.
          device, dtype:    placement for fresh cache allocations.
          ride_loader:      zero-arg callable returning one ride's tensors
                            or None if the rank's data supply is exhausted.
                            Must return a dict with keys
                                {"latents", "z_actions", "prompt_embeds",
                                 "zarr_path", "n_latent_frames"}.
                            ``latents`` and ``z_actions`` must have batch
                            dim 1 (shape [1, T, ...]).
        """
        self.pipeline = pipeline
        self.slots_per_rank = int(slots_per_rank)
        self.device = device
        self.dtype = dtype
        self.ride_loader = ride_loader

        self.slots: List[SlotRideState] = [SlotRideState() for _ in range(self.slots_per_rank)]

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------
    def _allocate_fresh_caches(self) -> Tuple[list, list]:
        """Allocate a fresh (kv_cache1, crossattn_cache) pair WITHOUT
        disturbing whichever slot's caches are currently installed in the
        pipeline. We save the current pointers, run the pipeline's in-place
        allocators, steal the result, then restore the saved pointers.
        """
        saved_kv = getattr(self.pipeline, "kv_cache1", None)
        saved_cross = getattr(self.pipeline, "crossattn_cache", None)
        self.pipeline._initialize_kv_cache(batch_size=1, dtype=self.dtype, device=self.device)
        self.pipeline._initialize_crossattn_cache(batch_size=1, dtype=self.dtype, device=self.device)
        new_kv = self.pipeline.kv_cache1
        new_cross = self.pipeline.crossattn_cache
        # Restore — the caller is responsible for installing the fresh
        # caches on whichever slot they belong to.
        self.pipeline.kv_cache1 = saved_kv
        self.pipeline.crossattn_cache = saved_cross
        return new_kv, new_cross

    def _install_slot_caches(self, slot: SlotRideState) -> None:
        """Point the pipeline at this slot's caches. O(1) reference swap."""
        self.pipeline.kv_cache1 = slot.kv_cache1
        self.pipeline.crossattn_cache = slot.crossattn_cache

    # ------------------------------------------------------------------
    # Ride lifecycle
    # ------------------------------------------------------------------
    def _remaining_rolling_steps(self, slot: SlotRideState) -> int:
        """Number of rolling steps this slot can still do before exhausting
        its GT action stream (matches the termination check in
        ``RollingStaircaseTrainingPipeline.rollout_ride``)."""
        if slot.phase != "rolling" or slot.gt_actions is None:
            return 0
        npb = int(self.pipeline.num_frame_per_block)
        remaining_actions = int(slot.gt_actions.shape[1]) - int(slot.action_base)
        return max(0, remaining_actions // npb)

    @torch.no_grad()
    def _run_setup(self, slot: SlotRideState) -> bool:
        """Priming + warmup + transition on a freshly-loaded slot.

        Returns True on success, False if the ride is too short or
        fails any gating check (in which case the slot is left in
        phase='exhausted' so the next ``prepare_for_iter`` will try
        to refill).
        """
        if slot.gt_latents is None or slot.gt_actions is None or slot.prompt_embeds is None:
            slot.phase = "exhausted"
            return False

        self._install_slot_caches(slot)

        # Prime KV cache (9 clean GT frames).
        n_primed = self.pipeline.warmup_prime_kv_cache(
            gt_latents=slot.gt_latents,
            gt_actions=slot.gt_actions,
            prompt_embeds=slot.prompt_embeds,
        )
        if n_primed <= 0:
            slot.phase = "exhausted"
            return False

        npb = int(self.pipeline.num_frame_per_block)
        num_slots = int(self.pipeline.num_live_slots)
        passes_per_step = int(getattr(self.pipeline, "passes_per_step", 1))
        NW = num_slots * passes_per_step
        # Matches rollout_ride's minimum-actions gate: warmup's last pass
        # reads up to prime_end + (NW + num_slots - 1) * npb actions
        # (NW for 4x1 = 4, NS-1 = 3 → 7*npb; for 2x2 NW=4, NS-1=1 → 5*npb).
        min_actions = n_primed + (NW + num_slots - 1) * npb
        if slot.gt_actions.shape[1] < min_actions:
            slot.phase = "exhausted"
            return False

        # Warmup: lockstep-denoise 4 pure-noise blocks all the way to clean.
        clean_live, _ = self.pipeline.warmup_fill_staircase(
            gt_actions=slot.gt_actions,
            prompt_embeds=slot.prompt_embeds,
            prime_end=n_primed,
            enable_grad_on_final_clean=False,
        )
        # Transition: commit slot 0 + trim + slide + re-noise.
        (
            state,
            logical_start,
            action_base,
            initial_kv_anchor,
        ) = self.pipeline.transition_to_steady_state(
            clean_live=clean_live,
            gt_actions=slot.gt_actions,
            prompt_embeds=slot.prompt_embeds,
            prime_end=n_primed,
        )

        slot.live = state
        slot.logical_start = int(logical_start)
        slot.action_base = int(action_base)
        slot.prime_end = int(n_primed)
        slot.kv_anchor_chunk = initial_kv_anchor

        # Seed the fake-score commit-history buffer with the 3 primed GT
        # chunks (first 9 frames of the ride). These frames were processed
        # through the student at t=0 as part of priming, so they are the
        # natural "previous commits" the fake_score's left context
        # should see at the very first steady-state step. Actions here
        # are the GT (no per-slot decay) actions that drove priming.
        slot.commit_history = []
        for b in range(3):
            f0 = b * npb
            f1 = f0 + npb
            slot.commit_history.append(
                {
                    "latent": slot.gt_latents[:, f0:f1].detach().contiguous(),
                    "action": slot.gt_actions[:, f0:f1].detach().contiguous(),
                }
            )

        slot.phase = "rolling"
        slot.rolling_steps_done = 0
        return True

    def load_slot(self, slot_idx: int, ride: RideTensors) -> bool:
        """Load a fresh ride into slot_idx and run setup.

        ``ride`` is a dict matching the schema returned by the trainer's
        ride loader (see ``ride_loader`` in ``__init__``). Returns True
        if the slot is ready to roll, False if setup failed (ride too
        short etc.); in the failure case the slot is marked exhausted
        so the next ``prepare_for_iter`` will try another ride.
        """
        slot = self.slots[slot_idx]
        # Drop previous ride tensors to free memory before allocating new.
        slot.gt_latents = None
        slot.gt_actions = None
        slot.prompt_embeds = None
        slot.live = None
        slot.kv_anchor_chunk = None
        slot.commit_history = []

        # Fresh caches for this slot (previous slot caches, if any, would
        # be stale — wipe by reallocating instead of zeroing the old
        # buffers, which is simpler and lets GC reclaim old tensors).
        kv_cache1, crossattn_cache = self._allocate_fresh_caches()
        slot.kv_cache1 = kv_cache1
        slot.crossattn_cache = crossattn_cache

        # Install new tensors.
        slot.zarr_path = str(ride.get("zarr_path", ""))
        slot.n_latent_frames = int(ride.get("n_latent_frames", 0))
        slot.ride_id = slot.zarr_path
        slot.gt_latents = ride["latents"].to(device=self.device, dtype=self.dtype)
        slot.gt_actions = ride["z_actions"].to(device=self.device, dtype=self.dtype)
        slot.prompt_embeds = ride["prompt_embeds"].to(device=self.device, dtype=self.dtype)
        slot.phase = "empty"  # intermediate — setup will promote to 'rolling' or 'exhausted'

        ok = self._run_setup(slot)
        return ok

    def refill_one(self, slot_idx: int) -> bool:
        """Try to refill one exhausted slot by pulling from ``ride_loader``.

        Returns True if refilled successfully, False if the loader is
        exhausted (trainer should treat this as end-of-epoch).
        """
        # Try up to a few rides in case a loaded ride is too short etc.
        # We bail after 8 attempts to avoid infinite loops on a bad dataset.
        for _ in range(8):
            ride = self.ride_loader()
            if ride is None:
                return False
            if self.load_slot(slot_idx, ride):
                return True
            # load_slot failed (ride too short); try another.
        return False

    # ------------------------------------------------------------------
    # Lockstep iter preparation
    # ------------------------------------------------------------------
    def prepare_for_iter(self, *, required_rolling_steps: int) -> int:
        """Ensure every slot has >= required_rolling_steps rolling steps
        remaining. Any slot that is empty, exhausted, or has too few
        steps left is refilled (wasting <= required_rolling_steps - 1
        steps of that ride's tail — acceptable for typical
        rolling_steps_per_iter << ride_length).

        Returns the number of slots that ARE READY (i.e. successfully
        refilled or already had enough steps). If this is less than
        ``slots_per_rank`` the trainer should treat the epoch as
        complete and synchronize epoch-end across ranks.
        """
        ready = 0
        for slot_idx in range(self.slots_per_rank):
            slot = self.slots[slot_idx]
            rem = self._remaining_rolling_steps(slot)
            if rem >= int(required_rolling_steps):
                ready += 1
                continue
            # Refill. If this slot was still "rolling" but short of the
            # required count, we eat its tail. This is the cost of
            # strict lockstep.
            ok = self.refill_one(slot_idx)
            if ok:
                ready += 1
            # else: slot is marked exhausted; trainer decides what to do.
        return ready

    # ------------------------------------------------------------------
    # Per-slot step
    # ------------------------------------------------------------------
    def step_slot(self, slot_idx: int) -> Optional[list]:
        """Run one rolling step on slot_idx. Returns the step's grad
        records (list[RollingStepOutput]) or None if the slot is not
        ready (exhausted, empty, or out of actions).

        After this call, the slot's cache + live window + counters have
        all advanced. The persistent KV cache is trimmed to
        ``kv_committed_max_frames`` by the pipeline.

        The returned records' tensors belong to the autograd graph of
        THIS rolling step only — the caller should backward them
        (under no_sync or not, at the caller's discretion) before
        calling ``step_slot`` for any other slot, because the pipeline's
        kv_cache1 reference has been reassigned to this slot's buffer
        and the NEXT ``step_slot`` will reassign it again.

        In practice this ordering is fine: autograd only needs the
        tensors inside the records (and their closures over ``temp_k``
        clones), not the persistent cache. See the MAINTAINER NOTE on
        ``_trim_committed_kv_cache`` for the full invariant.
        """
        slot = self.slots[slot_idx]
        if slot.phase != "rolling" or slot.live is None:
            return None
        if self._remaining_rolling_steps(slot) <= 0:
            slot.phase = "exhausted"
            return None

        self._install_slot_caches(slot)

        # Compose the fake-score's S_n left context: last 3 committed
        # student chunks (oldest first) with their paired commanded
        # actions. The pipeline validates shapes (must be exactly 3*npb
        # frames each) — we rely on setup having seeded the buffer with
        # 3 primed GT chunks and on steady-state rolling maintaining the
        # invariant by pushing each commit and popping the oldest.
        if len(slot.commit_history) != 3:
            raise RuntimeError(
                f"Slot {slot.zarr_path}: commit_history has "
                f"{len(slot.commit_history)} entries, expected 3. The "
                f"batcher's setup must seed it with 3 GT chunks."
            )
        fake_ctx_chunks = torch.cat(
            [e["latent"] for e in slot.commit_history], dim=1
        ).contiguous()
        fake_ctx_actions = torch.cat(
            [e["action"] for e in slot.commit_history], dim=1
        ).contiguous()

        (
            state,
            records,
            logical_adv,
            action_adv,
            next_kv_anchor,
        ) = self.pipeline.rolling_step(
            state=slot.live,
            gt_actions=slot.gt_actions,
            prompt_embeds=slot.prompt_embeds,
            logical_start_frame=slot.logical_start,
            action_base_frame_idx=slot.action_base,
            kv_anchor_chunk=slot.kv_anchor_chunk,
            gt_latents=slot.gt_latents,
            fake_context_chunks=fake_ctx_chunks,
            fake_context_actions=fake_ctx_actions,
            grad_slot_indices=None,  # default: all 4 slots (Option 4 DMD)
        )

        # Update commit history: the step just committed slot-0 x0 with
        # decay[0] * a_{action_base} — push it, pop the oldest.
        npb = int(self.pipeline.num_frame_per_block)
        decay0 = float(self.pipeline.action_decay_per_slot[0])
        a_base = int(slot.action_base)
        # action_base WAS slot-0's ride frame index during the forward
        # just completed. slice [a_base : a_base + npb] with end-of-ride
        # padding (rollout_ride stops before this triggers but guard anyway).
        a_end = a_base + npb
        total_a = int(slot.gt_actions.shape[1])
        if a_end <= total_a:
            a_slot0 = slot.gt_actions[:, a_base:a_end]
        else:
            # Pad with last available action (defensive; see pipeline).
            avail = slot.gt_actions[:, a_base:total_a]
            pad = slot.gt_actions[:, -1:].expand(-1, a_end - total_a, -1)
            a_slot0 = torch.cat([avail, pad], dim=1)
        a_slot0 = (a_slot0.detach() * decay0).contiguous()
        slot.commit_history.append(
            {"latent": next_kv_anchor.detach().contiguous(), "action": a_slot0}
        )
        if len(slot.commit_history) > 3:
            slot.commit_history.pop(0)

        slot.live = state
        slot.logical_start += int(logical_adv)
        slot.action_base += int(action_adv)
        slot.kv_anchor_chunk = next_kv_anchor
        slot.rolling_steps_done += 1

        # Check if this step exhausts the ride (for bookkeeping — next
        # prepare_for_iter will refill if so).
        if self._remaining_rolling_steps(slot) <= 0:
            slot.phase = "exhausted"

        return records

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def exhausted_slot_indices(self) -> List[int]:
        """Slots that are empty or exhausted — need refill before next iter."""
        return [
            i for i, s in enumerate(self.slots)
            if s.phase in ("empty", "exhausted")
        ]

    def ready_slot_indices(self) -> List[int]:
        """Slots currently in 'rolling' phase with at least one step left."""
        return [
            i for i, s in enumerate(self.slots)
            if s.phase == "rolling" and self._remaining_rolling_steps(s) > 0
        ]

    def summary(self) -> str:
        """One-line human-readable state dump."""
        parts = []
        for i, s in enumerate(self.slots):
            rem = self._remaining_rolling_steps(s)
            ride = s.zarr_path.split("/")[-1] if s.zarr_path else "?"
            parts.append(f"s{i}[{s.phase},done={s.rolling_steps_done},rem={rem},ride={ride}]")
        return " | ".join(parts)
