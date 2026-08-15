"""KV-cache AR rollout ODE stage — the student trained the way 14e is INFERENCED.

WHY THIS EXISTS
---------------
The 14e teacher produced every LMDB target with ``utils/causal_chain_rollout.
stream_causal_chain``: a real KV-cache rollout. It writes 3 real seed chunks into
the cache at t=0, then generates 6 chunks of 3 frames each — fresh noise, the full
rung ladder, then a clean-prediction "commit" forward that writes the block's K/V
into the cache before ``current_start`` advances. ``clean_x`` does not appear
anywhere in that file: the teacher's ONLY conditioning on its own past is the
cache.

The previous ODE stage trained the student a completely different way: one target
chunk, ``clean_x`` teacher forcing over a fixed 21-frame window, and gradient on
the last 3 frames only. So the student was trained through a mechanism the teacher
never used to make the targets, and that the student itself never uses at serve
(``utils/eval_causal_AR.py`` is cache-driven). This module removes that mismatch:
the student is rolled out through the SAME cache path, at its own 4 rungs instead
of the teacher's 20, and EVERY generated frame is supervised.

WHAT IS MIRRORED FROM THE TEACHER (do not "simplify" these)
-----------------------------------------------------------
* ``local_attn_size = local_attn_chunks * nfb`` (21) and ``max_attention_size``,
  set on the base module AND every submodule, snapshotted and restored.
* ``cached_rope_action_aware = has_tokens`` — without it the interleaved per-frame
  action tokens are roped as spatial ones (~1 token/frame column shear).
* Cache sized ``LOCAL_ATTN_F + 2*BLOCK_F`` frames, ``cache_start=0``.
* Conditioning dict carries ONLY ``_action_modulation`` / ``_action_tokens`` — no
  ``*_clean`` streams, because there is no ``clean_x`` in this path.
* Seed chunks written one at a time, in order, at t=0.
* Per chunk: commit forward at t=0 at the SAME ``current_start``, then advance.

WHAT DIFFERS (deliberately)
---------------------------
* 4 student rungs with the restart sampler (predict x0 -> re-noise to the next
  rung with FRESH noise), exactly as ``eval_causal_AR`` serves — not the teacher's
  20-step ``scheduler.step`` integration.
* Gradients are ON. The cache is DETACHED after every forward, so back-prop never
  spans chunks (truncated BPTT): each chunk learns to reproduce the teacher's
  chunk GIVEN its context, and memory stays bounded.
* ``commit_mode``:
    - "teacher"  (default) commit the teacher's committed chunk. Context is then
      exactly the context the teacher had, so every stored target is precisely
      valid. Fixes the mechanism mismatch with zero drift risk.
    - "student"  commit the student's own prediction: true on-policy self-forcing,
      which is the only thing that fixes exposure bias — but the context then
      diverges from the teacher's and the later targets are only approximate.
    - "schedule" commit the student's own with probability ``p``, else teacher's.
"""
from typing import Any, Dict, List, Optional

import contextlib
import os
import torch


@contextlib.contextmanager
def _ckpt_debug():
    """AF_CKPT_DEBUG=1 -> op-level trace on a CheckpointError.

    The default CheckpointError only prints tensor SHAPES, which is why this
    bug has been misdiagnosed twice: a shape list that shifts by one slot is
    consistent with several different causes and identifies none of them.
    torch's debug mode logs the actual op stack for the forward and the
    recompute so the diverging operation can be READ rather than inferred.
    Off by default -- it is expensive and stores both traces.
    """
    if os.environ.get("AF_CKPT_DEBUG") != "1":
        yield
        return
    try:
        from torch.utils.checkpoint import set_checkpoint_debug_enabled
    except ImportError:        # older torch: no debug hook, run unchanged
        yield
        return
    with set_checkpoint_debug_enabled(True):
        yield


def _detach_cache(kv_cache: List[Dict[str, torch.Tensor]]) -> None:
    """Cut the graph at the cache boundary (truncated BPTT).

    The cache is written IN PLACE by every forward, so without this the loss of
    chunk 5 would back-propagate through all 24 preceding forwards.
    """
    for e in kv_cache:
        e["k"] = e["k"].detach()
        e["v"] = e["v"].detach()


@torch.no_grad()
def _alloc_caches(base, *, kv_frames: int, frame_seq: int, dtype, device):
    num_blocks = len(base.blocks)
    blk0 = base.blocks[0]
    num_heads = int(getattr(blk0.self_attn, "num_heads", getattr(base, "num_heads", 16)))
    head_dim = int(getattr(blk0.self_attn, "head_dim", base.dim // num_heads))
    text_len = int(getattr(base, "text_len", 512))
    kv_size = kv_frames * frame_seq
    kv_cache, crossattn_cache = [], []
    for _ in range(num_blocks):
        kv_cache.append({
            "k": torch.zeros([1, kv_size, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([1, kv_size, num_heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        })
        crossattn_cache.append({
            "k": torch.zeros([1, text_len, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([1, text_len, num_heads, head_dim], dtype=dtype, device=device),
            "is_init": False,
        })
    return kv_cache, crossattn_cache


def rollout_ode_loss(
    wrapper,
    action_projection,
    action_token_projection,
    *,
    prompt_embeds: torch.Tensor,      # [1, L, D]
    seed_lat: torch.Tensor,           # [1, seed_f, C, H, W]  REAL context
    z_actions: torch.Tensor,          # [1, tot_f, 2]
    committed: torch.Tensor,          # [1, n_chunks, nfb, C, H, W]  teacher targets
    denoising_step_list: torch.Tensor,  # student rungs, descending, e.g. [1000,625,357,208]
    scheduler,                        # needs .add_noise(x, noise, t)
    dtype,
    device,
    nfb: int = 3,
    local_attn_chunks: int = 7,
    seed_base: int = 1234,
    wrapper_call=None,
    commit_mode: str = "teacher",
    commit_p: float = 0.0,
    loss_fn=None,                     # (pred, target) -> scalar; default MSE
    edist_fn=None,                    # (pred, target) -> scalar | None, set-level
    edist_weight: float = 0.0,
    gexcl_fn=None,                    # (pred, target) -> scalar, contraction barrier
    gexcl_weight: float = 0.0,
    ddp_module=None,                  # DDP wrapper: no_sync on all but the last backward
    tail_loss_fn=None,                # () -> scalar|None, folded into the LAST backward
    vrfm=None,                        # VRFMLatent: conditional prior + posterior
    z_modulation=None,                # ZModulation: z -> AdaLN stream
    vrfm_beta: float = 0.0,           # KL(q||p) weight
    sync_last: bool = True,           # False for every call but the step's LAST
    loss_scale: float = 1.0,          # action/variance weight (one step lagged)
) -> Dict[str, Any]:
    """Roll the student through the teacher's cache path; supervise every chunk.

    Returns dict with ``loss`` (mean over chunks x rungs), ``n_terms``, and
    ``per_chunk`` (detached per-chunk final-rung error, for the curriculum).
    """
    if loss_fn is None:
        def loss_fn(p, t):
            return torch.nn.functional.mse_loss(p.float(), t.float())

    # `wrapper` may be a DDP wrapper: nn.Module.__getattr__ does not proxy
    # arbitrary names, so wrapper.model / .seq_len / .adjust_seq_len_* would
    # raise. Call THROUGH the DDP object (so its reducer arms) but mutate
    # attributes on the bare module. The reference is explicit that it wants
    # the DDP-unwrapped wrapper (causal_chain_rollout.py:29).
    call = wrapper_call if wrapper_call is not None else wrapper
    wrapper = getattr(wrapper, "module", wrapper)

    has_tokens = action_token_projection is not None
    has_adaln = action_projection is not None
    apf = 1 if has_tokens else 0

    base = wrapper.model
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    # Parameter coverage on this path DEPENDS on gradient checkpointing: the
    # checkpointed branch omits crossattn_cache, so cross-attn K/V are
    # recomputed each forward and receive gradients. With it OFF, the no_grad
    # seed forward primes the cross-attn cache and every block's cross_attn
    # k/v/norm_k plus text_embedding become permanently unused -> DDP
    # (find_unused_parameters=False) aborts on the second step.
    if not bool(getattr(base, "gradient_checkpointing", False)):
        raise RuntimeError(
            "ode_rollout requires gradient_checkpointing=true: without it the "
            "cross-attention K/V are cached by the no_grad seed forward and "
            "their parameters never receive gradients, which aborts DDP.")
    FRAME_SEQ = 1560 + int(getattr(base, "action_tokens_per_frame", 0))
    LOCAL_ATTN_F = local_attn_chunks * nfb
    BLOCK_F = nfb
    target_max = LOCAL_ATTN_F * FRAME_SEQ

    # --- snapshot every mutation, restore in finally (teacher parity) ---
    SENT = "__keep__"
    saved_base_las = getattr(base, "local_attn_size", SENT)
    saved_base_mas = getattr(base, "max_attention_size", SENT)
    saved_mods = []
    saved_seq_len = getattr(wrapper, "seq_len", None)
    base.local_attn_size = LOCAL_ATTN_F
    if hasattr(base, "max_attention_size"):
        base.max_attention_size = target_max
    for _n, m in base.named_modules():
        if m is base:
            continue          # already snapshotted above; including it here
                              # would restore the MUTATED values (the loop
                              # restore runs after the base restore)
        saved_mods.append((m,
                           getattr(m, "local_attn_size", SENT),
                           getattr(m, "max_attention_size", SENT),
                           getattr(m, "cached_rope_action_aware", SENT)))
        if hasattr(m, "local_attn_size"):
            m.local_attn_size = LOCAL_ATTN_F
        if hasattr(m, "max_attention_size"):
            m.max_attention_size = target_max
        if hasattr(m, "action_tokens_per_frame"):
            m.cached_rope_action_aware = has_tokens

    try:
        seed_f_pre = int(seed_lat.shape[1])
        n_chunks_pre = int(committed.shape[1])
        # Must exceed the FINAL pointer (seed + all generated) by one block, or
        # the recompute evicts where the forward did not. Span is unchanged.
        kv_frames = max(LOCAL_ATTN_F + 2 * BLOCK_F,
                        seed_f_pre + n_chunks_pre * nfb + BLOCK_F)
        kv_cache, crossattn_cache = _alloc_caches(
            base, kv_frames=kv_frames, frame_seq=FRAME_SEQ,
            dtype=dtype, device=device)

        def cond_for(z_block, nframes, z_lat=None):
            # NOTE: no *_clean streams — this path has no clean_x, exactly like
            # the teacher's generation.
            cond = {"prompt_embeds": prompt_embeds}
            if has_adaln:
                cond["_action_modulation"] = action_projection(z_block, num_frames=nframes)
            if has_tokens:
                cond["_action_tokens"] = action_token_projection(z_block)
            if z_lat is not None:
                # v(x_t, t) -> v(x_t, t, z). `_action_modulation` is ADDED to
                # time_projection's output inside action_model_patch.py, i.e.
                # exactly the paper's "z added to the time embedding before
                # computing shift and offset". Because that hook is a plain
                # sum, adding our z term into the same tensor is identical to
                # a second additive stream -- and touches no DiT code.
                _zm = z_modulation(z_lat, num_frames=nframes)
                if "_action_modulation" in cond:
                    cond["_action_modulation"] = cond["_action_modulation"] + _zm
                else:
                    cond["_action_modulation"] = _zm
            return cond

        def fwd(lat, cond, ts, cur_start):
            # cache_enabled=False -- THE CheckpointError fix, proven by the
            # AF_CKPT_DEBUG=1 op trace (run dbg1, 2026-08-13): the recompute
            # contained `aten._to_copy(bf16)` weight+bias casts that the
            # original forward did not. Mechanism: the trainer wraps the whole
            # rollout in ONE outer autocast region, so the NO-GRAD seed/commit
            # forwards populate autocast's weight-cast cache; the grad-enabled
            # rung forward then CACHE-HITS (no cast op recorded, cast has no
            # grad_fn), while the checkpoint recompute at backward time re-runs
            # the cast fresh (op recorded, grad_fn attached) -> one extra saved
            # activation -> the position-5 metadata shift. Worse, the cache-hit
            # cast severed the weight-gradient path entirely, so even when
            # shapes agreed the DiT weights trained at zero gradient.
            # Disabling the cast cache makes forward and recompute run the
            # SAME cast ops and restores grad flow to the fp32 master weights.
            with torch.amp.autocast(device_type="cuda", dtype=dtype,
                                    cache_enabled=False):
                out = call(lat, cond, ts,
                           kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                           current_start=cur_start, cache_start=0)
            return out

        seed_f = int(seed_lat.shape[1])
        assert seed_f % nfb == 0, (seed_lat.shape, nfb)
        n_chunks = int(committed.shape[1])
        # LATENT HANG: n_chunks is data-derived and drives the number of
        # collectives (edist gathers). A single short chain on ONE rank would
        # issue fewer gathers while every other rank blocks in all_gather until
        # the NCCL timeout. Assert global uniformity BEFORE any collective.
        import torch.distributed as _d
        if _d.is_available() and _d.is_initialized() and _d.get_world_size() > 1:
            _nc = torch.tensor([n_chunks, -n_chunks], device=device)
            _d.all_reduce(_nc, op=_d.ReduceOp.MAX)
            if int(_nc[0]) != -int(_nc[1]):
                raise RuntimeError(
                    f"n_chunks differs across ranks (max {int(_nc[0])}, min "
                    f"{-int(_nc[1])}): the rollout would deadlock in all_gather.")
        # Eviction MUST never fire: gradient checkpointing re-reads the cache at
        # recompute time, so a rolled cache would silently produce wrong
        # gradients with no error. kv_frames = max(27, seed+gen+3) = 30 today
        # (one full block of margin past the final pointer at 27f) -- assert
        # rather than rely on that.
        assert seed_f + n_chunks * nfb + BLOCK_F <= kv_frames, (
            f"cache {kv_frames}f cannot hold the final pointer "
            f"{seed_f + n_chunks * nfb}f + one block; eviction would fire on "
            "the gradient-checkpoint RECOMPUTE but not on the forward, which "
            "raises CheckpointError (observed: job 5978752)")
        # BUG 6: the reference asserts the action stream covers the whole span.
        assert int(z_actions.shape[1]) >= seed_f + n_chunks * nfb, (
            z_actions.shape, seed_f + n_chunks * nfb)

        # --- warm-start: write the REAL seed into the cache, one chunk at a
        # time, in order, at t=0 (teacher: causal_chain_rollout.py:127-131) ---
        wrapper.adjust_seq_len_for_action_tokens(num_frames=nfb, action_per_frame=apf)
        with torch.no_grad():
            sl = seed_lat.to(dtype)
            for s0 in range(0, seed_f, nfb):
                fwd(sl[:, s0:s0 + nfb],
                    cond_for(z_actions[:, s0:s0 + nfb], nfb),
                    torch.zeros([1, nfb], device=device, dtype=torch.float32),
                    s0 * FRAME_SEQ)
        _detach_cache(kv_cache)

        wrapper.adjust_seq_len_for_action_tokens(num_frames=BLOCK_F, action_per_frame=apf)
        C, H, W = seed_lat.shape[2], seed_lat.shape[3], seed_lat.shape[4]
        rungs = [float(t) for t in denoising_step_list.tolist()]

        total = None
        n_terms = 0
        ed_total = None
        n_ed = 0
        gx_total = None
        n_gx = 0
        per_chunk: List[float] = []
        chain: List[torch.Tensor] = []
        cur_frames = seed_f

        # --- per-rung backward bookkeeping ---------------------------------
        # n_chunks is asserted globally equal across ranks upstream, and every
        # rank runs the same rung list, so n_bwd_total is a GLOBAL constant --
        # required, because exactly one backward per rank may carry the DDP
        # all-reduce and every rank must agree on which one.
        kl_z_total = None
        n_kl_z = 0
        vrfm_stats = {}
        n_bwd_total = int(n_chunks) * int(len(rungs))
        bwd_done = [0]

        @contextlib.contextmanager
        def _nosync():
            # Suppress DDP's gradient all-reduce; grads accumulate locally and
            # the single final backward syncs the lot.
            if ddp_module is not None and hasattr(ddp_module, "no_sync"):
                with ddp_module.no_sync():
                    yield
            else:
                yield

        for c in range(n_chunks):
            F0 = cur_frames
            cur_start = F0 * FRAME_SEQ
            z_block = z_actions[:, F0:F0 + BLOCK_F]
            # NOTE: cond is rebuilt PER RUNG inside the loop below. It carries
            # live action_projection/action_token_projection graph, and each
            # rung's backward frees it -- reusing one cond across rungs raises
            # "backward through the graph a second time" at chunk 0 rung 1.
            # Rebuilding also gives those projections their correct
            # 1/n_bwd_total-weighted gradient, so no rescale is needed.
            tgt = committed[:, c].to(device)

            # Per-chunk LOCAL generator seeded exactly as the teacher did
            # (causal_chain_rollout.py:147) so chunk c starts from the SAME
            # noise the teacher used, without perturbing the training RNG.
            g = torch.Generator(device=device).manual_seed(seed_base + c)
            # Separate stream for z: drawing it from `g` would advance the same
            # generator that produces the restart noise and the commit coin, so
            # a VRFM run would no longer share initial noise with the teacher
            # or with the non-VRFM arms it is being compared against.
            gz = torch.Generator(device=device).manual_seed(seed_base + c + 977)
            lat = torch.randn([1, BLOCK_F, C, H, W], dtype=torch.float32,
                              device=device, generator=g)
            # x_0 for the VRFM encoders: the chunk's INITIAL pure noise,
            # available identically at training and inference.
            x0_lat = lat

            pred_x0 = None
            for i, t in enumerate(rungs):
                # DDP's no_sync() must span the FORWARD as well as the
                # backward: the reducer is armed inside DDP.forward, so
                # wrapping only the backward leaves EVERY rung all-reducing
                # (torch's own docstring says so). Only the step's very last
                # backward may sync.
                _will_sync = (bwd_done[0] + 1 >= n_bwd_total) and sync_last
                _ctx = contextlib.nullcontext() if _will_sync else _nosync()
                # _ckpt_debug spans BOTH the forward and the backward: torch
                # records the forward's op trace when the flag is live, so
                # enabling it only around .backward() yields no comparison.
                with _ctx, _ckpt_debug():
                    ts = torch.full([1, BLOCK_F], t, device=device, dtype=torch.float32)
                    _kl_z = None
                    _cond = cond_for(z_block, BLOCK_F)
                    if vrfm is not None and z_modulation is not None:
                        # TRAINING: z ~ q(.|x0, x1, xt, t, a) -- x1 is the
                        # teacher's clean chunk, which inference does not have,
                        # so it enters q ONLY. KL(q||p) pulls the conditional
                        # prior onto the posterior, and at inference the probe
                        # samples z ~ p(.|x0, xt, t, a).
                        # fp32 explicitly: the surrounding autocast would run
                        # these encoders in bf16 at TRAIN while eval runs them
                        # in fp32, giving different mu/logvar for identical
                        # inputs -- a train/serve mismatch in the one module
                        # whose whole job is to agree across the two.
                        with torch.amp.autocast("cuda", enabled=False):
                            _z, _kl_z, _zs = vrfm(
                                x0=x0_lat.float(), xt=lat.float(),
                                t=ts.reshape(-1)[:1],
                                action=z_block.reshape(1, -1)[:, :vrfm.action_dim],
                                x1=tgt.float(), generator=gz,
                                stats=(c == 0 and i == 0))
                        if _zs:
                            vrfm_stats.update(_zs)
                        _cond = cond_for(z_block, BLOCK_F, z_lat=_z)
                    out = fwd(lat.to(dtype), _cond, ts, cur_start)
                    # ---- DETACH BEFORE THE BACKWARD -----------------------
                    # NOT a fix for the CheckpointError: tried in job 6004035
                    # and the failure was byte-identical (same position 5,
                    # same shapes), so the cache's requires_grad state is NOT
                    # what diverges. Kept only because it matches the stated
                    # truncated-BPTT intent and costs nothing. The real cause
                    # is still open -- run with AF_CKPT_DEBUG=1 for the op
                    # trace rather than inferring from shapes again.
                    # The block does NOT mutate the cache in place: it does
                    # `temp_k = kv_cache["k"].clone()` and attends over the
                    # clone, then `_apply_cache_updates` writes the live
                    # `new_k` into the persistent cache AFTER every block has
                    # run (causal_model.py:1455). That write makes
                    # kv_cache["k"] require grad. Gradient checkpointing then
                    # RECOMPUTES each block during backward, re-runs the same
                    # `.clone()`, and now reads a requires_grad=True cache --
                    # so the recompute's attention saves one EXTRA tensor and
                    # the saved/recomputed lists shift by one slot (observed:
                    # "tensor at position 5: saved [1536,1536] vs recomputed
                    # [4683,1536]", jobs 5989253 and 5998622).
                    #
                    # Backwarding per rung cannot fix this: the mutation
                    # happens inside the forward, before any backward runs.
                    # Detaching HERE restores the exact requires_grad state the
                    # forward read. Values are unchanged (detach shares
                    # storage) and gradients are unaffected: this rung's live
                    # k/v path runs through `temp_k`, not through the
                    # persistent cache, which is truncated-BPTT context only.
                    _detach_cache(kv_cache)
                    if not (isinstance(out, tuple) and len(out) >= 2):
                        raise RuntimeError(
                            "rollout ODE expects the wrapper to return "
                            "(flow_pred, pred_x0, ...); got "
                            f"{type(out).__name__} len="
                            f"{len(out) if isinstance(out, tuple) else 'N/A'}")
                    pred_x0 = out[1]
                    # Explicitly OUTSIDE autocast: guarantees the difference, the
                    # square and the reduction are all fp32 regardless of the
                    # surrounding autocast region. NOTE: pred_x0 itself comes out of
                    # the forward in bf16 -- recovering that would mean running the
                    # DiT in fp32, which is a different (much costlier) decision.
                    with torch.amp.autocast("cuda", enabled=False):
                        term = loss_fn(pred_x0.float(), tgt.float())
                    n_terms += 1
                    # ---- PER-RUNG BACKWARD (the CheckpointError fix) ----------
                    # Gradient checkpointing re-runs each block at BACKWARD time,
                    # and the block closes over the MUTABLE kv_cache -- so a
                    # deferred backward recomputes against whatever the cache
                    # holds THEN, not what this forward wrote. Every rung writes
                    # the SAME positions (cur_start..+BLOCK_F) and the commit
                    # forward overwrites them again, so one backward at the end
                    # recomputes at the wrong extent (the [1536,1536] vs
                    # [4683,1536] mismatch) and, when shapes happen to agree, on
                    # the wrong VALUES -- silent and worse. Backward each rung
                    # NOW, while the cache still holds what this forward wrote.
                    bwd_loss = loss_scale * (term / max(n_bwd_total, 1))
                    if _kl_z is not None and vrfm_beta > 0.0:
                        bwd_loss = bwd_loss + vrfm_beta * (_kl_z / max(n_bwd_total, 1))
                        kl_z_total = (_kl_z.detach() if kl_z_total is None
                                      else kl_z_total + _kl_z.detach())
                        n_kl_z += 1
                    # Fold the chunk-level terms into the FINAL rung's backward:
                    # both read `pred_x0`, whose graph that backward frees, so
                    # backwarding them separately afterwards raises "backward
                    # through the graph a second time" -- and would place a
                    # backward AFTER the DDP-syncing one, breaking DDP's
                    # one-reduction-per-iteration contract.
                    # REPULSOR RUNG GATING (user directive 2026-08-14): fire at
                    # EVERY rung EXCEPT the first — final-rung-only is "too
                    # little too late"; rung 0 is excluded because its pred_x0
                    # comes from near-pure noise and sits closest to the mu=0
                    # attractor, where the 1/d^2 force would be outsized.
                    # Dose: each firing is divided by n_chunks*(n_rungs-1) so
                    # the per-step total stays `gexcl_weight` once (the same
                    # calibrated dose the final-rung-only version applied),
                    # spread across the ladder where it can still steer the
                    # trajectory.
                    if i >= 1 and gexcl_fn is not None and gexcl_weight > 0.0:
                        _gx = gexcl_fn(pred_x0, tgt)
                        if _gx is not None:
                            n_gx += 1
                            bwd_loss = bwd_loss + loss_scale * gexcl_weight * (
                                _gx / max(n_chunks * max(len(rungs) - 1, 1), 1))
                            gx_total = (_gx.detach() if gx_total is None
                                        else gx_total + _gx.detach())
                    if i == len(rungs) - 1:
                        if edist_fn is not None and edist_weight > 0.0:
                            # Divided by n_chunks so the per-step dose is
                            # `edist_weight` once, not once per chunk. NOT added to
                            # n_terms: the pointwise mean stays comparable to the
                            # no-edist arms.
                            _ed = edist_fn(pred_x0, tgt)
                            if _ed is not None:
                                n_ed += 1
                                bwd_loss = bwd_loss + loss_scale * edist_weight * (
                                    _ed / max(n_chunks, 1))
                                ed_total = (_ed.detach() if ed_total is None
                                            else ed_total + _ed.detach())
                    # Touch the state-probe params on EVERY backward. DDP runs
                    # with find_unused_parameters=False, so each ARMED backward
                    # must mark every trainable param ready; touching only the
                    # final one leaves rung 0 unfinalized and rung 1's forward
                    # raises "Expected to have finished reduction in the prior
                    # iteration before starting a new one".
                    if tail_loss_fn is not None:
                        _tl = tail_loss_fn()
                        if _tl is not None:
                            bwd_loss = bwd_loss + _tl
                    bwd_loss.backward()
                # Belt-and-braces: the pre-backward detach above already left
                # the cache grad-free, so this is normally a no-op. It stays
                # because the commit forward below writes into the cache too,
                # and a future edit that moves work between the two must not
                # silently reintroduce a grad-carrying cache entry.
                _detach_cache(kv_cache)
                bwd_done[0] += 1
                total = (term.detach() if total is None
                         else total + term.detach())
                if i < len(rungs) - 1:
                    # STUDENT sampler (matches eval_causal_AR): predict x0, then
                    # re-noise to the next rung with FRESH noise. Detached so the
                    # graph never spans rungs.
                    flat = pred_x0.detach().flatten(0, 1).float()
                    nz = torch.randn(flat.shape, device=device, dtype=flat.dtype,
                                     generator=g)
                    tt = torch.full((flat.shape[0],), rungs[i + 1],
                                    device=device, dtype=torch.float32)
                    lat = scheduler.add_noise(flat, nz, tt).view(
                        1, BLOCK_F, C, H, W).float()

            # CONTRACTION BARRIER on the COMMITTED chunk. Teacher-referenced
            # deadband (per-channel log sigma ratio, squared hinge outside
            # +-band): zero loss and zero gradient while the student's
            # dispersion sits inside the teacher's band, real force once it
            # contracts below it. Local + cheap, no collectives.
            # SET-LEVEL distribution matching on the COMMITTED chunk: each
            # rank holds a different commanded direction, so the gathered set
            # is the output distribution induced by the action distribution.
            # Called once per chunk on EVERY rank -> collective-safe.
            with torch.no_grad():
                chain.append(pred_x0.detach().float())
                per_chunk.append(float(
                    (pred_x0.detach().float() - tgt.float()).pow(2).mean()))

            # --- commit into the cache at t=0, SAME current_start, then advance
            if commit_mode == "student":
                commit_lat = pred_x0.detach()
            elif commit_mode == "schedule":
                use_student = torch.rand((), device=device, generator=g).item() < commit_p
                commit_lat = pred_x0.detach() if use_student else tgt
            else:                                   # "teacher"
                commit_lat = tgt
            with torch.no_grad():
                fwd(commit_lat.to(dtype), cond_for(z_block, BLOCK_F),
                    torch.zeros([1, BLOCK_F], device=device, dtype=torch.float32),
                    cur_start)
            cur_frames += BLOCK_F

        # Every gradient has ALREADY been applied by the per-rung backwards;
        # this scalar is detached and exists only for logging. The caller must
        # NOT call .backward() on it -- hence the explicit marker below.
        loss = total / max(n_terms, 1)
        if ed_total is not None:
            # LOG-ONLY correction (review H3/M3): _edist_core returns
            # world_size * D_E so the BACKWARDED gradient dose is
            # world-invariant under DDP's 1/W averaging -- but the raw value
            # is x W inflated, so an 8-node log would read 4x a 2-node smoke
            # for identical behaviour. Divide by W here (and by n_chunks, the
            # same divisor the backward used) so the logged loss states the
            # optimized dose. The backwarded quantity is untouched.
            import torch.distributed as _d2
            _w = (_d2.get_world_size()
                  if _d2.is_available() and _d2.is_initialized() else 1)
            loss = loss + edist_weight * (ed_total / _w / max(n_chunks, 1))
        if gx_total is not None:
            loss = loss + gexcl_weight * (gx_total / max(n_gx, 1))
        if bwd_done[0] != n_bwd_total:
            raise RuntimeError(
                f"rollout backward count {bwd_done[0]} != planned "
                f"{n_bwd_total}; DDP would hang or skip its all-reduce.")
        return {"loss": loss, "backwarded": True,
                "vrfm_stats": vrfm_stats,
                "kl_z": (float(kl_z_total / max(n_kl_z, 1))
                         if kl_z_total is not None else 0.0),
                "n_terms": n_terms, "n_ed": n_ed,
                "per_chunk": per_chunk, "n_gx": n_gx,
                "chain": torch.cat(chain, dim=1) if chain else None}
    finally:
        if saved_base_las != SENT:
            base.local_attn_size = saved_base_las
        if saved_base_mas != SENT and hasattr(base, "max_attention_size"):
            base.max_attention_size = saved_base_mas
        for m, las, mas, cra in saved_mods:
            if las != SENT:
                m.local_attn_size = las
            if mas != SENT and hasattr(m, "max_attention_size"):
                m.max_attention_size = mas
            if cra != SENT:
                m.cached_rope_action_aware = cra
        if saved_seq_len is not None:
            wrapper.seq_len = saved_seq_len
