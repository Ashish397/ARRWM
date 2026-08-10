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

import torch


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
    for _, m in base.named_modules():
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
        kv_cache, crossattn_cache = _alloc_caches(
            base, kv_frames=LOCAL_ATTN_F + 2 * BLOCK_F, frame_seq=FRAME_SEQ,
            dtype=dtype, device=device)

        def cond_for(z_block, nframes):
            # NOTE: no *_clean streams — this path has no clean_x, exactly like
            # the teacher's generation.
            cond = {"prompt_embeds": prompt_embeds}
            if has_adaln:
                cond["_action_modulation"] = action_projection(z_block, num_frames=nframes)
            if has_tokens:
                cond["_action_tokens"] = action_token_projection(z_block)
            return cond

        def fwd(lat, cond, ts, cur_start):
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = call(lat, cond, ts,
                           kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                           current_start=cur_start, cache_start=0)
            return out

        seed_f = int(seed_lat.shape[1])
        assert seed_f % nfb == 0, (seed_lat.shape, nfb)
        n_chunks = int(committed.shape[1])
        # Eviction MUST never fire: gradient checkpointing re-reads the cache at
        # recompute time, so a rolled cache would silently produce wrong
        # gradients with no error. kv_frames = 21 + 2*3 = 27 exactly equals
        # seed(9) + gen(18) today -- assert rather than rely on that.
        assert seed_f + n_chunks * nfb <= LOCAL_ATTN_F + 2 * BLOCK_F, (
            f"rollout span {seed_f + n_chunks * nfb} frames exceeds the cache "
            f"({LOCAL_ATTN_F + 2 * BLOCK_F}); eviction would corrupt "
            "gradient-checkpoint recompute")
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
        per_chunk: List[float] = []
        cur_frames = seed_f

        for c in range(n_chunks):
            F0 = cur_frames
            cur_start = F0 * FRAME_SEQ
            cond = cond_for(z_actions[:, F0:F0 + BLOCK_F], BLOCK_F)
            tgt = committed[:, c].to(device)

            # Per-chunk LOCAL generator seeded exactly as the teacher did
            # (causal_chain_rollout.py:147) so chunk c starts from the SAME
            # noise the teacher used, without perturbing the training RNG.
            g = torch.Generator(device=device).manual_seed(seed_base + c)
            lat = torch.randn([1, BLOCK_F, C, H, W], dtype=torch.float32,
                              device=device, generator=g)

            pred_x0 = None
            for i, t in enumerate(rungs):
                ts = torch.full([1, BLOCK_F], t, device=device, dtype=torch.float32)
                out = fwd(lat.to(dtype), cond, ts, cur_start)
                if not (isinstance(out, tuple) and len(out) >= 2):
                    raise RuntimeError(
                        "rollout ODE expects the wrapper to return "
                        "(flow_pred, pred_x0, ...); got "
                        f"{type(out).__name__} len="
                        f"{len(out) if isinstance(out, tuple) else 'N/A'}")
                pred_x0 = out[1]
                term = loss_fn(pred_x0, tgt)
                total = term if total is None else total + term
                n_terms += 1
                # Cut the graph at the cache after EVERY forward: the cache is
                # written in place, so otherwise chunk c's loss would reach back
                # through every earlier forward.
                _detach_cache(kv_cache)
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

            with torch.no_grad():
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
                fwd(commit_lat.to(dtype), cond,
                    torch.zeros([1, BLOCK_F], device=device, dtype=torch.float32),
                    cur_start)
            _detach_cache(kv_cache)
            cur_frames += BLOCK_F

        loss = total / max(n_terms, 1)
        return {"loss": loss, "n_terms": n_terms, "per_chunk": per_chunk}
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
