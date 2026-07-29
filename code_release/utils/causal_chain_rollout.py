"""Lightweight KV-cache streaming causal-chain rollout (hardened, action-aware).

Portable extraction of the rollout in ``logs/stream_kv.sh`` so it can be driven
by the LIVE training model (not a freshly-built ChainPipeline). Drives the
model's native ``_forward_inference`` KV-cache path: frame-sink + bounded
local-attention window with eviction, per-block denoise then a clean-pred
"commit" that writes the block's clean K/V into the cache. The cached RoPE is
run ACTION-AWARE (matches the teacher-forced training path; removes the
~1-token/frame spatial drift the bare cached RoPE would introduce).

All model mutations (per-module ``local_attn_size`` / ``max_attention_size`` /
``cached_rope_action_aware`` and the wrapper ``seq_len``) are snapshotted and
restored in a ``finally`` so training continues byte-identically afterwards.
"""
import torch

from utils.scheduler import FlowMatchScheduler


@torch.no_grad()
def stream_causal_chain(
    wrapper, action_proj, token_proj, prompt_embeds, seed_lat, z_actions,
    *, gen_chunks, eval_steps, dtype, device,
    nfb=3, local_attn_chunks=7, block_chunks=1, seed_base=1234,
    step_recorder=None,
):
    """Roll out a causal chain under the per-frame action commands in z_actions.

    Args:
        wrapper: the (LoRA-wrapped) WanDiffusionWrapper (unwrapped from DDP).
        action_proj / token_proj: action modulation / token projections.
        prompt_embeds: [1, L, D] text embeds (already on device, dtype).
        seed_lat: [1, nfb, C, H, W] clean seed chunk (real context).
        z_actions: [1, TOT_F, A] per-frame action command (A = raw_action_dim,
            e.g. 2 for [z2,z7]); TOT_F must be >= nfb + gen_chunks*nfb. The first
            nfb frames condition the seed; flip only the frames AFTER nfb to keep
            the starting state identical across GT/flipped runs.

    Returns:
        full_lat: [1, TOT_F, C, H, W] committed latents (seed + generated).
    """
    BLOCK_F = block_chunks * nfb
    LOCAL_ATTN_F = local_attn_chunks * nfb
    GEN_F = gen_chunks * nfb
    SEED_F = int(seed_lat.shape[1])                # 1+ chunks of real context
    assert SEED_F % nfb == 0, (seed_lat.shape, nfb)
    TOT_F = SEED_F + GEN_F
    assert z_actions.shape[1] >= TOT_F, (z_actions.shape, TOT_F)

    # AdaLN-only ablation (action_conditioning_mode="adaln"): the action-token
    # projection is None. Mirror the trainer's forward exactly -- it omits the
    # "_action_tokens" stream and never reserves token seq-len (action_per_frame
    # == 0 -> seq_len == _base_seq_len). Strictly gated on token_proj so the
    # token runs ("both") are byte-identical (has_tokens=True -> _apf=1).
    has_tokens = token_proj is not None
    has_adaln = action_proj is not None          # noadaln ablation: AdaLN off, tokens only
    _apf = 1 if has_tokens else 0

    base = wrapper.model
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    FRAME_SEQ = 1560 + int(getattr(base, "action_tokens_per_frame", 0))
    target_max = LOCAL_ATTN_F * FRAME_SEQ

    # --- snapshot everything we mutate, for restore ---
    _SENT = "__keep__"
    saved_base_las = getattr(base, "local_attn_size", _SENT)
    saved_base_mas = getattr(base, "max_attention_size", _SENT)
    saved_mods = []
    base.local_attn_size = LOCAL_ATTN_F
    if hasattr(base, "max_attention_size"):
        base.max_attention_size = target_max
    for _, m in base.named_modules():
        rec = (m,
               getattr(m, "local_attn_size", _SENT),
               getattr(m, "max_attention_size", _SENT),
               getattr(m, "cached_rope_action_aware", _SENT))
        if hasattr(m, "local_attn_size"):
            m.local_attn_size = LOCAL_ATTN_F
        if hasattr(m, "max_attention_size"):
            m.max_attention_size = target_max
        if hasattr(m, "action_tokens_per_frame"):
            m.cached_rope_action_aware = has_tokens
        saved_mods.append(rec)
    saved_seq_len = getattr(wrapper, "seq_len", None)

    try:
        # --- allocate KV + cross-attn caches (mirror _allocate_slot_caches) ---
        num_blocks = len(base.blocks)
        blk0 = base.blocks[0]
        num_heads = int(getattr(blk0.self_attn, "num_heads", getattr(base, "num_heads", 16)))
        head_dim = int(getattr(blk0.self_attn, "head_dim", base.dim // num_heads))
        text_len = int(getattr(base, "text_len", 512))
        kv_frames = LOCAL_ATTN_F + 2 * BLOCK_F
        kv_size = kv_frames * FRAME_SEQ
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

        def cond_for(z_block, nframes):
            cond = {"prompt_embeds": prompt_embeds}
            if has_adaln:                         # omit for tokens-only (noadaln); mirrors training
                cond["_action_modulation"] = action_proj(z_block, num_frames=nframes)
            if has_tokens:                        # omit for AdaLN-only (noatok); mirrors training
                cond["_action_tokens"] = token_proj(z_block)
            return cond

        def fwd(lat, cond, ts, cur_start):
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = wrapper(lat, cond, ts, kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                              current_start=cur_start, cache_start=0)
            return out[0] if isinstance(out, (tuple, list)) else out

        # --- warm-start: write the clean seed into the cache at position 0 ---
        # (one cache-refresh forward per nfb-frame seed chunk, in order, so a
        # multi-chunk real context is supported: SEED_F = k * nfb, k >= 1)
        seed_lat = seed_lat.to(dtype)
        wrapper.adjust_seq_len_for_action_tokens(num_frames=nfb, action_per_frame=_apf)
        for s0 in range(0, SEED_F, nfb):
            fwd(seed_lat[:, s0:s0 + nfb], cond_for(z_actions[:, s0:s0 + nfb], nfb),
                torch.zeros([1, nfb], device=device, dtype=torch.float32),
                s0 * FRAME_SEQ)
        cur_frames = SEED_F
        lat_all = [seed_lat.float()]

        sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        sched.set_timesteps(num_inference_steps=eval_steps, denoising_strength=1.0)
        sched.sigmas = sched.sigmas.to(device)
        C, H, W = seed_lat.shape[2], seed_lat.shape[3], seed_lat.shape[4]

        wrapper.adjust_seq_len_for_action_tokens(num_frames=BLOCK_F, action_per_frame=_apf)
        n_blocks = GEN_F // BLOCK_F
        for b in range(n_blocks):
            F0 = cur_frames
            cond = cond_for(z_actions[:, F0:F0 + BLOCK_F], BLOCK_F)
            cur_start = cur_frames * FRAME_SEQ
            # local generator -> does NOT perturb the training RNG stream
            g = torch.Generator(device=device).manual_seed(seed_base + b)
            lat = torch.randn([1, BLOCK_F, C, H, W], dtype=torch.float32, device=device, generator=g)
            if step_recorder is not None:            # flow-viz: record initial noise
                step_recorder(b, -1, lat)
            for si, t in enumerate(sched.timesteps):
                ts = t * torch.ones([1, BLOCK_F], device=device, dtype=torch.float32)
                flow = fwd(lat, cond, ts, cur_start)
                lat = sched.step(flow.flatten(0, 1), ts.flatten(0, 1),
                                 lat.flatten(0, 1)).unflatten(dim=0, sizes=flow.shape[:2])
                if step_recorder is not None:        # flow-viz: record x_t after each step
                    step_recorder(b, si, lat)
            # commit clean prediction into the cache (context_noise = 0)
            fwd(lat.to(dtype), cond, torch.zeros([1, BLOCK_F], device=device, dtype=torch.float32), cur_start)
            cur_frames += BLOCK_F
            lat_all.append(lat.float())

        return torch.cat(lat_all, dim=1)
    finally:
        if saved_base_las != _SENT:
            base.local_attn_size = saved_base_las
        if saved_base_mas != _SENT:
            base.max_attention_size = saved_base_mas
        for m, las, mas, cra in saved_mods:
            if las != _SENT:
                m.local_attn_size = las
            if mas != _SENT:
                m.max_attention_size = mas
            if cra != _SENT:
                m.cached_rope_action_aware = cra
        if saved_seq_len is not None:
            wrapper.seq_len = saved_seq_len
