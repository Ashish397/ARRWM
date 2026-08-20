"""GPU check of the Infinity-RoPE buffer-relative anchor ACROSS the
cache-fill boundary.

THE CLAIM UNDER TEST
--------------------
``utils/infinity_rope.py`` rotates cache K at ``k_rel[i] = i`` for
``i in [0, num_cache_frames)`` and anchors Q at
``q_start = num_cache_frames - num_new_frames``. If that holds, then
once the buffer is SATURATED (``num_cache_frames == buffer_frames``)
the frames-back -> query/key offset map is a constant: a key ``d``
frames older than the newest query frame always sits ``d`` rotation
steps behind it, on every step, forever. The pre-fix code anchored Q
window-relatively after the first roll, which sheared every offset by
``local_attn_size - num_cache_frames`` from that roll onward.

HOW IT MEASURES
---------------
It drives a real DiT forward through the real KV-cache path with a
buffer deliberately smaller than the rollout, so the cache rolls, and
reads the numbers out of the ``ARRWM_ROPE_DEBUG`` hook already in
``utils/infinity_rope.py``. That hook is gated on
``torch.is_grad_enabled()`` (it is a TRAINING-path probe), so the
forwards here run under ``enable_grad`` — with every parameter's
``requires_grad`` cleared, so no autograd graph is actually built and
the memory profile stays inference-sized. The hook's own print budget
(8 lines per process) is reset before each chunk so every step is
observed rather than only the first forward.

Nothing in the training path is modified: this file only reads.
"""
import argparse
import io
import os
import re
import sys
from contextlib import redirect_stdout

os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ["ARRWM_ROPE_DEBUG"] = "1"

import torch  # noqa: E402

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
for _p in (ARR, f"{ARR}/action-forcing"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DBG_RE = re.compile(
    r"\[ROPE-DBG gen\] rolled=(\S+) num_new_frames=(\d+) q_rel=(\[[^\]]*\]) "
    r"num_cache_frames=(\d+) local_start_frame=(-?\d+) local_attn_size=(-?\d+)"
)


def _base_dit(wrapper):
    m = wrapper.model
    return m.get_base_model() if hasattr(m, "get_base_model") else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ARR}/logs/v14e_pca8_raw/"
                                      "causal_lora_step0005000.pt")
    ap.add_argument("--config", default=f"{ARR}/configs/action_ode_distill_F.yaml")
    ap.add_argument("--npb", type=int, default=3)
    ap.add_argument("--buffer_frames", type=int, default=12,
                    help="KV buffer size; the rollout must exceed it")
    ap.add_argument("--window_frames", type=int, default=12,
                    help="local_attn_size in frames")
    ap.add_argument("--chunks", type=int, default=10,
                    help="chunks to roll (must exceed buffer/npb)")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    cfg.generator_ckpt = args.ckpt
    dtype = torch.bfloat16

    from utils.infinity_rope import install as ir_install
    from utils import infinity_rope as ir
    ir_install(None)
    print("[rope] infinity-RoPE patch installed", flush=True)

    from af_model.ode_regression import ODERegression
    ode = ODERegression(cfg, device=device).eval()
    ode.use_motion_pipeline = False
    wrapper = ode.generator
    base = _base_dit(wrapper)
    # No autograd graph: the hook only needs is_grad_enabled(), not grads.
    for p in ode.parameters():
        p.requires_grad_(False)

    has_tokens = ode.action_token_projection is not None
    fsl = 1560 + int(getattr(base, "action_tokens_per_frame", 0))
    npb = args.npb
    base.block_mask = None
    base.local_attn_size = args.window_frames
    target_max = args.window_frames * fsl
    if hasattr(base, "max_attention_size"):
        base.max_attention_size = target_max
    for _, m in base.named_modules():
        if hasattr(m, "local_attn_size"):
            m.local_attn_size = args.window_frames
        if hasattr(m, "max_attention_size"):
            m.max_attention_size = target_max
        if hasattr(m, "action_tokens_per_frame"):
            m.cached_rope_action_aware = has_tokens
    wrapper.adjust_seq_len_for_action_tokens(
        num_frames=npb, action_per_frame=1 if has_tokens else 0)

    n_blocks = len(base.blocks)
    blk0 = base.blocks[0]
    n_heads = int(getattr(blk0.self_attn, "num_heads",
                          getattr(base, "num_heads", 12)))
    head_dim = int(getattr(blk0.self_attn, "head_dim", base.dim // n_heads))
    text_len = int(getattr(base, "text_len", 512))
    kv_size = args.buffer_frames * fsl
    kv, xa = [], []
    for _ in range(n_blocks):
        kv.append({
            "k": torch.zeros([1, kv_size, n_heads, head_dim], dtype=dtype,
                             device=device),
            "v": torch.zeros([1, kv_size, n_heads, head_dim], dtype=dtype,
                             device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long,
                                             device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long,
                                            device=device),
        })
        xa.append({
            "k": torch.zeros([1, text_len, n_heads, head_dim], dtype=dtype,
                             device=device),
            "v": torch.zeros([1, text_len, n_heads, head_dim], dtype=dtype,
                             device=device),
            "is_init": False,
        })

    pe = torch.randn([1, 512, base.text_dim if hasattr(base, "text_dim")
                      else 4096], device=device, dtype=dtype)
    C = int(getattr(cfg, "latent_channels", 16))
    H = W = None
    # latent spatial size implied by frame_seq_length 1560 = (H/2)*(W/2)
    # for patch (1,2,2): 1560 = 40*39 -> H=80, W=78 in the 14e geometry.
    H, W = 60, 104
    if (H // 2) * (W // 2) != 1560:
        H, W = 40, 156
    z = torch.zeros([1, npb, 2], device=device, dtype=dtype)

    def cond():
        c = {"prompt_embeds": pe}
        if ode.action_projection is not None:
            c["_action_modulation"] = ode.action_projection(
                z, num_frames=npb)
        if ode.action_token_projection is not None:
            c["_action_tokens"] = ode.action_token_projection(z)
        return c

    records = []
    buf = io.StringIO()
    with torch.enable_grad():
        for k_i in range(args.chunks):
            f0 = k_i * npb
            lat = torch.randn([1, npb, C, H, W], device=device,
                              dtype=torch.float32)
            ts = torch.full([1, npb], 1000.0, device=device,
                            dtype=torch.float32)
            ir.patched_forward._rope_dbg_gen = 0     # re-arm the hook
            with redirect_stdout(buf):
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    out = wrapper(lat.to(dtype), cond(), ts, kv_cache=kv,
                                  crossattn_cache=xa,
                                  current_start=f0 * fsl)
            del out
            lines = [ln for ln in buf.getvalue().splitlines()
                     if "[ROPE-DBG gen]" in ln]
            buf.seek(0), buf.truncate(0)
            m = DBG_RE.search(lines[0]) if lines else None
            if m is None:
                print(f"[rope] step {k_i}: NO DEBUG LINE (hook silent) "
                      f"-- got {lines[:1]}", flush=True)
                continue
            rec = {
                "step": k_i, "f0": f0, "rolled": m.group(1),
                "num_new_frames": int(m.group(2)),
                "q_rel": eval(m.group(3)),
                "num_cache_frames": int(m.group(4)),
                "local_start_frame": int(m.group(5)),
                "local_attn_size": int(m.group(6)),
            }
            records.append(rec)
            print(f"[rope] step={k_i:>2} f0={f0:>3} rolled={rec['rolled']:>5} "
                  f"num_cache_frames={rec['num_cache_frames']:>3} "
                  f"q_rel={rec['q_rel']} "
                  f"local_start_frame={rec['local_start_frame']}", flush=True)

    print("\n[rope] ================= VERDICT =================")
    sat = [r for r in records
           if r["num_cache_frames"] >= args.buffer_frames]
    unsat = [r for r in records
             if r["num_cache_frames"] < args.buffer_frames]
    print(f"[rope] buffer={args.buffer_frames}f window={args.window_frames}f "
          f"npb={npb} | {len(unsat)} filling step(s), {len(sat)} saturated")
    if not sat or len(sat) < 2:
        print("[rope] FAIL/INCONCLUSIVE: fewer than 2 saturated steps -- "
              "the rollout never crossed the cache-fill boundary.")
        return 2
    qs = {tuple(r["q_rel"]) for r in sat}
    ns = {r["num_cache_frames"] for r in sat}
    ok = (len(qs) == 1 and len(ns) == 1)
    print(f"[rope] saturated q_rel set          = {sorted(qs)}")
    print(f"[rope] saturated num_cache_frames   = {sorted(ns)}")
    # frames-back -> offset map, derived from the invariant k_rel[i]=i:
    # a key d frames older than the newest query frame sits at buffer
    # index (num_cache_frames-1-d), so offset = q_last - k_rel = d.
    if ok:
        q_last = max(next(iter(qs)))
        ncf = next(iter(ns))
        m = {d: q_last - (ncf - 1 - d) for d in range(0, min(6, ncf))}
        print(f"[rope] frames-back -> offset map (saturated) = {m}")
        print("[rope] PASS: q_rel and num_cache_frames are IDENTICAL on "
              "every saturated step -> the frames-back -> offset map is "
              "constant across the cache-fill boundary.")
        return 0
    print("[rope] FAIL: the query anchor MOVES between saturated steps -- "
          "offsets are sheared after the roll.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
