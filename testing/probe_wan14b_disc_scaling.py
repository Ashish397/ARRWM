"""WP-14B — memory/time scaling probe for the 14B-backbone LADD disc.

The unit tests pin correctness; this answers the question that decides
whether the arm survives step 1: how the disc's peak GPU memory and step
time grow with the number of disc ROWS at dim 5120, for both shapes the
trainer runs —

  D-update  : disc.train(),  forward on the combined real+fake batch,
              backward to the disc parameters (CCM/CSM/heads).
  G-guidance: disc.eval(),   forward on the fake rows, backward to the
              INPUT latent (this is the gradient the generator consumes),
              optionally micro-batched the way
              ``ladd_gen_guidance_micro_batch_groups`` does.

Run on one GPU inside an existing allocation, e.g.

  srun --jobid=<holder> --overlap --nodelist=<node> --nodes=1 --ntasks=1 \
       --export=ALL,CUDA_VISIBLE_DEVICES=3 \
       python testing/probe_wan14b_disc_scaling.py --rows 1 2 4 6 8 12

OOM at a given row count is caught and reported, not fatal — the point of
the probe is to find the ceiling.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ladd_disc import build_ladd_disc  # noqa: E402
from model.wan14b_prefix import (  # noqa: E402
    WAN14B_DEFAULT_PATH,
    load_wan14b_prefix,
)


def _fmt_gib(n_bytes):
    return f"{n_bytes / 2 ** 30:6.2f}"


def _sync():
    torch.cuda.synchronize()


def _timed(fn, warmup=1, iters=2):
    for _ in range(warmup):
        fn()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        fn()
    _sync()
    dt = (time.time() - t0) / iters
    return dt, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=WAN14B_DEFAULT_PATH)
    ap.add_argument("--taps", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    ap.add_argument("--frames", type=int, default=3, help="F per disc chunk")
    ap.add_argument("--height", type=int, default=60)
    ap.add_argument("--width", type=int, default=104)
    ap.add_argument("--dim-proj", type=int, default=256)
    ap.add_argument("--scalar-output", action="store_true", default=True)
    ap.add_argument("--gen-micro-groups", type=int, nargs="+", default=[1, 4])
    ap.add_argument(
        "--max-block", type=int, default=None,
        help="blocks to load (default: max(taps)). Use with "
             "deep taps to emulate today's FULL-depth teacher tap, e.g. "
             "--taps 6 12 18 24 29 on the 1.3B.",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "probe needs a GPU"
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    backbone = load_wan14b_prefix(
        args.path,
        max_block=(max(args.taps) if args.max_block is None
                   else int(args.max_block)),
        device=dev, dtype=torch.bfloat16,
    )
    w_bytes = sum(p.numel() * p.element_size() for p in backbone.parameters())
    disc = build_ladd_disc(
        real_score=None, block_indices=args.taps,
        dim_teacher=int(backbone.dim), dim_proj=args.dim_proj,
        use_csm=True, cmap_dim=0, patch_size=(1, 2, 2),
        action_tokens_per_frame=0, scalar_output=bool(args.scalar_output),
        backbone=backbone,
    ).to(device=dev, dtype=torch.float32)
    n_train = sum(p.numel() for p in disc.parameters() if p.requires_grad)

    n_tok = args.frames * (args.height // 2) * (args.width // 2)
    print(
        f"\nbackbone: {len(backbone.blocks)} blocks, dim={backbone.dim}, "
        f"weights={_fmt_gib(w_bytes)} GiB bf16\n"
        f"disc: taps={args.taps} dim_proj={args.dim_proj} "
        f"scalar_output={bool(args.scalar_output)} "
        f"trainable={n_train / 1e6:.2f}M\n"
        f"chunk: F={args.frames} {args.height}x{args.width} "
        f"-> {n_tok} tokens/row\n"
    )

    pe = torch.randn(1, 512, 4096, device=dev)

    def _batch(rows, grad):
        x = torch.randn(
            rows, args.frames, 16, args.height, args.width, device=dev,
            requires_grad=grad,
        )
        t = torch.full((rows, args.frames), 500, dtype=torch.long, device=dev)
        return x, t, pe.expand(rows, -1, -1)

    print(f"{'rows':>5} {'phase':<22} {'ms/iter':>9} {'peak_alloc':>11} "
          f"{'peak_resv':>10}")
    print("-" * 62)

    for rows in args.rows:
        # ---- D-update: train mode, combined real+fake, grads to params
        def _d_step(rows=rows):
            x, t, p = _batch(2 * rows, grad=False)
            disc.train()
            logits = disc(x_noisy=x, timestep=t, prompt_embeds=p)
            loss = torch.nn.functional.softplus(
                logits[rows:].mean() - logits[:rows].mean()
            )
            disc.zero_grad(set_to_none=True)
            loss.backward()

        # ---- G-guidance: eval mode, fake rows only, grad to the input
        def _g_step(rows=rows, groups=1):
            x, t, p = _batch(rows, grad=True)
            disc.eval()
            if groups <= 1:
                logits = disc(x_noisy=x, timestep=t, prompt_embeds=p)
            else:
                cuts = [(g * rows) // groups for g in range(groups + 1)]
                parts = [
                    disc(x_noisy=x[a:b], timestep=t[a:b], prompt_embeds=p[a:b])
                    for a, b in zip(cuts, cuts[1:]) if b > a
                ]
                logits = torch.cat(parts, dim=0)
            torch.nn.functional.softplus(-logits.mean()).backward()

        phases = [("D-update (2x rows)", _d_step)]
        for g in args.gen_micro_groups:
            if g > rows:
                continue
            phases.append(
                (f"G-guidance groups={g}", lambda g=g, r=rows: _g_step(r, g))
            )

        for name, fn in phases:
            try:
                dt, pa, pr = _timed(fn)
                print(f"{rows:>5} {name:<22} {dt * 1e3:9.1f} "
                      f"{_fmt_gib(pa):>11} {_fmt_gib(pr):>10}")
            except torch.cuda.OutOfMemoryError:
                print(f"{rows:>5} {name:<22} {'OOM':>9} "
                      f"{'-':>11} {'-':>10}")
            finally:
                disc.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    total = torch.cuda.get_device_properties(0).total_memory
    print(f"\ndevice total = {_fmt_gib(total)} GiB; "
          f"backbone weights = {_fmt_gib(w_bytes)} GiB (resident always)")


if __name__ == "__main__":
    main()
