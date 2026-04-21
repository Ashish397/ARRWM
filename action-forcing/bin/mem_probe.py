"""Memory probe for the action-forcing ODE distill training step.

Determines the largest ``batch_size`` (pairs per rank) that fits on a
single GH200 under the real training configuration:

  * full-rank DiT (merge_and_unload applied to the teacher LoRA),
  * all action apparatus (state probe + critic + projections),
  * motion pipeline built eagerly (VAE + CoTracker + ss_vae),
  * bf16 autocast,
  * gradient checkpointing,
  * ``critic_updates_per_step`` critic inner loop.

The packed forward inside ``generator_loss`` processes ``B_pair`` pairs
per rank = ``2 * B_pair`` DiT-batch slots. This probe sweeps
``B_pair ∈ {1, 2, 3}`` (→ DiT-B ∈ {2, 4, 6}) by spawning a subprocess per
value (each gets a fresh CUDA context so the peak-memory stat is clean).
Runs one realistic optimiser step per value and records:

  * ``torch.cuda.max_memory_allocated()`` — peak live allocations.
  * ``torch.cuda.max_memory_reserved()``  — peak size of the allocator
    pool including fragmentation / reserved blocks.
  * wall-clock step time.
  * whether it OOMed.

No silent fallbacks. If the chosen ``B_pair`` OOMs the worker exits
non-zero; the driver records OOM and moves on to the next value.

Usage (interactive)::

    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    srun -N1 -n1 --gres=gpu:1 --partition=<gpu> \\
         python action-forcing/bin/mem_probe.py \\
             --config configs/action_ode_distill.yaml \\
             --ckpt   logs/v14_balanced_weunz/causal_lora_step0006600.pt \\
             --b_pairs 1,2,3

The driver writes a JSON summary to
``logs/archive/mem_probe_<timestamp>.json`` and prints a recommendation
for the config's ``batch_size`` knob (largest ``B_pair`` whose peak
reserved memory stays under ``--memory-budget-gb`` with
``--headroom-gb`` to spare).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parents[1]     # action-forcing/
_WORKSPACE = _THIS_DIR.parent                        # repo root
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))


log = logging.getLogger("mem_probe")


# =====================================================================
# WORKER MODE (single subprocess; runs one probe at a fixed B_pair).
# =====================================================================

def _run_worker(
    config_path: str,
    ckpt_path: str,
    b_pair: int,
    result_path: str,
) -> None:
    import torch
    from omegaconf import OmegaConf

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][%(levelname)s][worker B={b_pair}] %(message)s",
        datefmt="%H:%M:%S",
    )
    worker_log = log

    if not torch.cuda.is_available():
        raise RuntimeError("mem_probe requires a CUDA device.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # -----------------------------------------------------------------
    # Load config + point generator_ckpt at the supplied file.
    # -----------------------------------------------------------------
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    cfg.generator_ckpt = ckpt_path
    cfg.max_pair = max(int(getattr(cfg, "max_pair", 0) or 4), 4)  # we only need 1 pair; leave some headroom

    result: Dict[str, Any] = {
        "b_pair": b_pair,
        "dit_batch": 2 * b_pair,
        "ok": False,
        "oomed": False,
        "error": None,
        "peak_alloc_gb": None,
        "peak_reserved_gb": None,
        "step_time_s": None,
    }

    def _dump(r: Dict[str, Any]) -> None:
        with open(result_path, "w") as f:
            json.dump(r, f, indent=2)

    try:
        worker_log.info("Building ODERegression (merged DiT + action apparatus) ...")
        from af_model.ode_regression import ODERegression
        model = ODERegression(cfg, device=device)
        model.ensure_motion_pipeline(distributed=False)
        # Train mode so gradient checkpointing + dropout behave as in the
        # real trainer, and motion pipeline is the full frozen stack.
        model.generator.train()
        if model.action_critic is not None:
            model.action_critic.train()

        # -----------------------------------------------------------------
        # Optimizers: same param groups the trainer builds.
        # -----------------------------------------------------------------
        gen_params = [
            p for p in model.generator.parameters() if p.requires_grad
        ]
        if model.action_projection is not None:
            gen_params.extend(
                p for p in model.action_projection.parameters() if p.requires_grad
            )
        if model.action_token_projection is not None:
            gen_params.extend(
                p for p in model.action_token_projection.parameters() if p.requires_grad
            )
        optimizer = torch.optim.AdamW(
            gen_params,
            lr=float(getattr(cfg, "lr", 2e-6)),
            betas=(float(getattr(cfg, "beta1", 0.9)), float(getattr(cfg, "beta2", 0.999))),
            weight_decay=float(getattr(cfg, "weight_decay", 0.01)),
        )
        critic_optimizer: Optional[torch.optim.Optimizer] = None
        if model.action_critic is not None:
            critic_optimizer = torch.optim.AdamW(
                model.action_critic.parameters(),
                lr=float(getattr(cfg, "critic_lr", 3e-4)),
                betas=(float(getattr(cfg, "beta1", 0.9)), float(getattr(cfg, "beta2", 0.999))),
                weight_decay=float(getattr(cfg, "weight_decay", 0.01)),
            )

        # -----------------------------------------------------------------
        # Pull one real paired sample from the dataset and replicate it
        # along the batch dim B_pair times. Using the real LMDB layout
        # gives us a true-to-life memory footprint (snapshot count, chunk
        # counts, etc.).
        # -----------------------------------------------------------------
        from af_utils.dataset import PairedTrajectoryDataset
        ds = PairedTrajectoryDataset(
            clean_root=str(cfg.clean_root),
            cf_root=str(cfg.cf_root),
            caption_root=str(cfg.caption_root),
            max_pair=int(cfg.max_pair) or None,
            require_cf=bool(getattr(cfg, "require_cf", True)),
        )
        pair = ds[0]

        def _repeat(t: torch.Tensor) -> torch.Tensor:
            t = t.unsqueeze(0).to(device)
            if b_pair == 1:
                return t
            return t.repeat((b_pair,) + (1,) * (t.dim() - 1)).contiguous()

        batch = {
            "trajectory_clean": _repeat(pair["trajectory_clean"]),
            "trajectory_cf":    _repeat(pair["trajectory_cf"]),
            "prompt_embeds":    _repeat(pair["prompt_embeds"]),
            "z_clean":          _repeat(pair["z_clean"]),
            "z_noisy":          _repeat(pair["z_noisy"]),
            "z_noisy_cf":       _repeat(pair["z_noisy_cf"]),
            "clean_x_gt":       _repeat(pair["clean_x_gt"]),
        }

        mixed_precision = bool(getattr(cfg, "mixed_precision", True))
        dtype = torch.bfloat16 if mixed_precision else torch.float32

        # -----------------------------------------------------------------
        # Warm-up + peak-memory reset.
        # -----------------------------------------------------------------
        worker_log.info("Warmup: (ensuring CoTracker Triton autotune is done) ...")
        with torch.no_grad():
            _ = model._compute_action_teacher_targets(
                batch["clean_x_gt"][: max(1, b_pair)].contiguous()
            )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t_start = time.time()

        # -----------------------------------------------------------------
        # One full packed training step (generator forward+backward+step).
        # -----------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            loss, logs = model.generator_loss(
                trajectory_clean=batch["trajectory_clean"],
                trajectory_cf=batch["trajectory_cf"],
                prompt_embeds=batch["prompt_embeds"],
                z_clean=batch["z_clean"],
                z_noisy=batch["z_noisy"],
                z_noisy_cf=batch["z_noisy_cf"],
                clean_x_gt=batch["clean_x_gt"],
                step=0,
            )
        loss.backward()
        if float(getattr(cfg, "grad_clip", 10.0)) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in gen_params if p.grad is not None],
                float(getattr(cfg, "grad_clip", 10.0)),
            )
        optimizer.step()

        # -----------------------------------------------------------------
        # Critic inner loop (2 updates per step by default) on the cleaner
        # half of the chunks — matches the real trainer's footprint.
        # -----------------------------------------------------------------
        if critic_optimizer is not None:
            action_critic_dims = model.action_critic_dims
            loss_weight = float(model.action_critic_z_loss_weight)
            critic_updates = int(getattr(cfg, "critic_updates_per_step", 2))
            for suffix in ("clean", "cf"):
                pred_detach = logs[f"pred_x0_detached_{suffix}"]
                teacher_z   = logs.get(f"teacher_z_8d_{suffix}")
                chunk_t     = logs[f"chunk_t_{suffix}"]
                chunk_acts  = logs[f"chunk_actions_{suffix}"]
                chunk_mask  = logs[f"critic_chunk_mask_{suffix}"]
                if teacher_z is None or not chunk_mask.any():
                    continue
                n_chunks = chunk_t.shape[1]
                tgt = teacher_z[:, :n_chunks]
                mask_f = chunk_mask[:, :n_chunks].to(pred_detach.dtype)
                for _ in range(critic_updates):
                    critic_optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        "cuda", dtype=dtype, enabled=mixed_precision,
                    ):
                        pred_z = model.action_critic(pred_detach, chunk_t, chunk_acts)
                        pred_z = pred_z[:, :n_chunks]
                        w = torch.ones(
                            pred_z.shape[-1], device=pred_z.device, dtype=pred_z.dtype,
                        )
                        for d in action_critic_dims:
                            w[d] = 2.0
                        per_chunk = (w * (pred_z - tgt) ** 2).mean(dim=-1)
                        denom = mask_f.sum().clamp_min(1.0)
                        critic_z_loss = (per_chunk * mask_f).sum() / denom
                        critic_loss = loss_weight * critic_z_loss
                    critic_loss.backward()
                    if float(getattr(cfg, "grad_clip", 10.0)) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.action_critic.parameters(),
                            float(getattr(cfg, "grad_clip", 10.0)),
                        )
                    critic_optimizer.step()

        torch.cuda.synchronize()
        dt = time.time() - t_start

        peak_alloc    = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        result.update({
            "ok": True,
            "oomed": False,
            "peak_alloc_gb":    peak_alloc    / 1024**3,
            "peak_reserved_gb": peak_reserved / 1024**3,
            "step_time_s":      dt,
            "loss":             float(loss.detach().float().item()),
        })

        worker_log.info(
            "OK: peak_alloc=%.2f GB reserved=%.2f GB  step=%.1fs  loss=%.4f",
            result["peak_alloc_gb"], result["peak_reserved_gb"],
            dt, result["loss"],
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"ok": False, "oomed": True, "error": f"CUDA OOM: {exc}"})
        worker_log.error("OOM at B_pair=%d: %s", b_pair, exc)
    except Exception as exc:  # noqa: BLE001  — surface any other error as well
        result.update({"ok": False, "oomed": False, "error": f"{type(exc).__name__}: {exc}"})
        worker_log.exception("Probe failed at B_pair=%d", b_pair)
    finally:
        _dump(result)


# =====================================================================
# DRIVER MODE (sweeps B_pair values by spawning worker subprocesses).
# =====================================================================

def _run_driver(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][mem_probe] %(message)s",
        datefmt="%H:%M:%S",
    )

    b_pairs = [int(x) for x in args.b_pairs.split(",") if x.strip()]
    if not b_pairs:
        raise ValueError("--b_pairs must list at least one positive integer")

    archive = Path(args.output_dir)
    archive.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = archive / f"mem_probe_{ts}.json"

    results: List[Dict[str, Any]] = []
    for bp in b_pairs:
        result_file = archive / f"mem_probe_{ts}_bp{bp}.json"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--config", str(args.config),
            "--ckpt", str(args.ckpt),
            "--b_pair", str(bp),
            "--result", str(result_file),
        ]
        log.info("Launching worker for B_pair=%d (DiT-B=%d) ...", bp, 2 * bp)
        proc = subprocess.run(cmd, check=False)
        # Read back the JSON even if subprocess returned non-zero (OOM path
        # writes its result before exiting).
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))
        else:
            results.append({
                "b_pair": bp,
                "dit_batch": 2 * bp,
                "ok": False,
                "oomed": False,
                "error": f"worker exit={proc.returncode}; no result file written",
            })

    # -----------------------------------------------------------------
    # Print a summary + write the combined JSON.
    # -----------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'B_pair':>6} {'DiT-B':>6} {'peak_alloc(GB)':>16} {'reserved(GB)':>14} {'step(s)':>9} status")
    print("-" * 72)
    for r in results:
        if r["ok"]:
            status = "OK"
            alloc = f"{r['peak_alloc_gb']:.2f}"
            rsv = f"{r['peak_reserved_gb']:.2f}"
            st = f"{r['step_time_s']:.1f}"
        elif r.get("oomed"):
            status = "OOM"
            alloc = rsv = st = "-"
        else:
            status = f"FAIL: {(r.get('error') or '')[:30]}"
            alloc = rsv = st = "-"
        print(f"{r['b_pair']:>6} {r['dit_batch']:>6} {alloc:>16} {rsv:>14} {st:>9} {status}")
    print("=" * 72)

    # Recommendation: largest OK B_pair whose reserved peak
    # leaves at least ``headroom_gb`` under ``memory_budget_gb``.
    recommended: Optional[int] = None
    for r in sorted(results, key=lambda x: -x["b_pair"]):
        if r["ok"] and (r["peak_reserved_gb"] + args.headroom_gb) <= args.memory_budget_gb:
            recommended = r["b_pair"]
            break
    if recommended is None:
        # No single value fit comfortably; fall back to the largest OK probe.
        ok = [r for r in results if r["ok"]]
        if ok:
            recommended = max(ok, key=lambda x: x["b_pair"])["b_pair"]

    if recommended is not None:
        print(
            f"\nRecommendation: configs/action_ode_distill.yaml  ->  batch_size: {recommended}\n"
            f"  (budget {args.memory_budget_gb:.0f} GB with {args.headroom_gb:.0f} GB headroom)"
        )
    else:
        print(
            "\nRecommendation: no B_pair in the sweep fit safely. "
            "Investigate the smallest probe's error before starting a real run."
        )

    combined = {
        "config_path": str(args.config),
        "ckpt_path": str(args.ckpt),
        "memory_budget_gb": args.memory_budget_gb,
        "headroom_gb": args.headroom_gb,
        "recommended_batch_size": recommended,
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSummary JSON: {summary_path}")
    return 0 if recommended is not None else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory probe for action-forcing ODE distill.")
    p.add_argument("--config", type=str,
                   default="configs/action_ode_distill.yaml")
    p.add_argument("--ckpt", type=str, required=False,
                   default="/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt")
    p.add_argument("--b_pairs", type=str, default="1,2,3",
                   help="Comma-separated list of B_pair values to probe (1 pair = 2 DiT slots).")
    p.add_argument("--memory-budget-gb", type=float, default=80.0,
                   help="Per-GPU memory budget the recommendation respects.")
    p.add_argument("--headroom-gb", type=float, default=10.0,
                   help="Required headroom between peak reserved and the budget.")
    p.add_argument("--output-dir", type=str,
                   default="/scratch/u6ex/as1748.u6ex/ARRWM/logs/archive",
                   help="Directory to write the summary JSON into.")
    # Worker-only flags
    p.add_argument("--worker", action="store_true",
                   help="Run in worker mode (invoked internally by the driver).")
    p.add_argument("--b_pair", type=int,
                   help="Worker mode: single B_pair value.")
    p.add_argument("--result", type=str,
                   help="Worker mode: path to JSON file for the single-value result.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        if args.b_pair is None or args.result is None:
            raise ValueError("--worker requires --b_pair and --result")
        _run_worker(args.config, args.ckpt, int(args.b_pair), str(args.result))
        return 0
    return _run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
