"""Standalone distilled-eval smoke test for the action-forcing student.

Validates the few-step student inference path against a saved checkpoint
*before* the same logic is wired into the training-loop evaluator. Single
GPU. Writes ``clean.mp4``, ``cf.mp4`` and ``sidebyside.mp4`` to disk and
prints action-aware diagnostics.

Usage (inside an interactive GPU shell)::

    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    python action-forcing/bin/smoke_eval.py \
        --ckpt logs/v14_balanced_weunz/causal_lora_step0006600.pt \
        --config configs/action_ode_distill.yaml \
        --pair-index 0 \
        --output /scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_eval

Acceptance checks on run:
  * Three non-empty mp4 files written.
  * ``teacher_z2/z7 clean vs cf`` differ (motion pipeline can distinguish
    the CF edit on the student's output).
  * ``pred_x0 cf_vs_clean mse`` > 0 (student responds to action edit).
  * No NaNs anywhere.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Make ``action-forcing/`` importable as top-level.
_THIS_DIR = Path(__file__).resolve().parents[1]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_WORKSPACE = _THIS_DIR.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))


log = logging.getLogger("smoke_eval")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distilled-eval smoke test.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to the *teacher* checkpoint (.pt). Must be a "
                        "v14-era file containing a 'lora' key — the base "
                        "DiT is initialised from this and then the LoRA is "
                        "merged in before any student-format overlay.")
    p.add_argument("--student-ckpt", type=str, default=None,
                   help="Optional path to a student-format checkpoint "
                        "(.pt) written by af_trainer.ode. Must contain a "
                        "'generator' key. When provided, its weights are "
                        "loaded on top of the teacher-merged base DiT + "
                        "heads so we can preview a mid-training snapshot. "
                        "Use this to render eval videos for any "
                        "action_ode_step*.pt or latest_*.pt file.")
    p.add_argument("--config", type=str,
                   default="configs/action_ode_distill.yaml",
                   help="YAML config (matches training).")
    p.add_argument("--pair-index", type=int, default=0,
                   help="Dataset pair index to evaluate on.")
    p.add_argument("--output", type=str,
                   default="/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_eval",
                   help="Output directory for mp4 files.")
    p.add_argument("--seed", type=int, default=0,
                   help="Manual-seed for noise sampling (shared between "
                        "clean and CF branches).")
    p.add_argument("--fps", type=int, default=5,
                   help="Frames-per-second for the output mp4s.")
    p.add_argument("--max-pair", type=int, default=8,
                   help="Dataset cap (only needs a handful).")
    return p.parse_args()


@torch.no_grad()
def decode_to_np(latents: torch.Tensor, frozen_vae) -> np.ndarray:
    """``[1, F, C, H, W]`` latents -> ``[F, H*s, W*s, 3]`` uint8 ndarray."""
    dummy = latents[:, 0:1]
    lat_wd = torch.cat([dummy, latents], dim=1)
    px = frozen_vae.decode_to_pixel(lat_wd.float())[:, 1:, ...]
    vid = (0.5 * (px.float() + 1.0)).clamp(0, 1)
    vid_np = (vid[0].cpu().numpy() * 255).astype(np.uint8)
    if vid_np.shape[-1] != 3:
        vid_np = vid_np.transpose(0, 2, 3, 1)  # [F, H, W, 3]
    return vid_np


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio  # old API is simpler for mimwrite
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Distilled eval requires a CUDA device.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # ------------------------------------------------------------------
    # Config + model
    # ------------------------------------------------------------------
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    # Point generator_ckpt at whatever the user asked to load.
    cfg.generator_ckpt = args.ckpt
    cfg.max_pair = int(args.max_pair)

    log.info("Building ODERegression on %s ...", device)
    from af_model.ode_regression import ODERegression
    model = ODERegression(cfg, device=device)
    model.eval()
    model.ensure_motion_pipeline(distributed=False)

    # Optional student-checkpoint overlay. The teacher init above gets
    # us a merged, full-rank DiT with initial action heads + probe +
    # critic. A student checkpoint written by ``af_trainer.ode`` has
    # the *same shapes* for every key (generator, action_projection,
    # action_token_projection, state_probe, action_critic) so we just
    # overwrite them. We refuse to load legacy ``{'lora': ...}`` blobs
    # here — that path belongs in ``load_teacher_checkpoint``.
    if args.student_ckpt:
        sp = Path(args.student_ckpt)
        if not sp.exists():
            raise FileNotFoundError(f"Student checkpoint not found: {sp}")
        sk = torch.load(sp, map_location="cpu", weights_only=False)
        if "generator" not in sk:
            raise RuntimeError(
                f"Student checkpoint {sp} is missing the 'generator' key; "
                "smoke_eval --student-ckpt expects the full-rank trainer "
                "format. Pass a v14 teacher ckpt via --ckpt if you want "
                "the LoRA-era path."
            )
        ck_cfg_name = sk.get("config_name")
        if ck_cfg_name is not None and str(ck_cfg_name) != str(cfg.config_name):
            log.warning(
                "Student ckpt config_name=%r != cfg.config_name=%r; "
                "proceeding but the overlay may have subtle shape mismatches.",
                ck_cfg_name, cfg.config_name,
            )
        model.generator.model.load_state_dict(sk["generator"], strict=True)
        if model.action_projection is not None and "action_projection" in sk:
            model.action_projection.load_state_dict(sk["action_projection"])
        if model.action_token_projection is not None and "action_token_projection" in sk:
            model.action_token_projection.load_state_dict(sk["action_token_projection"])
        if hasattr(model.generator, "_state_probe") and model.generator._state_probe is not None:
            if "state_probe" not in sk:
                raise RuntimeError(
                    f"Student ckpt {sp} is missing 'state_probe'; probe-mode "
                    "student requires it."
                )
            miss, unex = model.generator._state_probe.load_state_dict(
                sk["state_probe"], strict=False,
            )
            if miss or unex:
                raise RuntimeError(
                    f"state_probe load from {sp} mismatched: missing={list(miss)[:3]} "
                    f"unexpected={list(unex)[:3]}"
                )
        if model.action_critic is not None:
            if "action_critic" not in sk:
                raise RuntimeError(
                    f"Student ckpt {sp} is missing 'action_critic'; "
                    "critic-mode student requires it."
                )
            miss, unex = model.action_critic.load_state_dict(
                sk["action_critic"], strict=False,
            )
            if miss or unex:
                raise RuntimeError(
                    f"action_critic load from {sp} mismatched: missing={list(miss)[:3]} "
                    f"unexpected={list(unex)[:3]}"
                )
        model.to(device)
        model.eval()
        step_loaded = int(sk.get("step", -1))
        log.info("Loaded student overlay from %s (step=%d)", sp.name, step_loaded)

    dtype = model.dtype  # bf16 if mixed_precision else fp32

    # ------------------------------------------------------------------
    # Post-load assertions: the teacher LoRA must be fully merged and
    # the base DiT fully trainable (this is the invariant the full-rank
    # path relies on).
    # ------------------------------------------------------------------
    from peft.tuners.lora import LoraLayer
    residual = [
        n for n, m in model.generator.model.named_modules()
        if isinstance(m, LoraLayer)
    ]
    assert not residual, (
        f"Found residual LoraLayer modules after load — "
        f"merge_and_unload() did not strip them: {residual[:5]}"
    )
    n_trainable_dit = sum(
        p.numel() for p in model.generator.model.parameters() if p.requires_grad
    )
    log.info(
        "DiT trainable params: %.2fM (expect ~1300M for Wan2.1-T2V-1.3B).",
        n_trainable_dit / 1e6,
    )
    assert n_trainable_dit > 1e9, (
        f"Expected the full DiT (~1.3B params) to be trainable after "
        f"merge_and_unload(); got only {n_trainable_dit} trainable DiT "
        f"parameters. Something failed the merge."
    )

    # ------------------------------------------------------------------
    # Dataset + pair selection
    # ------------------------------------------------------------------
    from af_utils.dataset import PairedTrajectoryDataset
    ds = PairedTrajectoryDataset(
        clean_root=str(cfg.clean_root),
        cf_root=str(cfg.cf_root),
        caption_root=str(cfg.caption_root),
        max_pair=int(cfg.max_pair) or None,
        require_cf=bool(getattr(cfg, "require_cf", True)),
    )
    idx = int(args.pair_index) % len(ds)
    pair = ds[idx]
    log.info("Pair index=%d filename=%s", idx, pair["meta"]["filename"])

    # ------------------------------------------------------------------
    # Prepare tensors (bf16-cast inputs; fp32 noise)
    # ------------------------------------------------------------------
    def _add_batch(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).to(device)

    prompt_embeds = _add_batch(pair["prompt_embeds"])
    z_clean = _add_batch(pair["z_clean"])
    z_noisy = _add_batch(pair["z_noisy"])       # real-ride actions
    z_noisy_cf = _add_batch(pair["z_noisy_cf"])  # CF-edited actions
    clean_x = _add_batch(pair["clean_x_gt"])     # teacher-forced context
    B, F_, C, H, W = clean_x.shape
    assert B == 1 and F_ == 21, (B, F_)

    log.info(
        "Shapes: clean_x=%s z_clean=%s z_noisy=%s z_noisy_cf=%s prompt=%s",
        tuple(clean_x.shape), tuple(z_clean.shape),
        tuple(z_noisy.shape), tuple(z_noisy_cf.shape),
        tuple(prompt_embeds.shape),
    )

    # ------------------------------------------------------------------
    # Four-stream conditionals — teacher-parity routing.
    #   * clean branch uses z_noisy for the noisy-window action stream.
    #   * CF branch uses z_noisy_cf for the noisy-window action stream.
    #   * z_clean is identical in both (preceding context is the same).
    # ------------------------------------------------------------------
    cond_clean = model._build_conditional(
        prompt_embeds.to(dtype), z_noisy.to(dtype), z_clean.to(dtype), num_frames=F_,
    )
    cond_cf = model._build_conditional(
        prompt_embeds.to(dtype), z_noisy_cf.to(dtype), z_clean.to(dtype), num_frames=F_,
    )

    # ------------------------------------------------------------------
    # Paired noise sampling: shared seed so clean vs CF diff is purely
    # the action edit.
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    noise = torch.randn([B, F_, C, H, W], dtype=torch.float32, device=device)

    log.info("Running generate_eval on clean branch ...")
    gen_clean = model.generate_eval(cond_clean, clean_x, noise)

    log.info("Running generate_eval on CF branch ...")
    torch.manual_seed(args.seed)  # same seed ⇒ identical starting noise
    gen_cf = model.generate_eval(cond_cf, clean_x, noise)

    # ------------------------------------------------------------------
    # NaN / finiteness check.
    # ------------------------------------------------------------------
    def _fstat(name: str, x: torch.Tensor) -> None:
        f = x.float()
        log.info(
            "%s: shape=%s  mean=%.4f std=%.4f min=%.4f max=%.4f  nan=%s",
            name, tuple(f.shape), float(f.mean()), float(f.std()),
            float(f.min()), float(f.max()), bool(torch.isnan(f).any()),
        )
    _fstat("gen_clean", gen_clean)
    _fstat("gen_cf", gen_cf)

    # ------------------------------------------------------------------
    # Decode + side-by-side mp4
    # ------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    vid_clean = decode_to_np(gen_clean, model._frozen_vae)
    vid_cf = decode_to_np(gen_cf, model._frozen_vae)
    vid_sbs = np.concatenate([vid_clean, vid_cf], axis=2)  # concat along width

    _write_mp4(out_dir / "clean.mp4", vid_clean, fps=args.fps)
    _write_mp4(out_dir / "cf.mp4", vid_cf, fps=args.fps)
    _write_mp4(out_dir / "sidebyside.mp4", vid_sbs, fps=args.fps)
    log.info("Wrote %s / clean.mp4, cf.mp4, sidebyside.mp4", out_dir)

    # ------------------------------------------------------------------
    # Motion-pipeline diagnostics on the *generated* videos
    # ------------------------------------------------------------------
    tz_clean = model._compute_action_teacher_targets(gen_clean)
    tz_cf = model._compute_action_teacher_targets(gen_cf)
    d0, d1 = int(model.action_critic_dims[0]), int(model.action_critic_dims[1])

    print("\n=== Diagnostics ===")
    print(f"random_steps         : {model.random_steps}")
    print(f"denoising_step_list  : {[round(float(x),2) for x in model.denoising_step_list.tolist()]}")
    print(f"teacher_z{d0} clean  : {tz_clean[:, :, d0].mean().item():+.4f}")
    print(f"teacher_z{d0} cf     : {tz_cf[:, :, d0].mean().item():+.4f}")
    print(f"teacher_z{d1} clean  : {tz_clean[:, :, d1].mean().item():+.4f}")
    print(f"teacher_z{d1} cf     : {tz_cf[:, :, d1].mean().item():+.4f}")
    print(f"pred_x0 CF-vs-clean mse : {F.mse_loss(gen_clean, gen_cf).item():.6f}")
    print(f"teacher_z CF-vs-clean mse: {F.mse_loss(tz_clean, tz_cf).item():.6f}")

    # ------------------------------------------------------------------
    # Packed-vs-sequential DiT parity probe.
    #
    # The production path is always the packed B=2 forward (one DiT
    # call on ``torch.cat([clean_slot, cf_slot], dim=0)``). This probe
    # reconstructs a throwaway two-pass reference inline (two sequential
    # B=1 DiT forwards with the same input tensors and the same
    # ``_build_conditional`` tensors) and asserts their per-slot
    # outputs agree with the packed outputs within bf16 noise.
    # Catches any silent conditional-stacking / batching bug in
    # ``generator_loss`` without leaving a fallback in production code.
    # ------------------------------------------------------------------
    log.info("Running packed-vs-sequential DiT parity probe ...")

    # Freeze all RNG for the probe so the two paths see the same
    # dropout draws / gradient-checkpoint noise (the DiT should be in
    # ``eval()`` mode already, so dropout is zero, but this belts-and-
    # braces it).
    torch.manual_seed(args.seed + 123)

    noise_slot = torch.randn(
        [1, F_, C, H, W], dtype=torch.float32, device=device,
    )
    clean_x_slot = clean_x.to(dtype)

    # Packed conditional and a single B=2 forward.
    prompt_pack  = torch.cat([prompt_embeds.to(dtype),  prompt_embeds.to(dtype)],  dim=0)
    clean_x_pack = torch.cat([clean_x_slot,             clean_x_slot],             dim=0)
    z_noisy_pack = torch.cat([z_noisy.to(dtype),        z_noisy_cf.to(dtype)],     dim=0)
    z_clean_pack = torch.cat([z_clean.to(dtype),        z_clean.to(dtype)],        dim=0)
    noisy_pack   = torch.cat([noise_slot.to(dtype),     noise_slot.to(dtype)],     dim=0)
    # Timestep: pick the mid-pool level so we're not at t=0 (which
    # collapses to identity and would give vacuous parity).
    t_mid = float(model.denoising_step_list[min(1, model.denoising_step_list.shape[0] - 1)].item())
    t_pack = torch.full((2, F_), t_mid, device=device, dtype=torch.float32)

    cond_pack = model._build_conditional(
        prompt_pack, z_noisy_pack, z_clean_pack, num_frames=F_,
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            out_pack = model.generator(
                noisy_image_or_video=noisy_pack,
                conditional_dict=cond_pack,
                timestep=t_pack,
                clean_x=clean_x_pack,
                aug_t=None,
            )
        pred_pack = out_pack[1]  # pred_x0

    # Sequential reference: two B=1 calls with independently-built
    # conditional dicts (same inputs).
    cond_clean_seq = model._build_conditional(
        prompt_embeds.to(dtype), z_noisy.to(dtype),    z_clean.to(dtype), num_frames=F_,
    )
    cond_cf_seq = model._build_conditional(
        prompt_embeds.to(dtype), z_noisy_cf.to(dtype), z_clean.to(dtype), num_frames=F_,
    )
    t_seq = torch.full((1, F_), t_mid, device=device, dtype=torch.float32)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            out_seq_clean = model.generator(
                noisy_image_or_video=noise_slot.to(dtype),
                conditional_dict=cond_clean_seq,
                timestep=t_seq,
                clean_x=clean_x_slot,
                aug_t=None,
            )
            out_seq_cf = model.generator(
                noisy_image_or_video=noise_slot.to(dtype),
                conditional_dict=cond_cf_seq,
                timestep=t_seq,
                clean_x=clean_x_slot,
                aug_t=None,
            )
        pred_seq_clean = out_seq_clean[1]
        pred_seq_cf = out_seq_cf[1]

    # Compare on both absolute-max and relative-RMS scales. Flex-attention
    # under bf16 accumulates matmul reductions in a tile-size / batch-size
    # dependent order, so the per-element absolute delta between packed
    # and sequential can easily reach a few percent of the output scale
    # even though both runs are numerically correct. A relative-RMS
    # metric (Δ-RMS / pred-RMS) is a more honest check because it
    # normalises out the output magnitude and is robust to the per-pixel
    # outliers that dominate max|Δ|.
    ref_clean = pred_seq_clean.float()
    ref_cf = pred_seq_cf.float()
    err_clean = pred_pack[0:1].float() - ref_clean
    err_cf = pred_pack[1:2].float() - ref_cf
    diff_clean = err_clean.abs().max().item()
    diff_cf = err_cf.abs().max().item()
    rel_rms_clean = (err_clean.pow(2).mean().sqrt().item()
                     / max(ref_clean.pow(2).mean().sqrt().item(), 1e-6))
    rel_rms_cf = (err_cf.pow(2).mean().sqrt().item()
                  / max(ref_cf.pow(2).mean().sqrt().item(), 1e-6))
    print(f"\n=== Packed-vs-sequential DiT parity ===")
    print(f"  max|Δ| clean slot: {diff_clean:.3e}   rel-RMS: {rel_rms_clean:.3e}")
    print(f"  max|Δ| cf slot   : {diff_cf:.3e}   rel-RMS: {rel_rms_cf:.3e}")
    # bf16 flex-attention + large hidden-dim MLPs can drift by ~3e-2
    # rel-RMS; fp32 shuts everything down to ~1e-5. A real conditional
    # stacking bug would show either a systematic per-slot bias
    # (rel-RMS near 1.0) or huge absolute deltas (max|Δ| >> output std).
    if dtype == torch.bfloat16:
        rel_tol, abs_tol = 5e-2, 5e-1
    else:
        rel_tol, abs_tol = 1e-4, 1e-3
    assert rel_rms_clean < rel_tol and diff_clean < abs_tol, (
        f"Packed vs sequential DiT disagree on clean slot: max|Δ|={diff_clean:.3e} "
        f"(abs_tol {abs_tol:.1e}), rel-RMS={rel_rms_clean:.3e} (rel_tol {rel_tol:.1e}). "
        "Probable bug in conditional stacking."
    )
    assert rel_rms_cf < rel_tol and diff_cf < abs_tol, (
        f"Packed vs sequential DiT disagree on cf slot: max|Δ|={diff_cf:.3e} "
        f"(abs_tol {abs_tol:.1e}), rel-RMS={rel_rms_cf:.3e} (rel_tol {rel_tol:.1e}). "
        "Probable bug in conditional stacking."
    )

    # ------------------------------------------------------------------
    # Batched-vs-serial CoTracker parity probe.
    #
    # CoTracker itself cannot safely take B>1 (its internal
    # ``coords.view(B*T, N, 2)`` fails on non-contiguous strided batches),
    # so ``_compute_action_teacher_targets`` loops B samples through
    # CoTracker one at a time and then runs ss_vae as a single batched
    # encoder call. Assert that passing [B=2, T, C, H, W] into this
    # function gives the same per-sample result as two sequential B=1
    # calls (both operate at fp32 under no_grad after autocast, so the
    # delta should be essentially zero; the small tol below just guards
    # against bf16 autocast non-determinism inside CoTracker).
    # ------------------------------------------------------------------
    log.info("Running batched-vs-serial CoTracker parity probe ...")
    tz_batched = model._compute_action_teacher_targets(
        torch.cat([gen_clean, gen_cf], dim=0)
    )                                                          # [2, n_c, 8]
    tz_clean_serial = model._compute_action_teacher_targets(gen_clean)  # [1, n_c, 8]
    tz_cf_serial = model._compute_action_teacher_targets(gen_cf)        # [1, n_c, 8]
    diff_clean_ct = (tz_batched[0:1].float() - tz_clean_serial.float()).abs().max().item()
    diff_cf_ct = (tz_batched[1:2].float() - tz_cf_serial.float()).abs().max().item()
    print(f"\n=== Batched-vs-serial CoTracker parity ===")
    print(f"  max|Δ| clean slot: {diff_clean_ct:.3e}")
    print(f"  max|Δ| cf slot   : {diff_cf_ct:.3e}")
    ct_tol = 1e-3  # CoTracker runs in autocast(bf16) internally; give it a little slack
    assert diff_clean_ct < ct_tol, (
        f"Batched vs serial CoTracker disagree on clean slot by "
        f"{diff_clean_ct:.3e} (tol {ct_tol:.1e})."
    )
    assert diff_cf_ct < ct_tol, (
        f"Batched vs serial CoTracker disagree on cf slot by "
        f"{diff_cf_ct:.3e} (tol {ct_tol:.1e})."
    )

    # ------------------------------------------------------------------
    # Critic diagnostics
    # ------------------------------------------------------------------
    if model.action_critic is not None:
        n_c = model.n_chunks
        chunk_actions_clean = (
            z_clean.reshape(B, n_c, model.num_frame_per_block, -1).mean(dim=2)
        )  # commanded z per chunk (clean)
        chunk_t = torch.zeros(B, n_c, device=device, dtype=torch.float32)
        pz_clean = model.action_critic(gen_clean, chunk_t, chunk_actions_clean)[:, :n_c]
        pz_cf = model.action_critic(gen_cf, chunk_t, chunk_actions_clean)[:, :n_c]
        print("\n=== Critic ===")
        print(f"critic_z{d0} clean vs teacher : {F.mse_loss(pz_clean[..., d0], tz_clean[..., d0]).item():.6f}")
        print(f"critic_z{d0} cf    vs teacher : {F.mse_loss(pz_cf[..., d0],    tz_cf[..., d0]).item():.6f}")
        print(f"critic_z{d1} clean vs teacher : {F.mse_loss(pz_clean[..., d1], tz_clean[..., d1]).item():.6f}")
        print(f"critic_z{d1} cf    vs teacher : {F.mse_loss(pz_cf[..., d1],    tz_cf[..., d1]).item():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
