"""Sweep F (real_score with clean_x_GT) across many rides + ride offsets to
calibrate ``teacher_freeze_threshold``.

The freeze detector flags frames where ``per_frame_mae > threshold *
median(per_frame_mae)``. The right threshold should:
  * NOT flag the structural last-``npb`` frames (RoPE OOD boundary; their
    MAE is ~1.5–2x median; already gradient-masked so flagging is
    redundant but not harmful).
  * NOT flag legitimate motion (intra-window dynamics produce some MAE
    spread; threshold must be above the typical max-vs-median ratio).
  * DO flag actual teacher-freeze frames (= washed-out / static frames
    the user observed in F_real_GT.mp4).

This script runs F on N rides × M offsets, logs per-frame MAE + the
per-window ratio (max / median, p95 / median), saves an mp4 per
(ride, offset) so the user can visually identify freeze frames and
cross-reference the MAE ratio. Aggregate stats at the end suggest a
threshold.

Usage:
    DIAG_FREEZE_NUM_RIDES=8 DIAG_FREEZE_OFFSETS_PER_RIDE=2 \\
        conda run -n flash python _diag_p1/freeze_threshold_calibration.py
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("freeze_calib")


def _init_dist_singlerank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    import socket, random
    for _ in range(50):
        port = 30000 + random.randint(0, 30000)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            os.environ.setdefault("MASTER_PORT", str(port))
            break
        except OSError:
            s.close()
    else:
        os.environ.setdefault("MASTER_PORT", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


def _write_mp4(path: Path, frames_uint8: np.ndarray, fps: int = 20) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames_uint8, fps=fps, codec="libx264", quality=8)


@torch.no_grad()
def _decode_to_uint8(latents: torch.Tensor, vae) -> np.ndarray:
    dummy = latents[:, 0:1]
    lat_wd = torch.cat([dummy, latents], dim=1).float()
    px = vae.decode_to_pixel(lat_wd)[:, 1:, ...]
    vid = (0.5 * (px.float() + 1.0)).clamp(0, 1)
    arr = (vid[0].cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[-1] != 3:
        arr = arr.transpose(0, 2, 3, 1)
    return arr


def _build_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_REPO / "configs" / "action_forcing_phase1_aux_dmdctx.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.wan_model_path = "/home/ashish/Wan2.1/"
    cfg.encoded_root = "/home/ashish/frodobots/frodobots_encoded"
    cfg.caption_root = "/home/ashish/frodobots/frodobots_captions/train"
    cfg.motion_root = "/home/ashish/frodobots/frodobots_motion"
    cfg.ss_vae_checkpoint = "/home/ashish/ARRWM/action_query/checkpoints/ss_vae_8free.pt"
    cfg.ode_generator_checkpoint = "/home/ashish/action_ode_step0001000.pt"
    cfg.v14_teacher_checkpoint = "/home/ashish/Downloads/causal_lora_step0006600.pt"
    cfg.cotracker_checkpoint_path = ""
    cfg.cotracker_source_dir = ""
    cfg.action_teacher_mode = "off"
    cfg.gan_enabled = False
    cfg.sc_dmd_enabled = False
    cfg.mae_extension_max_extra_chunks = 0
    cfg.mae_extension_threshold = None
    cfg.last_step_only = False
    cfg.gradient_checkpointing = False
    cfg.mixed_precision = True
    cfg.text_pre_encoded = True
    cfg.dmd_context = "GT"
    cfg.clean_x_aug_t = 0
    if hasattr(cfg, "dmd_real_GT"):
        del cfg.dmd_real_GT
    if hasattr(cfg, "dmd_real_GT_aug_t"):
        del cfg.dmd_real_GT_aug_t
    return cfg


def main():
    _init_dist_singlerank()
    import utils.wan_wrapper as _ww
    _ww._default_wan_model_path = "/home/ashish/Wan2.1/"

    cfg = _build_config()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    num_rides = int(os.environ.get("DIAG_FREEZE_NUM_RIDES", "8"))
    offsets_per_ride = int(os.environ.get("DIAG_FREEZE_OFFSETS_PER_RIDE", "2"))
    seed = int(os.environ.get("DIAG_FREEZE_SEED", "42"))

    log.info("Sweep config: rides=%d offsets/ride=%d seed=%d",
             num_rides, offsets_per_ride, seed)

    log.info("Building ActionForcingDMD ...")
    from model.dmd_action_forcing import ActionForcingDMD
    model = ActionForcingDMD(args=cfg, device=device)
    model.generator.model.to(device=device, dtype=dtype)
    model.fake_score.model.to(device=device, dtype=dtype)
    model.real_score.model.to(device=device, dtype=dtype)
    if model.action_projection is not None:
        model.action_projection.to(device=device, dtype=dtype)
    if model.action_token_projection is not None:
        model.action_token_projection.to(device=device, dtype=dtype)
    if getattr(model, "vae", None) is not None:
        try:
            model.vae.to(device=device)
        except AttributeError:
            inner_vae = getattr(model.vae, "model", None)
            if inner_vae is not None:
                inner_vae.to(device=device)
    model.eval()

    from pipeline.action_forcing_training import ActionForcingTrainingPipeline
    pipe = ActionForcingTrainingPipeline(
        denoising_step_list=list(cfg.denoising_step_list),
        scheduler=model.scheduler,
        generator=model.generator,
        num_frame_per_block=int(cfg.num_frame_per_block),
        chunks_per_rolling_step=int(cfg.chunks_per_rolling_step),
        same_step_across_blocks=bool(cfg.same_step_across_blocks),
        last_step_only=bool(cfg.last_step_only),
        num_max_frames=int(cfg.num_training_frames),
        rollout_frames=int(getattr(cfg, "rollout_frames", cfg.num_training_frames)),
        mae_extension_threshold=None,
        mae_extension_max_extra_chunks=0,
        context_noise=int(cfg.context_noise),
    )
    model.inference_pipeline = pipe

    if bool(getattr(cfg, "infinity_rope", True)):
        from utils.infinity_rope import install as _install
        _install(model.generator.model)

    cf = int(cfg.dmd_context_clean_frames)
    N = int(cfg.num_training_frames)
    npb = int(cfg.num_frame_per_block)

    # Load up to ``num_rides`` rides via the small-scan dataset.
    log.info("Scanning %d rides ...", num_rides)
    from utils.zarr_dataset import ZarrRideDataset
    ds = ZarrRideDataset(
        encoded_root=cfg.encoded_root,
        caption_root=cfg.caption_root,
        motion_root=cfg.motion_root,
        ss_vae_checkpoint=cfg.ss_vae_checkpoint,
        min_ride_frames=cf + N + 200,  # need headroom for several offsets
        device="cpu",
        ss_vae_device="cuda:0",
        max_rides=num_rides,
    )
    if len(ds) == 0:
        raise RuntimeError("No rides found.")

    out_dir = _REPO / "_diag_p1" / "out" / "freeze_calib"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    aug_t_gt = torch.zeros(1, N, dtype=torch.long, device=device)  # aug_t=0

    # Per-frame stats accumulator. ``ratios`` collects per-frame
    # (mae / median_mae) values across all (ride, offset) windows so
    # we can compute global percentiles for the threshold.
    all_ratios = []          # per-frame ratio across all runs
    boundary_ratios = []     # per-frame ratio for the LAST npb frames
    middle_ratios = []       # per-frame ratio for the MIDDLE frames (= excludes last npb)
    per_run_summaries = []   # one tuple per (ride, offset)

    rides_processed = 0
    for ride_idx, (zpath, prompt_embeds, attrs, n_lat) in enumerate(ds._rides):
        if rides_processed >= num_rides:
            break
        log.info("=" * 60)
        log.info("Ride %d/%d: %s n_lat=%d", ride_idx + 1, num_rides, zpath.name, n_lat)
        action_dims = list(getattr(cfg, "action_dims", [2, 7]))

        # Sample offsets_per_ride distinct ride starts.
        max_s = max(0, n_lat - cf - N - 1)
        if max_s <= 0:
            continue
        offsets = rng.integers(0, max_s, size=offsets_per_ride).tolist()

        for off_idx, s in enumerate(offsets):
            need = s + cf + N
            latents_full = ZarrRideDataset.load_latent_chunk(
                str(zpath), 0, need
            ).unsqueeze(0).to(device=device, dtype=dtype)
            z_actions_full = ds.encode_z_actions_window(str(zpath), need, 0, need)
            z_actions_full = z_actions_full[..., action_dims].unsqueeze(0).to(
                device=device, dtype=dtype,
            )
            latents = latents_full[:, s:s + cf + N]
            z_actions = z_actions_full[:, s:s + cf + N]
            pe = prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype) if prompt_embeds.dim() == 2 else prompt_embeds.to(device=device, dtype=dtype)

            # GT slices (single 21-frame window; no anchor — we run F
            # ONE-SHOT against a 21-frame noisy_x and a 21-frame
            # clean_x_GT shifted back by npb. Standalone scorer call,
            # not a multi-batch rollout, so no anchor needed).
            seed_latents = latents[:, :cf]
            gt_clean = latents[:, cf - npb : cf - npb + N]   # ride[s+cf-npb : s+cf+N-npb]
            gt_noisy = latents[:, cf : cf + N]                # ride[s+cf       : s+cf+N    ]
            gt_clean_act = z_actions[:, cf - npb : cf - npb + N]
            gt_noisy_act = z_actions[:, cf : cf + N]

            cond_noisy, _ = model.build_action_conditional(
                prompt_embeds=pe, gt_actions=gt_noisy_act,
            )
            cond_clean, _ = model.build_action_conditional(
                prompt_embeds=pe, gt_actions=gt_clean_act,
            )
            # Merge clean streams.
            cond_for_F = dict(cond_noisy)
            for k, v in cond_clean.items():
                if k in ("_action_modulation", "_action_tokens"):
                    cond_for_F[k + "_clean"] = v

            # Roll the student to get a noisy_x source. We re-noise the
            # GT_noisy directly (skip the student rollout) — the freeze
            # behaviour is a property of real_score's forward, not the
            # student. This makes the script ride-agnostic + faster.
            torch.manual_seed(seed * 1000 + ride_idx * 10 + off_idx)
            timestep = torch.full(
                (1, N), fill_value=500, dtype=torch.long, device=device,  # mid-range t
            )
            noise_for_dmd = torch.randn_like(gt_noisy)
            noisy_input = model.scheduler.add_noise(
                gt_noisy.flatten(0, 1),
                noise_for_dmd.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, (1, N))

            # F: real_score with GT clean_x at aug_t=0.
            with torch.no_grad():
                _, pred_F = model.real_score(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=cond_for_F,
                    timestep=timestep,
                    clean_x=gt_clean.to(dtype=dtype),
                    aug_t=aug_t_gt,
                )

            # Per-frame MAE between F's pred and GT.
            per_frame_mae = (
                pred_F.float() - gt_noisy.float()
            ).abs().mean(dim=[2, 3, 4]).cpu().numpy()  # [1, N]
            mae_vec = per_frame_mae[0]  # [N]
            median = float(np.median(mae_vec))
            ratio = mae_vec / max(median, 1e-9)
            mx = float(ratio.max())
            p95 = float(np.percentile(ratio, 95))

            # Boundary vs middle.
            boundary_ratios.extend(ratio[-npb:].tolist())
            middle_ratios.extend(ratio[:-npb].tolist())
            all_ratios.extend(ratio.tolist())

            # Find which frame index has the max ratio.
            argmax_frame = int(ratio.argmax())
            log.info(
                "  ride=%s off=%d s=%d  median_mae=%.4f  max=%.2fx (frame %d)  "
                "p95=%.2fx  per-frame MAE: %s",
                zpath.name, off_idx, s, median, mx, argmax_frame, p95,
                " ".join(f"{v:.3f}" for v in mae_vec),
            )
            log.info(
                "                                            per-frame ratio: %s",
                " ".join(f"{v:.2f}x" for v in ratio),
            )
            per_run_summaries.append({
                "ride": zpath.name,
                "off": off_idx,
                "s": int(s),
                "median": median,
                "max_ratio": mx,
                "max_frame": argmax_frame,
                "p95_ratio": p95,
                "per_frame_mae": mae_vec.tolist(),
                "per_frame_ratio": ratio.tolist(),
            })

            # Save mp4 (F + GT side-by-side as separate files).
            stem = f"r{ride_idx:02d}_o{off_idx}_s{s:04d}_max{mx:.2f}x_f{argmax_frame:02d}"
            _write_mp4(out_dir / f"F_{stem}.mp4", _decode_to_uint8(pred_F, model.vae))
            _write_mp4(out_dir / f"GT_{stem}.mp4", _decode_to_uint8(gt_noisy, model.vae))

            # Free this offset's tensors.
            del pred_F, noisy_input, gt_clean, gt_noisy, cond_for_F, cond_noisy, cond_clean
            torch.cuda.empty_cache()

        rides_processed += 1

    # Aggregate.
    log.info("=" * 60)
    log.info("AGGREGATE STATS over %d (ride, offset) windows = %d frames total",
             len(per_run_summaries), len(all_ratios))
    arr_all = np.asarray(all_ratios)
    arr_mid = np.asarray(middle_ratios)
    arr_bnd = np.asarray(boundary_ratios)

    def _stats(name, a):
        if a.size == 0:
            log.info("  %-12s (empty)", name)
            return
        log.info(
            "  %-12s n=%d  mean=%.2fx  median=%.2fx  p90=%.2fx  p95=%.2fx  "
            "p99=%.2fx  max=%.2fx",
            name, a.size, a.mean(), float(np.median(a)),
            float(np.percentile(a, 90)), float(np.percentile(a, 95)),
            float(np.percentile(a, 99)), a.max(),
        )

    _stats("all frames", arr_all)
    _stats("middle (no last npb)", arr_mid)
    _stats("last npb (boundary)", arr_bnd)

    # Sort runs by max_ratio descending so user sees suspected freezes first.
    log.info("=" * 60)
    log.info("Top 10 windows by max_ratio (likely freeze candidates):")
    top = sorted(per_run_summaries, key=lambda r: -r["max_ratio"])[:10]
    for r in top:
        log.info(
            "  max_ratio=%.2fx at frame %2d  median=%.4f  ride=%s off=%d s=%d",
            r["max_ratio"], r["max_frame"], r["median"], r["ride"], r["off"], r["s"],
        )

    # Suggested threshold: choose a value that flags the top ~5% of
    # frames in the MIDDLE region (= excludes the structural boundary,
    # which is already gradient-masked). p95 of the middle is the
    # natural "above this is anomalous" boundary; the user should
    # cross-reference top-ranked windows visually before locking in.
    log.info("=" * 60)
    log.info(
        "SUGGESTED teacher_freeze_threshold (excluding structural last-%d boundary):",
        npb,
    )
    log.info("  p95(middle) = %.2fx  (current default = 2.00x)", float(np.percentile(arr_mid, 95)) if arr_mid.size else float("nan"))
    log.info("  p99(middle) = %.2fx", float(np.percentile(arr_mid, 99)) if arr_mid.size else float("nan"))
    log.info("Pick a value just above p95-p99 of MIDDLE; verify by inspecting "
             "the top-ratio mp4s in %s for visible freeze frames.", out_dir)

    log.info("DONE. Videos in %s", out_dir)


if __name__ == "__main__":
    main()
