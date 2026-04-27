"""Scan a handful of (ride, offset) combos for the lowest cos(R, S)
window. Used to find a candidate for ``test_freeze_action_mode.py``
to visualise.

Reuses the freeze_action diag's model + teacher build but only computes
the cosine similarity (no decode / mp4 / overlay). Reports the top-k
worst-cos windows so the caller can pick one and pass via DIAG_RIDE_NAME
+ DIAG_RIDE_START to the full diag.
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
log = logging.getLogger("scan_freeze")

# Reuse build helpers from the full diag.
from _diag_p1.test_freeze_action_mode import (
    _init_dist_singlerank,
    _build_config,
    _build_action_teacher_for_diag,
    _make_teacher_z_fn,
)


def main():
    _init_dist_singlerank()
    import utils.wan_wrapper as _ww
    _ww._default_wan_model_path = "/home/ashish/Wan2.1/"

    cfg = _build_config()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    log.info("Building model + teacher ...")
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

    cot, ss_vae, ss_vae_scale = _build_action_teacher_for_diag(device)
    npb = int(cfg.num_frame_per_block)
    teacher_z_fn = _make_teacher_z_fn(model, cot, ss_vae, ss_vae_scale, npb)
    model._action_teacher_fn = teacher_z_fn

    from pipeline.action_forcing_training import ActionForcingTrainingPipeline
    pipe = ActionForcingTrainingPipeline(
        denoising_step_list=list(cfg.denoising_step_list),
        scheduler=model.scheduler, generator=model.generator,
        num_frame_per_block=int(cfg.num_frame_per_block),
        chunks_per_rolling_step=int(cfg.chunks_per_rolling_step),
        same_step_across_blocks=bool(cfg.same_step_across_blocks),
        last_step_only=bool(cfg.last_step_only),
        num_max_frames=int(cfg.num_training_frames),
        rollout_frames=int(getattr(cfg, "rollout_frames", cfg.num_training_frames)),
        mae_extension_threshold=None, mae_extension_max_extra_chunks=0,
        context_noise=int(cfg.context_noise),
    )
    model.inference_pipeline = pipe

    if bool(getattr(cfg, "infinity_rope", True)):
        from utils.infinity_rope import install as _install
        _install(model.generator.model)

    cf = int(cfg.dmd_context_clean_frames)
    N = int(cfg.num_training_frames)

    # Scan: pull `num_rides` zarrs, sample `offsets_per_ride` random
    # starts each, compute cos(R, S) per slot for ONE 21-frame window
    # at each (ride, offset). Report top-k worst.
    num_rides = int(os.environ.get("DIAG_SCAN_RIDES", "10"))
    offsets_per_ride = int(os.environ.get("DIAG_SCAN_OFFSETS_PER_RIDE", "3"))
    seed = int(os.environ.get("DIAG_SCAN_SEED", "0"))
    rng = np.random.default_rng(seed)

    from utils.zarr_dataset import ZarrRideDataset
    # Two modes:
    #   * DIAG_SCAN_RIDE set -> single-ride deep scan (sweep many
    #     offsets across one ride to find the worst-cos sub-window).
    #   * else -> multi-ride wide scan (one offset cluster per ride).
    single_ride = os.environ.get("DIAG_SCAN_RIDE")
    if single_ride:
        from utils.eval_chain import _build_ts_to_ride_dir, _load_ride_entry_from_disk
        ts_map = _build_ts_to_ride_dir(Path(cfg.caption_root))
        ride_dict = _load_ride_entry_from_disk(
            single_ride, Path(cfg.encoded_root), Path(cfg.caption_root), ts_map,
        )
        ds = ZarrRideDataset.from_manifest(
            rides_data=[ride_dict], motion_root=cfg.motion_root,
            ss_vae_checkpoint=cfg.ss_vae_checkpoint,
            device="cpu", ss_vae_device="cuda:0",
        )
        log.info("Single-ride scan: %s", single_ride)
    else:
        log.info("Indexing %d rides ...", num_rides)
        ds = ZarrRideDataset(
            encoded_root=cfg.encoded_root,
            caption_root=cfg.caption_root,
            motion_root=cfg.motion_root,
            ss_vae_checkpoint=cfg.ss_vae_checkpoint,
            min_ride_frames=cf + N + 100,
            device="cpu", ss_vae_device="cuda:0",
            max_rides=num_rides,
        )
        if len(ds) == 0:
            raise RuntimeError("No rides found.")
        log.info("Loaded %d rides", len(ds._rides))

    aug_t_zero = torch.zeros(1, N, dtype=torch.long, device=device)
    timestep = torch.full(
        (1, N), fill_value=int(os.environ.get("DIAG_F_T", "500")),
        dtype=torch.long, device=device,
    )

    findings = []  # (cos_min, cos_mean, ride_name, s, slot_argmin)

    for ride_idx, (zpath, prompt_embeds, attrs, n_lat) in enumerate(ds._rides):
        max_s = max(0, n_lat - cf - N - 1)
        if max_s <= 0:
            continue
        if single_ride:
            # Dense sweep: sample evenly-spaced offsets covering the
            # whole ride. Each window is N=21 latents (~4s) so step
            # by N to get non-overlapping windows.
            step = max(N, 1)
            offs = list(range(0, max_s, step))
            log.info(
                "single-ride deep scan: ride=%s n_lat=%d -> %d windows "
                "(every %d latent frames)",
                zpath.name, n_lat, len(offs), step,
            )
        else:
            offs = rng.integers(0, max_s, size=offsets_per_ride).tolist()
        for off in offs:
            need = off + cf + N
            latents = ZarrRideDataset.load_latent_chunk(
                str(zpath), 0, need
            ).unsqueeze(0).to(device=device, dtype=dtype)
            z_acts = ds.encode_z_actions_window(str(zpath), need, 0, need)
            ad = list(getattr(cfg, "action_dims", [2, 7]))
            z_acts = z_acts[..., ad].unsqueeze(0).to(device=device, dtype=dtype)
            latents = latents[:, off:off + cf + N]
            z_acts = z_acts[:, off:off + cf + N]
            pe = prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype) if prompt_embeds.dim() == 2 else prompt_embeds.to(device=device, dtype=dtype)

            seed_lats = latents[:, :cf]
            gt_clean = latents[:, cf - npb : cf - npb + N]
            gt_noisy = latents[:, cf : cf + N]
            cond_full, _ = model.build_action_conditional(
                prompt_embeds=pe, gt_actions=z_acts[:, : cf + N],
            )
            cond_noisy_d, _ = model.build_action_conditional(
                prompt_embeds=pe, gt_actions=z_acts[:, cf : cf + N],
            )
            cond_clean_d, _ = model.build_action_conditional(
                prompt_embeds=pe, gt_actions=z_acts[:, cf - npb : cf - npb + N],
            )
            cond_F = dict(cond_noisy_d)
            for k, v in cond_clean_d.items():
                if k in ("_action_modulation", "_action_tokens"):
                    cond_F[k + "_clean"] = v

            torch.manual_seed(seed + ride_idx * 17 + off)
            noise = torch.randn([1, N, *latents.shape[2:]], dtype=dtype, device=device)
            with torch.no_grad():
                student, _, _ = pipe.inference_with_trajectory(
                    noise=noise, gt_latents=None, enable_mae_extension=False,
                    seed_latents=seed_lats, prefer_cache_pred_in_output=True,
                    **cond_full,
                )
                # Re-noise + run F.
                noise_dmd = torch.randn_like(student)
                noisy_in = model.scheduler.add_noise(
                    student.flatten(0, 1), noise_dmd.flatten(0, 1),
                    timestep.flatten(0, 1),
                ).unflatten(0, (1, N))
                _, pred_F = model.real_score(
                    noisy_image_or_video=noisy_in,
                    conditional_dict=cond_F,
                    timestep=timestep,
                    clean_x=gt_clean.to(dtype=dtype),
                    aug_t=aug_t_zero,
                )
                z_s = teacher_z_fn(student)
                z_r = teacher_z_fn(pred_F)
                if z_s is None or z_r is None:
                    continue
                cos = torch.nn.functional.cosine_similarity(
                    z_r.float(), z_s.float(), dim=-1,
                )[0].cpu().numpy()
            cos_min = float(cos.min())
            cos_mean = float(cos.mean())
            slot_argmin = int(cos.argmin())
            findings.append((cos_min, cos_mean, zpath.name, off, slot_argmin, cos.tolist()))
            log.info(
                "  ride=%s s=%4d  cos_min=%+.3f (slot %d)  cos_mean=%+.3f  per-slot=%s",
                zpath.name, off, cos_min, slot_argmin, cos_mean,
                " ".join(f"{v:+.2f}" for v in cos),
            )
            del student, pred_F, noisy_in, noise_dmd, noise, latents, z_acts, gt_clean, gt_noisy
            torch.cuda.empty_cache()

    # Sort by cos_min ascending = WORST first.
    findings.sort(key=lambda x: x[0])
    log.info("=" * 78)
    log.info("Top 10 worst cos(R,S) windows (= candidates for action-mode flagging):")
    for cmin, cmean, ride, s, slot, cos_list in findings[:10]:
        log.info(
            "  cos_min=%+.3f  cos_mean=%+.3f  ride=%s  s=%d  worst_slot=%d",
            cmin, cmean, ride, s, slot,
        )
    log.info("=" * 78)
    log.info(
        "Run the full 3-roll diag on the worst window via:\n"
        "  DIAG_RIDE_NAME=%s DIAG_RIDE_START=%d DIAG_OUT_SUFFIX=_worst \\\n"
        "  conda run -n flash python _diag_p1/test_freeze_action_mode.py",
        findings[0][2] if findings else "<none>",
        findings[0][3] if findings else 0,
    )


if __name__ == "__main__":
    main()
