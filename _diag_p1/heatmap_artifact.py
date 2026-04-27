"""Generate per-pixel error heatmaps to localize the F-mode top-left
artifact in latent space.

Runs the same setup as test_dmd_inference.py but DUMPS the latent
tensors to disk for offline analysis, then renders heatmaps showing
|F_pred - GT_noisy| per (frame, h, w) — averaged over channels — so we
can pinpoint the spatial extent of the artifact and dial in the
gradient_mask region precisely.

We average over multiple DMD-noise seeds so the per-pixel deviation
isolates the SYSTEMATIC artifact (LoRA misbehaviour at boundary
positions) from the per-seed noise variance.
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
log = logging.getLogger("heatmap")


def _init_dist_singlerank():
    import socket, random
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
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
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


def main():
    _init_dist_singlerank()

    import utils.wan_wrapper as _ww
    _ww._default_wan_model_path = "/home/ashish/Wan2.1/"

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
    cfg.gradient_checkpointing = False
    cfg.mixed_precision = True
    cfg.text_pre_encoded = True
    cfg.dmd_context = "GT"
    cfg.clean_x_aug_t = 20
    if hasattr(cfg, "dmd_real_GT"):
        del cfg.dmd_real_GT
    if hasattr(cfg, "dmd_real_GT_aug_t"):
        del cfg.dmd_real_GT_aug_t

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    log.info("Building model + pipeline ...")
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

    from utils.infinity_rope import install as _install
    _install(model.generator.model)
    log.info("infinity_rope INSTALLED")

    log.info("Loading ride 20240211153640.zarr ...")
    from utils.zarr_dataset import ZarrRideDataset
    ds = ZarrRideDataset(
        encoded_root=cfg.encoded_root,
        caption_root=cfg.caption_root,
        motion_root=cfg.motion_root,
        ss_vae_checkpoint=cfg.ss_vae_checkpoint,
        min_ride_frames=int(cfg.num_training_frames) + 3,
        device="cpu", ss_vae_device="cuda:0", max_rides=10,
    )
    chosen_idx = 0
    for i, (zp, _, _, _) in enumerate(ds._rides):
        if zp.name == "20240211153640.zarr":
            chosen_idx = i
            break
    zpath, prompt_embeds, attrs, n_lat = ds._rides[chosen_idx]

    cf = int(cfg.dmd_context_clean_frames)
    N = int(cfg.num_training_frames)
    npb = int(cfg.num_frame_per_block)
    rollout_total = N + npb
    s = 100
    need = s + cf + rollout_total

    latents_full = ZarrRideDataset.load_latent_chunk(str(zpath), 0, need).unsqueeze(0).to(device=device, dtype=dtype)
    z_actions_full = ds.encode_z_actions_window(str(zpath), need, 0, need)
    action_dims = list(getattr(cfg, "action_dims", [2, 7]))
    z_actions_full = z_actions_full[..., action_dims].unsqueeze(0).to(device=device, dtype=dtype)
    latents = latents_full[:, s:s + cf + rollout_total]
    z_actions = z_actions_full[:, s:s + cf + rollout_total]
    prompt_embeds = prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype) if prompt_embeds.dim() == 2 else prompt_embeds.to(device=device, dtype=dtype)

    seed_latents = latents[:, :cf]
    gt_clean_context_latents = latents[:, cf : cf + N]
    gt_clean_context_actions = z_actions[:, cf : cf + N]
    gt_noisy_window_latents = latents[:, cf + npb : cf + npb + N]
    gt_noisy_window_actions = z_actions[:, cf + npb : cf + npb + N]
    full_actions = z_actions[:, : cf + rollout_total]

    cond_noisy, _ = model.build_action_conditional(prompt_embeds=prompt_embeds, gt_actions=gt_noisy_window_actions)
    cond_clean, _ = model.build_action_conditional(prompt_embeds=prompt_embeds, gt_actions=gt_clean_context_actions)
    cond_full, _ = model.build_action_conditional(prompt_embeds=prompt_embeds, gt_actions=full_actions)

    # Roll the student once
    log.info("Rolling student (24 frames) ...")
    torch.manual_seed(0)
    noise = torch.randn([1, rollout_total, *latents.shape[2:]], dtype=dtype, device=device)
    with torch.no_grad():
        student_full, denoised_t_from, denoised_t_to = pipe.inference_with_trajectory(
            noise=noise, gt_latents=None, enable_mae_extension=False,
            seed_latents=seed_latents, **cond_full,
        )
    student_pred = student_full[:, npb:npb + N].contiguous()

    # Build cond_for_dmdctx with merged clean streams
    def _merge_clean_streams(cn, cc):
        out = dict(cn)
        for k, v in cc.items():
            if k in ("_action_modulation", "_action_tokens"):
                out[k + "_clean"] = v
        return out
    cond_for_dmdctx = _merge_clean_streams(cond_noisy, cond_clean)

    # Compute F's pred for MULTIPLE DMD-noise seeds, then average + variance
    aug_t_gt = torch.full((1, N), fill_value=int(cfg.clean_x_aug_t), device=device, dtype=torch.long)
    gt_view = gt_clean_context_latents.to(dtype=dtype)
    n_seeds = 8
    f_preds = []
    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        # Sample DMD timestep
        timestep = model._sample_dmd_timestep(
            batch_size=1, num_frame=N,
            denoised_timestep_from=denoised_t_from, denoised_timestep_to=denoised_t_to,
            device=device,
        )
        # Re-noise student_pred
        noise_dmd = torch.randn_like(student_pred)
        noisy_input = model.scheduler.add_noise(
            student_pred.flatten(0, 1), noise_dmd.flatten(0, 1), timestep.flatten(0, 1),
        ).unflatten(0, (1, N))
        # Noise the GT clean_x at clean_x_aug_t
        gt_noise = torch.randn_like(gt_view)
        gt_view_noised = model.scheduler.add_noise(
            gt_view.flatten(0, 1), gt_noise.flatten(0, 1), aug_t_gt.flatten(0, 1),
        ).unflatten(0, gt_view.shape[:2]).to(dtype=dtype)
        # Run F
        with torch.no_grad():
            _, pred_x0 = model.real_score(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond_for_dmdctx,
                timestep=timestep,
                clean_x=gt_view_noised,
                aug_t=aug_t_gt,
            )
        f_preds.append(pred_x0.float().cpu())
        log.info(f"  seed {seed}: pred_x0 std={pred_x0.float().std().item():.4f}")

    f_stack = torch.stack(f_preds, dim=0)  # [n_seeds, 1, F, C, H, W]
    f_mean = f_stack.mean(dim=0)            # [1, F, C, H, W] — averaged over noise seeds
    gt_cpu = gt_noisy_window_latents.float().cpu()
    err = (f_mean - gt_cpu).abs().mean(dim=2)  # [1, F, H, W] — channel-averaged abs error
    err_np = err[0].numpy()  # [F, H, W]
    log.info(f"err shape: {err_np.shape}; range [{err_np.min():.4f}, {err_np.max():.4f}]")

    # Save heatmap as a tiled image: each row = one latent frame's error map (60x104),
    # show first ~6 frames so we can see where the artifact is.
    out_dir = _REPO / "_diag_p1" / "heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    # Normalize error to [0, 255] for visualization. Per-frame normalization
    # would hide which frames are worse; use a global normalization.
    err_max = float(err_np.max())
    err_min = float(err_np.min())
    norm = lambda x: ((x - err_min) / max(err_max - err_min, 1e-8) * 255).astype(np.uint8)
    # Save first 8 frames as separate PNGs (60x104 latent space, upsampled 4x for visibility).
    import torch.nn.functional as F
    for i in range(min(8, err_np.shape[0])):
        h_arr = err_np[i]
        big = F.interpolate(torch.from_numpy(h_arr)[None, None].float(), scale_factor=4, mode="nearest")[0, 0].numpy()
        big = norm(big.astype(np.float32))
        # Apply a heatmap colormap (matplotlib magma-like, but simple)
        # Channel mapping: low error = dark blue, high error = yellow/red
        h, w = big.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = big                     # R: high in hot regions
        rgb[:, :, 1] = (big.astype(np.uint16) * 200 // 255).astype(np.uint8)  # G: mid
        rgb[:, :, 2] = (255 - big).astype(np.uint8)  # B: high in cold regions
        imageio.imwrite(str(out_dir / f"err_frame{i:02d}.png"), rgb)
        log.info(f"  frame {i:02d}: err mean={h_arr.mean():.4f} max={h_arr.max():.4f} max@{np.unravel_index(h_arr.argmax(), h_arr.shape)}")

    # Also dump the raw error tensor for offline analysis
    np.save(out_dir / "err_F_vs_GT.npy", err_np)
    log.info(f"saved heatmaps + raw to {out_dir}")

    # Print column-wise and row-wise error sums for the first frame to
    # nail the artifact's spatial extent.
    f0 = err_np[0]
    print("\nFrame 0 row-wise mean (first 30 rows):")
    for r in range(30):
        bar_len = int(60 * f0[r].mean() / max(f0.max(), 1e-8))
        print(f"  row {r:2d}: mean={f0[r].mean():.4f}  {'█' * bar_len}")
    print("\nFrame 0 col-wise mean (first 30 cols):")
    for c in range(30):
        bar_len = int(60 * f0[:, c].mean() / max(f0.max(), 1e-8))
        print(f"  col {c:2d}: mean={f0[:, c].mean():.4f}  {'█' * bar_len}")


if __name__ == "__main__":
    main()
