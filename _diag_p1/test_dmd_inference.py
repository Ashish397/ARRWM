"""Phase-1 DMD diagnostic: 6-test inference matrix on a real ride.

Tests (all on the same ride, same student rollout):
  A. Student rolls 21 frames via ActionForcingTrainingPipeline -> student.mp4
  B. Re-noise student pred, fake_score denoises (no clean_x) -> fake.mp4
  C. Re-noise student pred, real_score denoises (no clean_x) -> real.mp4
  D. Re-noise student pred, fake_score denoises with student clean_x -> fake_studentctx.mp4
  E. Re-noise student pred, real_score denoises with student clean_x -> real_studentctx.mp4
  F. Re-noise student pred, real_score denoises with GT clean_x -> real_gtctx.mp4

Goal: see which combination produces sensible video and which produces
garbage. The DMD gradient = (fake - real) on the same noisy_input is
what trains the generator. If fake and real agree visually on Tests B/C
or D/E or D+F, DMD is consistent. If real with GT-context (F) looks
correct but real with student-context (E) is junk, the dmd_context
student-context plumbing is broken. Etc.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

# Make repo importable
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dmd_diag")


# ---------------- single-rank dist init -----------------------------
def _init_dist_singlerank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    import socket, random
    # Pick a free port
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


# ---------------- video write -----------------------------
def _write_mp4(path: Path, frames_uint8: np.ndarray, fps: int = 20) -> None:
    """frames_uint8: [F, H, W, 3] uint8."""
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames_uint8, fps=fps, codec="libx264", quality=8)
    log.info("wrote %s (%d frames, %dx%d)", path, *frames_uint8.shape[:3])


@torch.no_grad()
def _decode_to_uint8(latents: torch.Tensor, vae) -> np.ndarray:
    """latents: [1, F, C, H, W] -> uint8 [F, H*s, W*s, 3]."""
    dummy = latents[:, 0:1]
    lat_wd = torch.cat([dummy, latents], dim=1).float()
    px = vae.decode_to_pixel(lat_wd)[:, 1:, ...]  # [1, F, 3, H, W] in [-1, 1]
    vid = (0.5 * (px.float() + 1.0)).clamp(0, 1)
    arr = (vid[0].cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[-1] != 3:
        arr = arr.transpose(0, 2, 3, 1)
    return arr


# ---------------- config build -----------------------------
def _build_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_REPO / "configs" / "action_forcing_phase1_aux_dmdctx.yaml")
    OmegaConf.set_struct(cfg, False)

    # Local overrides
    cfg.wan_model_path = "/home/ashish/Wan2.1/"
    cfg.encoded_root = "/home/ashish/frodobots/frodobots_encoded"
    cfg.caption_root = "/home/ashish/frodobots/frodobots_captions/train"
    cfg.motion_root = "/home/ashish/frodobots/frodobots_motion"
    cfg.ss_vae_checkpoint = "/home/ashish/ARRWM/action_query/checkpoints/ss_vae_8free.pt"
    cfg.ode_generator_checkpoint = "/home/ashish/action_ode_step0001000.pt"
    cfg.v14_teacher_checkpoint = "/home/ashish/Downloads/causal_lora_step0006600.pt"
    cfg.cotracker_checkpoint_path = ""
    cfg.cotracker_source_dir = ""

    # Force action teacher off (we don't need it for inference)
    cfg.action_teacher_mode = "off"
    cfg.gan_enabled = False
    cfg.sc_dmd_enabled = False
    cfg.mae_extension_max_extra_chunks = 0  # no extensions
    cfg.mae_extension_threshold = None

    # Default: stay aligned with the trainer — random-rung exit per
    # block (``last_step_only`` from YAML, typically False). The new
    # pipeline runs the no-grad continuation past the random exit
    # rung so the cache K/V is built from a fully-denoised x0
    # estimate (no noise compounding across the rolling cache). The
    # diagnostic mp4 visualises ``cache_pred`` (= the post-finish-
    # denoise clean output) by default so the rolling student looks
    # crisp; the DMD scorer tests in this script still use whatever
    # the pipeline returns, so the D/E/F numbers operate on
    # ``cache_pred`` rather than the grad-active ``denoised_pred``.
    # Set DIAG_FULL_DENOISE=1 to force ``last_step_only=True`` (every
    # block exits at the last rung, no random-exit, eval-style).
    # Set DIAG_VIZ_CACHE_PRED=0 to suppress the cache_pred swap and
    # decode the raw exit-rung ``denoised_pred`` instead.
    if os.environ.get("DIAG_FULL_DENOISE", "0") == "1":
        cfg.last_step_only = True

    # Optional denoising-schedule override via DIAG_DENOISING_STEPS env
    # (comma-separated floats, e.g. "1000,683,367,50"). Lets us A/B the
    # YAML's training-schedule against eval-style schedules without
    # editing the config.
    _ds_override = os.environ.get("DIAG_DENOISING_STEPS")
    if _ds_override:
        cfg.denoising_step_list = [
            float(x.strip()) for x in _ds_override.split(",") if x.strip()
        ]

    cfg.gradient_checkpointing = False  # no need for inference test
    cfg.mixed_precision = True

    # Required by base
    cfg.text_pre_encoded = True

    # New dmd_context API (post Bug #1 fix). Drop legacy dmd_real_GT keys
    # (no longer recognised; validators error if present).
    cfg.dmd_context = "GT"
    cfg.clean_x_aug_t = 20
    if hasattr(cfg, "dmd_real_GT"):
        del cfg.dmd_real_GT
    if hasattr(cfg, "dmd_real_GT_aug_t"):
        del cfg.dmd_real_GT_aug_t

    # Patch in-memory wan_model_path resolution
    return cfg


# ---------------- main -----------------------------
def main():
    _init_dist_singlerank()

    # Patch wan_model_path BEFORE any wan_wrapper import resolves it
    import utils.wan_wrapper as _ww
    _ww._default_wan_model_path = "/home/ashish/Wan2.1/"

    cfg = _build_config()
    log.info("[diag_cfg] denoising_step_list=%s", list(cfg.denoising_step_list))
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ---- Build model -----
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
    log.info("Model built.")

    # ---- Build pipeline -----
    log.info("Building ActionForcingTrainingPipeline ...")
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

    # ---- Install infinity_rope (default ON in YAML) -----
    irope_env = os.environ.get("DIAG_INFINITY_ROPE", None)
    if irope_env is not None:
        infinity_rope_enabled = irope_env.lower() in ("1","true","yes","on")
    else:
        infinity_rope_enabled = bool(getattr(cfg, "infinity_rope", True))
    if infinity_rope_enabled:
        from utils.infinity_rope import install as _install
        _install(model.generator.model)
        log.info("infinity_rope INSTALLED on student")
    else:
        log.info("infinity_rope DISABLED (running with original RoPE)")
    out_suffix = os.environ.get("DIAG_OUT_SUFFIX", "")

    # ---- Load one ride -----
    # Two paths:
    #   * DIAG_RIDE_NAME set → fast disk-fallback (no scan; loads only the
    #     named zarr's prompt_embeds + attrs). Mirrors what
    #     ``utils.eval_chain._load_ride_entry_from_disk`` does for the
    #     eval pipeline. ~0.5s instead of 18min for a 3000-ride scan.
    #   * Otherwise → small scan (max_rides=4), pick the first ride.
    from utils.zarr_dataset import ZarrRideDataset
    from pathlib import Path as _Path
    ride_name_override = os.environ.get("DIAG_RIDE_NAME")
    if ride_name_override:
        log.info("Fast-path loading single ride %s (no scan)", ride_name_override)
        from utils.eval_chain import (
            _build_ts_to_ride_dir,
            _load_ride_entry_from_disk,
        )
        ts_map = _build_ts_to_ride_dir(_Path(cfg.caption_root))
        ride_dict = _load_ride_entry_from_disk(
            ride_name_override,
            _Path(cfg.encoded_root),
            _Path(cfg.caption_root),
            ts_map,
        )
        ds = ZarrRideDataset.from_manifest(
            rides_data=[ride_dict],
            motion_root=cfg.motion_root,
            ss_vae_checkpoint=cfg.ss_vae_checkpoint,
            device="cpu",
            ss_vae_device="cuda:0",
        )
        chosen_idx = 0
    else:
        log.info("Loading one ride from local zarr (small scan, max_rides=4)...")
        ds = ZarrRideDataset(
            encoded_root=cfg.encoded_root,
            caption_root=cfg.caption_root,
            motion_root=cfg.motion_root,
            ss_vae_checkpoint=cfg.ss_vae_checkpoint,
            min_ride_frames=int(cfg.num_training_frames) + int(cfg.dmd_context_clean_frames),
            device="cpu",
            ss_vae_device="cuda:0",
            max_rides=4,
        )
        if len(ds) == 0:
            raise RuntimeError("No rides found in local dataset.")
        chosen_idx = 0
    zpath, prompt_embeds, attrs, n_lat = ds._rides[chosen_idx]
    log.info("Selected ride: %s n_latent_frames=%d (idx=%d, override=%r)",
             zpath.name, n_lat, chosen_idx, ride_name_override)

    # We need num_training_frames=21 + dmd_context_clean_frames=cf frames of GT.
    cf = int(cfg.dmd_context_clean_frames)
    N = int(cfg.num_training_frames)
    npb = int(cfg.num_frame_per_block)
    # Optional ride-start offset (s) so we can probe what happens at
    # high temporal positions. Default 0 = ride[0:cf+N]. With s=97 we
    # get noisy_x at ride[100:121], stressing rope_apply's freq table
    # at higher absolute positions.
    s = int(os.environ.get("DIAG_RIDE_START", "0"))
    need = s + cf + N
    if need > n_lat:
        raise RuntimeError(
            f"ride {zpath.name} only has {n_lat} latent frames, need s+cf+N={need} "
            f"(s={s} cf={cf} N={N}). Pick a longer ride or smaller s."
        )

    latents_full = ZarrRideDataset.load_latent_chunk(str(zpath), 0, need).unsqueeze(0).to(device=device, dtype=dtype)
    z_actions_full = ds.encode_z_actions_window(str(zpath), need, 0, need)
    action_dims = list(getattr(cfg, "action_dims", [2, 7]))
    z_actions_full = z_actions_full[..., action_dims].unsqueeze(0).to(device=device, dtype=dtype)
    # Slice down to the cf+N window starting at offset s.
    latents = latents_full[:, s:s + cf + N]
    z_actions = z_actions_full[:, s:s + cf + N]
    prompt_embeds = prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype) if prompt_embeds.dim() == 2 else prompt_embeds.to(device=device, dtype=dtype)

    log.info("ride_start s=%d (noisy_x at absolute frames [%d:%d])", s, s+cf, s+cf+N)
    log.info("latents=%s z_actions=%s prompt_embeds=%s", tuple(latents.shape), tuple(z_actions.shape), tuple(prompt_embeds.shape))

    # GT slices.
    # Seed              = ride[s : s+cf]                          (cf GT frames, KV-cache prefill at t=0)
    # Noisy_x window    = ride[s+cf : s+cf+N]                     (N=21 frames the student rolls)
    # Clean context     = ride[s+cf-shift : s+cf+N-shift]         (N-frame view shifted back by 1 chunk = ``shift`` frames)
    # ``shift`` is hardcoded to ``num_frame_per_block`` (= 1 chunk = 3 frames)
    # to match the model — separate from ``cf`` (which is the seed prefill size).
    shift = npb
    seed_latents = latents[:, :cf]                                            # [1, cf, 16, h, w]
    gt_noisy_window_latents = latents[:, cf : cf + N]                         # [1, 21, 16, h, w]
    gt_noisy_window_actions = z_actions[:, cf : cf + N]                       # [1, 21, A]
    gt_clean_context_latents = latents[:, cf - shift : cf + N - shift]        # [1, 21, 16, h, w]
    gt_clean_context_actions = z_actions[:, cf - shift : cf + N - shift]      # [1, 21, A]

    # Build conditional dicts for the noisy half (student rollout)
    cond_noisy, uncond_noisy = model.build_action_conditional(
        prompt_embeds=prompt_embeds,
        gt_actions=gt_noisy_window_actions,
    )
    # Build conditional dicts for the clean half (used by dmd_context tests)
    cond_clean, uncond_clean = model.build_action_conditional(
        prompt_embeds=prompt_embeds,
        gt_actions=gt_clean_context_actions,
    )
    # Build conditional dicts over the FULL ride window (seed + rollout =
    # cf + N frames) — pipeline needs per-frame action streams covering
    # both the seed prefill forwards and the rollout chunks.
    full_actions = z_actions[:, : cf + N]
    cond_full, uncond_full = model.build_action_conditional(
        prompt_embeds=prompt_embeds,
        gt_actions=full_actions,
    )

    # Reference MAE: between the SHIFTED clean GT and the noised target
    # the scorer actually sees. Decomposes into three contributors so
    # we can attribute scorer error to noise vs. content-shift vs. pred-
    # error separately. Computed AFTER noisy_input is constructed (see
    # below).
    _gt_window_mae = float(
        (gt_clean_context_latents.float() - gt_noisy_window_latents.float()).abs().mean().item()
    )
    log.info(
        "MAE(gt_clean_context [s:s+N], gt_noisy_window [s+cf:s+cf+N]) = %.4f "
        "(scene shift over cf=%d frames at ride positions [%d:%d] vs [%d:%d])",
        _gt_window_mae, cf, s, s+N, s+cf, s+cf+N,
    )

    # ---- Test A: roll the student (N=21 frames = 7 chunks) with seed prefill -----
    log.info("=" * 60)
    log.info("Test A: student rollout (%d frames, %d chunks) WITH cf=%d-frame KV-cache prefill",
             N, N // npb, cf)
    torch.manual_seed(0)
    noise = torch.randn(
        [1, N, *latents.shape[2:]],
        dtype=dtype, device=device,
    )
    # By default the pipeline writes the FULLY-DENOISED ``cache_pred``
    # (post no-grad continuation past the random exit rung) to
    # ``output[]`` instead of the grad-active exit-rung
    # ``denoised_pred`` — gives a crisp diagnostic mp4 even with
    # random-rung exit. Set ``DIAG_VIZ_CACHE_PRED=0`` to render the
    # raw exit-rung ``denoised_pred`` (= what training sees as
    # ``pred_image``); invalid for training (no gradient signal in
    # cache_pred), only intended for diagnostic visualisation.
    _viz_cache_pred = os.environ.get("DIAG_VIZ_CACHE_PRED", "1") == "1"
    with torch.no_grad():
        student_pred, denoised_t_from, denoised_t_to = pipe.inference_with_trajectory(
            noise=noise, gt_latents=None, enable_mae_extension=False,
            seed_latents=seed_latents,
            prefer_cache_pred_in_output=_viz_cache_pred,
            **cond_full,
        )
    log.info("student_pred shape=%s mean=%.3f std=%.3f", tuple(student_pred.shape), float(student_pred.mean()), float(student_pred.std()))
    log.info("denoised_t [%s, %s]", denoised_t_from, denoised_t_to)
    log.info("GT slice    mean=%.3f std=%.3f", float(gt_noisy_window_latents.float().mean()), float(gt_noisy_window_latents.float().std()))

    out_dir = _REPO / "_diag_p1" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    student_uint8 = _decode_to_uint8(student_pred, model.vae)
    _write_mp4(out_dir / f"A_student{out_suffix}.mp4", student_uint8)

    # Also save GT for visual reference
    gt_uint8 = _decode_to_uint8(gt_noisy_window_latents, model.vae)
    _write_mp4(out_dir / "_gt_noisy_window.mp4", gt_uint8)
    gt_clean_uint8 = _decode_to_uint8(gt_clean_context_latents, model.vae)
    _write_mp4(out_dir / "_gt_clean_context.mp4", gt_clean_uint8)

    # ---- Pick a DMD timestep and re-noise student_pred once -----
    # Use the same noisy_input for all denoise tests so they're directly comparable.
    log.info("=" * 60)
    log.info("Sampling shared DMD timestep + noising student pred for Tests B-F")
    with torch.no_grad():
        timestep = model._sample_dmd_timestep(
            batch_size=1, num_frame=N,
            denoised_timestep_from=denoised_t_from, denoised_timestep_to=denoised_t_to,
            device=device,
        )
        log.info("DMD timestep[0,0]=%d (shape=%s)", int(timestep[0, 0]), tuple(timestep.shape))
        noise_for_dmd = torch.randn_like(student_pred)
        noisy_input = model.scheduler.add_noise(
            student_pred.flatten(0, 1), noise_for_dmd.flatten(0, 1), timestep.flatten(0, 1),
        ).unflatten(0, (1, N))
        log.info("noisy_input mean=%.3f std=%.3f", float(noisy_input.float().mean()), float(noisy_input.float().std()))

        # MAE decomposition. Three lines — what the scorer must bridge:
        #  1) shifted_clean → noisy_input (= total gap between real_score's
        #     "GT" mode clean_x and the noisy target it's denoising)
        #  2) gt_noisy_window → noisy_input (= no-shift; pure pred-error +
        #     scheduler noise. gt_noisy_window is the CLEAN GT version
        #     of the same time positions as noisy_input.)
        #  3) gt_noisy_window → noised(gt_noisy_window) at the SAME t (=
        #     pure scheduler-noise contribution, isolated)
        gt_noisy_window_noised = model.scheduler.add_noise(
            gt_noisy_window_latents.float().flatten(0, 1),
            torch.randn_like(gt_noisy_window_latents.float()).flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (1, N))
        mae_shifted_vs_noisy = float(
            (gt_clean_context_latents.float() - noisy_input.float()).abs().mean().item()
        )
        mae_gtwin_vs_noisy = float(
            (gt_noisy_window_latents.float() - noisy_input.float()).abs().mean().item()
        )
        mae_pure_noise = float(
            (gt_noisy_window_latents.float() - gt_noisy_window_noised.float()).abs().mean().item()
        )
        log.info(
            "MAE breakdown @ t=%d:\n"
            "  shifted_clean → noisy_input = %.4f  (= scene shift + pred-err + noise)\n"
            "  gt_noisy_win  → noisy_input = %.4f  (= pred-err + noise; no shift)\n"
            "  gt_noisy_win  → noised(gt_noisy_win) = %.4f  (= pure scheduler noise)\n"
            "  scene-shift baseline (clean→noisy_window, both clean) = %.4f",
            int(timestep[0, 0].item()),
            mae_shifted_vs_noisy, mae_gtwin_vs_noisy, mae_pure_noise, _gt_window_mae,
        )

    def _run_scorer_one_shot(scorer, cond_dict, *, clean_x=None, aug_t=None, label=""):
        """Single-shot denoise: predict x0 with the scorer at the sampled timestep."""
        kwargs = {}
        if clean_x is not None:
            kwargs["clean_x"] = clean_x
            kwargs["aug_t"] = aug_t
        with torch.no_grad():
            _, pred_x0 = scorer(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond_dict,
                timestep=timestep,
                **kwargs,
            )
        log.info("[%s] pred_x0 mean=%.3f std=%.3f abs-diff-vs-student=%.4f",
                 label, float(pred_x0.float().mean()), float(pred_x0.float().std()),
                 float((pred_x0.float() - student_pred.float()).abs().mean()))
        return pred_x0

    # ---- Tests B & C skipped (per user) — go straight to D/E/F. -----
    fakeB = None
    realC = None

    # ---- For Tests D-F: dmd_context style. ActionForcingDMD.__init__
    #      already set context_shift=1 chunk and tf_rope_offset_frames=
    #      num_frame_per_block on both scorers. No manual override needed.
    log.info("=" * 60)
    log.info("Tests D-F: dmd_context (cs=1 chunk, tf_rope_offset_frames=%d, cf=%d)",
             npb, cf)

    # aug_t for clean_x:
    #  * fake's clean_x is ALWAYS the unnoised self-view (aug_t=0).
    #  * real's clean_x in "self" mode = same self-view (aug_t=0).
    #  * real's clean_x in "GT" mode = scheduler.add_noise(GT, n,
    #    clean_x_aug_t) — a small symmetry-breaking noise on the GT view.
    aug_t_zero = torch.zeros(1, N, dtype=torch.long, device=device)

    # Build the "self" clean_x view for batch-1 of the rollout:
    # [last seed chunk GT (= last ``shift`` = npb = 3 frames), sdn[:N-shift]=18 frames] = 21 frames.
    # Same time positions as ride[s+cf-shift : s+cf+N-shift] in absolute coords,
    # so its action streams are ``gt_clean_context_actions`` (already wired into cond_clean).
    student_clean_x = torch.cat(
        [seed_latents[:, -shift:], student_pred[:, : N - shift]], dim=1,
    ).to(dtype=dtype)
    log.info(
        "student_clean_x (self-view, batch-1) shape=%s mean=%.3f std=%.3f",
        tuple(student_clean_x.shape),
        float(student_clean_x.float().mean()),
        float(student_clean_x.float().std()),
    )

    # Merge clean cond streams into the noisy cond_dict (model expects
    # _action_modulation_clean / _action_tokens_clean). The clean
    # action streams cover ride[0:N] = same time positions as both
    # student_clean_x (self-view) AND gt_clean_context_latents (GT-view),
    # so a single cond_clean serves both modes.
    def _merge_clean_streams(cond_noisy_dict, cond_clean_dict):
        out = dict(cond_noisy_dict)
        for k, v in cond_clean_dict.items():
            if k in ("_action_modulation", "_action_tokens"):
                out[k + "_clean"] = v
        return out

    cond_for_dmdctx = _merge_clean_streams(cond_noisy, cond_clean)

    # Detailed per-frame comparison: clean vs noisy values at the
    # OVERLAPPING RoPE positions [shift:N] vs [0:N-shift]. Clean is
    # shifted back by 1 chunk (= ``shift`` = npb), so clean[i] should
    # equal noisy[i-shift] for i ≥ shift.
    am = cond_for_dmdctx.get("_action_modulation")
    am_c = cond_for_dmdctx.get("_action_modulation_clean")
    at = cond_for_dmdctx.get("_action_tokens")
    at_c = cond_for_dmdctx.get("_action_tokens_clean")
    if am is not None and am_c is not None and at is not None and at_c is not None:
        log.info("PER-FRAME OVERLAP CHECK (clean[i] vs noisy[i-shift], shift=%d):", shift)
        log.info("  i  |clean_mod-noisy_mod|  |clean_tok-noisy_tok|  clean_mod_norm  clean_tok_norm")
        for i in range(am_c.shape[1]):
            cmod = am_c[0, i].float()
            ctok = at_c[0, i].float()
            if i >= shift:
                ndi = i - shift
                mod_diff = (cmod - am[0, ndi].float()).abs().mean().item()
                tok_diff = (ctok - at[0, ndi].float()).abs().mean().item()
                log.info("  %2d  %20.6f  %20.6f  %14.4f  %14.4f",
                         i, mod_diff, tok_diff,
                         cmod.abs().mean().item(), ctok.abs().mean().item())
            else:
                log.info("  %2d  %20s  %20s  %14.4f  %14.4f  (clean-only, no noisy counterpart)",
                         i, "—", "—",
                         cmod.abs().mean().item(), ctok.abs().mean().item())

    # ---- Test D: fake_score with self-view clean_x (batch-1) -----
    log.info("Test D: fake_score with SELF-view clean_x")
    fakeD = _run_scorer_one_shot(
        model.fake_score, cond_for_dmdctx,
        clean_x=student_clean_x, aug_t=aug_t_zero, label="D/fake_self",
    )
    _write_mp4(out_dir / f"D_fake_self{out_suffix}.mp4", _decode_to_uint8(fakeD, model.vae))

    # ---- Test E: real_score in "self" mode = same view as fake -----
    log.info("Test E: real_score with SELF-view clean_x ('self' mode)")
    realE = _run_scorer_one_shot(
        model.real_score, cond_for_dmdctx,
        clean_x=student_clean_x, aug_t=aug_t_zero, label="E/real_self",
    )
    _write_mp4(out_dir / f"E_real_self{out_suffix}.mp4", _decode_to_uint8(realE, model.vae))

    # ---- Test F: real_score in "GT" mode = noised GT clean_x -----
    aug_t_gt_val = int(getattr(cfg, "clean_x_aug_t", 20))
    aug_t_gt = torch.full(
        (1, N), fill_value=aug_t_gt_val, device=device, dtype=torch.long,
    )
    gt_view = gt_clean_context_latents.to(dtype=dtype)
    gt_noise = torch.randn_like(gt_view)
    gt_view_noised = model.scheduler.add_noise(
        gt_view.flatten(0, 1),
        gt_noise.flatten(0, 1),
        aug_t_gt.flatten(0, 1),
    ).unflatten(0, gt_view.shape[:2]).to(dtype=dtype)
    log.info("Test F: real_score with NOISED-GT clean_x ('GT' mode, aug_t=%d)", aug_t_gt_val)
    realF = _run_scorer_one_shot(
        model.real_score, cond_for_dmdctx,
        clean_x=gt_view_noised, aug_t=aug_t_gt, label="F/real_GT",
    )
    _write_mp4(out_dir / f"F_real_GT{out_suffix}.mp4", _decode_to_uint8(realF, model.vae))

    # ---- Numeric summary -----
    log.info("=" * 60)
    log.info("Per-test x0 stats:")
    for name, t in [
        ("student_pred (A) seeded", student_pred),
        ("fake D (self-view)", fakeD),
        ("real E (self mode)", realE),
        ("real F (GT mode + aug_t)", realF),
        ("GT noisy window", gt_noisy_window_latents),
    ]:
        t = t.float()
        log.info(
            "  %-22s  mean=%+.3f std=%.3f  L2-vs-GT=%.4f  L2-vs-student=%.4f",
            name, float(t.mean()), float(t.std()),
            float((t - gt_noisy_window_latents.float()).pow(2).mean().sqrt()),
            float((t - student_pred.float()).pow(2).mean().sqrt()),
        )

    log.info("DONE. Videos in %s", out_dir)


if __name__ == "__main__":
    main()
