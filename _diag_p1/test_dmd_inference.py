"""Phase-1 DMD diagnostic: multi-batch (3-rollout) inference matrix.

Rolls the student for ``DIAG_NUM_ROLLOUTS * N`` (= 3 * 21 = 63 by default)
frames, sliced into ``DIAG_NUM_ROLLOUTS`` consecutive 21-frame DMD scoring
windows. For each batch we run the standard 6-test matrix (A is the long
student rollout; B-F are per-batch scorer denoises) and concatenate B-F
outputs along F to produce a contiguous video of the same length as A.

Per-batch geometry (matches the trainer's multi-batch contract):
  i = 1..NUM_ROLLOUTS
    noisy_x_i  = sdn[(i-1)*N        : i*N        ]   (= 21 frames)
    clean_x_i  = sdn[(i-1)*N - shift: i*N - shift]   (= shift frames behind)
  with shift = num_frame_per_block (= 3). For i=1 the clean window dips
  into the seed prefill (sdn[-3:0] doesn't exist), so clean_x_1's first
  ``shift`` frames are taken from the GT seed_latents tail — same hack
  the trainer uses for batch 1.

Each batch's cond_dict re-slices ``z_actions`` so the action streams
match the absolute ride positions of THAT batch's noisy and clean
windows. Per-frame timestep is shared across all batches so the
concatenated B-F videos have a consistent noise level along F.

Knobs:
  DIAG_NUM_ROLLOUTS  (int, default 3)  — number of N-frame batches.
  DIAG_RIDE_NAME     (zarr basename)   — fast-path single-ride load.
  DIAG_RIDE_START    (int, default 0)  — absolute ride offset s.
  DIAG_BIDIR_TF_NO_MASK=1              — disable v14 TF mask in bidir scorer.
  DIAG_F_AUG_T       (int, default 20) — clean_x_aug_t in GT mode.
  DIAG_OUT_SUFFIX    (str)             — suffix appended to mp4 filenames.
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
    # if os.environ.get("DIAG_FULL_DENOISE", "0") == "1":
    #     cfg.last_step_only = True
    cfg.last_step_only = False

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

    # ---- Multi-batch geometry --------------------------------------------------
    # Roll ONE EXTRA chunk (= npb frames) at the FRONT of the student
    # rollout so batch 1's clean_x_i lives entirely in the rolled-out
    # range — no seed-dip. Per-batch slicing then becomes
    #   noisy_x_i  = sdn[(i-1)*N + npb : i*N + npb]
    #   clean_x_i  = sdn[(i-1)*N        : i*N        ]
    # i.e. clean leads noisy by ``npb`` frames within sdn (matching v14's
    # context_shift=1-chunk TF contract). For i=1 this gives
    #   noisy_x_1 = sdn[3:24]  (= ride[s+cf+npb : s+cf+npb+N])
    #   clean_x_1 = sdn[0:21]  (= ride[s+cf     : s+cf+N    ])
    # entirely from the student's own rollout — no GT/student quality
    # discontinuity in the first ``shift`` frames.
    cf = int(cfg.dmd_context_clean_frames)
    N = int(cfg.num_training_frames)
    npb = int(cfg.num_frame_per_block)
    shift = npb
    num_rollouts = int(os.environ.get("DIAG_NUM_ROLLOUTS", "3"))
    if num_rollouts < 1:
        raise RuntimeError(f"DIAG_NUM_ROLLOUTS must be >= 1, got {num_rollouts}")
    total_rollout_frames = num_rollouts * N + npb  # +npb extra leading chunk for clean_x_1
    s = int(os.environ.get("DIAG_RIDE_START", "0"))
    need = s + cf + total_rollout_frames
    if need > n_lat:
        raise RuntimeError(
            f"ride {zpath.name} only has {n_lat} latent frames, need "
            f"s+cf+(num_rollouts*N+npb)={need} (s={s} cf={cf} N={N} "
            f"num_rollouts={num_rollouts} npb={npb}). Pick a longer ride."
        )

    latents_full = ZarrRideDataset.load_latent_chunk(str(zpath), 0, need).unsqueeze(0).to(device=device, dtype=dtype)
    z_actions_full = ds.encode_z_actions_window(str(zpath), need, 0, need)
    action_dims = list(getattr(cfg, "action_dims", [2, 7]))
    z_actions_full = z_actions_full[..., action_dims].unsqueeze(0).to(device=device, dtype=dtype)
    # Slice ride to [s : s + cf + total_rollout_frames].
    latents = latents_full[:, s:s + cf + total_rollout_frames]
    z_actions = z_actions_full[:, s:s + cf + total_rollout_frames]
    prompt_embeds = prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype) if prompt_embeds.dim() == 2 else prompt_embeds.to(device=device, dtype=dtype)

    log.info(
        "ride_start s=%d  num_rollouts=%d  total_rollout=%d frames "
        "(= %d chunks of %d). seed prefill = %d frames (cf).",
        s, num_rollouts, total_rollout_frames, total_rollout_frames // npb, npb, cf,
    )
    log.info("latents=%s z_actions=%s prompt_embeds=%s",
             tuple(latents.shape), tuple(z_actions.shape), tuple(prompt_embeds.shape))

    seed_latents = latents[:, :cf]  # [1, cf, 16, h, w]

    # cond_dicts spanning the FULL window (seed + rollout) — pipeline
    # needs per-frame action streams covering the seed prefill forwards
    # AND every rollout chunk.
    full_actions = z_actions[:, : cf + total_rollout_frames]
    cond_full, _ = model.build_action_conditional(
        prompt_embeds=prompt_embeds, gt_actions=full_actions,
    )

    # ---- Test A: roll the student for total_rollout_frames frames -------------
    log.info("=" * 60)
    log.info("Test A: student rollout (%d frames, %d chunks) WITH cf=%d-frame KV-cache prefill",
             total_rollout_frames, total_rollout_frames // npb, cf)
    torch.manual_seed(0)
    noise = torch.randn(
        [1, total_rollout_frames, *latents.shape[2:]],
        dtype=dtype, device=device,
    )
    _viz_cache_pred = os.environ.get("DIAG_VIZ_CACHE_PRED", "1") == "1"
    with torch.no_grad():
        student_full, denoised_t_from, denoised_t_to = pipe.inference_with_trajectory(
            noise=noise, gt_latents=None, enable_mae_extension=False,
            seed_latents=seed_latents,
            prefer_cache_pred_in_output=_viz_cache_pred,
            **cond_full,
        )
    log.info("student_full shape=%s mean=%.3f std=%.3f",
             tuple(student_full.shape), float(student_full.mean()), float(student_full.std()))
    log.info("denoised_t [%s, %s]", denoised_t_from, denoised_t_to)

    out_dir = _REPO / "_diag_p1" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A_student.mp4 = the NOISY-X span of the rollout (= student_full[npb:],
    # same num_rollouts * N = 63 frames the scorers operate on). Skips the
    # leading ``npb``-frame extra chunk that anchors batch 1's clean_x.
    _write_mp4(out_dir / f"A_student{out_suffix}.mp4",
               _decode_to_uint8(student_full[:, npb:], model.vae))

    # GT reference video aligned with A's NOISY-X span
    # (= ride[s+cf+npb : s+cf+total_rollout]).
    _write_mp4(out_dir / f"_gt_rollout_window{out_suffix}.mp4",
               _decode_to_uint8(latents[:, cf + npb:], model.vae))

    # ---- Pick ONE shared DMD timestep + ONE shared noise tensor for B-F --------
    # Shared across all batches so the concatenated B-F outputs have a
    # consistent noise level along F (otherwise per-batch t differences
    # cause visible discontinuities at chunk boundaries).
    log.info("=" * 60)
    log.info("Sampling shared DMD timestep + per-frame noise for tests B-F")
    with torch.no_grad():
        timestep_full = model._sample_dmd_timestep(
            batch_size=1, num_frame=total_rollout_frames,
            denoised_timestep_from=denoised_t_from, denoised_timestep_to=denoised_t_to,
            device=device,
        )
        # Reuse the FIRST sampled t across all frames for visual consistency.
        t0 = int(timestep_full[0, 0].item())
        timestep_full = torch.full(
            (1, total_rollout_frames), fill_value=t0, dtype=torch.long, device=device,
        )
        log.info("DMD timestep (shared) = %d", t0)
        noise_for_dmd_full = torch.randn(
            [1, total_rollout_frames, *latents.shape[2:]],
            dtype=dtype, device=device,
        )
        noisy_input_full = model.scheduler.add_noise(
            student_full.flatten(0, 1),
            noise_for_dmd_full.flatten(0, 1),
            timestep_full.flatten(0, 1),
        ).unflatten(0, (1, total_rollout_frames))
        # GT clean_x (shifted-back) noise — pre-sample for all frames so
        # batches share the noise pattern.
        aug_t_gt_val = int(os.environ.get(
            "DIAG_F_AUG_T",
            str(int(getattr(cfg, "clean_x_aug_t", 20))),
        ))
        log.info("F clean_x aug_t = %d", aug_t_gt_val)

    aug_t_zero_per_batch = torch.zeros(1, N, dtype=torch.long, device=device)
    aug_t_gt_per_batch = torch.full(
        (1, N), fill_value=aug_t_gt_val, device=device, dtype=torch.long,
    )
    timestep_per_batch = torch.full(
        (1, N), fill_value=t0, dtype=torch.long, device=device,
    )

    def _merge_clean_streams(cond_noisy_dict, cond_clean_dict):
        out = dict(cond_noisy_dict)
        for k, v in cond_clean_dict.items():
            if k in ("_action_modulation", "_action_tokens"):
                out[k + "_clean"] = v
        return out

    def _run_scorer_one_shot(scorer, cond_dict, noisy_input, *, clean_x=None, aug_t=None, label=""):
        """Single-shot denoise: predict x0 with the scorer at the shared timestep."""
        kwargs = {}
        if clean_x is not None:
            kwargs["clean_x"] = clean_x
            kwargs["aug_t"] = aug_t
        with torch.no_grad():
            _, pred_x0 = scorer(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond_dict,
                timestep=timestep_per_batch,
                **kwargs,
            )
        log.info("[%s] pred_x0 mean=%.3f std=%.3f",
                 label, float(pred_x0.float().mean()), float(pred_x0.float().std()))
        return pred_x0

    # ---- Per-batch B/C/D/E/F loop ---------------------------------------------
    log.info("=" * 60)
    log.info("Running %d sequential rollouts (batch i ∈ 1..%d). Per-batch "
             "noisy_x_i = sdn[(i-1)*N + npb : i*N + npb], clean_x_i = "
             "sdn[(i-1)*N : i*N] (clean leads noisy by npb=%d in sdn). "
             "Extra leading +npb chunk in sdn anchors batch-1's clean_x — "
             "no seed-dip.", num_rollouts, num_rollouts, npb)

    fakeB_list, realC_list = [], []
    fakeD_list, realE_list, realF_list = [], [], []

    for i in range(1, num_rollouts + 1):
        # ----- per-batch slice indices in sdn (= student_full) and ride.
        # sdn has total_rollout_frames = num_rollouts*N + npb elements:
        # the FIRST npb-frame chunk is the "leading" chunk reserved for
        # batch-1's clean_x; everything past index npb is the noisy_x
        # span sliced by num_rollouts.
        noisy_start_sdn = (i - 1) * N + npb        # 3, 24, 45 for i=1..3
        noisy_end_sdn   = i * N       + npb        # 24, 45, 66
        clean_start_sdn = noisy_start_sdn - shift   # 0, 21, 42
        clean_end_sdn   = noisy_end_sdn   - shift   # 21, 42, 63

        # absolute ride positions (for picking action streams + GT slices)
        abs_noisy_start = cf + noisy_start_sdn   # cf+npb, cf+npb+N, cf+npb+2N
        abs_noisy_end   = cf + noisy_end_sdn
        abs_clean_start = cf + clean_start_sdn   # cf, cf+N, cf+2N (no seed dip)
        abs_clean_end   = cf + clean_end_sdn

        log.info("---- Batch %d ----", i)
        log.info("  noisy_x_i  = sdn[%d:%d]    (= ride[%d:%d])",
                 noisy_start_sdn, noisy_end_sdn,
                 s + abs_noisy_start, s + abs_noisy_end)
        log.info("  clean_x_i  = sdn[%d:%d]    (= ride[%d:%d])",
                 clean_start_sdn, clean_end_sdn,
                 s + abs_clean_start, s + abs_clean_end)

        # ----- noisy_x_i: re-noised student rollout slice (shared noise/t)
        noisy_input_i = noisy_input_full[:, noisy_start_sdn:noisy_end_sdn].contiguous()
        student_pred_i = student_full[:, noisy_start_sdn:noisy_end_sdn].contiguous()

        # ----- clean_x_self_i: the student's own rollout, shifted back by
        # ``shift`` frames in sdn. clean_start_sdn >= 0 for ALL batches now
        # (= entire clean_x lives in the rolled-out range, no seed mixing).
        student_clean_x_i = student_full[:, clean_start_sdn:clean_end_sdn].contiguous()

        # ----- clean_x_GT_i: GT slice at the shifted absolute ride positions
        gt_clean_i = latents[:, abs_clean_start:abs_clean_end].contiguous()
        # Noise the GT view at clean_x_aug_t (small symmetry-breaking noise)
        gt_noise_i = torch.randn_like(gt_clean_i)
        gt_view_noised_i = model.scheduler.add_noise(
            gt_clean_i.float().flatten(0, 1),
            gt_noise_i.float().flatten(0, 1),
            aug_t_gt_per_batch.flatten(0, 1),
        ).unflatten(0, gt_clean_i.shape[:2]).to(dtype=dtype)

        # ----- per-batch action streams. cond_noisy_i covers the noisy
        # window's ride positions; cond_clean_i covers the clean window's
        # ride positions. CRITICAL: these MUST track absolute ride
        # positions for action conditioning to be sensible across the
        # 3 sequential rollouts.
        actions_noisy_i = z_actions[:, abs_noisy_start:abs_noisy_end]
        actions_clean_i = z_actions[:, abs_clean_start:abs_clean_end]
        cond_noisy_i, _ = model.build_action_conditional(
            prompt_embeds=prompt_embeds, gt_actions=actions_noisy_i,
        )
        cond_clean_i, _ = model.build_action_conditional(
            prompt_embeds=prompt_embeds, gt_actions=actions_clean_i,
        )
        cond_for_dmdctx_i = _merge_clean_streams(cond_noisy_i, cond_clean_i)

        # ----- Run B/C: NO clean_x (plain bidir, no TF) ------
        fakeB_i = _run_scorer_one_shot(
            model.fake_score, cond_noisy_i, noisy_input_i,
            label=f"B-{i}/fake_no_ctx",
        )
        fakeB_list.append(fakeB_i)
        realC_i = _run_scorer_one_shot(
            model.real_score, cond_noisy_i, noisy_input_i,
            label=f"C-{i}/real_no_ctx",
        )
        realC_list.append(realC_i)

        # ----- Run D/E/F: WITH clean_x (TF mode) ------
        fakeD_i = _run_scorer_one_shot(
            model.fake_score, cond_for_dmdctx_i, noisy_input_i,
            clean_x=student_clean_x_i, aug_t=aug_t_zero_per_batch,
            label=f"D-{i}/fake_self",
        )
        fakeD_list.append(fakeD_i)
        realE_i = _run_scorer_one_shot(
            model.real_score, cond_for_dmdctx_i, noisy_input_i,
            clean_x=student_clean_x_i, aug_t=aug_t_zero_per_batch,
            label=f"E-{i}/real_self",
        )
        realE_list.append(realE_i)
        realF_i = _run_scorer_one_shot(
            model.real_score, cond_for_dmdctx_i, noisy_input_i,
            clean_x=gt_view_noised_i, aug_t=aug_t_gt_per_batch,
            label=f"F-{i}/real_GT",
        )
        realF_list.append(realF_i)

    # ---- Concatenate per-batch outputs along F and decode ---------------------
    log.info("=" * 60)
    log.info("Concatenating %d per-batch outputs and writing videos", num_rollouts)

    def _cat(name, lst, fname):
        full = torch.cat(lst, dim=1).contiguous()
        log.info("  %s  shape=%s mean=%.3f std=%.3f",
                 name, tuple(full.shape), float(full.float().mean()), float(full.float().std()))
        _write_mp4(out_dir / f"{fname}{out_suffix}.mp4", _decode_to_uint8(full, model.vae))
        return full

    fakeB_full = _cat("B fake_no_ctx",  fakeB_list, "B_fake_no_ctx")
    realC_full = _cat("C real_no_ctx",  realC_list, "C_real_no_ctx")
    fakeD_full = _cat("D fake_self",    fakeD_list, "D_fake_self")
    realE_full = _cat("E real_self",    realE_list, "E_real_self")
    realF_full = _cat("F real_GT",      realF_list, "F_real_GT")

    # ---- Per-frame std diagnostic (last-chunk-noise check) -------------
    # Hypothesis: at the END of every 21-frame scorer window the last
    # ``npb=3`` frames have systematically inflated std vs the GT.
    # If True, every i*N - 1 .. i*N - npb position should spike. This
    # log lets us verify before masking.
    log.info("=" * 60)
    log.info("PER-FRAME std (and |pred - GT|) for each tested scorer "
             "across the %d-frame composite. Watch frames %s "
             "(= last chunk of each rollout) for outlier inflation.",
             num_rollouts * N,
             ", ".join(str(i * N - 1) for i in range(1, num_rollouts + 1)))
    gt_for_per_frame = latents[:, cf + npb : cf + total_rollout_frames].float()
    for name, t in [
        ("D fake_self",   fakeD_full),
        ("E real_self",   realE_full),
        ("F real_GT",     realF_full),
        ("B fake_no_ctx", fakeB_full),
        ("C real_no_ctx", realC_full),
    ]:
        tf = t.float()
        per_frame_std = tf.std(dim=(0, 2, 3, 4)).cpu().numpy()
        per_frame_mae = (tf - gt_for_per_frame).abs().mean(dim=(0, 2, 3, 4)).cpu().numpy()
        std_str = " ".join(f"{v:.3f}" for v in per_frame_std)
        mae_str = " ".join(f"{v:.3f}" for v in per_frame_mae)
        log.info("  %-14s std: %s", name, std_str)
        log.info("  %-14s mae: %s", name, mae_str)

    # ---- Numeric summary -------------------------------------------------------
    log.info("=" * 60)
    # Compare everything on the NOISY-X span (= num_rollouts * N = 63 frames)
    # so all tensors share shape. student_noisy_span aligns with B/C/D/E/F.
    student_noisy_span = student_full[:, npb:].contiguous()
    gt_noisy_span = latents[:, cf + npb : cf + total_rollout_frames]
    span_frames = student_noisy_span.shape[1]
    log.info("Per-test x0 stats (over the %d-frame noisy span):", span_frames)
    for name, t in [
        ("student (noisy span)",      student_noisy_span),
        ("fake B (no clean_x)",       fakeB_full),
        ("real C (no clean_x)",       realC_full),
        ("fake D (self-view)",        fakeD_full),
        ("real E (self mode)",        realE_full),
        ("real F (GT mode + aug_t)",  realF_full),
        ("GT (ride-aligned)",         gt_noisy_span),
    ]:
        t = t.float()
        log.info(
            "  %-26s  mean=%+.3f std=%.3f  L2-vs-GT=%.4f  L2-vs-student=%.4f",
            name, float(t.mean()), float(t.std()),
            float((t - gt_noisy_span.float()).pow(2).mean().sqrt()),
            float((t - student_noisy_span.float()).pow(2).mean().sqrt()),
        )

    log.info("DONE. Videos in %s", out_dir)


if __name__ == "__main__":
    main()
