"""Diagnostic for the ``teacher_freeze_mode='action'`` decision.

Runs F (real_score with clean_x_GT) on a single ride window, runs the
frozen action teacher (CoTracker + ss_vae) on:
  * the GT video  → z_gt   [B, n_slots, 8]
  * the student's rollout  → z_student
  * the real_score's pred  → z_real
computes per-slot cosine_similarity(z_real, z_student), and overlays
the result on the F mp4 frame-by-frame:

  Top-left  per chunk:
      slot k  cos=±x.xx  → FREEZE YES/NO
  Top-right per chunk:
      cmd_action [a₀ a₁]
      z_student[:3] [...]
      z_real[:3]    [...]

The user can scrub the mp4 to find slots where the teacher disagrees
hard with the student and verify the threshold is sane.
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
log = logging.getLogger("freeze_action")


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
    log.info("wrote %s (%d frames)", path, frames_uint8.shape[0])


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


def _overlay_text(
    frames: np.ndarray,
    *,
    npb: int,
    cmd_actions: np.ndarray,        # [F, A]
    z_student: np.ndarray,          # [n_slots, 8]
    z_real: np.ndarray,             # [n_slots, 8]
    cos_per_slot: np.ndarray,       # [n_slots]
    freeze_per_slot: np.ndarray,    # [n_slots] bool — action-cos DROP
    last_chunk_per_slot: np.ndarray,  # [n_slots] bool — structural last-chunk DROP
    rollout_idx_per_slot: np.ndarray,  # [n_slots] int — which rollout each slot belongs to
    threshold: float,
) -> np.ndarray:
    """Burn per-slot info onto the frames.

    Three decision states per slot:
      * action_cos      DROP_GRAD/KEEP_GRAD (cos < threshold)
      * last_chunk      DROP_GRAD/KEEP_GRAD (= last npb frames of every
                        rollout's scoring window — always DROP under the
                        structural ``_dmd_score_grad_mask``)
      * EFFECTIVE       (action_cos KEEP) AND (last_chunk KEEP)
    """
    from PIL import Image, ImageDraw, ImageFont
    F_pix, H, W = frames.shape[:3]
    F_lat = cmd_actions.shape[0]
    pix_per_lat = max(1, F_pix // F_lat)
    pix_per_slot = pix_per_lat * npb
    n_slots = z_student.shape[0]
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    out = np.empty_like(frames)
    for f in range(F_pix):
        img = Image.fromarray(frames[f])
        draw = ImageDraw.Draw(img)
        slot = min(n_slots - 1, f // pix_per_slot)
        lat_idx = min(F_lat - 1, f // pix_per_lat)
        cos = float(cos_per_slot[slot])
        action_drop = bool(freeze_per_slot[slot])
        lastchunk_drop = bool(last_chunk_per_slot[slot])
        effective_drop = action_drop or lastchunk_drop
        ro_idx = int(rollout_idx_per_slot[slot])
        cmd_a = cmd_actions[lat_idx]
        zs = z_student[slot]
        zr = z_real[slot]

        # ---- Top-left box: action-cos decision ------------------
        action_decision = "DROP_GRAD" if action_drop else "KEEP_GRAD"
        action_color = (255, 64, 64) if action_drop else (64, 220, 64)
        draw.rectangle([(0, 0), (270, 76)], fill=(0, 0, 0))
        draw.text(
            (4, 2), f"frame {f:3d}  rollout {ro_idx}  slot {slot}",
            fill=(220, 220, 220), font=font,
        )
        draw.text(
            (4, 16), f"cos(real,student) = {cos:+.3f}",
            fill=(220, 220, 220), font=font,
        )
        draw.text(
            (4, 30), f"thresh = {threshold:+.2f}",
            fill=(180, 180, 180), font=font,
        )
        draw.text(
            (4, 44), f"ACTION mask:  {action_decision}",
            fill=action_color, font=font,
        )
        # ---- Top-left lower: structural last-chunk decision ---
        lc_decision = "DROP_GRAD" if lastchunk_drop else "KEEP_GRAD"
        lc_color = (255, 64, 64) if lastchunk_drop else (64, 220, 64)
        draw.text(
            (4, 60), f"LAST-CHUNK:   {lc_decision}",
            fill=lc_color, font=font,
        )
        # ---- Effective box (right under the two masks) --------
        eff_text = "EFFECTIVE: DROP_GRAD" if effective_drop else "EFFECTIVE: KEEP_GRAD"
        eff_color = (255, 96, 96) if effective_drop else (64, 240, 96)
        draw.rectangle([(0, 78), (270, 96)], fill=(0, 0, 0))
        draw.text(
            (4, 80), eff_text, fill=eff_color, font=font,
        )

        # ---- Top-right: cmd_action + z_student / z_real ---------
        rx = W - 320
        draw.rectangle([(rx, 0), (W, 76)], fill=(0, 0, 0))
        draw.text(
            (rx + 4, 2),
            "cmd: " + " ".join(f"{a:+.2f}" for a in cmd_a[:4]),
            fill=(180, 220, 180), font=font,
        )
        draw.text(
            (rx + 4, 18),
            "z_stud[:4]: " + " ".join(f"{v:+.2f}" for v in zs[:4]),
            fill=(180, 220, 220), font=font,
        )
        draw.text(
            (rx + 4, 32),
            "z_real[:4]: " + " ".join(f"{v:+.2f}" for v in zr[:4]),
            fill=(220, 180, 220), font=font,
        )
        draw.text(
            (rx + 4, 50),
            f"|z_real|={np.linalg.norm(zr):.2f}  |z_stud|={np.linalg.norm(zs):.2f}",
            fill=(180, 180, 180), font=font,
        )
        out[f] = np.asarray(img)
    return out


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
    cfg.cotracker_source_dir = os.environ.get(
        "DIAG_COTRACKER_SOURCE_DIR",
        "/home/ashish/.cache/torch/hub/facebookresearch_co-tracker_main",
    )
    cfg.cotracker_checkpoint_path = os.environ.get(
        "DIAG_COTRACKER_CHECKPOINT",
        "/home/ashish/.cache/torch/hub/checkpoints/scaled_offline.pth",
    )
    # Action-mode freeze prerequisites — must all be live or the
    # ActionForcingDMD __init__ assert fails.
    cfg.action_teacher_mode = "all"
    cfg.action_critic_aux_enabled = True
    cfg.state_probe_aux_enabled = True
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
    cfg.teacher_freeze_detect_enabled = True
    cfg.teacher_freeze_mode = "action"
    cfg.teacher_freeze_action_threshold = float(os.environ.get(
        "DIAG_FREEZE_COS_THRESH", "0.0",
    ))
    if hasattr(cfg, "dmd_real_GT"):
        del cfg.dmd_real_GT
    if hasattr(cfg, "dmd_real_GT_aug_t"):
        del cfg.dmd_real_GT_aug_t
    return cfg


def _build_action_teacher_for_diag(device):
    """Manually load CoTracker + ss_vae (mirrors the trainer's
    ``_build_action_teacher`` since we're not running the full trainer)."""
    src_dir = os.environ.get(
        "DIAG_COTRACKER_SOURCE_DIR",
        "/home/ashish/.cache/torch/hub/facebookresearch_co-tracker_main",
    )
    ckpt = os.environ.get(
        "DIAG_COTRACKER_CHECKPOINT",
        "/home/ashish/.cache/torch/hub/checkpoints/scaled_offline.pth",
    )
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"CoTracker checkpoint not found at {ckpt}. Set "
            "DIAG_COTRACKER_CHECKPOINT to the correct path."
        )
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    from cotracker.predictor import CoTrackerPredictor  # type: ignore
    cot = CoTrackerPredictor(checkpoint=ckpt, window_len=60, v2=False).to(device)
    cot.eval()
    for p in cot.parameters():
        p.requires_grad_(False)
    from action_query.ss_vae_model import load_ss_vae
    ss_vae, scale = load_ss_vae(
        "/home/ashish/ARRWM/action_query/checkpoints/ss_vae_8free.pt",
        device=str(device),
    )
    ss_vae.eval()
    ss_vae.requires_grad_(False)
    return cot, ss_vae, float(scale)


def _make_teacher_z_fn(model, cot, ss_vae, ss_vae_scale, npb):
    """Closure that returns ``[B, n_slots, 8]`` z per slot for any
    ``[B, F, C, H, W]`` latent — matches the trainer's
    ``_compute_teacher_z_per_slot`` signature.
    """
    @torch.no_grad()
    def fn(pred_x0_all_slots: torch.Tensor):
        if pred_x0_all_slots is None:
            return None
        x0 = pred_x0_all_slots.detach()
        if x0.dim() != 5:
            return None
        B, F = x0.shape[0], x0.shape[1]
        if F % npb != 0:
            return None
        n_slots = F // npb
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pixels = model.vae.decode_to_pixel(x0.float())
        video = (255.0 * 0.5 * (pixels + 1.0)).clamp(0, 255).float()
        T_pix = video.shape[1]
        if T_pix < 2:
            return None
        per_slot_pix = T_pix // n_slots
        if per_slot_pix < 2:
            return None
        used_pix = per_slot_pix * n_slots
        vid = video[:, :used_pix]
        with torch.amp.autocast(device_type="cuda", enabled=True):
            pred_tracks, pred_vis = cot(vid, grid_size=10)
        d = pred_tracks[:, 1:] - pred_tracks[:, :-1]
        if pred_vis.dim() == 3:
            vis = pred_vis.unsqueeze(-1)
        else:
            vis = pred_vis
        vis = vis[:, 1:]
        eff_len = used_pix - 1
        base = eff_len // n_slots
        if base < 1:
            return None
        d = d[:, : base * n_slots]
        vis = vis[:, : base * n_slots]
        d = d.reshape(B, n_slots, base, 100, 2).mean(dim=2)
        xy = d.reshape(B * n_slots, 10, 10, 2)
        x_in = xy.permute(0, 3, 1, 2).float() / ss_vae_scale
        mu, _ = ss_vae.encoder(x_in.to(x0.device))
        z = mu.squeeze(-1).squeeze(-1)
        from utils.zarr_dataset import _tanh_squash
        z = _tanh_squash(z)
        z = z.reshape(B, n_slots, -1)
        return z.detach().contiguous()
    return fn


def main():
    _init_dist_singlerank()
    import utils.wan_wrapper as _ww
    _ww._default_wan_model_path = "/home/ashish/Wan2.1/"

    cfg = _build_config()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    log.info("Building ActionForcingDMD (with action-mode freeze prerequisites)...")
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

    log.info("Building action teacher (cotracker + ss_vae)...")
    cot, ss_vae, ss_vae_scale = _build_action_teacher_for_diag(device)
    npb = int(cfg.num_frame_per_block)
    teacher_z_fn = _make_teacher_z_fn(model, cot, ss_vae, ss_vae_scale, npb)
    model._action_teacher_fn = teacher_z_fn
    log.info("Attached _action_teacher_fn to model.")

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
    num_rollouts = int(os.environ.get("DIAG_NUM_ROLLOUTS", "3"))
    shift = npb
    total_rollout_frames = num_rollouts * N + npb  # +npb leading anchor

    # Load ride.
    log.info("Loading ride %s ...", os.environ.get("DIAG_RIDE_NAME", "20240408152948.zarr"))
    from utils.zarr_dataset import ZarrRideDataset
    from utils.eval_chain import _build_ts_to_ride_dir, _load_ride_entry_from_disk
    ride_name = os.environ.get("DIAG_RIDE_NAME", "20240408152948.zarr")
    ts_map = _build_ts_to_ride_dir(Path(cfg.caption_root))
    ride_dict = _load_ride_entry_from_disk(
        ride_name, Path(cfg.encoded_root), Path(cfg.caption_root), ts_map,
    )
    ds = ZarrRideDataset.from_manifest(
        rides_data=[ride_dict], motion_root=cfg.motion_root,
        ss_vae_checkpoint=cfg.ss_vae_checkpoint,
        device="cpu", ss_vae_device="cuda:0",
    )
    zpath, prompt_embeds, attrs, n_lat = ds._rides[0]

    s = int(os.environ.get("DIAG_RIDE_START", "0"))
    need = s + cf + total_rollout_frames
    if need > n_lat:
        raise RuntimeError(
            f"ride {zpath.name} has only {n_lat} latent frames; need "
            f"s+cf+(num_rollouts*N+npb)={need}. Pick a longer ride or "
            f"smaller s."
        )

    latents_full = ZarrRideDataset.load_latent_chunk(str(zpath), 0, need).unsqueeze(0).to(device=device, dtype=dtype)
    z_actions_full = ds.encode_z_actions_window(str(zpath), need, 0, need)
    action_dims = list(getattr(cfg, "action_dims", [2, 7]))
    z_actions_full = z_actions_full[..., action_dims].unsqueeze(0).to(device=device, dtype=dtype)
    latents = latents_full[:, s:s + cf + total_rollout_frames]
    z_actions = z_actions_full[:, s:s + cf + total_rollout_frames]
    pe = prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype) if prompt_embeds.dim() == 2 else prompt_embeds.to(device=device, dtype=dtype)
    seed_latents = latents[:, :cf]

    log.info(
        "ride_start s=%d  num_rollouts=%d  total_rollout=%d frames "
        "(= %d-frame anchor + %d × %d-frame scoring windows)",
        s, num_rollouts, total_rollout_frames, npb, num_rollouts, N,
    )

    # cond_dicts spanning the FULL window (seed + anchor + rollouts).
    full_actions = z_actions[:, : cf + total_rollout_frames]
    cond_full, _ = model.build_action_conditional(
        prompt_embeds=pe, gt_actions=full_actions,
    )

    # Roll the student over all rollouts in one pipeline call.
    log.info("Rolling student (%d frames)...", total_rollout_frames)
    torch.manual_seed(42)
    noise = torch.randn(
        [1, total_rollout_frames, *latents.shape[2:]],
        dtype=dtype, device=device,
    )
    with torch.no_grad():
        student_full, _, _ = pipe.inference_with_trajectory(
            noise=noise, gt_latents=None, enable_mae_extension=False,
            seed_latents=seed_latents, prefer_cache_pred_in_output=True,
            **cond_full,
        )
    log.info("student rollout shape=%s", tuple(student_full.shape))

    # Per-rollout: build slices, run F, run action teacher, collect decisions.
    F_chunks = []
    student_chunks = []
    cmd_actions_per_rollout = []
    cos_per_slot_all = []
    freeze_per_slot_all = []
    last_chunk_per_slot_all = []
    rollout_idx_per_slot_all = []
    z_student_all = []
    z_real_all = []
    timestep_per_batch = torch.full(
        (1, N), fill_value=int(os.environ.get("DIAG_F_T", "500")),
        dtype=torch.long, device=device,
    )
    aug_t_zero_per_batch = torch.zeros(1, N, dtype=torch.long, device=device)
    threshold = float(cfg.teacher_freeze_action_threshold)
    n_slots_per_batch = N // npb  # = 7 with N=21, npb=3

    log.info("=" * 78)
    log.info(
        "Per-slot decisions (threshold = %+.2f). FREEZE METRIC = cos(R,GT) < threshold "
        "(teacher direction wrong vs GT). 'LASTCHUNK' = structural mask on "
        "the final %d frames of every rollout. EFFECTIVE = OR of the two. "
        "(cos(R,S) and cos(S,GT) are context only.)",
        threshold, npb,
    )
    log.info(
        "  ro slot frames        cmd_action            cos(R,GT)  cos(R,S)  cos(S,GT)  ACTION     LAST     EFFECTIVE"
    )

    for i in range(1, num_rollouts + 1):
        noisy_start_sdn = (i - 1) * N + npb
        noisy_end_sdn   = i * N + npb
        clean_start_sdn = noisy_start_sdn - shift
        clean_end_sdn   = noisy_end_sdn - shift

        # Per-rollout slices in absolute ride coords (for cond + GT).
        abs_noisy_start = cf + noisy_start_sdn
        abs_noisy_end   = cf + noisy_end_sdn
        abs_clean_start = cf + clean_start_sdn
        abs_clean_end   = cf + clean_end_sdn

        student_pred_i = student_full[:, noisy_start_sdn:noisy_end_sdn].contiguous()
        student_chunks.append(student_pred_i)
        student_clean_x_i = student_full[:, clean_start_sdn:clean_end_sdn].contiguous()

        gt_clean_i = latents[:, abs_clean_start:abs_clean_end].contiguous()
        gt_noisy_i = latents[:, abs_noisy_start:abs_noisy_end].contiguous()
        actions_noisy_i = z_actions[:, abs_noisy_start:abs_noisy_end]
        actions_clean_i = z_actions[:, abs_clean_start:abs_clean_end]

        cond_noisy_i, _ = model.build_action_conditional(
            prompt_embeds=pe, gt_actions=actions_noisy_i,
        )
        cond_clean_i, _ = model.build_action_conditional(
            prompt_embeds=pe, gt_actions=actions_clean_i,
        )
        cond_for_F = dict(cond_noisy_i)
        for k, v in cond_clean_i.items():
            if k in ("_action_modulation", "_action_tokens"):
                cond_for_F[k + "_clean"] = v

        # Re-noise + run F (real_score with clean_x_GT, aug_t=0).
        torch.manual_seed(42 + i)
        noise_for_dmd = torch.randn_like(student_pred_i)
        noisy_input_i = model.scheduler.add_noise(
            student_pred_i.flatten(0, 1),
            noise_for_dmd.flatten(0, 1),
            timestep_per_batch.flatten(0, 1),
        ).unflatten(0, (1, N))
        with torch.no_grad():
            _, pred_F_i = model.real_score(
                noisy_image_or_video=noisy_input_i,
                conditional_dict=cond_for_F,
                timestep=timestep_per_batch,
                clean_x=gt_clean_i.to(dtype=dtype),
                aug_t=aug_t_zero_per_batch,
            )
        F_chunks.append(pred_F_i)

        # Action teacher per-rollout.
        z_gt_i = teacher_z_fn(gt_noisy_i)
        z_student_i = teacher_z_fn(student_pred_i)
        z_real_i = teacher_z_fn(pred_F_i)
        if z_gt_i is None or z_student_i is None or z_real_i is None:
            raise RuntimeError("Action teacher returned None on rollout %d" % i)

        # FREEZE METRIC: cos(z_real, z_gt). The teacher is "wrong" iff
        # its predicted action direction is OPPOSITE to GT. Cosine sim
        # is magnitude-invariant, so a teacher that gets the magnitude
        # wrong but the direction right stays unmasked.
        cos_RG = torch.nn.functional.cosine_similarity(
            z_real_i.float(), z_gt_i.float(), dim=-1,
        )[0].cpu().numpy()
        # Telemetry only:
        cos_RS = torch.nn.functional.cosine_similarity(
            z_real_i.float(), z_student_i.float(), dim=-1,
        )[0].cpu().numpy()
        cos_SG = torch.nn.functional.cosine_similarity(
            z_student_i.float(), z_gt_i.float(), dim=-1,
        )[0].cpu().numpy()
        action_freeze = (cos_RG < threshold)  # [n_slots_per_batch]
        # Structural last-chunk mask: last slot of EVERY rollout drops.
        last_chunk = np.zeros(n_slots_per_batch, dtype=bool)
        last_chunk[-1] = True

        cmd_per_lat_i = z_actions[0, abs_noisy_start:abs_noisy_end].float().cpu().numpy()  # [N, A]
        cmd_actions_per_rollout.append(cmd_per_lat_i)
        cos_per_slot_all.append(cos_RS)
        freeze_per_slot_all.append(action_freeze)
        last_chunk_per_slot_all.append(last_chunk)
        rollout_idx_per_slot_all.append(np.full(n_slots_per_batch, i, dtype=int))
        z_student_all.append(z_student_i[0].float().cpu().numpy())
        z_real_all.append(z_real_i[0].float().cpu().numpy())

        # Per-slot log row.
        for k in range(n_slots_per_batch):
            f0 = k * npb
            cmd_str = " ".join(f"{v:+.2f}" for v in cmd_per_lat_i[f0][:4])
            ad = "DROP" if action_freeze[k] else "KEEP"
            ld = "DROP" if last_chunk[k] else "KEEP"
            ed = "DROP" if (action_freeze[k] or last_chunk[k]) else "KEEP"
            log.info(
                "   %d  %2d   [%2d:%2d]  cmd=[%s]  %+.3f    %+.3f    %+.3f    %s     %s     %s",
                i, k, f0, f0 + npb, cmd_str,
                cos_RG[k], cos_RS[k], cos_SG[k], ad, ld, ed,
            )

    # Aggregate per-rollout outputs.
    F_full = torch.cat(F_chunks, dim=1)              # [1, num_rollouts*N, ...]
    student_full_pred = torch.cat(student_chunks, dim=1)
    cmd_actions_lat = np.concatenate(cmd_actions_per_rollout, axis=0)  # [num_rollouts*N, A]
    cos_per_slot_full = np.concatenate(cos_per_slot_all)
    freeze_per_slot_full = np.concatenate(freeze_per_slot_all)
    last_chunk_per_slot_full = np.concatenate(last_chunk_per_slot_all)
    rollout_idx_per_slot_full = np.concatenate(rollout_idx_per_slot_all)
    z_student_full = np.concatenate(z_student_all, axis=0)
    z_real_full = np.concatenate(z_real_all, axis=0)

    n_slots_total = int(freeze_per_slot_full.size)
    action_drops = int(freeze_per_slot_full.sum())
    last_drops = int(last_chunk_per_slot_full.sum())
    eff_drops = int((freeze_per_slot_full | last_chunk_per_slot_full).sum())
    log.info("=" * 78)
    log.info(
        "ACTION mask:   %d/%d slots → %d/%d frames",
        action_drops, n_slots_total, action_drops * npb, n_slots_total * npb,
    )
    log.info(
        "LAST-CHUNK:    %d/%d slots → %d/%d frames "
        "(structural; the last slot of EVERY rollout)",
        last_drops, n_slots_total, last_drops * npb, n_slots_total * npb,
    )
    log.info(
        "EFFECTIVE:     %d/%d slots → %d/%d frames "
        "(union of action and last-chunk masks)",
        eff_drops, n_slots_total, eff_drops * npb, n_slots_total * npb,
    )

    out_dir = _REPO / "_diag_p1" / "out" / "freeze_action"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Decoding videos + overlaying per-slot info...")
    F_uint8 = _decode_to_uint8(F_full, model.vae)
    student_uint8 = _decode_to_uint8(student_full_pred, model.vae)
    GT_full = latents[:, cf + npb : cf + total_rollout_frames]
    GT_uint8 = _decode_to_uint8(GT_full, model.vae)

    overlay_kwargs = dict(
        npb=npb,
        cmd_actions=cmd_actions_lat,
        z_student=z_student_full,
        z_real=z_real_full,
        cos_per_slot=cos_per_slot_full,
        freeze_per_slot=freeze_per_slot_full,
        last_chunk_per_slot=last_chunk_per_slot_full,
        rollout_idx_per_slot=rollout_idx_per_slot_full,
        threshold=threshold,
    )
    F_overlay = _overlay_text(F_uint8, **overlay_kwargs)
    student_overlay = _overlay_text(student_uint8, **overlay_kwargs)

    suffix = os.environ.get("DIAG_OUT_SUFFIX", "")
    _write_mp4(out_dir / f"F_real_GT_with_decision{suffix}.mp4", F_overlay)
    _write_mp4(out_dir / f"student_with_decision{suffix}.mp4", student_overlay)
    _write_mp4(out_dir / f"GT_ride{suffix}.mp4", GT_uint8)

    log.info("DONE. Videos in %s", out_dir)


if __name__ == "__main__":
    main()
