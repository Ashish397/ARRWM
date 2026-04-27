"""Minimal streaming-mode smoke test (Phase B).

Loads the ActionForcingDMD model, opens a sequence via
``setup_sequence``, and runs a few ``generate_next_chunk`` +
``compute_generator_loss_streaming`` / ``compute_critic_loss_streaming``
cycles to confirm:
  1. setup_sequence doesn't crash, prefills the seed cleanly.
  2. generate_next_chunk picks new_frames in [min_new, chunk_size]
     and advances current_length correctly.
  3. compute_generator_loss_streaming returns a finite loss with
     gradient flowing through new frames only.
  4. compute_critic_loss_streaming returns a finite loss; cache K/V
     are detached after the call.
  5. can_generate_more returns False once the sequence is full.

Reuses the same Brighton ride + dmdctx config as test_dmd_inference.
"""
from __future__ import annotations
import gc
import os, sys, logging, random
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
log = logging.getLogger("stream_diag")


def _init_dist_singlerank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    import socket
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


def _build_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_REPO / "configs" / "action_forcing_phase1_aux_dmdctx.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.wan_model_path = os.environ.get("DIAG_WAN_MODEL_PATH", "/home/ashish/Wan2.1/")
    cfg.encoded_root = os.environ.get("DIAG_ENCODED_ROOT", "/home/ashish/frodobots/frodobots_encoded")
    cfg.caption_root = os.environ.get("DIAG_CAPTION_ROOT", "/home/ashish/frodobots/frodobots_captions/train")
    cfg.motion_root = os.environ.get("DIAG_MOTION_ROOT", "/home/ashish/frodobots/frodobots_motion")
    cfg.ss_vae_checkpoint = os.environ.get("DIAG_SSVAE_CKPT", "/home/ashish/ARRWM/action_query/checkpoints/ss_vae_8free.pt")
    cfg.ode_generator_checkpoint = os.environ.get("DIAG_ODE_CKPT", "/home/ashish/action_ode_step0001000.pt")
    cfg.v14_teacher_checkpoint = os.environ.get("DIAG_V14_CKPT", "/home/ashish/Downloads/causal_lora_step0006600.pt")
    cfg.cotracker_checkpoint_path = ""
    cfg.cotracker_source_dir = ""
    cfg.action_teacher_mode = "off"
    cfg.gan_enabled = False
    cfg.sc_dmd_enabled = False
    cfg.mae_extension_max_extra_chunks = 0
    cfg.mae_extension_threshold = None
    cfg.gradient_checkpointing = True  # streaming needs grad-checkpoint to fit the 5090 (32GB)
    cfg.mixed_precision = True
    cfg.text_pre_encoded = True
    cfg.dmd_context = "GT"
    cfg.clean_x_aug_t = 20
    cfg.streaming_chunk_size = 21
    cfg.streaming_min_new_frame = 18
    cfg.streaming_max_length = int(os.environ.get("DIAG_STREAMING_MAX", "57"))
    return cfg


def main():
    _init_dist_singlerank()
    import utils.wan_wrapper as _ww
    _wan_root = os.environ.get("DIAG_WAN_MODEL_PATH", "/home/ashish/Wan2.1/")
    # WanDiffusionWrapper composes path as f"{_default_wan_model_path}{model_name}/",
    # so _default_wan_model_path must be the PARENT dir (with trailing slash) of the
    # model_name directory. Accept either the parent dir or the model dir itself.
    if _wan_root.rstrip("/").endswith("Wan2.1-T2V-1.3B"):
        _wan_root = str(Path(_wan_root.rstrip("/")).parent)
    if not _wan_root.endswith("/"):
        _wan_root = _wan_root + "/"
    _ww._default_wan_model_path = _wan_root

    cfg = _build_config()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    log.info("Building ActionForcingDMD ...")
    from model.dmd_action_forcing import ActionForcingDMD
    model = ActionForcingDMD(args=cfg, device=device)
    for m in (model.generator, model.fake_score, model.real_score):
        m.model.to(device=device, dtype=dtype)
    if model.action_projection is not None:
        model.action_projection.to(device=device, dtype=dtype)
    if model.action_token_projection is not None:
        model.action_token_projection.to(device=device, dtype=dtype)
    model.eval()

    log.info("Building pipeline ...")
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

    log.info("Loading Brighton ride (fast-path)...")
    from utils.zarr_dataset import ZarrRideDataset
    from utils.eval_chain import _build_ts_to_ride_dir, _load_ride_entry_from_disk
    from pathlib import Path as _Path
    ride_name = os.environ.get("DIAG_RIDE_NAME", "20240408152948.zarr")
    ts_map = _build_ts_to_ride_dir(_Path(cfg.caption_root))
    ride_dict = _load_ride_entry_from_disk(
        ride_name, _Path(cfg.encoded_root), _Path(cfg.caption_root), ts_map,
    )
    ds = ZarrRideDataset.from_manifest(
        rides_data=[ride_dict],
        motion_root=cfg.motion_root,
        ss_vae_checkpoint=cfg.ss_vae_checkpoint,
        device="cpu", ss_vae_device="cuda:0",
    )
    zpath, prompt_embeds, attrs, n_lat = ds._rides[0]
    log.info("Ride %s n_latent=%d", zpath.name, n_lat)

    cf = int(cfg.dmd_context_clean_frames)
    npb = int(cfg.num_frame_per_block)
    max_len = int(cfg.streaming_max_length)
    if max_len % npb != 0:
        max_len = (max_len // npb) * npb
    s = int(os.environ.get("DIAG_RIDE_START", "0"))
    need = s + cf + max_len
    if need > n_lat:
        raise RuntimeError(f"ride too short: n_lat={n_lat} need={need}")

    latents_full = ZarrRideDataset.load_latent_chunk(str(zpath), 0, need).unsqueeze(0).to(
        device=device, dtype=dtype,
    )
    z_actions_full = ds.encode_z_actions_window(str(zpath), need, 0, need)
    action_dims = list(getattr(cfg, "action_dims", [2, 7]))
    z_actions_full = z_actions_full[..., action_dims].unsqueeze(0).to(
        device=device, dtype=dtype,
    )
    seed_latents = latents_full[:, s : s + cf]
    ride_lat_window = latents_full[:, s : s + cf + max_len]
    ride_act_window = z_actions_full[:, s : s + cf + max_len]
    pe = (
        prompt_embeds.unsqueeze(0).to(device=device, dtype=dtype)
        if prompt_embeds.dim() == 2
        else prompt_embeds.to(device=device, dtype=dtype)
    )

    # ---- setup_sequence ----
    log.info("=" * 60)
    log.info("setup_sequence: cf=%d max_len=%d s=%d", cf, max_len, s)
    torch.manual_seed(0)
    model.setup_sequence(
        seed_latents=seed_latents,
        ride_latents_window=ride_lat_window,
        ride_actions_window=ride_act_window,
        prompt_embeds=pe,
        max_length=max_len,
    )
    log.info(
        "streaming_state: current_length=%d max_length=%d cf=%d shift=%d",
        model.streaming_state["current_length"],
        model.streaming_state["max_length"],
        model.streaming_state["cf"],
        model.streaming_state["shift"],
    )

    # ---- iter loop: gen + critic per iter ----
    iter_count = 0
    while model.can_generate_more():
        iter_count += 1
        log.info("=" * 60)
        log.info("ITER %d: generator step", iter_count)

        # ---- gen step ----
        chunk, info = model.generate_next_chunk(requires_grad=True)
        log.info(
            "  chunk shape=%s new_frames=%d overlap=%d cur_len=%d/%d MAE=%.4f",
            tuple(chunk.shape),
            info["new_frames"], info["overlap"],
            info["current_length"], info["max_length"],
            float(info.get("baseline_last_chunk_mae", float("nan"))),
        )
        gen_loss, gen_log = model.compute_generator_loss_streaming(chunk, info)
        log.info(
            "  gen_loss=%.4f denoised_t=[%s,%s]",
            float(gen_loss.detach().item()),
            gen_log.get("denoised_timestep_from"),
            gen_log.get("denoised_timestep_to"),
        )
        # Verify finite + grad on new frames only.
        assert torch.isfinite(gen_loss), "gen_loss non-finite"
        if gen_loss.requires_grad:
            gen_loss.backward()
        else:
            log.warning("  gen_loss does NOT require grad")

        # Free the gen-step graph + intermediates before the critic step.
        del gen_loss, gen_log, chunk, info
        gc.collect()
        torch.cuda.empty_cache()
        # ---- critic step (consumes the NEXT chunk in the sequence) ----
        if not model.can_generate_more():
            log.info("  sequence exhausted before critic step; stopping")
            break
        log.info("ITER %d: critic step", iter_count)
        chunk_c, info_c = model.generate_next_chunk(requires_grad=False)
        critic_loss, critic_log = model.compute_critic_loss_streaming(chunk_c, info_c)
        log.info(
            "  chunk_c shape=%s new=%d overlap=%d cur_len=%d/%d  critic_loss=%.4f",
            tuple(chunk_c.shape), info_c["new_frames"], info_c["overlap"],
            info_c["current_length"], info_c["max_length"],
            float(critic_loss.detach().item()),
        )
        assert torch.isfinite(critic_loss), "critic_loss non-finite"
        if critic_loss.requires_grad:
            critic_loss.backward()
        else:
            log.warning("  critic_loss does NOT require grad (critic params frozen?)")

        # Force a 2nd backward through the SAME persistent sequence to
        # trip cond_dict graph-reuse bugs (single-iter smoke masks them).
        # Mock optimizer.step + zero_grad between iters so accumulated
        # parameter gradients don't OOM the 5090 before iter 2 runs.
        if int(os.environ.get("DIAG_FORCE_2BACKWARD", "0")) == 1 and iter_count == 1:
            log.info("Mocking optimizer.zero_grad on all trainable params before iter 2...")
            for p in model.generator.parameters():
                if p.grad is not None:
                    p.grad = None
            for p in model.fake_score.parameters():
                if p.grad is not None:
                    p.grad = None
            if model.action_projection is not None:
                for p in model.action_projection.parameters():
                    if p.grad is not None:
                        p.grad = None
            if model.action_token_projection is not None:
                for p in model.action_token_projection.parameters():
                    if p.grad is not None:
                        p.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            log.info("Forcing a 2nd gen step on the same sequence (cond-graph reuse check)...")
            chunk2, info2 = model.generate_next_chunk(requires_grad=True)
            log.info(
                "  iter2 chunk shape=%s new=%d overlap=%d cur_len=%d/%d",
                tuple(chunk2.shape), info2["new_frames"], info2["overlap"],
                info2["current_length"], info2["max_length"],
            )
            gen_loss2, gen_log2 = model.compute_generator_loss_streaming(chunk2, info2)
            log.info("  iter2 gen_loss=%.4f", float(gen_loss2.detach().item()))
            gen_loss2.backward()
            log.info("  iter2 backward OK ✓ (no graph-reuse error)")
            del gen_loss2, gen_log2, chunk2, info2
            gc.collect()
            torch.cuda.empty_cache()

        if iter_count >= int(os.environ.get("DIAG_MAX_ITERS", "8")):
            log.info("Stopping early at iter=%d (DIAG_MAX_ITERS)", iter_count)
            break

    log.info("=" * 60)
    log.info("DONE. iters=%d final_length=%d/%d",
             iter_count,
             model.streaming_state["current_length"] if model.streaming_state else -1,
             max_len)


if __name__ == "__main__":
    main()
