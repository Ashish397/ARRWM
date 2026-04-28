"""Side-by-side comparison: streaming-mode iter-1 inputs vs the diag's
per-batch inputs (test_dmd_inference.py path the user verified).

Goal: locate the divergence that makes pred_real OOD when we score the
student's chunk via the streaming code path, even though the same
real_score on the same student rollout produces good output via the
batch-style diag code path.

Plan:
  1. Build the model from configs/smoke_extender.yaml (= the failing
     config) and load the same ride loader the trainer uses.
  2. Open a streaming sequence (seed=0, deterministic offset).
  3. Call ``model.generate_next_chunk(requires_grad=False)`` for iter 1
     and capture all the inputs that flow into ``real_score``.
  4. Independently re-derive the same inputs using the diag's per-batch
     formulas on the SAME ``streaming_state`` tensors (ride_latents_window,
     ride_actions_window, prompt_embeds).
  5. Print per-tensor diffs (shape, mean abs, max abs, top-k mismatched
     positions) for: clean_x_GT, noisy-half action streams (modulation +
     tokens), clean-half action streams.
  6. Add the same noise pattern + DMD timestep to ``chunk`` to build
     ``noisy_input``, run real_score under each cond/clean_x set, and
     compare ``pred_real`` outputs (mean abs diff and saved as mp4 for
     visual diff).

Run on the existing srun:
  srun --jobid=<JOB> --overlap --ntasks=1 --gres=gpu:1 \
       bash -c '
         cd /scratch/u6ex/as1748.u6ex/ARRWM
         set --
         source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
         conda activate arrwm
         export PYTHONPATH=$(pwd):$PYTHONPATH
         python _diag_p1/compare_streaming_vs_diag.py
       '
"""
from __future__ import annotations

import logging
import os
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_REPO = Path("/scratch/u6ex/as1748.u6ex/ARRWM").resolve()
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("compare")


def _ensure_cuda_device() -> None:
    """``torchrun`` sets RANK/WORLD_SIZE/LOCAL_RANK + master env; the
    trainer's ``_setup_distributed`` gates on RANK in env, so when this
    script is launched via ``torchrun --nproc_per_node=1`` the trainer
    handles ``dist.init_process_group`` for us. We only need to bind
    this process to the GPU."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def _write_mp4(path: Path, frames_uint8: np.ndarray, fps: int = 5) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path), frames_uint8, fps=fps, codec="libx264", quality=8,
    )
    log.info(
        "wrote %s (%d frames, %dx%d)",
        path, *frames_uint8.shape[:3],
    )


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


def _diff_tensor(
    name: str,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    if a is None and b is None:
        log.info("[diff] %s: BOTH NONE", name)
        return
    if a is None or b is None:
        log.error(
            "[diff] %s: ONE NONE (streaming=%s diag=%s)",
            name,
            "None" if a is None else f"shape={tuple(a.shape)}",
            "None" if b is None else f"shape={tuple(b.shape)}",
        )
        return
    if a.shape != b.shape:
        log.error(
            "[diff] %s: SHAPE MISMATCH streaming=%s diag=%s",
            name, tuple(a.shape), tuple(b.shape),
        )
        return
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    mean_abs = float(diff.mean().item())
    max_abs = float(diff.max().item())
    rel_max = max_abs / (float(bf.abs().max().item()) + 1e-12)
    matches = torch.allclose(af, bf, rtol=rtol, atol=atol)
    tag = "OK    " if matches else "DIFF  "
    log.info(
        "[diff] %s %s shape=%s mean_abs=%.3e max_abs=%.3e rel_max=%.3e",
        tag, name, tuple(a.shape), mean_abs, max_abs, rel_max,
    )
    if not matches:
        # Top-k mismatches by abs diff for diagnostic.
        flat_diff = diff.flatten()
        k = min(5, flat_diff.numel())
        top_v, top_i = torch.topk(flat_diff, k)
        log.info(
            "    top-%d abs diffs: %s",
            k,
            ", ".join(f"{float(v):.3e}" for v in top_v),
        )
        # Per-leading-dim slice means (helps localise where the diff
        # is concentrated — first frame? all frames?).
        if a.dim() >= 2:
            per_dim = (
                diff.flatten(2).mean(dim=-1)
                if diff.dim() >= 3
                else diff.mean(dim=-1)
            )
            log.info(
                "    per-frame mean_abs (first 21): %s",
                per_dim[0, :21].cpu().tolist()
                if per_dim.dim() >= 2
                else per_dim[:21].cpu().tolist(),
            )


def _diff_dict(
    name: str,
    a: dict,
    b: dict,
    *,
    keys: list[str] | None = None,
) -> None:
    log.info("=== diff dict: %s ===", name)
    if keys is None:
        keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        ta = a.get(k)
        tb = b.get(k)
        if torch.is_tensor(ta) or torch.is_tensor(tb):
            _diff_tensor(f"{name}[{k}]", ta, tb)
        else:
            same = ta == tb
            log.info(
                "[diff] %-6s %s[%s] streaming=%r diag=%r",
                "OK" if same else "DIFF",
                name, k, ta, tb,
            )


def main() -> None:
    _ensure_cuda_device()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ---- Load smoke config (this is the path that produces the OOD pred_real).
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_REPO / "configs" / "smoke_extender.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.encoded_root = "/projects/u6ex/fbots/frodobots_encoded_smoke_low"
    cfg.disable_wandb = True
    cfg.gradient_checkpointing = False  # forward-only diag
    cfg.mixed_precision = True
    cfg.strict_resume_load = False
    cfg.strict_ode_load = False
    cfg.action_teacher_mode = "off"
    cfg.gan_loss_weight = 0.0
    cfg.mae_extension_max_extra_chunks = 0
    cfg.mae_extension_threshold = None

    # Determinism: seed=0 throughout.
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # ---- Build the trainer just enough to use its ride loader and
    # the model's _streaming_setup_sequence_from_ride.
    from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
    log.info("Building ActionForcingDMDTrainer ...")
    trainer = ActionForcingDMDTrainer(cfg)
    trainer.model.eval()

    rollout_frames = int(cfg.num_training_frames)
    cf_dmdctx = int(getattr(cfg, "dmd_context_clean_frames", 9))
    max_total_rollout_frames = rollout_frames

    # ---- Open a streaming sequence on a fixed ride.
    trainer._epoch = 0
    trainer._ride_iter = trainer._fresh_ride_iter(0)
    log.info("Opening streaming sequence (cf=%d, rollout=%d) ...",
             cf_dmdctx, rollout_frames)
    ok = trainer._streaming_setup_sequence_from_ride(
        rollout_frames=rollout_frames,
        max_total_rollout_frames=max_total_rollout_frames,
        cf_dmdctx=cf_dmdctx,
    )
    if not ok:
        log.error("Failed to open streaming sequence (no ride available).")
        return
    s = trainer.model.streaming_state
    log.info("streaming_state opened: cf=%d shift=%d chunk_size=%d "
             "current_length=%d max_length=%d ride_latents=%s "
             "ride_actions=%s",
             int(s["cf"]), int(s["shift"]), int(s["chunk_size"]),
             int(s["current_length"]), int(s["max_length"]),
             tuple(s["ride_latents_window"].shape),
             tuple(s["ride_actions_window"].shape))

    # ---- Iter 1: generate the next chunk under no_grad (deterministic
    # — pipeline picks new_frames=chunk_size on iter 1, see
    # generate_next_chunk).
    torch.manual_seed(1234)  # deterministic noise for the chunk
    with torch.no_grad():
        chunk, info = trainer.model.generate_next_chunk(requires_grad=False)
    log.info(
        "iter-1 chunk: shape=%s new_frames=%d overlap=%d current_length=%d",
        tuple(chunk.shape), info["new_frames"], info["overlap"],
        info["current_length"],
    )

    # ---- Capture STREAMING-side scorer inputs for this chunk.
    M = trainer.model
    cf = int(s["cf"])
    npb = int(s["shift"])
    N = int(s["chunk_size"])

    clean_x_self_str = M._streaming_build_clean_x_self(chunk, info)
    clean_x_GT_str = (
        M._streaming_build_clean_x_GT(info)
        if M.dmd_context == "GT"
        else None
    )
    cond_for_scoring_str, uncond_for_scoring_str = (
        M._streaming_noisy_cond_slice(info)
    )
    clean_cond_str, clean_uncond_str = M._streaming_clean_cond_slice(info)

    (
        sc_clean_x_str, sc_aug_t_str,
        sc_clean_x_real_str, sc_aug_t_real_str,
        cond_merged_str, uncond_merged_str,
    ) = M._build_dmd_context_kwargs(
        clean_x_self=clean_x_self_str,
        clean_x_GT=clean_x_GT_str,
        clean_conditional_dict=clean_cond_str,
        clean_unconditional_dict=clean_uncond_str,
        cond_for_scoring=cond_for_scoring_str,
        uncond_for_scoring=uncond_for_scoring_str,
        device=chunk.device, dtype=chunk.dtype,
        build_real_view=True,
    )
    log.info("streaming inputs captured.")

    # ---- Build DIAG-side equivalents from the same streaming_state.
    # Diag's batch i=1 (model index): noisy_start_sdn=npb, clean_start_sdn=0
    # → abs_noisy_start=cf+npb, abs_clean_start=cf.
    # NB: streaming's iter-1 ALSO has noisy_start_sdn=npb (because
    # setup_sequence pre-rolled a +npb anchor → current_length=npb at
    # entry → noisy_start_sdn = npb + 21 - 21 - 0 = ... wait check).
    noisy_start_sdn_str = int(
        info["current_length"] - info["new_frames"] - info["overlap"]
    )
    log.info(
        "STREAMING: noisy_start_sdn=%d clean_start_in_ride=%d "
        "abs_noisy_start=%d",
        noisy_start_sdn_str,
        cf + noisy_start_sdn_str - npb,
        cf + noisy_start_sdn_str,
    )

    # The diag uses iter-1 noisy_start_sdn=npb (because of leading-chunk
    # anchor at the FRONT of student_full). Streaming's setup_sequence
    # ALSO pre-rolls a +npb anchor (current_length init = npb) so iter
    # 1 should match. Verify:
    abs_noisy_start_diag = cf + npb
    abs_clean_start_diag = cf
    log.info(
        "DIAG     : noisy_start_sdn=%d clean_start_in_ride=%d "
        "abs_noisy_start=%d",
        npb, abs_clean_start_diag, abs_noisy_start_diag,
    )

    ride_latents = s["ride_latents_window"]
    ride_actions = s["ride_actions_window"]
    prompt_embeds = s["prompt_embeds"]

    clean_x_GT_diag = ride_latents[
        :, abs_clean_start_diag : abs_clean_start_diag + N
    ].contiguous()

    actions_noisy_diag = ride_actions[
        :, abs_noisy_start_diag : abs_noisy_start_diag + N
    ].contiguous()
    actions_clean_diag = ride_actions[
        :, abs_clean_start_diag : abs_clean_start_diag + N
    ].contiguous()

    cond_noisy_diag, uncond_noisy_diag = M.build_action_conditional(
        prompt_embeds=prompt_embeds, gt_actions=actions_noisy_diag,
    )
    cond_clean_diag, uncond_clean_diag = M.build_action_conditional(
        prompt_embeds=prompt_embeds, gt_actions=actions_clean_diag,
    )
    # Merge clean streams in the same way _build_dmd_context_kwargs does.
    cond_merged_diag = dict(cond_noisy_diag)
    uncond_merged_diag = dict(uncond_noisy_diag)
    for src_dict, dst_dict in (
        (cond_clean_diag, cond_merged_diag),
        (uncond_clean_diag, uncond_merged_diag),
    ):
        am = src_dict.get("_action_modulation")
        at = src_dict.get("_action_tokens")
        if am is not None:
            dst_dict["_action_modulation_clean"] = am
        if at is not None:
            dst_dict["_action_tokens_clean"] = at

    log.info("diag inputs captured.")

    # ---- Diff each pair.
    log.info("=" * 70)
    log.info("DIFF: clean_x_GT")
    _diff_tensor("clean_x_GT", clean_x_GT_str, clean_x_GT_diag)

    log.info("=" * 70)
    _diff_dict(
        "cond_noisy",
        cond_for_scoring_str,
        cond_noisy_diag,
        keys=["_action_modulation", "_action_tokens", "prompt_embeds"],
    )
    _diff_dict(
        "uncond_noisy",
        uncond_for_scoring_str,
        uncond_noisy_diag,
        keys=["_action_modulation", "_action_tokens", "prompt_embeds"],
    )

    log.info("=" * 70)
    _diff_dict(
        "cond_clean",
        clean_cond_str,
        cond_clean_diag,
        keys=["_action_modulation", "_action_tokens", "prompt_embeds"],
    )

    log.info("=" * 70)
    _diff_dict(
        "cond_MERGED (the dict actually passed to real_score)",
        cond_merged_str,
        cond_merged_diag,
        keys=[
            "_action_modulation",
            "_action_tokens",
            "_action_modulation_clean",
            "_action_tokens_clean",
            "prompt_embeds",
        ],
    )

    # ---- Run real_score under each input set on the SAME noisy_input
    # + same DMD timestep. Pre-compute so streaming/diag share noise.
    timestep = M._sample_dmd_timestep(
        batch_size=1, num_frame=N,
        denoised_timestep_from=info.get("denoised_timestep_from"),
        denoised_timestep_to=info.get("denoised_timestep_to"),
        device=device,
    )
    # Force same-step-across-frames so the visual is consistent.
    t0 = int(timestep.flatten()[0].item())
    timestep = torch.full(
        (1, N), fill_value=t0, dtype=torch.long, device=device,
    )
    log.info("real_score forward at shared timestep=%d", t0)

    torch.manual_seed(5678)
    noise_for_dmd = torch.randn_like(chunk)
    noisy_input = M.scheduler.add_noise(
        chunk.flatten(0, 1), noise_for_dmd.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, chunk.shape[:2])

    # clean_x_real for both paths (streaming's was built inside
    # _build_dmd_context_kwargs with its own randn; rebuild the diag's
    # using the same aug_t but a fresh randn — divergence in the
    # randn is expected and not the bug we're hunting). Use the same
    # streaming sc_clean_x_real for the streaming branch and rebuild
    # an analogous one for the diag branch with a FIXED seed so it's
    # also reproducible.
    aug_t_real = sc_aug_t_real_str  # both paths share aug_t (= clean_x_aug_t)
    torch.manual_seed(9999)
    gt_view_diag = clean_x_GT_diag.to(dtype=dtype, device=device)
    real_noise_diag = torch.randn_like(gt_view_diag)
    sc_clean_x_real_diag = M.scheduler.add_noise(
        gt_view_diag.flatten(0, 1),
        real_noise_diag.flatten(0, 1),
        aug_t_real.flatten(0, 1),
    ).unflatten(0, gt_view_diag.shape[:2]).to(dtype=dtype)

    scale = float(M.real_guidance_scale)
    log.info(
        "real_guidance_scale=%.3f → pred_real_cfg = cond + scale*(cond-uncond)",
        scale,
    )

    log.info("running real_score (streaming inputs) ...")
    with torch.no_grad():
        _, pred_real_cond_str = M.real_score(
            noisy_image_or_video=noisy_input,
            conditional_dict=cond_merged_str,
            timestep=timestep,
            clean_x=sc_clean_x_real_str, aug_t=sc_aug_t_real_str,
        )
        _, pred_real_uncond_str = M.real_score(
            noisy_image_or_video=noisy_input,
            conditional_dict=uncond_merged_str,
            timestep=timestep,
            clean_x=sc_clean_x_real_str, aug_t=sc_aug_t_real_str,
        )
    pred_real_cfg_str = pred_real_cond_str + (
        pred_real_cond_str - pred_real_uncond_str
    ) * scale

    log.info("running real_score (diag inputs) ...")
    with torch.no_grad():
        _, pred_real_cond_diag = M.real_score(
            noisy_image_or_video=noisy_input,
            conditional_dict=cond_merged_diag,
            timestep=timestep,
            clean_x=sc_clean_x_real_diag, aug_t=aug_t_real,
        )
        _, pred_real_uncond_diag = M.real_score(
            noisy_image_or_video=noisy_input,
            conditional_dict=uncond_merged_diag,
            timestep=timestep,
            clean_x=sc_clean_x_real_diag, aug_t=aug_t_real,
        )
    pred_real_cfg_diag = pred_real_cond_diag + (
        pred_real_cond_diag - pred_real_uncond_diag
    ) * scale

    log.info("=" * 70)
    _diff_tensor(
        "pred_real_COND_only (streaming vs diag) — what the diag's F_real_GT "
        "actually decodes",
        pred_real_cond_str, pred_real_cond_diag,
    )
    _diff_tensor(
        "pred_real_UNCOND_only (streaming vs diag)",
        pred_real_uncond_str, pred_real_uncond_diag,
    )
    _diff_tensor(
        "pred_real_CFG (4*cond - 3*uncond) (streaming vs diag) — what "
        "production / smoke decodes",
        pred_real_cfg_str, pred_real_cfg_diag,
    )
    _diff_tensor(
        "(uncond - cond) MAGNITUDE on streaming side — how much CFG "
        "amplification happens",
        pred_real_uncond_str, pred_real_cond_str,
    )

    # ---- Decode each + chunk + clean_x_GT to mp4 for visual confirmation.
    out_dir = _REPO / "_diag_p1" / "out" / "compare_streaming_vs_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("decoding videos to %s ...", out_dir)
    _write_mp4(
        out_dir / "00_chunk_student.mp4",
        _decode_to_uint8(chunk.to(torch.float32), M.vae),
    )
    _write_mp4(
        out_dir / "01_noisy_input.mp4",
        _decode_to_uint8(noisy_input.to(torch.float32), M.vae),
    )
    _write_mp4(
        out_dir / "10_clean_x_GT_streaming.mp4",
        _decode_to_uint8(clean_x_GT_str.to(torch.float32), M.vae),
    )
    _write_mp4(
        out_dir / "11_clean_x_GT_diag.mp4",
        _decode_to_uint8(clean_x_GT_diag.to(torch.float32), M.vae),
    )
    # Cond-only: what the diag's F_real_GT.mp4 shows (no CFG). Should
    # look ~ identical between streaming and diag — confirms the v14
    # cond branch is in-distribution on the streaming inputs.
    _write_mp4(
        out_dir / "20_pred_real_COND_streaming.mp4",
        _decode_to_uint8(pred_real_cond_str.to(torch.float32), M.vae),
    )
    _write_mp4(
        out_dir / "21_pred_real_COND_diag.mp4",
        _decode_to_uint8(pred_real_cond_diag.to(torch.float32), M.vae),
    )
    # Uncond-only: this is what CFG subtracts. If it looks like noise
    # / garbage, then the 3* amplification in the CFG combination is
    # propagating that garbage into the production pred_real.
    _write_mp4(
        out_dir / "22_pred_real_UNCOND_streaming.mp4",
        _decode_to_uint8(pred_real_uncond_str.to(torch.float32), M.vae),
    )
    # CFG-amplified: what the smoke's pred_real.mp4 decodes (what the
    # user reported as "really bad"). If THIS looks bad while
    # COND_streaming above looks fine, CFG is the culprit.
    _write_mp4(
        out_dir / "30_pred_real_CFG_streaming.mp4",
        _decode_to_uint8(pred_real_cfg_str.to(torch.float32), M.vae),
    )
    _write_mp4(
        out_dir / "31_pred_real_CFG_diag.mp4",
        _decode_to_uint8(pred_real_cfg_diag.to(torch.float32), M.vae),
    )
    log.info("done. Compare mp4s in %s", out_dir)


if __name__ == "__main__":
    main()
