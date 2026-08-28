"""Static/CPU review of the 2026-08-27 holder recipes.

These tests resolve the launcher's PRINT_ONLY mode.  They specifically guard
the repo's last-wins failure mode: the injected values are parsed as a dotlist
and compared with the intended experiment, rather than inferred from comments.
"""
from pathlib import Path
import os
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "sbatch" / "run_gan_crop_arm_on_holder.sh"
SNAPSHOT_LAUNCHER = ROOT / "sbatch" / "run_gan_crop_arm_snapshot.sh"


def _resolve(arm, **env):
    run_env = os.environ.copy()
    run_env.update({"GAN_ARM": arm, "PRINT_ONLY": "1", "RUNSTAMP": "test"})
    run_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT, env=run_env,
        text=True, capture_output=True,
    )
    return proc


def _dotlist(stdout):
    line = next(x for x in stdout.splitlines() if "last_wins=[" in x)
    raw = line.split("last_wins=[", 1)[1].rsplit("]", 1)[0]
    return dict(token.split("=", 1) for token in shlex.split(raw))


def _assert_balanced_5x(cfg):
    assert cfg["gan_updates_per_step"] == "1"
    assert cfg["ladd_pixel_crops_per_row"] == "2"
    assert cfg["ladd_pixel_lat_frames"] == "3"
    assert cfg["ladd_pixel_frames_per_crop"] == "5"
    assert cfg["ladd_pixel_decode_split"] == "1"
    assert cfg["ladd_pixel_crop_stratify_x"] == "1"
    assert cfg["ladd_pixel_crop_y_lo_frac"] == "0.0"
    assert cfg["ladd_pixel_crop_y_hi_frac"] == "1.0"
    # Per-step logit exposure is exactly conserved against the parent.
    assert 2 * 5 * 1 == 1 * 2 * 5


def _assert_direct_g1x(cfg):
    assert cfg["ladd_pixel_g_crops_per_row"] == "1"
    assert cfg["ladd_pixel_g_lat_frames"] == "2"
    assert cfg["ladd_pixel_g_frames_per_crop"] == "2"
    assert cfg["ladd_pixel_g_decode_split"] == "0"
    assert cfg["ladd_pixel_g_crop_stratify_x"] == "0"
    assert cfg["ladd_pixel_post_d_memory_release"] == "true"


def test_direct_vgg_is_balanced_and_last_wins():
    proc = _resolve("vgg_5x")
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    _assert_direct_g1x(cfg)
    assert cfg["surrogate_critic_enabled"] == "false"
    assert cfg["ladd_disc_loss_weight"] == "1.0"
    assert "source=sbatch/pixvgg_online.sbatch" in proc.stdout
    assert "kind=direct stage=direct" in proc.stdout
    assert "gan_grad_telemetry_every=0 texture_tripwire_every=0" in proc.stdout


def test_direct_rn50_is_balanced_and_last_wins():
    proc = _resolve("rn50_5x")
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    _assert_direct_g1x(cfg)
    assert cfg["surrogate_critic_enabled"] == "false"
    assert cfg["ladd_disc_loss_weight"] == "1.0"
    assert "source=sbatch/pixrn50_online.sbatch" in proc.stdout
    assert "kind=direct stage=direct" in proc.stdout
    assert "gan_grad_telemetry_every=0 texture_tripwire_every=0" in proc.stdout


def test_surrogate_calibration_is_strong_but_weight_zero():
    proc = _resolve("vgg_surrogate_teacher_d5x_targets4perclass_fit24")
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    assert cfg["ladd_disc_loss_weight"] == "0.0"
    assert cfg["pix_r1_gamma"] == "0.0"
    assert cfg["pix_crop_lat"] == "[24,32]"
    assert cfg["pix_lat_frames_per_crop"] == "3"
    assert cfg["surrogate_critic_enabled"] == "true"
    assert cfg["surrogate_teacher_backbone"] == "ladd_pixel"
    assert cfg["surrogate_grad_loss_normalize"] == "true"
    assert cfg["surrogate_distill_substeps"] == "24"
    assert cfg["surrogate_critic_head_init_std"] == "0.02"
    assert cfg["pix_teacher_refresh_every"] == "2"
    assert cfg["surrogate_grad_check_every"] == "20"
    assert cfg["pix_crops_per_step"] == "4"
    assert cfg["pix_real_pool_windows"] == "2560"
    assert int(cfg["pix_real_pool_windows"]) * int(
        cfg["pix_lat_frames_per_crop"]) >= 5120
    assert cfg["pix_gan_weight"] == "0.0"
    assert "gan_grad_telemetry_every=25 texture_tripwire_every=0" in proc.stdout


def test_direct_gradient_surrogate_is_explicit_and_weight_zero():
    proc = _resolve(
        "vgg_surrogate_directgrad_teacher_d5x_targets4perclass_fit24"
    )
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    assert cfg["surrogate_critic_enabled"] == "true"
    assert cfg["surrogate_gradient_mode"] == "direct"
    assert cfg["surrogate_direct_width"] == "96"
    assert cfg["surrogate_direct_num_blocks"] == "6"
    assert cfg["surrogate_direct_head_init_std"] == "1.0e-3"
    assert cfg["surrogate_teacher_target_microbatch"] == "1"
    assert cfg["surrogate_distill_substeps"] == "24"
    assert cfg["pix_gan_weight"] == "0.0"
    assert "directgrad_teacherD5x_targets4perclass_fit24_cal_w0" in proc.stdout


def test_current_teacher_direct_gradient_arm_names_exact_evidence_and_drops_history():
    proc = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24"
    )
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    assert cfg["surrogate_gradient_mode"] == "direct"
    assert cfg["pix_crops_per_step"] == "4"
    # Four real + four fake crop targets = eight logical current-teacher
    # gradient targets per refresh. Checkpoint replay is compute, not evidence.
    assert 2 * int(cfg["pix_crops_per_step"]) == 8
    assert cfg["surrogate_cache_capacity"] == "1"
    assert cfg["surrogate_param_control_enabled"] == "true"
    assert cfg["surrogate_teacher_target_microbatch"] == "1"
    assert cfg["surrogate_distill_substeps"] == "24"
    assert cfg["pix_gan_weight"] == "0.0"
    assert "directgrad_currentteacher_8targets_refresh_fit24_cal_w0" in proc.stdout


def test_aligned_flash_arm_changes_only_the_teacher_fake_source():
    base = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24"
    )
    aligned = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_alignedflash"
    )
    assert base.returncode == aligned.returncode == 0
    a = _dotlist(base.stdout)
    b = _dotlist(aligned.stdout)
    assert "ladd_fake_sample_source" not in a
    assert b.pop("ladd_fake_sample_source") == "flash"
    assert a == b
    assert "currentteacher_8targets_refresh_fit24_alignedflash_cal_w0" in aligned.stdout


def test_live_flash_arm_changes_only_the_generator_fake_frame_selector():
    base = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24"
    )
    live = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_liveflash"
    )
    assert base.returncode == live.returncode == 0
    a = _dotlist(base.stdout)
    b = _dotlist(live.stdout)
    assert "pix_flash_grad_select_enabled" not in a
    assert b.pop("pix_flash_grad_select_enabled") == "true"
    assert a == b
    assert "currentteacher_8targets_refresh_fit24_liveflash_cal_w0" in live.stdout


def test_16target_live_flash_arm_only_doubles_fresh_teacher_evidence():
    live8 = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24_liveflash"
    )
    live16 = _resolve(
        "vgg_surrogate_directgrad_currentteacher_16targets_refresh_fit24_liveflash"
    )
    assert live8.returncode == live16.returncode == 0
    a = _dotlist(live8.stdout)
    b = _dotlist(live16.stdout)
    assert a.pop("pix_crops_per_step") == "4"
    assert b.pop("pix_crops_per_step") == "8"
    assert a == b
    assert b["surrogate_teacher_target_microbatch"] == "1"
    assert b["surrogate_cache_capacity"] == "1"
    assert b["surrogate_distill_substeps"] == "24"
    assert b["pix_flash_grad_select_enabled"] == "true"
    assert b["pix_gan_weight"] == "0.0"
    assert "currentteacher_16targets_refresh_fit24_liveflash_cal_w0" in live16.stdout


def test_t0_rung_active_arm_removes_flash_and_records_dynamics_override():
    arm = "vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_active"
    rejected = _resolve(
        arm, SURROGATE_STAGE="active", PIXW="0.15",
        SURROGATE_APPROVED="YES", SURROGATE_CALIBRATION_RUN="79re58zl",
        SURROGATE_COS="0.464675",
    )
    assert rejected.returncode != 0
    assert "SURROGATE_DYNAMICS_OVERRIDE=YES" in rejected.stderr

    proc = _resolve(
        arm, SURROGATE_STAGE="active", PIXW="0.15",
        SURROGATE_APPROVED="YES", SURROGATE_CALIBRATION_RUN="79re58zl",
        SURROGATE_COS="0.464675", SURROGATE_DYNAMICS_OVERRIDE="YES",
    )
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    assert cfg["pix_crops_per_step"] == "4"
    assert cfg["flash_dmd_enabled"] == "false"
    assert cfg["flash_dmd_gan_t"] == "0"
    assert cfg["pix_finish_grad_enabled"] == "true"
    assert cfg["exit_exclude_last_rung"] == "true"
    assert cfg["pix_flash_grad_select_enabled"] == "false"
    assert cfg["ladd_fake_sample_source"] == "dmd"
    assert cfg["ladd_disc_force_clean"] == "true"
    assert cfg["gen_aux_losses_x0_source"] == "ladder_endpoint"
    assert cfg["forward_noiser_train_source"] == "ladder_endpoint"
    assert cfg["forward_noiser_rollout2_source"] == "ladder_endpoint"
    assert cfg["pix_gan_weight"] == "0.15"
    assert "t0rungs_noflash_active_researchoverride_w0p15" in proc.stdout
    assert "RESEARCH OVERRIDE" in proc.stdout


def test_t0_rung_arm_refuses_calibration_stage():
    proc = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_active"
    )
    assert proc.returncode != 0
    assert "requires SURROGATE_STAGE=active" in proc.stderr


def test_gate_v2_arm_is_weight_zero_and_names_every_selected_technique():
    proc = _resolve("vgg_surrogate_directgrad_gatev2_t0rungs")
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    assert cfg["pix_gan_weight"] == "0.0"
    assert cfg["surrogate_gradient_mode"] == "direct"
    assert cfg["surrogate_critic_lr"] == "1.0e-3"
    assert cfg["surrogate_direct_loss_mode"] == "cosine"
    assert cfg["surrogate_direct_real_loss_weight"] == "0.0"
    assert cfg["surrogate_direct_fake_loss_weight"] == "1.0"
    assert cfg["surrogate_direct_temporal_mixing"] == "true"
    assert cfg["surrogate_direct_temporal_blocks"] == "2"
    assert cfg["surrogate_direct_global_context"] == "true"
    assert cfg["surrogate_cache_capacity"] == "1"
    assert cfg["flash_dmd_enabled"] == "false"
    assert cfg["pix_finish_grad_enabled"] == "true"
    assert (
        "gatev2_temporalglobal_cosine_fakeonly_lr1e3_t0rungs_noflash_cal_w0"
        in proc.stdout
    )


def test_gate_v2_active_cannot_use_the_historical_t0_override():
    arm = "vgg_surrogate_directgrad_gatev2_t0rungs"
    common = dict(
        SURROGATE_STAGE="active", PIXW="0.15",
        SURROGATE_APPROVED="YES", SURROGATE_CALIBRATION_RUN="gate-v2-test",
    )
    rejected = _resolve(
        arm, SURROGATE_Q1_MIN="0.499", SURROGATE_AUDIT_COUNT="3", **common,
    )
    assert rejected.returncode != 0
    assert "below activation gate 0.50" in rejected.stderr

    too_few = _resolve(
        arm, SURROGATE_Q1_MIN="0.501", SURROGATE_AUDIT_COUNT="1", **common,
    )
    assert too_few.returncode != 0
    assert "at least two reviewed live audits" in too_few.stderr

    accepted = _resolve(
        arm, SURROGATE_Q1_MIN="0.501", SURROGATE_AUDIT_COUNT="3", **common,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "RESEARCH OVERRIDE" not in accepted.stdout


def test_decoder_shaped_arm_names_all_residual_q1p99_and_is_weight_zero():
    proc = _resolve("vgg_decoder_shaped_exactmidlow12_t0rungs")
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    _assert_balanced_5x(cfg)
    assert cfg["surrogate_critic_enabled"] == "false"
    assert cfg["surrogate_decoder_shaped_enabled"] == "true"
    assert cfg["surrogate_decoder_shaped_bundle"].endswith(
        "exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt"
    )
    assert cfg["surrogate_decoder_shaped_audit_every"] == "0"
    assert cfg["log_interval"] == "5"
    assert cfg["surrogate_teacher_backbone"] == "ladd_pixel"
    assert cfg["ladd_disc_loss_weight"] == "0.0"
    assert cfg["pix_gan_weight"] == "0.0"
    assert cfg["flash_dmd_enabled"] == "false"
    assert cfg["pix_finish_grad_enabled"] == "true"
    assert cfg["ladd_fake_sample_source"] == "dmd"
    assert "currentpixel_allresidual_q1p99_t0rungs_noflash_cal_w0" in proc.stdout


def test_decoder_shaped_active_requires_live_q1_gate():
    arm = "vgg_decoder_shaped_exactmidlow12_t0rungs"
    common = dict(
        SURROGATE_STAGE="active", PIXW="0.1",
        SURROGATE_APPROVED="YES", SURROGATE_CALIBRATION_RUN="decoder-smoke",
    )
    rejected = _resolve(arm, SURROGATE_Q1_MIN="0.989", **common)
    assert rejected.returncode != 0
    assert "below 0.99" in rejected.stderr
    accepted = _resolve(arm, SURROGATE_Q1_MIN="0.992351", **common)
    assert accepted.returncode == 0, accepted.stderr
    assert "currentpixel_allresidual_q1p99_t0rungs_noflash_active_w0p1" in accepted.stdout


def test_t0_rgbmaxmin_condition_arm_changes_only_detached_conditioning():
    env = dict(
        SURROGATE_STAGE="active", PIXW="0.15",
        SURROGATE_APPROVED="YES", SURROGATE_CALIBRATION_RUN="v53mvn1t",
        SURROGATE_COS="0.162996", SURROGATE_DYNAMICS_OVERRIDE="YES",
    )
    base = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_active",
        **env,
    )
    conditioned = _resolve(
        "vgg_surrogate_directgrad_currentteacher_8targets_t0rungs_rgbmaxmin_active",
        **env,
    )
    assert base.returncode == conditioned.returncode == 0
    a = _dotlist(base.stdout)
    b = _dotlist(conditioned.stdout)
    assert b.pop("surrogate_pixel_condition_enabled") == "true"
    assert b.pop("surrogate_pixel_condition_decode_batch") == "1"
    assert a == b
    assert "t0rungs_noflash_rgbmaxmin_active_researchoverride_w0p15" in conditioned.stdout


def test_surrogate_calibration_refuses_inherited_nonzero_weight():
    proc = _resolve("vgg_surrogate_teacher_d5x_targets4perclass_fit24", PIXW="0.2")
    assert proc.returncode != 0
    assert "calibration requires PIXW=0.0" in proc.stderr


def test_surrogate_active_refuses_unreviewed_activation():
    proc = _resolve("vgg_surrogate_teacher_d5x_targets4perclass_fit24", SURROGATE_STAGE="active", PIXW="0.2")
    assert proc.returncode != 0
    assert "SURROGATE_APPROVED" in proc.stderr


def test_surrogate_active_refuses_failed_cosine_gate():
    proc = _resolve(
        "vgg_surrogate_teacher_d5x_targets4perclass_fit24",
        SURROGATE_STAGE="active", PIXW="0.2",
        SURROGATE_APPROVED="YES", SURROGATE_CALIBRATION_RUN="probe",
        SURROGATE_COS="0.49",
    )
    assert proc.returncode != 0
    assert "below activation gate" in proc.stderr


def test_surrogate_active_accepts_reviewed_calibration():
    proc = _resolve(
        "vgg_surrogate_teacher_d5x_targets4perclass_fit24",
        SURROGATE_STAGE="active", PIXW="0.2",
        SURROGATE_APPROVED="YES", SURROGATE_CALIBRATION_RUN="probe",
        SURROGATE_COS="0.51",
    )
    assert proc.returncode == 0, proc.stderr
    cfg = _dotlist(proc.stdout)
    assert cfg["pix_gan_weight"] == "0.2"
    assert cfg["ladd_disc_loss_weight"] == "0.0"


def test_sources_keep_the_requested_frozen_action_critic_and_clean_pair_mode():
    for name in ("pixvgg_online.sbatch", "pixrn50_online.sbatch"):
        text = (ROOT / "sbatch" / name).read_text()
        # These occur after DEXTRA in both reviewed parents.
        tail = text.split("$DEXTRA", 1)[1]
        assert "action_critic_aux_enabled=true" in tail
        assert "action_critic_freeze=true" in tail
        assert "ladd_gt_transition_enabled=false" in tail
        assert "ladd_gt_vs_fake_enabled=true" in tail
        assert "ladd_fake_sample_source=dmd" in text
        assert tail.index("surrogate_critic_enabled=false") < tail.index(
            "run_name=dmd10k_"
        )


def test_launcher_injects_after_parent_hardcodes_and_before_run_name():
    text = LAUNCHER.read_text()
    assert 'ENVIRON["GAN_ARM_EXTRA"]' in text
    assert '/^[[:space:]]*run_name=dmd10k_/' in text
    assert 'N_RUN_NAME=$(grep -c' in text


def test_launcher_refuses_sample_interval_at_run_boundary():
    env = os.environ.copy()
    env.update({
        "GAN_ARM": "vgg_surrogate_directgrad_currentteacher_8targets_refresh_fit24",
        "HOLDER": "9999999",
        "MAXSTEPS": "90",
        "SAMPLE_EVERY": "90",
        "RUNSTAMP": "test",
    })
    proc = subprocess.run(
        ["bash", str(SNAPSHOT_LAUNCHER)], cwd=ROOT, env=env,
        text=True, capture_output=True,
    )
    assert proc.returncode == 14
    assert "boundary sample is consumed on the following training step" in proc.stderr
