"""Static review for the one-node CARN-base decoder-surrogate sweep."""

from pathlib import Path
import os
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "sbatch" / "run_carn_decoder_surrogate_node.sh"


def _resolve(**extra):
    env = os.environ.copy()
    env.update({
        "PRINT_ONLY": "1", "HOLDER": "6161257", "NODE": "nid010292",
        "PIXW": "0", "TARGET_SHARE": "calibrationw0", "RUNSTAMP": "test",
    })
    env.update({key: str(value) for key, value in extra.items()})
    return subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT, env=env,
        text=True, capture_output=True,
    )


def _cfg(stdout):
    line = next(line for line in stdout.splitlines() if "last_wins=[" in line)
    raw = line.split("last_wins=[", 1)[1].rsplit("]", 1)[0]
    return dict(token.split("=", 1) for token in shlex.split(raw))


def test_sweep_is_derived_from_exact_requested_base_and_keeps_base_consumers():
    proc = _resolve()
    assert proc.returncode == 0, proc.stderr
    assert "source=sbatch/carncommit_long.sbatch base_wandb=g5ndc0fz" in proc.stdout
    cfg = _cfg(proc.stdout)
    assert cfg["flash_dmd_enabled"] == "true"
    assert cfg["flash_dmd_gan_t"] == "60"
    assert cfg["ladd_fake_sample_source"] == "flash"
    assert cfg["ladd_disc_force_clean"] == "true"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "false"
    # These inherited base settings must not be silently overridden.
    for key in (
        "action_critic_aux_enabled", "action_critic_freeze",
        "generator_action_z_guidance_weight", "forward_noiser_cycle_enabled",
        "ladd_gt_vs_fake_enabled", "ladd_gt_transition_enabled", "seed",
    ):
        assert key not in cfg


def test_exact_q1p99_route_and_weight_zero_calibration_are_explicit():
    proc = _resolve()
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["surrogate_critic_enabled"] == "false"
    assert cfg["gan_pixel_texture_enabled"] == "false"
    assert cfg["gan_of_enabled"] == "false"
    assert cfg["fake_alt_head_enabled"] == "false"
    assert cfg["real_teacher_train_online"] == "false"
    assert cfg["dmd_frozen_teacher_pass_enabled"] == "false"
    assert cfg["boundary_vae_roundtrip"] == "false"
    assert cfg["state_probe_aux_enabled"] == "false"
    assert cfg["surrogate_decoder_shaped_enabled"] == "true"
    assert cfg["surrogate_decoder_shaped_bundle"].endswith(
        "exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt"
    )
    assert cfg["surrogate_decoder_shaped_audit_every"] == "0"
    assert cfg["surrogate_teacher_backbone"] == "ladd_pixel"
    assert cfg["pix_gan_weight"] == "0"
    assert cfg["pix_finish_grad_enabled"] == "false"
    assert cfg["pix_flash_grad_select_enabled"] == "true"
    assert cfg["exit_exclude_last_rung"] == "true"
    assert cfg["ladd_disc_loss_weight"] == "0.0"
    assert cfg["ladd_feature_source"] == "vgg"
    assert cfg["ladd_pixel_encoder_trainable"] == "false"
    assert cfg["ladd_pixel_decode_cache"] == "false"
    assert cfg["ladd_defer_disc_update"] == "false"
    assert cfg["ladd_pixel_input_filter"] == "none"
    assert cfg["ladd_pixel_swt_strength"] == "1.0"
    assert cfg["stat_anchor_loss_weight"] == "1.0"
    assert cfg["gan_lr"] == "2e-5"


def test_brightness_wavelet_and_anchor_ablation_are_explicit():
    dc = _cfg(_resolve(PIXEL_FILTER="dc").stdout)
    assert dc["ladd_pixel_input_filter"] == "dc"
    assert dc["stat_anchor_loss_weight"] == "1.0"

    swt = _cfg(_resolve(PIXEL_FILTER="swt", SWT_STRENGTH="0.75").stdout)
    assert swt["ladd_pixel_input_filter"] == "swt"
    assert swt["ladd_pixel_swt_strength"] == "0.75"

    no_anchor = _cfg(_resolve(PIXEL_FILTER="dc", STAT_ANCHOR="0").stdout)
    assert no_anchor["ladd_pixel_input_filter"] == "dc"
    assert no_anchor["stat_anchor_loss_weight"] == "0"


def test_discriminator_learning_rate_is_an_explicit_sweep_parameter():
    proc = _resolve(GANLR="1e-3")
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["gan_lr"] == "1e-3"
    assert "_dlr1em3_" in proc.stdout


def test_discriminator_inner_updates_are_an_explicit_sweep_parameter():
    proc = _resolve(GANUPDATES="3")
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["gan_updates_per_step"] == "3"
    assert "_du3_" in proc.stdout


def test_ladd_r1_strength_is_an_explicit_sweep_parameter():
    default = _cfg(_resolve().stdout)
    assert default["ladd_r1_gamma"] == "10"

    proc = _resolve(LADD_R1_GAMMA="1")
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["ladd_r1_gamma"] == "1"
    assert "_r11_" in proc.stdout


def test_fresh_discriminator_order_is_default_and_deferred_control_is_named():
    inline = _resolve()
    assert inline.returncode == 0, inline.stderr
    assert _cfg(inline.stdout)["ladd_defer_disc_update"] == "false"
    assert "_dinline_DthenG_" in inline.stdout

    deferred = _resolve(LADD_DEFER_DISC_UPDATE="true")
    assert deferred.returncode == 0, deferred.stderr
    assert _cfg(deferred.stdout)["ladd_defer_disc_update"] == "true"
    assert "_ddefer_gthenD_" in deferred.stdout
    assert _resolve(LADD_DEFER_DISC_UPDATE="sometimes").returncode != 0


def test_carn_commit_mode_changes_only_the_named_consumer_policy():
    proc = _resolve(CARNMODE="commit")
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "true"
    assert cfg["reverse_noiser_internalize_weight"] == "0.0"
    assert "_commit_share" in proc.stdout
    # The proven g5 cycle remains inherited rather than re-specified.
    assert "forward_noiser_cycle_enabled" not in cfg


def test_carn_aux_minus_mode_reproduces_known_good_internalisation_contract():
    proc = _resolve(CARNMODE="aux_minus", CARN_AUX_WEIGHT="0.25")
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["forward_noiser_train_source"] == "rollout"
    assert cfg["forward_noiser_rollout2_source"] == "legacy"
    assert cfg["forward_noiser_reverse"] == "true"
    assert cfg["forward_noiser_cycle_enabled"] == "false"
    assert cfg["forward_noiser_apply_decoupled"] == "false"
    assert cfg["reverse_noiser_dedrift_enabled"] == "true"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "false"
    assert cfg["reverse_noiser_dedrift_apply_to_flash"] == "false"
    assert cfg["reverse_noiser_dedrift_apply_to_train_score"] == "false"
    assert cfg["reverse_noiser_internalize_weight"] == "0.25"
    assert "_aux_minus_share" in proc.stdout


def test_carn_commit_aux_combines_only_the_two_proven_consumers():
    proc = _resolve(CARNMODE="commit_aux", CARN_AUX_WEIGHT="0.2")
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "true"
    assert cfg["reverse_noiser_internalize_weight"] == "0.2"
    assert cfg["forward_noiser_train_source"] == "rollout"
    assert "_commit_aux_share" in proc.stdout


def test_recent_transition_modes_remain_explicit_ablation_routes():
    former = _cfg(_resolve(CARNMODE="former_plus").stdout)
    latter = _cfg(_resolve(CARNMODE="latter_minus").stdout)

    assert former["ladd_gt_vs_fake_enabled"] == "false"
    assert former["ladd_gt_transition_enabled"] == "true"
    assert former["ladd_gt_transition_carn_former"] == "true"
    assert former["ladd_gt_transition_carn_latter_reverse"] == "false"
    assert former["forward_noiser_reverse"] == "false"

    assert latter["ladd_gt_vs_fake_enabled"] == "false"
    assert latter["ladd_gt_transition_enabled"] == "true"
    assert latter["ladd_gt_transition_carn_former"] == "false"
    assert latter["ladd_gt_transition_carn_latter_reverse"] == "true"
    assert latter["forward_noiser_reverse"] == "true"


def test_per_step_discriminator_evidence_matches_base_budget():
    cfg = _cfg(_resolve().stdout)
    assert cfg["gan_updates_per_step"] == "1"
    assert cfg["ladd_pixel_crops_per_row"] == "2"
    assert cfg["ladd_pixel_lat_frames"] == "3"
    assert cfg["ladd_pixel_frames_per_crop"] == "5"
    assert cfg["ladd_pixel_decode_split"] == "1"
    assert 2 * 5 * 1 == 1 * 2 * 5


def test_bad_weight_and_video_cadence_fail_before_slurm():
    assert _resolve(PIXW="0").returncode == 0
    assert _resolve(PIXW="0.1").returncode != 0
    assert _resolve(PIXW="-0.01").returncode != 0
    assert _resolve(GANLR="0").returncode != 0
    assert _resolve(GANUPDATES="0").returncode != 0
    assert _resolve(LADD_R1_GAMMA="-1").returncode != 0
    assert _resolve(CARNMODE="not_a_mode").returncode != 0
    assert _resolve(CARN_AUX_WEIGHT="-1").returncode != 0
    assert _resolve(PIXEL_FILTER="latent_hh").returncode != 0
    assert _resolve(SWT_STRENGTH="-1").returncode != 0
    assert _resolve(STAT_ANCHOR="-1").returncode != 0
    assert _resolve(MAXSTEPS="15", SAMPLE_EVERY="15").returncode != 0


def test_nonzero_stage_requires_reviewed_q1p99_calibration():
    common = {
        "SURROGATE_STAGE": "active",
        "PIXW": "0.2",
        "SURROGATE_APPROVED": "YES",
        "SURROGATE_CALIBRATION_RUN": "nze64p52",
    }
    assert _resolve(SURROGATE_Q1_MIN="0.989", **common).returncode != 0
    proc = _resolve(SURROGATE_Q1_MIN="0.992351", **common)
    assert proc.returncode == 0, proc.stderr
    assert "_q1p99_active_reference_" in proc.stdout


def test_holder_step_does_not_combine_mutually_exclusive_slurm_flags():
    source = LAUNCHER.read_text()
    assert "--overlap --exclusive" not in source
    assert "--overlap --nodelist=" in source
