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
    # The action-critic defaults deliberately restate the htfhu37j/base
    # treatment so the frozen/online experiment is a named launcher axis.
    assert cfg["action_critic_aux_enabled"] == "true"
    assert cfg["action_critic_freeze"] == "true"
    assert cfg["action_teacher_mode"] == "off"
    assert cfg["generator_action_z_guidance_weight"] == "0.3"
    # Other inherited base settings must not be silently overridden.
    for key in (
        "forward_noiser_cycle_enabled", "ladd_gt_vs_fake_enabled",
        "ladd_gt_transition_enabled", "seed",
    ):
        assert key not in cfg


def test_action_critic_frozen_strength_and_online_training_are_explicit_axes():
    frozen = _resolve(
        ACTION_CRITIC_FREEZE="true",
        ACTION_CRITIC_GUIDANCE_WEIGHT=".6",
        ACTION_TEACHER_MODE="off",
    )
    assert frozen.returncode == 0, frozen.stderr
    frozen_cfg = _cfg(frozen.stdout)
    assert frozen_cfg["action_critic_freeze"] == "true"
    assert frozen_cfg["action_teacher_mode"] == "off"
    assert frozen_cfg["generator_action_z_guidance_weight"] == ".6"
    assert "_acfrozen_gp6_" in frozen.stdout

    online = _resolve(
        ACTION_CRITIC_FREEZE="false",
        ACTION_CRITIC_GUIDANCE_WEIGHT=".3",
        ACTION_TEACHER_MODE="all",
        ACTION_CRITIC_UPDATES="2",
        ACTION_CRITIC_Z_LOSS_WEIGHT=".5",
        ACTION_CRITIC_LR="3e-4",
        TEACHER_ACTION_ENCODER="pca_raw",
    )
    assert online.returncode == 0, online.stderr
    online_cfg = _cfg(online.stdout)
    assert online_cfg["action_critic_freeze"] == "false"
    assert online_cfg["action_teacher_mode"] == "all"
    assert online_cfg["teacher_action_encoder"] == "pca_raw"
    assert online_cfg["critic_updates_per_step"] == "2"
    assert online_cfg["action_critic_z_loss_weight"] == ".5"
    assert online_cfg["critic_lr"] == "3e-4"
    assert "_aconline_all_u2_gp3_" in online.stdout

    assert _resolve(
        ACTION_CRITIC_FREEZE="false", ACTION_TEACHER_MODE="off",
    ).returncode != 0
    assert _resolve(
        ACTION_CRITIC_FREEZE="true", ACTION_TEACHER_MODE="all",
    ).returncode != 0


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
    assert cfg["surrogate_decoder_fresh_disc_order"] == "true"
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
    # Production keeps the raw/evolving mean visible to D while blocking
    # broad brightness cotangents on the generator side.
    assert cfg["ladd_pixel_input_filter"] == "raw_grad_hp"
    assert cfg["ladd_pixel_swt_strength"] == "1.0"
    assert cfg["stat_anchor_loss_weight"] == "1.0"
    assert cfg["gan_lr"] == "2e-5"


def test_brightness_wavelet_and_anchor_ablation_are_explicit():
    default = _cfg(_resolve().stdout)
    assert default["ladd_pixel_input_filter"] == "raw_grad_hp"
    assert default["stat_anchor_loss_weight"] == "1.0"

    dc = _cfg(_resolve(PIXEL_FILTER="dc").stdout)
    assert dc["ladd_pixel_input_filter"] == "dc"
    assert dc["stat_anchor_loss_weight"] == "1.0"

    swt = _cfg(_resolve(PIXEL_FILTER="swt", SWT_STRENGTH="0.75").stdout)
    assert swt["ladd_pixel_input_filter"] == "swt"
    assert swt["ladd_pixel_swt_strength"] == "0.75"

    no_anchor = _cfg(_resolve(PIXEL_FILTER="dc", STAT_ANCHOR="0").stdout)
    assert no_anchor["ladd_pixel_input_filter"] == "dc"
    assert no_anchor["stat_anchor_loss_weight"] == "0"

    texture_grad = _cfg(_resolve(PIXEL_FILTER="dc_grad_hp").stdout)
    assert texture_grad["ladd_pixel_input_filter"] == "dc_grad_hp"
    assert texture_grad["stat_anchor_loss_weight"] == "1.0"

    raw_texture_grad = _cfg(_resolve(PIXEL_FILTER="raw_grad_hp").stdout)
    assert raw_texture_grad["ladd_pixel_input_filter"] == "raw_grad_hp"
    assert raw_texture_grad["stat_anchor_loss_weight"] == "1.0"


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


def test_fresh_discriminator_order_is_default_and_controls_are_named():
    fresh = _resolve()
    assert fresh.returncode == 0, fresh.stderr
    assert _cfg(fresh.stdout)["ladd_defer_disc_update"] == "false"
    assert _cfg(fresh.stdout)["surrogate_decoder_fresh_disc_order"] == "true"
    assert "_dfresh_DthenG_" in fresh.stdout

    deferred = _resolve(
        LADD_DEFER_DISC_UPDATE="true",
        SURROGATE_FRESH_DISC_ORDER="false",
    )
    assert deferred.returncode == 0, deferred.stderr
    assert _cfg(deferred.stdout)["ladd_defer_disc_update"] == "true"
    assert "_ddefer_GthenD_" in deferred.stdout

    inline_old_order = _resolve(SURROGATE_FRESH_DISC_ORDER="false")
    assert inline_old_order.returncode == 0, inline_old_order.stderr
    assert "_dinline_GthenD_" in inline_old_order.stdout

    assert _resolve(LADD_DEFER_DISC_UPDATE="true").returncode != 0
    assert _resolve(LADD_DEFER_DISC_UPDATE="sometimes").returncode != 0
    assert _resolve(SURROGATE_FRESH_DISC_ORDER="sometimes").returncode != 0


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


def test_carn_commit_aux_staged_is_the_fail_closed_promoted_recipe():
    staged = {
        "CARNMODE": "commit_aux_staged",
        "PIXEL_FILTER": "dc",
        "STAT_ANCHOR": "1.0",
        "GANLR": "1e-3",
        "GANUPDATES": "1",
        "LADD_R1_GAMMA": "1",
    }
    proc = _resolve(**staged)
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["forward_noiser_enabled"] == "true"
    assert cfg["forward_noiser_train_source"] == "rollout"
    assert cfg["forward_noiser_rollout2_source"] == "legacy"
    assert cfg["forward_noiser_reverse"] == "true"
    assert cfg["forward_noiser_cycle_enabled"] == "false"
    for key in (
        "forward_noiser_apply_decoupled", "forward_noiser_apply_gt_former",
        "forward_noiser_apply_gt_both", "forward_noiser_apply_in_aux",
    ):
        assert cfg[key] == "false"
    assert cfg["reverse_noiser_dedrift_enabled"] == "true"
    assert cfg["reverse_noiser_dedrift_level"] == "1"
    assert cfg["reverse_noiser_dedrift_min_level"] == "1"
    assert cfg["reverse_noiser_dedrift_steps"] == "1"
    assert cfg["reverse_noiser_dedrift_alpha0"] == "1.0"
    assert cfg["reverse_noiser_internalize_weight"] == "0.25"
    assert cfg["reverse_noiser_dedrift_apply_to_train_score"] == "false"
    assert cfg["reverse_noiser_dedrift_apply_to_flash"] == "false"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "true"
    assert cfg["reverse_noiser_commit_alpha"] == "0.25"
    assert cfg["reverse_noiser_commit_start_step"] == "100"
    assert cfg["reverse_noiser_commit_ramp_steps"] == "100"
    assert cfg["gen_aux_losses_x0_source"] == "flash"
    assert cfg["ladd_fake_sample_source"] == "flash"
    assert cfg["pix_flash_grad_select_enabled"] == "true"
    assert cfg["pix_finish_grad_enabled"] == "false"
    assert cfg["ladd_pixel_input_filter"] == "dc"
    assert cfg["ladd_pixel_swt_strength"] == "0.0"
    assert cfg["stat_anchor_loss_weight"] == "1.0"
    assert cfg["ladd_gt_transition_match"] == "false"
    assert cfg["stat_anchor_use_clean_match_offset"] == "true"
    assert cfg["stat_anchor_match_k"] == "2"
    assert cfg["action_critic_aux_enabled"] == "true"
    assert cfg["action_critic_freeze"] == "true"
    assert cfg["generator_action_z_guidance_weight"] == "0.3"
    assert cfg["ladd_pixel_decode_cache"] == "false"
    assert cfg["ladd_defer_disc_update"] == "false"
    assert cfg["surrogate_decoder_fresh_disc_order"] == "true"
    assert cfg["gan_updates_per_step"] == "1"
    assert cfg["gan_lr"] == "1e-3"
    assert cfg["ladd_r1_gamma"] == "1"
    assert "_commit_aux_staged_share" in proc.stdout

    for bad in (
        {"PIXEL_FILTER": "none"}, {"PIXEL_FILTER": "swt"},
        {"PIXEL_FILTER": "raw_grad_hp"},
        {"STAT_ANCHOR": "0"}, {"GANLR": "2e-5"},
        {"GANUPDATES": "2"}, {"LADD_R1_GAMMA": "10"},
        {"LADD_DEFER_DISC_UPDATE": "true",
         "SURROGATE_FRESH_DISC_ORDER": "false"},
    ):
        assert _resolve(**(staged | bad)).returncode != 0


def test_harmony_bridge_accepts_dc_off_brightness_default():
    proc = _resolve(
        CARNMODE="harmony_bridge", GANLR="1e-3", LADD_R1_GAMMA="1",
    )
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["ladd_pixel_input_filter"] == "raw_grad_hp"
    assert cfg["stat_anchor_loss_weight"] == "1.0"
    assert cfg["ladd_gt_transition_carn_bridge"] == "true"

    dc_control = _resolve(
        CARNMODE="harmony_bridge", PIXEL_FILTER="dc", GANLR="1e-3",
        LADD_R1_GAMMA="1",
    )
    assert dc_control.returncode == 0, dc_control.stderr

    for invalid_filter in ("none", "swt", "dc_grad_hp"):
        assert _resolve(
            CARNMODE="harmony_bridge", PIXEL_FILTER=invalid_filter,
            GANLR="1e-3", LADD_R1_GAMMA="1",
        ).returncode != 0


def test_carn_commit_aux_staged_schedule_knobs_are_validated():
    staged = {
        "CARNMODE": "commit_aux_staged", "PIXEL_FILTER": "dc",
        "GANLR": "1e-3", "LADD_R1_GAMMA": "1",
    }
    proc = _resolve(
        CARN_COMMIT_ALPHA="0.4", CARN_COMMIT_START_STEP="80",
        CARN_COMMIT_RAMP_STEPS="120", **staged,
    )
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["reverse_noiser_commit_alpha"] == "0.4"
    assert cfg["reverse_noiser_commit_start_step"] == "80"
    assert cfg["reverse_noiser_commit_ramp_steps"] == "120"
    assert _resolve(CARN_COMMIT_ALPHA="1.1", **staged).returncode != 0
    assert _resolve(CARN_COMMIT_ALPHA="-0.1", **staged).returncode != 0
    assert _resolve(CARN_COMMIT_START_STEP="-1", **staged).returncode != 0
    assert _resolve(CARN_COMMIT_RAMP_STEPS="1.5", **staged).returncode != 0


def test_carn_commit_aux_staged_active_requires_its_own_weight_zero_run():
    active = {
        "CARNMODE": "commit_aux_staged", "PIXEL_FILTER": "dc",
        "GANLR": "1e-3", "LADD_R1_GAMMA": "1",
        "SURROGATE_STAGE": "active", "PIXW": ".0125",
        "SURROGATE_APPROVED": "YES",
        "SURROGATE_CALIBRATION_RUN": "q1p99_offline_authority",
        "SURROGATE_Q1_MIN": ".992351",
    }
    assert _resolve(**active).returncode != 0
    proc = _resolve(
        CARN_STAGED_CALIBRATION_APPROVED="YES",
        CARN_STAGED_CALIBRATION_RUN="staged_weight_zero_run_id",
        **active,
    )
    assert proc.returncode == 0, proc.stderr
    assert "staged_calibration_run=staged_weight_zero_run_id" in proc.stdout
    assert "staged_calibration_approved=YES" in proc.stdout


def test_staged_weight_zero_wrapper_runs_past_every_schedule_landmark():
    source = (
        ROOT / "sbatch" /
        "run_carn_q1p99_commit_aux_staged_calibration_node.sh"
    ).read_text()
    assert "PIXW=0" in source
    assert "CARNMODE=commit_aux_staged" in source
    assert "CARN_AUX_WEIGHT=.25" in source
    assert "CARN_COMMIT_ALPHA=.25" in source
    assert "CARN_COMMIT_START_STEP=100" in source
    assert "CARN_COMMIT_RAMP_STEPS=100" in source
    assert 'MAXSTEPS="${MAXSTEPS:-225}"' in source
    assert "PIXEL_FILTER=dc SWT_STRENGTH=0.0 STAT_ANCHOR=1.0" in source


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


def test_htf_reference_rollout_uses_native_four_roll_geometry():
    proc = _resolve(HTF_REFERENCE_ROLLOUT="true")
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["num_training_frames"] == "21"
    assert cfg["streaming_chunk_size"] == "18"
    assert cfg["max_rolls_per_ride"] == "4"
    assert cfg["dmd_only_first_chunk_per_ride"] == "false"
    assert cfg["dmd_42f_rolling_sup_new"] == "true"
    assert cfg["dmd_42f_allroll_student_ctx"] == "true"
    assert cfg["num_chunks_roll_forward"] == "3"
    assert cfg["streaming_min_new_frame"] == "9"
    assert cfg["streaming_max_length"] == "60"
    assert cfg["sample_7chunk_enabled"] == "true"
    assert cfg["rollout_viz_source"] == "finish"

    assert _resolve(
        STATIONARY_FIRST7="true", HTF_REFERENCE_ROLLOUT="true",
    ).returncode != 0


def test_memory_reduced_calibration_keeps_d_budget_and_reduces_only_g():
    proc = _resolve(
        MEMORY_REDUCED_G="true", SAMPLE_EVERY="0", EMPTY_CACHE_INTERVAL="1",
    )
    assert proc.returncode == 0, proc.stderr
    cfg = _cfg(proc.stdout)
    assert cfg["sample_interval"] == "0"
    assert cfg["empty_cache_interval"] == "1"
    assert cfg["ladd_pixel_crops_per_row"] == "2"
    assert cfg["ladd_pixel_lat_frames"] == "3"
    assert cfg["ladd_pixel_frames_per_crop"] == "5"
    assert cfg["ladd_pixel_g_crops_per_row"] == "1"
    assert cfg["ladd_pixel_g_lat_frames"] == "2"
    assert cfg["ladd_pixel_g_frames_per_crop"] == "2"
    assert cfg["ladd_pixel_post_d_memory_release"] == "true"
    assert "_g1x_" in proc.stdout
    assert _resolve(MEMORY_REDUCED_G="sometimes").returncode != 0


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


def test_fresh_field_is_constructed_only_after_inline_disc_update():
    source = (
        ROOT / "trainer" / "causal_action_forcing_train.py"
    ).read_text()
    start = source.index("# The only fresh-order call site.")
    gan_call = source.rfind(
        "gen_gan_loss, gan_logs = self._compute_r3gan_losses(", 0, start,
    )
    field_call = source.index(
        "self._compute_pixel_texture_g_loss(", start,
    )
    fold = source.index("if _pix_g_w is not None:", field_call)
    assert gan_call < start < field_call < fold
    assert "surrogate_decoder_disc_updated_before_field" in source
    assert "_inline_disc_after - _inline_disc_before" in source
    assert "current-batch inline discriminator update completed" in source
    assert "surrogate_decoder_fresh_disc_order=true but the current" in source
    assert "streaming step has no live discriminator/optimizer" in source
