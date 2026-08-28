"""Focused CPU coverage for the CARN_VX_2708 comparison routes."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import ActionForcingDMD


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "sbatch" / "run_carn_vx_on_holder.sh"


def _resolve_launcher(mode: str, **overrides: str):
    env = os.environ.copy()
    env.update({
        "PRINT_ONLY": "1",
        "CARNVX_MODE": mode,
        "RUNSTAMP": "testroute",
    })
    env.update({key: str(value) for key, value in overrides.items()})
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc


def _last_wins(stdout: str):
    line = next(line for line in stdout.splitlines() if "last_wins=[" in line)
    tokens = shlex.split(line.split("last_wins=[", 1)[1].rsplit("]", 1)[0])
    return dict(token.split("=", 1) for token in tokens)


class _StepNet(nn.Module):
    """Deterministic raw delta: row level L moves by -(L+1)*scale."""

    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x, carn_step, residual=True):
        shape = (-1,) + (1,) * (x.dim() - 1)
        delta = -(carn_step.to(x.dtype) + 1).view(shape) * self.scale
        delta = delta.expand_as(x)
        return x + delta if residual else delta


class _Owner:
    _dedrift_with_reverse_noiser = ActionForcingDMD._dedrift_with_reverse_noiser
    _dedrift_streaming_slab_with_reverse_noiser = (
        ActionForcingDMD._dedrift_streaming_slab_with_reverse_noiser
    )
    _reverse_noiser_internalize_loss = (
        ActionForcingDMD._reverse_noiser_internalize_loss
    )

    def __init__(self, **kw):
        self.reverse_noiser_dedrift_enabled = True
        self.reverse_noiser_dedrift_min_level = 1
        self.reverse_noiser_dedrift_steps = 1
        self.reverse_noiser_dedrift_alpha0 = 1.0
        self.reverse_noiser_dedrift_alpha_decay = 0.5
        self.fn_pair_mode = "r1_vs_r2"
        self.forward_noiser_reverse = True
        self.forward_noiser_cycle_enabled = False
        self.forward_noiser = _StepNet()
        self.reverse_noiser = None
        self.num_frame_per_block = 3
        self.dmd_context_clean_frames = 9
        self.forward_noiser_max_carn_step = 8
        self.reverse_noiser_internalize_weight = 0.25
        for key, value in kw.items():
            setattr(self, key, value)


def test_direct_reverse_fn_is_a_real_consumer_not_identity():
    owner = _Owner()
    z = torch.ones(2, 3, 1, 1, 1, requires_grad=True)
    got = owner._dedrift_with_reverse_noiser(z, 1)
    assert not torch.equal(got, z)
    got.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert owner.forward_noiser.scale.grad is None
    assert owner.forward_noiser.scale.requires_grad


def test_per_row_levels_leave_level_zero_byte_identical():
    owner = _Owner()
    z = torch.ones(3, 3, 1, 1, 1)
    got = owner._dedrift_with_reverse_noiser(z, torch.tensor([0, 1, 2]))
    assert torch.equal(got[0], z[0])
    assert torch.all(got[2] < got[1])
    assert torch.all(got[1] < z[1])


def test_streaming_slab_is_chunked_and_position_conditioned():
    owner = _Owner()
    # abs [6,15) with a 9f seed -> levels [0,1,2].
    z = torch.ones(1, 9, 1, 1, 1)
    got = owner._dedrift_streaming_slab_with_reverse_noiser(
        z, {"abs_frame_start": 6, "overlap": 0},
    )
    assert torch.equal(got[:, :3], z[:, :3])
    assert torch.all(got[:, 6:9] < got[:, 3:6])
    assert owner._carn_slab_level_min == 0
    assert owner._carn_slab_level_max == 2


def test_aux_is_stop_grad_masked_and_nonzero_in_single_reverse_mode():
    owner = _Owner(reverse_noiser_internalize_weight=0.5)
    raw = torch.zeros(1, 6, 1, 1, 1, requires_grad=True)
    target = torch.full_like(raw, -2.0, requires_grad=True)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    mask[:, 3:] = True
    loss = owner._reverse_noiser_internalize_loss(raw, target, mask=mask)
    assert torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert raw.grad is not None
    assert torch.equal(raw.grad[:, :3], torch.zeros_like(raw.grad[:, :3]))
    assert target.grad is None


def test_aux_and_direct_score_are_separate_trainer_routes():
    src = inspect.getsource(
        __import__(
            "trainer.causal_action_forcing_train", fromlist=[
                "ActionForcingDMDTrainer",
            ],
        ).ActionForcingDMDTrainer._streaming_train_one_chunk
    )
    assert "reverse_noiser_dedrift_apply_to_train_score" in src
    assert "_dedrifted_train_chunk" in src
    assert "mask=train_info.get(\"gradient_mask\")" in src


def test_transition_both_signs_are_explicit_and_fail_loud():
    init_src = inspect.getsource(ActionForcingDMD.__init__)
    assert "ladd_gt_transition_carn_latter_reverse" in init_src
    trainer_src = inspect.getsource(
        __import__(
            "trainer.causal_action_forcing_train", fromlist=[
                "ActionForcingDMDTrainer",
            ],
        ).ActionForcingDMDTrainer._ladd_run_pair_mode
    )
    assert "[CARN-LATTER-REVERSE]" in trainer_src
    assert "[CARN-TX-BOTH]" in trainer_src
    assert "cycle_reverse_noiser" in trainer_src
    assert "tx_both requires forward_noiser_cycle_enabled=true" in trainer_src
    assert '"train/ladd_carn_former_rows_total"' in trainer_src
    assert '"train/ladd_carn_latter_reverse_rows_total"' in trainer_src


def test_original_launcher_modes_still_resolve_to_their_single_routes():
    plus = _last_wins(_resolve_launcher("tx_plus_former").stdout)
    assert plus["ladd_gt_transition_carn_former"] == "true"
    assert plus["ladd_gt_transition_carn_latter_reverse"] == "false"
    assert plus["forward_noiser_reverse"] == "false"
    assert plus["forward_noiser_train_source"] == "flash"

    minus = _last_wins(_resolve_launcher("tx_minus_latter").stdout)
    assert minus["ladd_gt_transition_carn_former"] == "false"
    assert minus["ladd_gt_transition_carn_latter_reverse"] == "true"
    assert minus["forward_noiser_reverse"] == "true"
    assert minus["forward_noiser_train_source"] == "flash"

    aux = _last_wins(_resolve_launcher("aux_minus").stdout)
    assert aux["ladd_gt_vs_fake_enabled"] == "true"
    assert aux["ladd_gt_transition_enabled"] == "false"
    assert aux["reverse_noiser_internalize_weight"] == "0.25"
    assert aux["forward_noiser_train_source"] == "rollout"


def test_tx_both_resolves_to_two_distinct_cycle_maps_and_300_steps():
    proc = _resolve_launcher("tx_both")
    assert proc.returncode == 0, proc.stderr
    cfg = _last_wins(proc.stdout)
    assert "max_steps=300" in proc.stdout
    assert "carnvx_tx_both_flash60" in proc.stdout
    assert cfg["sample_interval"] == "15"
    assert cfg["checkpoint_interval"] == "200"
    assert cfg["ladd_gt_vs_fake_enabled"] == "false"
    assert cfg["ladd_gt_transition_enabled"] == "true"
    assert cfg["ladd_fake_backbone_grad_scale"] == "0.2"
    assert cfg["forward_noiser_reverse"] == "false"
    assert cfg["forward_noiser_cycle_enabled"] == "true"
    assert cfg["ladd_gt_transition_carn_former"] == "true"
    assert cfg["ladd_gt_transition_carn_latter_reverse"] == "true"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "false"


def test_submitted_campaign_modes_pin_sampling_and_checkpoint_cadence():
    for mode in (
        "tx_both",
        "commit_aux",
        "commit_aux_flash60",
        "commit_aux_tx_minus",
        "aux_tx_minus",
    ):
        proc = _resolve_launcher(mode)
        assert proc.returncode == 0, (mode, proc.stderr)
        cfg = _last_wins(proc.stdout)
        assert cfg["sample_interval"] == "15", mode
        assert cfg["checkpoint_interval"] == "200", mode


def test_submitted_campaign_restores_old_flash_included_routing():
    for mode in (
        "tx_both",
        "commit_aux",
        "commit_aux_tx_minus",
        "aux_tx_minus",
    ):
        proc = _resolve_launcher(mode)
        assert proc.returncode == 0, (mode, proc.stderr)
        assert "flash60" in proc.stdout, mode
        cfg = _last_wins(proc.stdout)
        assert cfg["flash_dmd_enabled"] == "true", mode
        assert cfg["flash_dmd_gan_t"] == "60", mode
        assert cfg["anti_collapse_apply_to_flash_rung"] == "true", mode
        assert cfg["anti_collapse_apply_to_ladder_endpoint"] == "false", mode
        assert cfg["pix_finish_grad_enabled"] == "false", mode
        assert cfg["gen_aux_losses_x0_source"] == "flash", mode
        assert cfg["forward_noiser_rollout2_source"] == "legacy", mode
        assert cfg["ladd_fake_sample_source"] == "dmd", mode
        assert cfg["ladd_disc_force_clean"] == "true", mode
        expected_source = "flash" if mode == "tx_both" else "rollout"
        assert cfg["forward_noiser_train_source"] == expected_source, mode


def test_flash60_followup_restores_known_good_auxiliary_domains_only():
    proc = _resolve_launcher("commit_aux_flash60")
    assert proc.returncode == 0, proc.stderr
    assert "commit_aux_minus_flash60_tuned" in proc.stdout
    cfg = _last_wins(proc.stdout)
    assert cfg["sample_interval"] == "15"
    assert cfg["checkpoint_interval"] == "200"
    assert cfg["flash_dmd_enabled"] == "true"
    assert cfg["flash_dmd_gan_t"] == "60"
    assert cfg["ladd_fake_sample_source"] == "dmd"
    assert cfg["ladd_disc_force_clean"] == "true"
    assert cfg["anti_collapse_apply_to_flash_rung"] == "true"
    assert cfg["anti_collapse_apply_to_ladder_endpoint"] == "false"
    assert cfg["pix_finish_grad_enabled"] == "false"
    assert cfg["gen_aux_losses_x0_source"] == "flash"
    assert cfg["forward_noiser_train_source"] == "rollout"
    assert cfg["forward_noiser_rollout2_source"] == "legacy"
    assert cfg["reverse_noiser_internalize_weight"] == "0.25"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "true"


def test_staged_commit_preserves_aux_and_only_ramps_the_kv_consumer():
    proc = _resolve_launcher("commit_aux_staged")
    assert proc.returncode == 0, proc.stderr
    cfg = _last_wins(proc.stdout)
    assert cfg["sample_interval"] == "15"
    assert cfg["checkpoint_interval"] == "200"
    assert cfg["flash_dmd_enabled"] == "true"
    assert cfg["flash_dmd_gan_t"] == "60"
    assert cfg["forward_noiser_train_source"] == "rollout"
    assert cfg["forward_noiser_rollout2_source"] == "legacy"
    assert cfg["reverse_noiser_internalize_weight"] == "0.25"
    assert cfg["reverse_noiser_dedrift_apply_to_train_score"] == "false"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "true"
    assert cfg["reverse_noiser_commit_alpha"] == "0.25"
    assert cfg["reverse_noiser_commit_start_step"] == "100"
    assert cfg["reverse_noiser_commit_ramp_steps"] == "100"


def test_flash60_followup_is_a_direct_two_node_three_hour_batch_job():
    src = (ROOT / "sbatch" / "carn_commit_aux_flash60_3h.sbatch").read_text()
    assert "#SBATCH --nodes=2" in src
    assert "#SBATCH --gpus-per-node=4" in src
    assert "#SBATCH --time=03:00:00" in src
    assert "CARNVX_MODE=commit_aux_flash60" in src
    assert "MAXSTEPS=300" in src


def test_commit_aux_resolves_to_commit_and_internalize_without_transition():
    proc = _resolve_launcher("commit_aux")
    assert proc.returncode == 0, proc.stderr
    cfg = _last_wins(proc.stdout)
    assert cfg["ladd_gt_vs_fake_enabled"] == "true"
    assert cfg["ladd_gt_transition_enabled"] == "false"
    assert cfg["ladd_fake_backbone_grad_scale"] == "0.2"
    assert cfg["forward_noiser_reverse"] == "true"
    assert cfg["forward_noiser_train_source"] == "rollout"
    assert cfg["reverse_noiser_dedrift_apply_to_train_score"] == "false"
    assert cfg["reverse_noiser_internalize_weight"] == "0.25"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "true"


def test_aux_tx_minus_and_commit_variant_are_exactly_one_flag_apart():
    aux_tx = _last_wins(_resolve_launcher("aux_tx_minus").stdout)
    commit_aux_tx = _last_wins(_resolve_launcher("commit_aux_tx_minus").stdout)
    assert aux_tx["ladd_gt_vs_fake_enabled"] == "true"
    assert aux_tx["ladd_gt_transition_enabled"] == "true"
    assert aux_tx["ladd_fake_backbone_grad_scale"] == "0.1"
    assert aux_tx["ladd_gt_transition_carn_latter_reverse"] == "true"
    assert aux_tx["reverse_noiser_internalize_weight"] == "0.25"
    assert aux_tx["reverse_noiser_dedrift_apply_to_commit"] == "false"
    assert commit_aux_tx["reverse_noiser_dedrift_apply_to_commit"] == "true"
    differing = {
        key for key in aux_tx | commit_aux_tx
        if aux_tx.get(key) != commit_aux_tx.get(key)
    }
    assert differing == {"reverse_noiser_dedrift_apply_to_commit"}


def test_aux_tx_minus_action_critic_ablations_change_only_action_apparatus():
    base = _last_wins(_resolve_launcher("aux_tx_minus").stdout)
    online = _last_wins(_resolve_launcher("aux_tx_minus_ac_online").stdout)
    strong = _last_wins(
        _resolve_launcher("aux_tx_minus_ac_frozen055").stdout)

    scientific_keys = {
        "forward_noiser_train_source", "forward_noiser_rollout2_source",
        "ladd_gt_vs_fake_enabled", "ladd_gt_transition_enabled",
        "ladd_fake_backbone_grad_scale", "forward_noiser_reverse",
        "reverse_noiser_dedrift_enabled",
        "reverse_noiser_dedrift_apply_to_train_score",
        "reverse_noiser_internalize_weight",
        "ladd_gt_transition_carn_latter_reverse",
        "ladd_gt_transition_carn_steps",
        "ladd_gt_transition_gen_detach_former", "flash_dmd_enabled",
        "flash_dmd_gan_t", "sample_interval", "checkpoint_interval",
    }
    for key in scientific_keys:
        assert online[key] == base[key], key
        assert strong[key] == base[key], key

    assert online["action_critic_aux_enabled"] == "true"
    assert online["action_critic_freeze"] == "false"
    assert online["action_teacher_mode"] == "all"
    assert online["action_critic_z_loss_weight"] == "0.5"
    assert online["critic_updates_per_step"] == "2"
    assert online["generator_action_z_guidance_weight"] == "0.3"

    assert strong["action_critic_aux_enabled"] == "true"
    assert strong["action_critic_freeze"] == "true"
    assert strong["action_teacher_mode"] == "off"
    assert strong["generator_action_z_guidance_weight"] == "0.55"


def test_action_critic_ablation_jobs_are_two_node_three_hour_300_step_jobs():
    for filename, mode in (
        ("carn_aux_tx_minus_ac_online_3h.sbatch",
         "aux_tx_minus_ac_online"),
        ("carn_aux_tx_minus_ac_frozen055_3h.sbatch",
         "aux_tx_minus_ac_frozen055"),
    ):
        src = (ROOT / "sbatch" / filename).read_text()
        assert "#SBATCH --nodes=2" in src
        assert "#SBATCH --gpus-per-node=4" in src
        assert "#SBATCH --time=03:00:00" in src
        assert "MAXSTEPS=300" in src
        assert f"CARNVX_MODE={mode}" in src


def test_full_cycle_stack_keeps_two_pair_mode_dose_constant():
    cfg = _last_wins(_resolve_launcher("commit_aux_tx_both").stdout)
    assert cfg["ladd_gt_vs_fake_enabled"] == "true"
    assert cfg["ladd_gt_transition_enabled"] == "true"
    assert cfg["ladd_fake_backbone_grad_scale"] == "0.1"
    assert cfg["forward_noiser_reverse"] == "false"
    assert cfg["forward_noiser_cycle_enabled"] == "true"
    assert cfg["ladd_gt_transition_carn_former"] == "true"
    assert cfg["ladd_gt_transition_carn_latter_reverse"] == "true"
    assert cfg["reverse_noiser_internalize_weight"] == "0.25"
    assert cfg["reverse_noiser_dedrift_apply_to_commit"] == "true"


def test_full_cycle_stack_has_a_direct_two_node_three_hour_batch_job():
    src = (ROOT / "sbatch" / "carn_commit_aux_tx_both_3h.sbatch").read_text()
    assert "#SBATCH --nodes=2" in src
    assert "#SBATCH --gpus-per-node=4" in src
    assert "#SBATCH --time=03:00:00" in src
    assert "CARNVX_MODE=commit_aux_tx_both" in src
    assert "MAXSTEPS=300" in src
    assert "bash sbatch/run_carn_vx_snapshot.sh" in src


def test_launcher_rejects_unknown_mode_without_needing_a_holder():
    proc = _resolve_launcher("not_a_mode")
    assert proc.returncode == 2
    assert "unknown CARNVX_MODE=not_a_mode" in proc.stderr


def test_ladder_endpoint_sources_are_explicit_and_fail_loud_in_code():
    init_src = inspect.getsource(ActionForcingDMD.__init__)
    assert "forward_noiser_rollout2_source" in init_src
    assert "anti_collapse_apply_to_ladder_endpoint" in init_src
    prebuild_src = inspect.getsource(
        ActionForcingDMD._prebuild_rollout2_for_v24
    )
    assert 'r2_source == "ladder_endpoint"' in prebuild_src
    assert "did not publish its requested" in prebuild_src
    trainer_cls = __import__(
        "trainer.causal_action_forcing_train", fromlist=[
            "ActionForcingDMDTrainer",
        ],
    ).ActionForcingDMDTrainer
    tf_src = inspect.getsource(trainer_cls._train_forward_noiser_tf)
    assert 'elif _fn_source == "ladder_endpoint"' in tf_src
    step_src = inspect.getsource(trainer_cls._streaming_train_one_chunk)
    assert 'if _ac_x0_source == "ladder_endpoint"' in step_src
    assert "action_critic_x0_ladder_endpoint" in step_src
