from unittest.mock import patch
from types import SimpleNamespace

import torch

# Wan's T5 wrapper evaluates this default at import time even though this test
# never constructs a model.
with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
    from trainer.causal_rolling_staircase_train import RollingStaircaseDMDTrainer


def test_gan_only_checkpoint_does_not_reopen_parent_save(monkeypatch):
    trainer = ActionForcingDMDTrainer.__new__(ActionForcingDMDTrainer)
    trainer.is_main_process = True
    trainer.action_critic_loss_active = False
    trainer.real_teacher_train_online = False
    trainer.state_probe_aux_active = False
    trainer.gan_enabled = True
    trainer.config = SimpleNamespace()

    parent_calls = []
    monkeypatch.setattr(
        RollingStaircaseDMDTrainer,
        "_save_checkpoint",
        lambda _self: parent_calls.append(True),
    )

    ActionForcingDMDTrainer._save_checkpoint(trainer)

    assert parent_calls == [True]


def test_eval_only_checkpoint_can_skip_full_parent_save(monkeypatch):
    trainer = ActionForcingDMDTrainer.__new__(ActionForcingDMDTrainer)
    trainer.is_main_process = True
    trainer.config = SimpleNamespace(
        eval_checkpoint_path=None,
        save_full_checkpoint=False,
    )

    parent_calls = []
    monkeypatch.setattr(
        RollingStaircaseDMDTrainer,
        "_save_checkpoint",
        lambda _self: parent_calls.append(True),
    )

    ActionForcingDMDTrainer._save_checkpoint(trainer)

    assert parent_calls == []
