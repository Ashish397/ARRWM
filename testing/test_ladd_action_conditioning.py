"""CPU contracts for the action-aware VGG/LADD discriminator path."""

import contextlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ladd_disc import LADDActionConditioner, LADDDiscriminator
from model.ladd_pixel_features import LaddPixelFeatureSource, LaddPixelStatHead

with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer.causal_action_forcing_train import ActionForcingDMDTrainer


class _RaisingProjector:
    def __call__(self, **_kwargs):
        raise AssertionError("the VGG/pixel path must not call the WAN projector")


def _decode(latents, want_grad=False, **_kwargs):
    n, f, c, h, w = latents.shape
    x = latents if want_grad else latents.detach()
    x = x[:, :, :3] if c >= 3 else x.repeat(1, 1, 3, 1, 1)[:, :, :3]
    x = x.reshape(n * f, 3, h, w)
    x = torch.nn.functional.interpolate(x, scale_factor=8, mode="nearest")
    return x.reshape(n, f, 3, h * 8, w * 8)


def _pixel_disc(*, action_dim=6, cmap_dim=8):
    src = LaddPixelFeatureSource(
        "pixgan", pixgan_base_channels=8, common_stride=8,
    )
    disc = LADDDiscriminator(
        projector=_RaisingProjector(),
        block_indices=[0, 2],
        dim_teacher=32,
        dim_proj=16,
        use_csm=True,
        pixel_source=src,
        cmap_dim=cmap_dim,
        action_embed_dim=action_dim,
    )
    disc.pixel_decode_fn = _decode
    disc.pixel_cfg = {
        "crop_rows": 8,
        "crop_cols": 8,
        "crops_per_row": 1,
        "lat_frames": 2,
        "frames_per_crop": 2,
        "border": 2,
        "decode_batch": 4,
    }
    disc.eval()
    return disc, src


def _pooled_pixel_disc(*, action_dim=6, cmap_dim=8):
    # A lightweight pixel trunk with the exact orderless readout contract used
    # by VGG.  This avoids downloading/constructing VGG in the CPU seam test
    # while exercising the production pooled branch itself.
    src = LaddPixelFeatureSource(
        "pixgan", pixgan_base_channels=8, common_stride=8,
    )
    src.pooled_readout = LaddPixelStatHead(
        [src.tap_dims[i] for i in src.tap_indices],
        proj_dim=4, hidden_dim=12,
    )
    disc = LADDDiscriminator(
        projector=_RaisingProjector(),
        block_indices=src.tap_indices,
        dim_teacher=32,
        dim_proj=16,
        use_csm=True,
        pixel_source=src,
        cmap_dim=cmap_dim,
        action_embed_dim=action_dim,
    )
    disc.pixel_decode_fn = _decode
    disc.pixel_cfg = {
        "crop_rows": 8,
        "crop_cols": 8,
        "crops_per_row": 1,
        "lat_frames": 2,
        "frames_per_crop": 2,
        "border": 2,
        "decode_batch": 4,
    }
    return disc, src


def test_action_conditioner_preserves_temporal_order_information():
    torch.manual_seed(1)
    cond = LADDActionConditioner(action_embed_dim=6, cmap_dim=8).eval()
    actions = torch.randn(2, 4, 6)
    forward = cond(actions)
    reverse = cond(actions.flip(1))
    assert forward.shape == (2, 8)
    assert not torch.allclose(forward, reverse)


def test_pixel_disc_reuses_one_feature_forward_for_wrong_action_logits():
    torch.manual_seed(2)
    disc, src = _pixel_disc()
    x = torch.randn(2, 2, 16, 8, 10)
    t = torch.zeros(2, 2, dtype=torch.long)
    pe = torch.zeros(2, 4, 8)
    act = torch.randn(2, 2, 6)
    wrong = act.roll(1, 0)

    matched, mismatched = disc(
        x_noisy=x,
        timestep=t,
        prompt_embeds=pe,
        conditional_extra={
            "_action_tokens": act,
            "_ladd_action_tokens_mismatch": wrong,
        },
    )
    assert matched.shape == mismatched.shape
    assert matched.shape[0] == 2
    assert not torch.allclose(matched, mismatched)
    # The alternate condition reuses the decoded pixel/VGG features; it is
    # only a second cheap cmap projection through the dense heads.
    assert src.n_forward == 1


def test_train_mode_joint_projection_backwards_without_sn_version_drift():
    torch.manual_seed(22)
    disc, src = _pixel_disc()
    disc.train()
    x = torch.randn(2, 2, 16, 8, 10)
    act = torch.randn(2, 2, 6)
    matched, mismatched = disc(
        x_noisy=x,
        timestep=torch.zeros(2, 2, dtype=torch.long),
        prompt_embeds=torch.zeros(2, 4, 8),
        conditional_extra={
            "_action_tokens": act,
            "_ladd_action_tokens_mismatch": act.roll(1, 0),
        },
    )
    torch.nn.functional.softplus(mismatched - matched).mean().backward()
    assert src.n_forward == 1
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in disc.action_cmapper.parameters()
    )


def test_orderless_pooled_disc_is_action_conditioned_without_dense_heads():
    torch.manual_seed(23)
    disc, src = _pooled_pixel_disc()
    disc.train()
    actions = torch.randn(2, 2, 6)
    matched, mismatched = disc(
        x_noisy=torch.randn(2, 2, 16, 8, 10),
        timestep=torch.zeros(2, 2, dtype=torch.long),
        prompt_embeds=torch.zeros(2, 4, 8),
        conditional_extra={
            "_action_tokens": actions,
            "_ladd_action_tokens_mismatch": actions.roll(1, 0),
        },
    )
    assert matched.shape == mismatched.shape == (2, 2)
    assert not torch.allclose(matched, mismatched)
    assert src.n_forward == 1
    assert src.pooled_readout.n_forward == 1
    assert disc.heads is None
    assert disc.pixel_stats["dense_head_calls"] == 0.0
    torch.nn.functional.softplus(mismatched - matched).mean().backward()
    assert disc.pooled_action_proj.weight.grad is not None
    assert any(p.grad is not None for p in disc.action_cmapper.parameters())


def test_orderless_pooled_score_pixels_consumes_matching_action_rows():
    torch.manual_seed(24)
    disc, src = _pooled_pixel_disc()
    disc.eval()
    pixels = torch.randn(6, 3, 64, 64)
    actions = torch.randn(2, 3, 6)
    first = disc.score_pixels(pixels, action_tokens=actions)
    second = disc.score_pixels(pixels, action_tokens=actions.flip(1))
    assert first.shape == second.shape == (6,)
    assert not torch.allclose(first, second)
    assert src.n_forward == 2


def test_action_blind_pooled_disc_ignores_wan_required_matched_tokens():
    """WAN geometry may require action tensors even when VGG is blind.

    Passing those aligned tensors must preserve the blind control rather than
    aborting.  A wrong-action objective remains forbidden without an action
    projector (covered by the explicit error below).
    """
    torch.manual_seed(25)
    disc, _ = _pooled_pixel_disc(action_dim=0, cmap_dim=0)
    x = torch.randn(2, 2, 16, 8, 10)
    common = dict(
        x_noisy=x,
        timestep=torch.zeros(2, 2, dtype=torch.long),
        prompt_embeds=torch.zeros(2, 4, 8),
    )
    blind = disc(**common)
    with_tokens = disc(
        **common,
        conditional_extra={"_action_tokens": torch.randn(2, 2, 6)},
    )
    assert torch.equal(blind, with_tokens)
    with pytest.raises(ValueError, match="Wrong-action logits require"):
        disc(
            **common,
            conditional_extra={
                "_action_tokens": torch.randn(2, 2, 6),
                "_ladd_action_tokens_mismatch": torch.randn(2, 2, 6),
            },
        )


def test_action_conditioning_requires_tokens_fail_loud():
    disc, _ = _pixel_disc()
    x = torch.randn(2, 2, 16, 8, 10)
    with pytest.raises(ValueError, match="action conditioner is active"):
        disc(
            x_noisy=x,
            timestep=torch.zeros(2, 2, dtype=torch.long),
            prompt_embeds=torch.zeros(2, 4, 8),
        )


def test_wrong_action_builder_is_a_valid_row_derangement():
    actions = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    wrong, delta = ActionForcingDMDTrainer._ladd_wrong_action_tokens(actions)
    assert wrong.shape == actions.shape
    assert delta > 0.0
    assert all(not torch.equal(wrong[i], actions[i]) for i in range(4))
    assert sorted(wrong[:, 0, 0].tolist()) == sorted(actions[:, 0, 0].tolist())


def test_generator_action_window_tracks_selected_flash_subwindow():
    trainer = ActionForcingDMDTrainer.__new__(ActionForcingDMDTrainer)
    trainer.config = SimpleNamespace(ladd_use_action_cond=True)
    ride_actions = torch.arange(
        2 * 20 * 3, dtype=torch.float32,
    ).reshape(2, 20, 3)
    trainer.model = SimpleNamespace(
        streaming_state={
            "ride_actions_window": ride_actions,
            "last_chunk_lo_in_ride_window": 6,
            "prompt_embeds": torch.zeros(2, 1, 4),
        },
        action_token_projection=nn.Identity(),
    )
    fake = torch.zeros(2, 3, 16, 2, 2)
    tokens, logs = trainer._pix_action_tokens_for_fake(
        fake, relative_start=4,
    )
    assert torch.equal(tokens, ride_actions[:, 10:13])
    assert logs["train/pix_g_action_conditioned"] == 1.0
    assert logs["train/pix_g_action_abs_lo"] == 10.0
    assert logs["train/pix_g_action_frames"] == 3.0


class _ToyConditionalDisc(nn.Module):
    def __init__(self, width=5, tokens=3, action_dim=4):
        super().__init__()
        self.stat_logit_count = 0
        self.net = nn.Linear(width, tokens)
        self.action = nn.Linear(action_dim, tokens, bias=False)

    def forward(self, x_noisy=None, conditional_extra=None, **_kwargs):
        base = self.net(x_noisy)
        matched = base + self.action(
            conditional_extra["_action_tokens"].mean(dim=1)
        )
        wrong = conditional_extra.get("_ladd_action_tokens_mismatch")
        if wrong is None:
            return matched
        return matched, base + self.action(wrong.mean(dim=1))


class _RecordingOpt:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        pass


def _run_mismatch_helper(weight):
    torch.manual_seed(4)
    disc = _ToyConditionalDisc()
    trainer = ActionForcingDMDTrainer.__new__(ActionForcingDMDTrainer)
    trainer.config = SimpleNamespace()
    trainer.model = SimpleNamespace()
    trainer.r3gan_disc = disc
    trainer.r3gan_optimizer = _RecordingOpt(disc.parameters())
    trainer.gan_max_grad_norm = 0.0
    trainer._mem_step_snapshot = lambda *_a, **_kw: None
    b = 4
    actions = torch.randn(b, 2, 4)
    wrong = actions.roll(2, 0) if weight > 0 else None
    out = trainer._ladd_disc_update_positional_microbatched(
        _it=0,
        real_part=torch.randn(b, 5),
        fake_part=torch.randn(b, 5),
        t_disc=torch.zeros(b),
        prompt_embeds_eff=torch.zeros(b, 2, 3),
        pooled_prompt=None,
        real_action_tokens=actions,
        fake_action_tokens=actions,
        real_action_modulation=None,
        fake_action_modulation=None,
        disc_for_update=disc,
        _disc_no_sync=lambda: contextlib.nullcontext(),
        _K_stat=0,
        _W_stat=0.0,
        _do_r1=False,
        _r1_sigma=0.01,
        _r1_gamma=1.0,
        _r1_tok_norm=True,
        _r1_num_samples=0,
        current_step=10,
        _micro_groups=2,
        real_action_tokens_mismatch=wrong,
        action_mismatch_weight=weight,
    )
    return out


def test_microbatched_d_update_optimizes_and_logs_wrong_action_negative():
    off = _run_mismatch_helper(0.0)
    on = _run_mismatch_helper(1.0)
    assert off["d_loss_action_mismatch"] == 0.0
    assert off["d_wrong_action"] == 0.0
    assert on["d_loss_action_mismatch"] > 0.0
    assert on["d_wrong_action"] != 0.0
    assert on["d_loss"] > off["d_loss"]
