"""CPU invariants for the production GAN + CARN commit rehabilitation."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model.carn_commit import (  # noqa: E402
    absolute_carn_level,
    blend_carn_commit,
    resolved_commit_alpha,
    straight_through_value,
)

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import ActionForcingDMD  # noqa: E402


def test_production_schedule_is_meaningful_but_staged():
    kw = dict(target_alpha=0.125, start_step=150, ramp_steps=100)
    assert resolved_commit_alpha(step=149, **kw) == 0.0
    assert resolved_commit_alpha(step=200, **kw) == pytest.approx(0.0625)
    assert resolved_commit_alpha(step=225, **kw) == pytest.approx(0.09375)
    assert resolved_commit_alpha(step=250, **kw) == pytest.approx(0.125)
    assert resolved_commit_alpha(step=1000, **kw) == pytest.approx(0.125)


def test_trust_region_is_independent_per_committed_chunk():
    # Row 0 would move 10%; row 1 only 0.5%. The 1.5% cap must clip only
    # row 0 and leave the already-safe row 1 at the requested alpha.
    raw = torch.ones(2, 3, 2, 2, 2)
    corrected = raw.clone()
    corrected[0] += 0.8  # alpha=.125 -> 10%
    corrected[1] += 0.04  # alpha=.125 -> .5%
    out, effective = blend_carn_commit(
        raw, corrected, alpha=0.125, max_relative_shift=0.015,
    )
    rel = ((out - raw).flatten(1).norm(dim=1)
           / raw.flatten(1).norm(dim=1))
    assert rel[0].item() == pytest.approx(0.015, abs=1e-6)
    assert rel[1].item() == pytest.approx(0.005, abs=1e-6)
    assert effective[0].item() < 0.125
    assert effective[1].item() == pytest.approx(0.125)


def test_straight_through_is_exact_value_and_identity_gradient():
    graph = torch.randn(2, 6, requires_grad=True)
    committed = torch.randn_like(graph)
    aligned = straight_through_value(graph, committed)
    torch.testing.assert_close(aligned.detach(), committed, rtol=0, atol=0)
    grad, = torch.autograd.grad(aligned.sum(), graph)
    torch.testing.assert_close(grad, torch.ones_like(graph), rtol=0, atol=0)


def test_model_alignment_reuses_literal_pipeline_commit_without_recompute():
    stub = SimpleNamespace(
        gan_carn_commit_align_enabled=True,
        gan_carn_commit_straight_through=True,
        reverse_noiser_commit_alpha=0.125,
        reverse_noiser_commit_start_step=150,
        reverse_noiser_commit_ramp_steps=100,
        reverse_noiser_dedrift_apply_to_commit=True,
        reverse_noiser_commit_max_relative_shift=0.015,
        num_frame_per_block=3,
    )
    graph = torch.randn(2, 21, 4, 2, 2, requires_grad=True)
    committed = graph.detach() + 0.01 * torch.randn_like(graph)
    aligned, logs = ActionForcingDMD._gan_carn_commit_aligned_slab(
        stub,
        graph,
        {"committed_ladder_endpoint_chunk": committed},
        current_step=200,
    )
    torch.testing.assert_close(aligned.detach(), committed, rtol=0, atol=0)
    grad, = torch.autograd.grad(aligned.square().sum(), graph)
    torch.testing.assert_close(grad, 2 * committed, rtol=0, atol=0)
    assert logs["train/gan_carn_commit_alpha_target"] == pytest.approx(0.0625)
    assert logs["train/gan_carn_commit_forward_value_max_error"] == 0.0


def test_frozen_jacobian_alignment_uses_commit_transform_backward():
    class _Pipe:
        context_noise = 0

        @staticmethod
        def _reverse_noiser_dedrift_commit(x, *, frame_start):
            del frame_start
            return 2.0 * x

        @staticmethod
        def _carn_seam_correct(x, *, record):
            assert record is False
            return x

    stub = SimpleNamespace(
        gan_carn_commit_align_enabled=True,
        gan_carn_commit_straight_through=False,
        reverse_noiser_commit_alpha=0.125,
        reverse_noiser_commit_start_step=150,
        reverse_noiser_commit_ramp_steps=100,
        reverse_noiser_dedrift_apply_to_commit=True,
        reverse_noiser_commit_max_relative_shift=0.015,
        num_frame_per_block=3,
        inference_pipeline=_Pipe(),
    )
    graph = torch.randn(1, 6, 2, 2, 2, requires_grad=True)
    committed = 2.0 * graph.detach()
    aligned, logs = ActionForcingDMD._gan_carn_commit_aligned_slab(
        stub,
        graph,
        {
            "committed_ladder_endpoint_chunk": committed,
            "abs_frame_start": 9,
        },
        current_step=200,
    )
    torch.testing.assert_close(aligned.detach(), committed, rtol=0, atol=0)
    grad, = torch.autograd.grad(aligned.sum(), graph)
    torch.testing.assert_close(grad, 2.0 * torch.ones_like(graph))
    assert logs["train/gan_carn_commit_straight_through"] == 0.0
    assert logs["train/gan_carn_commit_frozen_jacobian"] == 1.0
    assert logs["train/gan_carn_commit_replay_max_error"] == 0.0


def test_alignment_refuses_approximation_when_actual_commit_is_missing():
    stub = SimpleNamespace(
        gan_carn_commit_align_enabled=True,
        gan_carn_commit_straight_through=True,
        reverse_noiser_commit_alpha=0.125,
        reverse_noiser_commit_start_step=150,
        reverse_noiser_commit_ramp_steps=100,
        reverse_noiser_dedrift_apply_to_commit=True,
        reverse_noiser_commit_max_relative_shift=0.015,
        num_frame_per_block=3,
    )
    graph = torch.randn(1, 21, 2, 2, 2, requires_grad=True)
    with pytest.raises(RuntimeError, match="refusing to recompute"):
        ActionForcingDMD._gan_carn_commit_aligned_slab(
            stub, graph, {}, current_step=200,
        )


def test_absolute_level_matches_relative_chunk_semantics():
    # Three GT seeds: final seed is level 0, generated chunks are 1,2,...
    assert absolute_carn_level(
        frame_start=6, frames_per_block=3, num_seed_chunks=3, max_level=8,
    ) == 0
    assert absolute_carn_level(
        frame_start=9, frames_per_block=3, num_seed_chunks=3, max_level=8,
    ) == 1
    assert absolute_carn_level(
        frame_start=30, frames_per_block=3, num_seed_chunks=3, max_level=8,
    ) == 8


def test_rolling_launcher_pins_the_one_shot_regime():
    text = open(
        os.path.join(REPO, "sbatch", "run_phase3_rolling_definitive_4node.sbatch"),
        encoding="utf-8",
    ).read()
    for token in (
        "pix_finish_grad_enabled=true",
        "ladd_fake_sample_source=commit",
        "gan_carn_commit_align_enabled=true",
        "gan_carn_commit_straight_through=true",
        "reverse_noiser_commit_alpha=0.125",
        "reverse_noiser_commit_start_step=150",
        "reverse_noiser_commit_ramp_steps=100",
        "reverse_noiser_commit_max_relative_shift=0.015",
        "gen_aux_losses_x0_source=commit",
        "ladd_gt_transition_enabled=false",
        "ladd_gt_transition_carn_former=false",
        "ladd_gt_transition_carn_latter_reverse=false",
    ):
        assert token in text
    assert 'ABLATION_EXTRA="rolling_window_curriculum_increment=6"' not in text


def test_commit_surface_is_prepared_before_action_critic_consumes_x0():
    text = open(
        os.path.join(REPO, "trainer", "causal_action_forcing_train.py"),
        encoding="utf-8",
    ).read()
    prep = text.index(
        "# Resolve the single recurrent-state surface before ANY generator"
    )
    aux = text.index("        if aux_active:", prep)
    action_commit = text.index(
        'elif _ac_x0_source == "commit":', aux,
    )
    pixel = text.index("# WP-PIXGAN T3-C", action_commit)
    assert prep < aux < action_commit < pixel
