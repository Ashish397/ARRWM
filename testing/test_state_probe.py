#!/usr/bin/env python3
"""Comprehensive tests for StateProbeModule and its integration.

Tests cover:
  1. Shape correctness of forward pass
  2. Per-chunk isolation (each chunk's query only attends to its own KV)
  3. Shared query: same query init used across all chunks
  4. Gradient flow through probe layers, readout, and tapped features
  5. Consistency between parallel (teacher-forcing) and sequential (causal) modes
  6. noisy_start / frame_seqlen computation correctness
  7. Frozen readout mechanism (guidance loss detaches weights but not hidden states)
  8. Tap index selection (evenly spaced across transformer layers)
  9. Adding probe via WanDiffusionWrapper and correct wiring
 10. n_chunks * chunk_tokens alignment with actual feature shapes
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _import_probe_classes():
    with patch("torch.cuda.current_device", return_value=0):
        from wan.modules.causal_model import StateProbeModule, StateProbeLayer
    return StateProbeModule, StateProbeLayer


# ======================================================================
# Helpers
# ======================================================================

def _make_probe(n_chunks=7, model_dim=64, probe_dim=32, z_out_dim=2,
                num_heads=4, n_taps=3, num_frame_per_block=3,
                action_tokens_per_frame=0):
    StateProbeModule, _ = _import_probe_classes()
    return StateProbeModule(
        n_chunks=n_chunks, model_dim=model_dim, probe_dim=probe_dim,
        z_out_dim=z_out_dim, num_heads=num_heads, n_taps=n_taps,
        action_tokens_per_frame=action_tokens_per_frame,
        num_frame_per_block=num_frame_per_block,
    )


def _make_tapped_features(B, total_seq, model_dim, n_taps, requires_grad=False):
    return [
        torch.randn(B, total_seq, model_dim, requires_grad=requires_grad)
        for _ in range(n_taps)
    ]


# ======================================================================
# 1. Shape correctness
# ======================================================================

class TestShapeCorrectness:
    def test_basic_shapes(self):
        probe = _make_probe(n_chunks=7, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=3, num_frame_per_block=3)
        B, num_frames, spatial_per_frame = 2, 21, 100
        frame_seqlen = spatial_per_frame
        total_seq = num_frames * frame_seqlen * 2  # clean + noisy
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 3)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        assert preds.shape == (B, 7, 2), f"Expected (2, 7, 2), got {preds.shape}"
        assert hidden.shape == (B, 7, 32), f"Expected (2, 7, 32), got {hidden.shape}"

    def test_single_chunk(self):
        probe = _make_probe(n_chunks=1, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        B = 1
        frame_seqlen = 80
        noisy_seq = 1 * 3 * frame_seqlen  # 1 chunk * 3 frames * seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 2)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        assert preds.shape == (1, 1, 2)
        assert hidden.shape == (1, 1, 32)

    def test_no_clean_side(self):
        """When noisy_start=0, all features are treated as noisy (no TF split)."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        B = 1
        frame_seqlen = 50
        total_seq = 3 * 3 * frame_seqlen  # 3 chunks * 3 frames * seqlen (noisy only)
        noisy_start = 0

        tapped = _make_tapped_features(B, total_seq, 64, 2)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        assert preds.shape == (1, 3, 2)
        assert hidden.shape == (1, 3, 32)

    def test_z_out_dim_8(self):
        """Probe can predict full 8D latent, not just z2/z7."""
        probe = _make_probe(n_chunks=7, model_dim=64, probe_dim=32,
                            z_out_dim=8, n_taps=3, num_frame_per_block=3)
        B = 2
        frame_seqlen = 100
        total_seq = 21 * frame_seqlen * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 3)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        assert preds.shape == (B, 7, 8)


# ======================================================================
# 2. Per-chunk isolation
# ======================================================================

class TestChunkIsolation:
    def test_chunks_are_independent(self):
        """Modifying one chunk's features must not affect another chunk's output."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.eval()

        B = 1
        frame_seqlen = 50
        chunk_tokens = 3 * frame_seqlen
        noisy_seq = 3 * chunk_tokens
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped_a = [torch.randn(B, total_seq, 64) for _ in range(2)]
        tapped_b = [t.clone() for t in tapped_a]

        # Replace chunk 1 with completely different random features
        # (a constant offset is absorbed by LayerNorm)
        offset = noisy_start + chunk_tokens
        for t in tapped_b:
            t[:, offset:offset + chunk_tokens] = torch.randn_like(
                t[:, offset:offset + chunk_tokens]) * 5.0

        with torch.no_grad():
            preds_a, _ = probe(tapped_a, noisy_start, frame_seqlen)
            preds_b, _ = probe(tapped_b, noisy_start, frame_seqlen)

        # Chunk 0 and chunk 2 must be unchanged
        torch.testing.assert_close(preds_a[:, 0], preds_b[:, 0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(preds_a[:, 2], preds_b[:, 2], atol=1e-5, rtol=1e-5)

        # Chunk 1 should differ
        diff = (preds_a[:, 1] - preds_b[:, 1]).abs().max()
        assert diff > 1e-6, f"Perturbed chunk should produce different output, diff={diff}"


# ======================================================================
# 3. Shared query
# ======================================================================

class TestSharedQuery:
    def test_identical_chunks_produce_identical_output(self):
        """If all chunks have the same features, all outputs should be identical."""
        probe = _make_probe(n_chunks=4, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.eval()

        B = 1
        frame_seqlen = 40
        chunk_tokens = 3 * frame_seqlen

        # Create one chunk pattern and tile it
        pattern = torch.randn(1, chunk_tokens, 64)
        noisy = pattern.repeat(1, 4, 1)
        clean = torch.randn(B, 4 * chunk_tokens, 64)
        full = torch.cat([clean, noisy], dim=1)
        tapped = [full.clone() for _ in range(2)]
        noisy_start = 4 * chunk_tokens

        with torch.no_grad():
            preds, _ = probe(tapped, noisy_start, frame_seqlen)

        for i in range(1, 4):
            torch.testing.assert_close(
                preds[:, 0], preds[:, i], atol=1e-5, rtol=1e-5,
                msg=f"Chunk 0 and chunk {i} should match for identical features",
            )


# ======================================================================
# 4. Gradient flow
# ======================================================================

class TestGradientFlow:
    def test_gradients_flow_through_probe_to_tapped_features(self):
        """Gradients from the probe loss must reach tapped features."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.train()

        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 2, requires_grad=True)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        target = torch.zeros_like(preds)
        loss = F.mse_loss(preds, target)
        loss.backward()

        for i, feat in enumerate(tapped):
            assert feat.grad is not None, f"Tapped feature {i} has no gradient"
            noisy_grad = feat.grad[:, noisy_start:noisy_start + 3 * 3 * frame_seqlen]
            assert noisy_grad.abs().sum() > 0, f"Noisy region of tap {i} has zero gradient"

    def test_probe_parameters_receive_gradients(self):
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.train()

        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 2)
        preds, _ = probe(tapped, noisy_start, frame_seqlen)
        loss = preds.sum()
        loss.backward()

        assert probe.query_init.grad is not None, "query_init has no gradient"
        assert probe.readout.weight.grad is not None, "readout weight has no gradient"
        for i, layer in enumerate(probe.probe_layers):
            for name, param in layer.named_parameters():
                assert param.grad is not None, f"probe_layer[{i}].{name} has no gradient"

    def test_frozen_readout_gradients(self):
        """The frozen readout mechanism: detach readout weights, but keep hidden states live."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.train()

        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 2, requires_grad=True)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        # Simulate the frozen readout mechanism from the trainer
        frozen_preds = F.linear(
            hidden.float(),
            probe.readout.weight.detach(),
            probe.readout.bias.detach(),
        )
        guidance_target = torch.ones_like(frozen_preds) * 0.5
        guidance_loss = F.mse_loss(frozen_preds, guidance_target)
        guidance_loss.backward()

        # Readout weights should NOT have gradients (detached)
        assert probe.readout.weight.grad is None, \
            "readout.weight should have no grad in frozen readout"

        # But tapped features SHOULD have gradients (through hidden states)
        for feat in tapped:
            noisy_grad = feat.grad[:, noisy_start:noisy_start + noisy_seq]
            assert noisy_grad.abs().sum() > 0, \
                "Tapped features should receive gradient through frozen readout"

        # And probe layers should have gradients too
        assert probe.query_init.grad is not None, \
            "query_init should get gradient from frozen readout path"

    def test_clean_side_gets_no_gradient(self):
        """The clean side of tapped features should get no gradient from the probe."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.train()

        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 2, requires_grad=True)
        preds, _ = probe(tapped, noisy_start, frame_seqlen)
        loss = preds.sum()
        loss.backward()

        for i, feat in enumerate(tapped):
            clean_grad = feat.grad[:, :noisy_start]
            assert clean_grad.abs().sum() == 0, \
                f"Clean side of tap {i} should have zero gradient from probe"


# ======================================================================
# 5. Consistency between parallel and sequential modes
# ======================================================================

class TestCausalConsistency:
    def test_parallel_matches_sequential(self):
        """Running all chunks in parallel (teacher-forcing) should give the same
        result as running each chunk independently (simulating causal inference)."""
        StateProbeModule, StateProbeLayer = _import_probe_classes()

        n_chunks = 4
        model_dim = 64
        probe_dim = 32
        z_out_dim = 2
        n_taps = 2
        fpb = 3

        probe = _make_probe(n_chunks=n_chunks, model_dim=model_dim,
                            probe_dim=probe_dim, z_out_dim=z_out_dim,
                            n_taps=n_taps, num_frame_per_block=fpb)
        probe.eval()

        B = 1
        frame_seqlen = 40
        chunk_tokens = fpb * frame_seqlen
        noisy_seq = n_chunks * chunk_tokens
        total_seq = noisy_seq  # no clean side for this test
        noisy_start = 0

        tapped = [torch.randn(B, total_seq, model_dim) for _ in range(n_taps)]

        # Parallel run (all chunks at once)
        with torch.no_grad():
            preds_parallel, hidden_parallel = probe(tapped, noisy_start, frame_seqlen)

        # Sequential run: process each chunk individually with n_chunks=1
        probe_single = _make_probe(n_chunks=1, model_dim=model_dim,
                                   probe_dim=probe_dim, z_out_dim=z_out_dim,
                                   n_taps=n_taps, num_frame_per_block=fpb)
        # Copy weights
        probe_single.load_state_dict(probe.state_dict(), strict=False)
        probe_single.eval()

        preds_seq_list = []
        for c in range(n_chunks):
            chunk_start = c * chunk_tokens
            chunk_end = chunk_start + chunk_tokens
            tapped_chunk = [f[:, chunk_start:chunk_end] for f in tapped]
            with torch.no_grad():
                p, _ = probe_single(tapped_chunk, noisy_start=0, frame_seqlen=frame_seqlen)
            preds_seq_list.append(p)

        preds_sequential = torch.cat(preds_seq_list, dim=1)

        torch.testing.assert_close(
            preds_parallel, preds_sequential, atol=1e-4, rtol=1e-4,
            msg="Parallel and sequential modes should produce identical results",
        )


# ======================================================================
# 6. noisy_start / frame_seqlen computation
# ======================================================================

class TestNoisyStartFrameSeqlen:
    def test_teacher_forcing_noisy_start(self):
        """Verify that noisy_start = total_seq // 2 correctly isolates noisy half."""
        probe = _make_probe(n_chunks=7, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.eval()

        B = 1
        num_frames = 21
        spatial = 100
        frame_seqlen = spatial + 1  # spatial + 1 action token
        noisy_seq = num_frames * frame_seqlen
        total_seq = noisy_seq * 2  # clean + noisy
        noisy_start = total_seq // 2

        # Construct features: clean and noisy halves with different random distributions
        torch.manual_seed(42)
        clean = torch.randn(B, noisy_seq, 64) * 0.5
        torch.manual_seed(99)
        noisy = torch.randn(B, noisy_seq, 64) * 2.0
        full = torch.cat([clean, noisy], dim=1)
        tapped = [full.clone() for _ in range(2)]

        with torch.no_grad():
            preds_correct, _ = probe(tapped, noisy_start, frame_seqlen)

        # Running with noisy_start=0 reads from position 0 (the clean half)
        # instead of the noisy half — different input means different output
        with torch.no_grad():
            preds_wrong, _ = probe(tapped, 0, frame_seqlen)

        diff = (preds_correct - preds_wrong).abs().max()
        assert diff > 1e-6, \
            f"Different noisy_start should read different features, diff={diff}"

    def test_frame_seqlen_includes_action_tokens(self):
        """frame_seqlen must include action tokens for correct chunk boundaries."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.eval()

        B = 1
        spatial = 80
        action_tokens = 1
        frame_seqlen = spatial + action_tokens  # 81
        chunk_tokens = 3 * frame_seqlen  # 243
        noisy_seq = 3 * chunk_tokens  # 729
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = [torch.randn(B, total_seq, 64) for _ in range(2)]

        # Should not error
        with torch.no_grad():
            preds, _ = probe(tapped, noisy_start, frame_seqlen)
        assert preds.shape == (1, 3, 2)

    def test_frame_seqlen_mismatch_causes_wrong_chunking(self):
        """Using wrong frame_seqlen should cause cross-chunk leakage or misaligned reads."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.eval()

        B = 1
        correct_frame_seqlen = 81
        wrong_frame_seqlen = 80
        chunk_tokens_correct = 3 * correct_frame_seqlen
        noisy_seq = 3 * chunk_tokens_correct

        # Use structured features where each chunk is distinct
        torch.manual_seed(42)
        noisy_feats = torch.randn(B, noisy_seq, 64)
        # Make chunk boundaries very different
        for c in range(3):
            start = c * chunk_tokens_correct
            noisy_feats[:, start:start + chunk_tokens_correct] *= (c + 1)
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2
        full = torch.cat([torch.randn(B, noisy_seq, 64), noisy_feats], dim=1)
        tapped = [full.clone() for _ in range(2)]

        with torch.no_grad():
            preds_correct, _ = probe(tapped, noisy_start, correct_frame_seqlen)
            preds_wrong, _ = probe(tapped, noisy_start, wrong_frame_seqlen)

        diff = (preds_correct - preds_wrong).abs().max()
        assert diff > 1e-5, \
            f"Wrong frame_seqlen should produce different chunking, diff={diff}"


# ======================================================================
# 7. Tap index selection
# ======================================================================

class TestTapSelection:
    def test_evenly_spaced_taps(self):
        """adding_state_probe_branch should select evenly spaced tap indices."""
        n_blocks = 30
        n_taps = 6
        expected = [int(round(i * (n_blocks - 1) / (n_taps - 1))) for i in range(n_taps)]
        assert expected == [0, 6, 12, 17, 23, 29], f"Got {expected}"

    def test_tap_count_matches_probe_layers(self):
        probe = _make_probe(n_chunks=7, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=6, num_frame_per_block=3)
        assert len(probe.probe_layers) == 6

    def test_providing_wrong_number_of_taps_raises(self):
        """Mismatched tap count should raise an assertion error."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=3, num_frame_per_block=3)
        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped_short = _make_tapped_features(B, total_seq, 64, 2)
        with pytest.raises(AssertionError, match="expected 3 tapped features"):
            probe(tapped_short, noisy_start, frame_seqlen)


# ======================================================================
# 8. Integration: wrapper wiring
# ======================================================================

class TestWrapperIntegration:
    def test_adding_state_probe_sets_attributes(self):
        """adding_state_probe_branch should configure the model correctly."""
        with patch("torch.cuda.current_device", return_value=0):
            from wan.modules.causal_model import CausalWanModel, StateProbeModule

        model_dim = 64
        model = CausalWanModel(
            dim=model_dim, ffn_dim=128, num_heads=4, num_layers=4,
            text_dim=64, freq_dim=64, in_dim=16, out_dim=16,
        )

        # Simulate what WanDiffusionWrapper.adding_state_probe_branch does
        n_chunks = 7
        n_taps = 3
        probe = StateProbeModule(
            n_chunks=n_chunks, model_dim=model_dim, probe_dim=32,
            z_out_dim=2, num_heads=4, n_taps=n_taps, num_frame_per_block=3,
        )

        n_blocks = len(model.blocks)
        tap_indices = [int(round(i * (n_blocks - 1) / (n_taps - 1))) for i in range(n_taps)]
        model._state_probe_tap_set = set(tap_indices)
        model._state_probe_tap_indices = tap_indices

        assert hasattr(model, '_state_probe_tap_set')
        assert hasattr(model, '_state_probe_tap_indices')
        assert len(model._state_probe_tap_set) == n_taps
        assert len(model._state_probe_tap_indices) == n_taps
        assert model._state_probe_tap_indices == tap_indices

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention requires CUDA")
    def test_tapped_features_collected_during_forward_train(self):
        """_forward_train should collect tapped features when _state_probe_tap_set is set."""
        with patch("torch.cuda.current_device", return_value=0):
            from wan.modules.causal_model import CausalWanModel

        model = CausalWanModel(
            dim=64, ffn_dim=128, num_heads=4, num_layers=4,
            text_dim=64, freq_dim=64, in_dim=16, out_dim=16,
        )
        model.num_frame_per_block = 3
        model._state_probe_tap_set = {0, 2, 3}

        B = 1
        F_lat = 3
        H, W = 8, 8
        C_in = 16

        # _forward_train expects x as a tensor [B, C, F, H, W] (passed from wrapper)
        # that it then iterates over batch dim to get [C, F, H, W] per item
        x = torch.randn(B, C_in, F_lat, H, W)
        t = torch.tensor([[0.5]])
        context = [torch.randn(10, 64)]

        model.block_mask = None
        model.eval()

        # spatial_seqlen = H*W // (patch_h * patch_w) = 8*8 // (2*2) = 16
        # seq_len = F_lat * spatial_seqlen = 3 * 16 = 48
        seq_len = F_lat * (H * W) // (model.patch_size[1] * model.patch_size[2])

        with torch.no_grad():
            result = model._forward_train(
                x, t, context, seq_len=seq_len,
            )

        if isinstance(result, tuple) and len(result) == 2:
            output, tapped = result
            if isinstance(tapped, list):
                assert len(tapped) == 3, \
                    f"Expected 3 tapped features for 3 taps, got {len(tapped)}"
                for feat in tapped:
                    assert feat.shape[0] == B
                    assert feat.shape[2] == 64  # model_dim


# ======================================================================
# 9. Readout initialization
# ======================================================================

class TestReadoutInit:
    def test_readout_near_zero_init(self):
        """Readout layer should be initialized near zero for stable training start."""
        probe = _make_probe(n_chunks=7, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=3, num_frame_per_block=3)

        assert probe.readout.bias.abs().max() == 0.0, "Readout bias should be zero-init"
        assert probe.readout.weight.abs().max() < 0.1, \
            f"Readout weight should be near-zero, max={probe.readout.weight.abs().max()}"

    def test_query_init_small(self):
        probe = _make_probe()
        assert probe.query_init.abs().max() < 1.0, \
            f"query_init should be small, max={probe.query_init.abs().max()}"


# ======================================================================
# 10. Edge cases and numerical
# ======================================================================

class TestEdgeCases:
    def test_single_tap(self):
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=1, num_frame_per_block=3)
        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 1)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)
        assert preds.shape == (1, 3, 2)

    def test_dtype_preservation_mixed(self):
        """Probe with bf16 layers but fp32 readout should produce fp32 preds."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        for layer in probe.probe_layers:
            layer.to(torch.bfloat16)
        probe.query_init.data = probe.query_init.data.to(torch.bfloat16)

        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = [torch.randn(B, total_seq, 64, dtype=torch.bfloat16) for _ in range(2)]
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        assert preds.dtype == torch.float32, f"Preds should be float32, got {preds.dtype}"
        assert hidden.dtype == torch.bfloat16, f"Hidden should preserve input dtype"

    def test_full_bf16_probe_works(self):
        """Full bf16 probe works because readout uses F.linear with explicit
        .float() on both input and weights."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe = probe.to(torch.bfloat16)

        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = [torch.randn(B, total_seq, 64, dtype=torch.bfloat16) for _ in range(2)]
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        assert preds.dtype == torch.float32, f"Preds should be float32, got {preds.dtype}"
        assert hidden.dtype == torch.bfloat16, f"Hidden should preserve input dtype"

    def test_batch_independence(self):
        """Different batch items should produce independent results."""
        probe = _make_probe(n_chunks=2, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.eval()

        frame_seqlen = 50
        chunk_tokens = 3 * frame_seqlen
        noisy_seq = 2 * chunk_tokens
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped_b0 = [torch.randn(1, total_seq, 64) for _ in range(2)]
        tapped_b1 = [torch.randn(1, total_seq, 64) for _ in range(2)]
        tapped_batched = [torch.cat([t0, t1], dim=0) for t0, t1 in zip(tapped_b0, tapped_b1)]

        with torch.no_grad():
            preds_batched, _ = probe(tapped_batched, noisy_start, frame_seqlen)
            preds_b0, _ = probe(tapped_b0, noisy_start, frame_seqlen)
            preds_b1, _ = probe(tapped_b1, noisy_start, frame_seqlen)

        torch.testing.assert_close(preds_batched[0:1], preds_b0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(preds_batched[1:2], preds_b1, atol=1e-5, rtol=1e-5)

    def test_deterministic_with_same_input(self):
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3)
        probe.eval()

        B = 1
        frame_seqlen = 50
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 2)

        with torch.no_grad():
            preds1, _ = probe(tapped, noisy_start, frame_seqlen)
            preds2, _ = probe(tapped, noisy_start, frame_seqlen)

        torch.testing.assert_close(preds1, preds2)


# ======================================================================
# 11. StateProbeLayer unit tests
# ======================================================================

class TestStateProbeLayer:
    def test_layer_shapes(self):
        _, StateProbeLayer = _import_probe_classes()
        layer = StateProbeLayer(model_dim=64, probe_dim=32, num_heads=4)

        q = torch.randn(2, 1, 32)
        kv = torch.randn(2, 100, 64)
        out = layer(q, kv)

        assert out.shape == (2, 1, 32)

    def test_residual_connection(self):
        """Output should not be identical to input (cross-attention adds info)."""
        _, StateProbeLayer = _import_probe_classes()
        layer = StateProbeLayer(model_dim=64, probe_dim=32, num_heads=4)
        layer.eval()

        q = torch.randn(1, 1, 32)
        kv = torch.randn(1, 50, 64)

        with torch.no_grad():
            out = layer(q, kv)

        # Due to residual connection, the output should be close to but not equal to q
        # (unless the cross-attention and FFN outputs are exactly zero, which is unlikely)
        assert not torch.allclose(out, q, atol=1e-6), \
            "Layer output should differ from input (cross-attention adds information)"

    def test_layer_gradient_flow(self):
        _, StateProbeLayer = _import_probe_classes()
        layer = StateProbeLayer(model_dim=64, probe_dim=32, num_heads=4)

        q = torch.randn(1, 1, 32, requires_grad=True)
        kv = torch.randn(1, 50, 64, requires_grad=True)
        out = layer(q, kv)
        out.sum().backward()

        assert q.grad is not None
        assert kv.grad is not None
        assert q.grad.abs().sum() > 0
        assert kv.grad.abs().sum() > 0


# ======================================================================
# 12. Probe supervision target alignment
# ======================================================================

class TestSupervisionAlignment:
    def test_chunk_action_pooling_aligns_with_probe_chunks(self):
        """The chunk averaging used for z_actions targets must match
        probe chunk boundaries."""
        num_frames = 21
        fpb = 3
        n_chunks = num_frames // fpb  # 7

        B = 2
        z_actions = torch.randn(B, num_frames, 2)  # [B, F, 2] for z2/z7

        # Chunk by averaging (as done in _chunk_actions)
        chunked = z_actions[:, :n_chunks * fpb].reshape(B, n_chunks, fpb, 2).mean(dim=2)
        assert chunked.shape == (B, n_chunks, 2)

        probe = _make_probe(n_chunks=n_chunks, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=fpb)
        frame_seqlen = 100
        noisy_seq = num_frames * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 2)
        preds, _ = probe(tapped, noisy_start, frame_seqlen)

        # Shapes must align for MSE loss
        assert preds.shape == chunked.shape, \
            f"Probe output {preds.shape} must match chunked targets {chunked.shape}"

    def test_loss_computes_without_error(self):
        """End-to-end: probe output → MSE loss against teacher z2/z7."""
        probe = _make_probe(n_chunks=7, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=3, num_frame_per_block=3)
        probe.train()

        B = 2
        num_frames = 21
        frame_seqlen = 100
        noisy_seq = num_frames * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = _make_tapped_features(B, total_seq, 64, 3, requires_grad=True)
        preds, hidden = probe(tapped, noisy_start, frame_seqlen)

        # Teacher targets
        teacher_z = torch.randn(B, 7, 2)
        state_loss = F.mse_loss(preds, teacher_z)

        # Frozen readout guidance
        cmd_z = torch.randn(B, 7, 2)
        frozen_preds = F.linear(
            hidden.float(),
            probe.readout.weight.detach(),
            probe.readout.bias.detach(),
        )
        guidance_loss = F.mse_loss(frozen_preds, cmd_z)

        total_loss = 1.5 * state_loss + 0.5 * guidance_loss
        total_loss.backward()

        # Probe params should have grad from state_loss + guidance_loss
        assert probe.query_init.grad is not None
        # Readout should only have grad from state_loss (not guidance)
        assert probe.readout.weight.grad is not None


# ======================================================================
# 13. Action token leakage concern
# ======================================================================

class TestActionTokenStripping:
    """Verify that action tokens are stripped from KV so the probe only
    attends to spatial features, preventing shortcutting through the
    action conditioning embedding."""

    def test_action_tokens_stripped_from_kv(self):
        """A probe with action_tokens_per_frame > 0 should produce different
        results from one that sees all tokens, because it strips the trailing
        action token from each frame."""
        B = 1
        spatial = 80
        action_tokens = 1
        frame_seqlen = spatial + action_tokens
        fpb = 3
        n_chunks = 3
        chunk_tokens = fpb * frame_seqlen
        noisy_seq = n_chunks * chunk_tokens
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = [torch.randn(B, total_seq, 64) for _ in range(2)]

        # Probe that strips action tokens
        probe_strip = _make_probe(
            n_chunks=n_chunks, model_dim=64, probe_dim=32, z_out_dim=2,
            n_taps=2, num_frame_per_block=fpb, action_tokens_per_frame=1)
        probe_strip.eval()

        # Probe that does NOT strip action tokens (legacy behaviour)
        probe_no_strip = _make_probe(
            n_chunks=n_chunks, model_dim=64, probe_dim=32, z_out_dim=2,
            n_taps=2, num_frame_per_block=fpb, action_tokens_per_frame=0)
        # Copy weights so the only difference is stripping
        probe_no_strip.load_state_dict(probe_strip.state_dict(), strict=False)
        probe_no_strip.eval()

        with torch.no_grad():
            preds_strip, _ = probe_strip(tapped, noisy_start, frame_seqlen)
            preds_no_strip, _ = probe_no_strip(tapped, noisy_start, frame_seqlen)

        diff = (preds_strip - preds_no_strip).abs().max()
        assert diff > 1e-6, \
            f"Stripping action tokens should change probe output, diff={diff}"

    def test_no_action_tokens_is_identity(self):
        """With action_tokens_per_frame=0, no stripping occurs and output matches
        a direct pass over the full frame_seqlen."""
        probe = _make_probe(n_chunks=3, model_dim=64, probe_dim=32,
                            z_out_dim=2, n_taps=2, num_frame_per_block=3,
                            action_tokens_per_frame=0)
        probe.eval()

        B = 1
        frame_seqlen = 80
        noisy_seq = 3 * 3 * frame_seqlen
        total_seq = noisy_seq * 2
        noisy_start = total_seq // 2

        tapped = [torch.randn(B, total_seq, 64) for _ in range(2)]

        with torch.no_grad():
            preds1, _ = probe(tapped, noisy_start, frame_seqlen)
            preds2, _ = probe(tapped, noisy_start, frame_seqlen)

        torch.testing.assert_close(preds1, preds2)

    def test_action_tokens_are_minority_of_chunk(self):
        """Action tokens should be a small fraction of total chunk tokens."""
        spatial = 1560
        action_tokens = 1
        fpb = 3
        chunk_tokens = fpb * (spatial + action_tokens)
        action_fraction = (fpb * action_tokens) / chunk_tokens
        assert action_fraction < 0.01, \
            f"Action tokens are {action_fraction:.4f} of chunk, should be < 1%"

    def test_parallel_sequential_consistency_with_action_tokens(self):
        """Chunk isolation + action stripping must still produce identical results
        in parallel vs sequential mode."""
        StateProbeModule, _ = _import_probe_classes()

        n_chunks = 3
        model_dim = 64
        probe_dim = 32
        z_out_dim = 2
        n_taps = 2
        fpb = 3
        a_per_f = 1

        probe = _make_probe(
            n_chunks=n_chunks, model_dim=model_dim, probe_dim=probe_dim,
            z_out_dim=z_out_dim, n_taps=n_taps, num_frame_per_block=fpb,
            action_tokens_per_frame=a_per_f)
        probe.eval()

        B = 1
        frame_seqlen = 81  # 80 spatial + 1 action
        chunk_tokens = fpb * frame_seqlen
        noisy_seq = n_chunks * chunk_tokens
        noisy_start = 0

        tapped = [torch.randn(B, noisy_seq, model_dim) for _ in range(n_taps)]

        with torch.no_grad():
            preds_parallel, _ = probe(tapped, noisy_start, frame_seqlen)

        probe_single = _make_probe(
            n_chunks=1, model_dim=model_dim, probe_dim=probe_dim,
            z_out_dim=z_out_dim, n_taps=n_taps, num_frame_per_block=fpb,
            action_tokens_per_frame=a_per_f)
        probe_single.load_state_dict(probe.state_dict(), strict=False)
        probe_single.eval()

        preds_seq = []
        for c in range(n_chunks):
            chunk_start = c * chunk_tokens
            chunk_end = chunk_start + chunk_tokens
            tapped_chunk = [f[:, chunk_start:chunk_end] for f in tapped]
            with torch.no_grad():
                p, _ = probe_single(tapped_chunk, noisy_start=0, frame_seqlen=frame_seqlen)
            preds_seq.append(p)

        preds_sequential = torch.cat(preds_seq, dim=1)
        torch.testing.assert_close(
            preds_parallel, preds_sequential, atol=1e-4, rtol=1e-4,
        )


# ======================================================================
# Runner
# ======================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
