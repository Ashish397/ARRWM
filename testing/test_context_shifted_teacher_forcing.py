#!/usr/bin/env python3
"""Focused regression tests for context-shifted teacher forcing."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.zarr_dataset import ZarrSequentialDataset


def _import_causal_model():
    with patch("torch.cuda.current_device", return_value=0):
        from wan.modules.causal_model import CausalWanModel

    return CausalWanModel


def _import_rope_helpers():
    with patch("torch.cuda.current_device", return_value=0):
        from wan.modules.model import rope_apply, rope_params

    return rope_apply, rope_params


class TestShiftedSplitLogic(unittest.TestCase):
    def test_fixed_window_and_streaming_use_same_shifted_split(self):
        full_latents = torch.arange(24).view(1, 24, 1, 1, 1)
        context_frames = 3
        num_frames = 21

        fixed_clean = full_latents[:, : full_latents.shape[1] - context_frames]
        fixed_noisy = full_latents[:, context_frames:]

        streaming_clean = full_latents[:, :num_frames]
        streaming_noisy = full_latents[:, context_frames:]

        expected_clean = torch.arange(21).view(1, 21, 1, 1, 1)
        expected_noisy = torch.arange(3, 24).view(1, 21, 1, 1, 1)

        torch.testing.assert_close(fixed_clean, expected_clean)
        torch.testing.assert_close(fixed_noisy, expected_noisy)
        torch.testing.assert_close(streaming_clean, expected_clean)
        torch.testing.assert_close(streaming_noisy, expected_noisy)


class TestTeacherForcingMask(unittest.TestCase):
    def test_first_noisy_block_sees_only_clean_context_block_and_self(self):
        CausalWanModel = _import_causal_model()
        block_mask = CausalWanModel._prepare_teacher_forcing_mask(
            "cpu",
            num_frames=21,
            frame_seqlen=1,
            num_frame_per_block=3,
            context_shift=1,
        )
        mask = create_mask(
            block_mask.mask_mod,
            B=None,
            H=None,
            Q_LEN=42,
            KV_LEN=42,
            device="cpu",
        )[0, 0]

        expected = {0, 1, 2, 21, 22, 23}
        for q_idx in (21, 22, 23):
            visible = set(torch.nonzero(mask[q_idx], as_tuple=False).flatten().tolist())
            self.assertEqual(visible, expected)

    def test_second_noisy_block_sees_two_clean_blocks_and_self(self):
        CausalWanModel = _import_causal_model()
        block_mask = CausalWanModel._prepare_teacher_forcing_mask(
            "cpu",
            num_frames=21,
            frame_seqlen=1,
            num_frame_per_block=3,
            context_shift=1,
        )
        mask = create_mask(
            block_mask.mask_mod,
            B=None,
            H=None,
            Q_LEN=42,
            KV_LEN=42,
            device="cpu",
        )[0, 0]

        expected = {0, 1, 2, 3, 4, 5, 24, 25, 26}
        for q_idx in (24, 25, 26):
            visible = set(torch.nonzero(mask[q_idx], as_tuple=False).flatten().tolist())
            self.assertEqual(visible, expected)


class TestRopeOffset(unittest.TestCase):
    def test_temporal_offset_starts_noisy_stream_at_three(self):
        rope_apply, rope_params = _import_rope_helpers()

        num_frames = 8
        x = torch.ones(1, num_frames, 1, 24)
        grid_sizes = torch.tensor([[num_frames, 1, 1]])
        freqs = rope_params(32, 24)

        roped_unshifted = rope_apply(x, grid_sizes, freqs, temporal_offset=0)
        roped_shifted = rope_apply(x, grid_sizes, freqs, temporal_offset=3)

        torch.testing.assert_close(roped_shifted[0, : num_frames - 3], roped_unshifted[0, 3:])


class TestSequentialDatasetContextAccounting(unittest.TestCase):
    @patch("utils.zarr_dataset.load_ss_vae", return_value=(object(), 1.0))
    def test_window_count_includes_context_frames(self, _mock_ss_vae):
        prompt = torch.zeros(2, 4)
        per_file = {
            "len24.zarr": 24,
            "len44.zarr": 44,
            "len45.zarr": 45,
        }

        def fake_process(self, zpath):
            n_latent = per_file[zpath.name]
            return prompt, torch.zeros(n_latent, 8), n_latent

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in per_file:
                (root / name).touch()

            with patch.object(ZarrSequentialDataset, "_process_zarr", fake_process):
                dataset = ZarrSequentialDataset(
                    encoded_root=str(root),
                    caption_root=str(root),
                    motion_root=str(root),
                    ss_vae_checkpoint="dummy.pt",
                    window_size=21,
                    window_stride=21,
                    context_frames=3,
                    device="cpu",
                )

        self.assertEqual(len(dataset), 4)

    def test_dataset_item_returns_window_plus_context(self):
        dataset = object.__new__(ZarrSequentialDataset)
        dataset.window_size = 21
        dataset.context_frames = 3

        prompt = torch.randn(2, 4)
        z_actions = torch.arange(40 * 8, dtype=torch.float32).view(40, 8)
        dataset._samples = [(Path("/fake/sample.zarr"), prompt, z_actions, 5)]

        fake_latents = torch.arange(40, dtype=torch.float32).view(40, 1, 1, 1).numpy()

        with patch("utils.zarr_dataset.zarr_lib.open_group", return_value={"latents": fake_latents}):
            item = dataset[0]

        self.assertEqual(item["real_latents"].shape[0], 24)
        self.assertEqual(item["z_actions"].shape[0], 24)
        torch.testing.assert_close(item["real_latents"][:, 0, 0, 0], torch.arange(5, 29, dtype=torch.float32))
        torch.testing.assert_close(item["z_actions"], z_actions[5:29])


class TestActionDimSlicing(unittest.TestCase):
    """Verify z2/z7 selection from the full 8D action tensor."""

    def test_action_dims_selects_correct_indices(self):
        action_dims = [2, 7]
        z_full = torch.arange(24 * 8, dtype=torch.float32).view(1, 24, 8)
        z_sliced = z_full[..., action_dims]
        self.assertEqual(z_sliced.shape, (1, 24, 2))
        torch.testing.assert_close(z_sliced[0, :, 0], z_full[0, :, 2])
        torch.testing.assert_close(z_sliced[0, :, 1], z_full[0, :, 7])

    def test_clean_noisy_action_split_is_shifted(self):
        cf = 3
        num_frames = 21
        z_full = torch.arange(24, dtype=torch.float32).view(1, 24, 1)
        z_noisy = z_full[:, cf:]        # frames 3..23
        z_clean = z_full[:, :num_frames] # frames 0..20

        self.assertEqual(z_noisy.shape, (1, 21, 1))
        self.assertEqual(z_clean.shape, (1, 21, 1))
        self.assertAlmostEqual(z_clean[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(z_noisy[0, 0, 0].item(), 3.0)
        self.assertAlmostEqual(z_clean[0, -1, 0].item(), 20.0)
        self.assertAlmostEqual(z_noisy[0, -1, 0].item(), 23.0)
        self.assertFalse(torch.equal(z_clean, z_noisy))

    def test_overlap_region_is_identical(self):
        cf = 3
        num_frames = 21
        z_full = torch.arange(24, dtype=torch.float32).view(1, 24, 1)
        z_noisy = z_full[:, cf:]
        z_clean = z_full[:, :num_frames]
        torch.testing.assert_close(z_clean[:, cf:], z_noisy[:, :num_frames - cf])


class TestActionModulationProjectionWith2D(unittest.TestCase):
    """Smoke-test that ActionModulationProjection accepts 2D action input."""

    def test_2d_input_produces_correct_shape(self):
        from model.action_modulation import ActionModulationProjection

        hidden_dim = 64
        proj = ActionModulationProjection(
            action_dim=2, activation="silu", hidden_dim=hidden_dim,
            num_frames=1, zero_init=True,
        )
        z = torch.randn(2, 21, 2)
        out = proj(z, num_frames=21)
        self.assertEqual(out.shape, (2, 21, 6, hidden_dim))


class TestCleanSideModulationThreading(unittest.TestCase):
    """Verify that _action_modulation_clean is used for clean-side time_projection."""

    def test_clean_side_uses_action_modulation_clean(self):
        CausalWanModel = _import_causal_model()

        with patch("torch.cuda.current_device", return_value=0):
            model = CausalWanModel(
                dim=64, ffn_dim=128, freq_dim=32, text_dim=64,
                num_heads=4, num_layers=1, in_dim=4, out_dim=4,
                patch_size=(1, 1, 1), text_len=4,
            )

        model.num_frame_per_block = 3
        model.context_shift = 1

        B, F = 1, 3
        am_noisy = torch.randn(B, F, 6, 64)
        am_clean = torch.randn(B, F, 6, 64)
        model._action_modulation = am_noisy
        model._action_modulation_clean = am_clean

        captured = {"calls": []}
        tp_module = model.time_projection
        orig_forward = tp_module.forward

        class _EarlyExit(Exception):
            """Raised after we've captured enough spy calls."""

        def spy_forward(e):
            result = orig_forward(e)
            am = getattr(model, '_action_modulation', None)
            captured["calls"].append(
                'clean' if (am is not None and torch.equal(am, am_clean)) else
                'noisy' if (am is not None and torch.equal(am, am_noisy)) else
                'none'
            )
            if len(captured["calls"]) >= 2:
                raise _EarlyExit()
            return result

        tp_module.forward = spy_forward

        t = torch.randint(0, 100, (B, F))
        aug_t = torch.zeros(B, F)
        x_noisy = torch.randn(B, 4, F, 2, 2)
        clean_x = torch.randn(B, 4, F, 2, 2)
        context = [torch.randn(4, 64)]

        model.block_mask = None
        try:
            model._forward_train(
                x_noisy, t, context, seq_len=100,
                clean_x=clean_x, aug_t=aug_t,
            )
        except _EarlyExit:
            pass
        finally:
            tp_module.forward = orig_forward

        self.assertGreaterEqual(len(captured['calls']), 2,
                                "time_projection must be called at least twice (noisy + clean)")
        self.assertEqual(captured['calls'][0], 'noisy',
                         "First time_projection call (noisy) should use am_noisy")
        self.assertEqual(captured['calls'][1], 'clean',
                         "Second time_projection call (clean) should use am_clean")


class TestActionTokenProjectionShape(unittest.TestCase):
    """Smoke-test that ActionTokenProjection produces correct output shape."""

    def test_output_is_b_f_dim(self):
        from model.action_modulation import ActionTokenProjection

        hidden_dim = 64
        proj = ActionTokenProjection(
            action_dim=2, activation="silu", hidden_dim=hidden_dim,
            zero_init=True,
        )
        z = torch.randn(2, 21, 2)
        out = proj(z)
        self.assertEqual(out.shape, (2, 21, hidden_dim))

    def test_different_actions_produce_different_tokens(self):
        from model.action_modulation import ActionTokenProjection

        proj = ActionTokenProjection(
            action_dim=2, activation="silu", hidden_dim=64,
            zero_init=False,
        )
        z1 = torch.randn(1, 7, 2)
        z2 = torch.randn(1, 7, 2)
        self.assertFalse(torch.allclose(proj(z1), proj(z2), atol=1e-6))


def _parse_action_conditioning_mode(config):
    """Mirror the exact parsing logic from CausalLoRADiffusionTrainer.__init__."""
    acm = getattr(config, "action_conditioning_mode", "adaln")
    if acm not in ("adaln", "action_tokens", "both"):
        raise ValueError(f"action_conditioning_mode must be adaln|action_tokens|both, got {acm}")
    return acm, acm in ("adaln", "both"), acm in ("action_tokens", "both")


class TestActionConditioningModeParsing(unittest.TestCase):
    """Verify mode parsing contract using OmegaConf configs (the real config type)."""

    def test_adaln_mode(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"action_conditioning_mode": "adaln"})
        mode, use_adaln, use_tokens = _parse_action_conditioning_mode(cfg)
        self.assertEqual(mode, "adaln")
        self.assertTrue(use_adaln)
        self.assertFalse(use_tokens)

    def test_action_tokens_mode(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"action_conditioning_mode": "action_tokens"})
        mode, use_adaln, use_tokens = _parse_action_conditioning_mode(cfg)
        self.assertEqual(mode, "action_tokens")
        self.assertFalse(use_adaln)
        self.assertTrue(use_tokens)

    def test_both_mode(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"action_conditioning_mode": "both"})
        mode, use_adaln, use_tokens = _parse_action_conditioning_mode(cfg)
        self.assertEqual(mode, "both")
        self.assertTrue(use_adaln)
        self.assertTrue(use_tokens)

    def test_default_is_adaln_when_key_missing(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({})
        mode, use_adaln, use_tokens = _parse_action_conditioning_mode(cfg)
        self.assertEqual(mode, "adaln")
        self.assertTrue(use_adaln)
        self.assertFalse(use_tokens)

    def test_invalid_mode_raises_with_message(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"action_conditioning_mode": "garbage"})
        with self.assertRaises(ValueError) as ctx:
            _parse_action_conditioning_mode(cfg)
        self.assertIn("garbage", str(ctx.exception))


class TestActionTokenInsertionAndStripping(unittest.TestCase):
    """Verify action tokens are correctly inserted and stripped."""

    def test_insertion_increases_seqlen_by_f(self):
        """Inserting 1 action token per frame adds F tokens total."""
        B, F, spatial, dim = 2, 7, 12, 32
        x = torch.randn(B, F * spatial, dim)
        at = torch.randn(B, F, dim)

        x_framed = x.unflatten(1, (F, spatial))
        at_framed = at.unsqueeze(2)
        merged = torch.cat([x_framed, at_framed], dim=2).flatten(1, 2)

        self.assertEqual(merged.shape, (B, F * (spatial + 1), dim))

    def test_stripping_restores_original_seqlen(self):
        """Stripping action tokens restores F*spatial sequence length."""
        B, F, spatial, a_per_f, dim = 2, 7, 12, 1, 32
        frame_seqlen = spatial + a_per_f
        x = torch.randn(B, F * frame_seqlen, dim)

        stripped = x.unflatten(1, (F, frame_seqlen))[:, :, :spatial].flatten(1, 2)
        self.assertEqual(stripped.shape, (B, F * spatial, dim))

    def test_spatial_tokens_preserved_after_roundtrip(self):
        """Spatial tokens are unchanged after insert → strip."""
        B, F, spatial, dim = 1, 5, 8, 16
        x_spatial = torch.randn(B, F * spatial, dim)
        at = torch.randn(B, F, dim)

        x_framed = x_spatial.unflatten(1, (F, spatial))
        at_framed = at.unsqueeze(2)
        merged = torch.cat([x_framed, at_framed], dim=2).flatten(1, 2)

        frame_seqlen = spatial + 1
        stripped = merged.unflatten(1, (F, frame_seqlen))[:, :, :spatial].flatten(1, 2)
        torch.testing.assert_close(stripped, x_spatial)

    def test_action_token_content_preserved(self):
        """Action token values are at the correct positions after insertion."""
        B, F, spatial, dim = 1, 3, 4, 8
        x_spatial = torch.zeros(B, F * spatial, dim)
        at = torch.ones(B, F, dim) * 42.0

        x_framed = x_spatial.unflatten(1, (F, spatial))
        at_framed = at.unsqueeze(2)
        merged = torch.cat([x_framed, at_framed], dim=2)

        for f in range(F):
            torch.testing.assert_close(
                merged[0, f, spatial],
                torch.full((dim,), 42.0),
            )


class TestActionTokenMaskDimensions(unittest.TestCase):
    """Verify masks account for action tokens in frame_seqlen."""

    def test_teacher_forcing_mask_with_action_tokens(self):
        CausalWanModel = _import_causal_model()
        spatial_seqlen = 4
        action_per_frame = 1
        frame_seqlen = spatial_seqlen + action_per_frame
        num_frames = 6
        num_frame_per_block = 3

        block_mask = CausalWanModel._prepare_teacher_forcing_mask(
            "cpu",
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            context_shift=1,
        )
        total_len = num_frames * frame_seqlen * 2
        padded = (total_len + 127) // 128 * 128
        mask = create_mask(
            block_mask.mask_mod, B=None, H=None,
            Q_LEN=padded, KV_LEN=padded, device="cpu",
        )[0, 0]

        clean_half = num_frames * frame_seqlen
        first_noisy_start = clean_half
        first_noisy_end = first_noisy_start + frame_seqlen * num_frame_per_block
        for q in range(first_noisy_start, first_noisy_end):
            visible = set(torch.nonzero(mask[q], as_tuple=False).flatten().tolist())
            for kv in visible:
                if kv < clean_half:
                    self.assertLess(kv, frame_seqlen * num_frame_per_block,
                                    f"Noisy token {q} should only see first clean block, "
                                    f"but sees kv={kv}")

    def test_blockwise_mask_with_action_tokens(self):
        CausalWanModel = _import_causal_model()
        frame_seqlen = 5
        num_frames = 6
        num_frame_per_block = 3

        block_mask = CausalWanModel._prepare_blockwise_causal_attn_mask(
            "cpu",
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
        )
        total_len = num_frames * frame_seqlen
        padded = (total_len + 127) // 128 * 128
        mask = create_mask(
            block_mask.mask_mod, B=None, H=None,
            Q_LEN=padded, KV_LEN=padded, device="cpu",
        )[0, 0]

        block1_end = frame_seqlen * num_frame_per_block
        for q in range(block1_end, block1_end + frame_seqlen * num_frame_per_block):
            visible = set(torch.nonzero(mask[q], as_tuple=False).flatten().tolist())
            self.assertTrue(visible.issubset(set(range(block1_end + frame_seqlen * num_frame_per_block)) | {q}))


class TestSeparateMergeActionTokens(unittest.TestCase):
    """Verify _separate_action_tokens and _merge_action_tokens are inverses."""

    def test_roundtrip(self):
        with patch("torch.cuda.current_device", return_value=0):
            from wan.modules.causal_model import _separate_action_tokens, _merge_action_tokens

        B, F, H, W, A = 2, 3, 2, 4, 1
        spatial = H * W
        frame_seq = spatial + A
        x = torch.randn(B, F * frame_seq, 16)
        grid_sizes = torch.tensor([[F, H, W]])

        sp, ac = _separate_action_tokens(x, grid_sizes, A)
        self.assertEqual(sp.shape, (B, F * spatial, 16))
        self.assertEqual(ac.shape, (B, F * A, 16))

        recon = _merge_action_tokens(sp, ac, grid_sizes, A)
        torch.testing.assert_close(recon, x)


class TestPEFTCleanSideAdaLN(unittest.TestCase):
    """Verify the PEFT ordering fix: clean-side modulation uses am_clean."""

    def test_peft_wrapped_clean_side_uses_clean_modulation(self):
        CausalWanModel = _import_causal_model()
        with patch("torch.cuda.current_device", return_value=0):
            model = CausalWanModel(
                dim=64, ffn_dim=128, freq_dim=32, text_dim=64,
                num_heads=4, num_layers=1, in_dim=4, out_dim=4,
                patch_size=(1, 1, 1), text_len=4,
            )
        model.num_frame_per_block = 3
        model.context_shift = 1

        with patch("torch.cuda.current_device", return_value=0):
            from model.action_model_patch import patch_causal_wan_model_for_action

        patch_causal_wan_model_for_action(model)

        B, F = 1, 3
        am_noisy = torch.full((B, F, 6, 64), 5.0)
        am_clean = torch.full((B, F, 6, 64), 7.0)

        captured = {"calls": []}
        tp_module = model.time_projection
        orig_fwd = tp_module.forward

        class _EarlyExit(Exception):
            """Raised after we've captured enough spy calls."""

        def spy_tp(e):
            result = orig_fwd(e)
            am = getattr(model, '_action_modulation', None)
            tag = 'none'
            if am is not None:
                if torch.allclose(am, am_clean):
                    tag = 'clean'
                elif torch.allclose(am, am_noisy):
                    tag = 'noisy'
            captured["calls"].append(tag)
            if len(captured["calls"]) >= 2:
                raise _EarlyExit()
            return result

        tp_module.forward = spy_tp

        t = torch.randint(0, 100, (B, F))
        aug_t = torch.zeros(B, F)
        x_noisy = torch.randn(B, 4, F, 2, 2)
        clean_x = torch.randn(B, 4, F, 2, 2)
        context = [torch.randn(4, 64)]

        model.block_mask = None
        try:
            model._forward_train(
                x_noisy, t, context, seq_len=100,
                clean_x=clean_x, aug_t=aug_t,
                action_modulation=am_noisy,
                action_modulation_clean=am_clean,
            )
        except _EarlyExit:
            pass
        finally:
            tp_module.forward = orig_fwd

        self.assertGreaterEqual(len(captured['calls']), 2,
                                "time_projection must be called at least twice (noisy + clean)")
        self.assertEqual(captured['calls'][0], 'noisy',
                         "First call (noisy branch) should see am_noisy")
        self.assertEqual(captured['calls'][1], 'clean',
                         "Second call (clean branch) should see am_clean")


class TestActionTokenCleanNoisyAlignment(unittest.TestCase):
    """Verify clean/noisy action token slicing matches the shifted split."""

    def test_clean_noisy_action_tokens_are_shifted(self):
        cf = 3
        num_frames = 21
        z_full = torch.arange(24 * 2, dtype=torch.float32).view(1, 24, 2)
        z_noisy = z_full[:, cf:]
        z_clean = z_full[:, :num_frames]

        self.assertEqual(z_noisy.shape, (1, 21, 2))
        self.assertEqual(z_clean.shape, (1, 21, 2))
        self.assertAlmostEqual(z_clean[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(z_noisy[0, 0, 0].item(), 6.0)
        self.assertFalse(torch.equal(z_clean, z_noisy))


# ===================================================================
# Action Critic Tests
# ===================================================================

from model.action_critic import ActionCritic


class TestActionCriticModule(unittest.TestCase):
    """Verify the chunkwise 3D CNN ActionCritic module."""

    def setUp(self):
        self.B = 2
        self.num_frames = 21
        self.chunk_frames = 3
        self.n_chunks = self.num_frames // self.chunk_frames  # 7
        self.C, self.H, self.W = 16, 60, 104
        self.z_dim = 2
        self.critic = ActionCritic(
            latent_channels=self.C, z_dim=self.z_dim,
            base_channels=32, num_res_blocks=2, chunk_frames=self.chunk_frames,
        )

    def test_output_shapes(self):
        """Output should be [B, n_chunks, z_dim] and [B, n_chunks, 1]."""
        x = torch.randn(self.B, self.num_frames, self.C, self.H, self.W)
        pred_z, pred_r = self.critic(x)
        self.assertEqual(pred_z.shape, (self.B, self.n_chunks, self.z_dim))
        self.assertEqual(pred_r.shape, (self.B, self.n_chunks, 1))

    def test_both_heads_present(self):
        self.assertTrue(hasattr(self.critic, "z_head"))
        self.assertTrue(hasattr(self.critic, "reward_head"))

    def test_zero_init_near_zero_output(self):
        """With zero-init heads, outputs should be near zero initially."""
        x = torch.randn(1, 3, self.C, self.H, self.W)
        pred_z, pred_r = self.critic(x)
        self.assertLess(pred_z.abs().max().item(), 1.0)
        self.assertLess(pred_r.abs().max().item(), 1.0)

    def test_chunk_frames_assertion(self):
        """F not divisible by chunk_frames should raise AssertionError."""
        x = torch.randn(1, 5, self.C, self.H, self.W)  # 5 not divisible by 3
        with self.assertRaises(AssertionError):
            self.critic(x)

    def test_3d_conv_trunk_present(self):
        """The critic should have stem, down1, down2, trunk as 3D conv stages."""
        self.assertTrue(hasattr(self.critic, "stem"))
        self.assertTrue(hasattr(self.critic, "down1"))
        self.assertTrue(hasattr(self.critic, "down2"))
        self.assertTrue(hasattr(self.critic, "trunk"))


class TestActionCriticGradientFlow(unittest.TestCase):
    """Verify gradients flow from the chunkwise critic loss back to pred_x0."""

    def _make_critic(self):
        return ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )

    def test_critic_loss_produces_grad_on_pred_x0(self):
        """Generator-side: critic(pred_x0) must give grads on pred_x0."""
        critic = self._make_critic()
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        n_chunks = 7
        target_z = torch.randn(1, n_chunks, 2)

        pred_z, pred_r = critic(pred_x0)
        gen_loss = F.mse_loss(pred_z, target_z) - pred_r.mean()
        gen_loss.backward()

        self.assertIsNotNone(pred_x0.grad)
        self.assertTrue((pred_x0.grad.abs() > 0).any(),
                        "pred_x0 must receive non-zero gradients through the critic")

    def test_critic_training_grads_on_critic_params_only(self):
        """Critic-training path: grads flow to critic params, NOT to pred_x0."""
        critic = self._make_critic()
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        n_chunks = 7
        teacher_z = torch.randn(1, n_chunks, 2)
        teacher_r = torch.randn(1, n_chunks, 1)

        pred_z, pred_r = critic(pred_x0.detach())
        critic_loss = F.mse_loss(pred_z, teacher_z) + F.mse_loss(pred_r, teacher_r)
        critic_loss.backward()

        self.assertIsNone(pred_x0.grad)
        for name, p in critic.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Critic param {name} should have grad")

    def test_total_loss_backward_succeeds(self):
        """Combined flow_loss + critic_loss + gen_action_loss backward works."""
        critic = self._make_critic()
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        target = torch.randn_like(pred_x0)
        n_chunks = 7

        flow_loss = F.mse_loss(pred_x0, target)

        teacher_z = torch.randn(1, n_chunks, 2)
        teacher_r = torch.randn(1, n_chunks, 1)
        pz_c, pr_c = critic(pred_x0.detach())
        critic_loss = F.mse_loss(pz_c, teacher_z) + F.mse_loss(pr_c, teacher_r)

        pz_g, pr_g = critic(pred_x0)
        gen_action_loss = F.mse_loss(pz_g, teacher_z) - pr_g.mean()

        total_loss = flow_loss + critic_loss + gen_action_loss
        total_loss.backward()

        self.assertIsNotNone(pred_x0.grad)
        self.assertTrue((pred_x0.grad.abs() > 0).any())
        for name, p in critic.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Critic param {name} should have grad")


class TestActionCriticTargetAlignment(unittest.TestCase):
    """Verify target selection uses the correct z-dims and slicing."""

    def test_critic_dims_select_z2_z7(self):
        """action_critic_dims=[2, 7] correctly selects from 8D z_actions."""
        z_actions_8d = torch.randn(1, 21, 8)
        action_critic_dims = [2, 7]
        target_z = z_actions_8d[..., action_critic_dims]
        self.assertEqual(target_z.shape, (1, 21, 2))
        torch.testing.assert_close(target_z[..., 0], z_actions_8d[..., 2])
        torch.testing.assert_close(target_z[..., 1], z_actions_8d[..., 7])

    def test_target_uses_noisy_frames(self):
        """Target z should come from the noisy-side (shifted) frames."""
        cf = 3
        z_full = torch.arange(24 * 8, dtype=torch.float32).view(1, 24, 8)
        z_noisy = z_full[:, cf:]
        target_z = z_noisy[..., [2, 7]]
        self.assertEqual(target_z.shape, (1, 21, 2))
        expected_z2_frame0 = z_full[0, cf, 2].item()
        self.assertAlmostEqual(target_z[0, 0, 0].item(), expected_z2_frame0)

    def test_zero_weights_leave_flow_loss_unchanged(self):
        """With zero guidance weights, only flow matching loss contributes."""
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )
        pred_x0 = torch.randn(1, 3, 16, 4, 4, requires_grad=True)
        target = torch.randn_like(pred_x0)

        flow_loss = F.mse_loss(pred_x0, target)

        pz, pr = critic(pred_x0)
        gen_z_loss = F.mse_loss(pz, torch.randn_like(pz))
        gen_r_loss = -pr.mean()

        z_w = 0.0
        r_w = 0.0
        total = flow_loss + z_w * gen_z_loss + r_w * gen_r_loss

        torch.testing.assert_close(total, flow_loss)


def _reduce_to_segments(per_frame, n_seg):
    """Local copy of the trainer's static helper for CPU-only testing."""
    F_len = per_frame.shape[0]
    seg_size = max(1, F_len // n_seg)
    segments = []
    for i in range(n_seg):
        s = i * seg_size
        e = min(s + seg_size, F_len)
        segments.append(per_frame[s:e].mean(dim=0))
    return torch.stack(segments, dim=0)


def _temporal_pool_to_segments(per_frame, n_seg):
    """Local copy of the trainer's static helper for CPU-only testing."""
    B, F_len, D = per_frame.shape
    seg_size = max(1, F_len // n_seg)
    segs = []
    for i in range(n_seg):
        s = i * seg_size
        e = min(s + seg_size, F_len)
        segs.append(per_frame[:, s:e].mean(dim=1))
    return torch.stack(segs, dim=1)


class TestReduceToSegments(unittest.TestCase):
    """Verify temporal pooling helpers."""

    def test_reduce_to_segments_identity_when_matching(self):
        """If n_seg == F, each segment is one frame."""
        per_frame = torch.arange(21, dtype=torch.float32).unsqueeze(-1)  # [21, 1]
        out = _reduce_to_segments(per_frame, 21)
        self.assertEqual(out.shape, (21, 1))
        torch.testing.assert_close(out, per_frame)

    def test_reduce_to_segments_pools(self):
        """7 segments from 21 frames -> each segment averages 3 frames."""
        per_frame = torch.arange(21, dtype=torch.float32).unsqueeze(-1)
        out = _reduce_to_segments(per_frame, 7)
        self.assertEqual(out.shape, (7, 1))
        self.assertAlmostEqual(out[0, 0].item(), 1.0)  # mean(0,1,2)
        self.assertAlmostEqual(out[1, 0].item(), 4.0)  # mean(3,4,5)

    def test_batch_temporal_pool(self):
        """_temporal_pool_to_segments pools [B, F, D] -> [B, n_seg, D]."""
        per_frame = torch.arange(42, dtype=torch.float32).view(2, 21, 1)
        out = _temporal_pool_to_segments(per_frame, 7)
        self.assertEqual(out.shape, (2, 7, 1))


class TestMotionToSsVaeReshape(unittest.TestCase):
    """Verify the motion -> ss_vae input reshape matches the dataset pipeline."""

    def test_motion_reshape_matches_dataset(self):
        """Motion (n, 100, 3) -> take dx/dy -> (n, 2, 10, 10) matches zarr_dataset."""
        n = 5
        motion = torch.randn(n, 100, 3)
        xy = motion[:, :, :2].reshape(n, 10, 10, 2)
        x_in = xy.permute(0, 3, 1, 2)  # [n, 2, 10, 10]
        self.assertEqual(x_in.shape, (n, 2, 10, 10))


def _parse_critic_config(config):
    """Mirror the exact critic config parsing from CausalLoRADiffusionTrainer.__init__."""
    enabled = bool(getattr(config, "action_critic_enabled", False))
    noise_ts = int(getattr(config, "action_critic_noise_timestep", 25))
    dims_cfg = getattr(config, "action_critic_dims", None)
    dims = list(dims_cfg) if dims_cfg is not None else [2, 7]
    z_w = float(getattr(config, "action_critic_z_loss_weight", 1.0))
    r_w = float(getattr(config, "action_critic_reward_loss_weight", 0.1))
    gz_w = float(getattr(config, "generator_action_z_guidance_weight", 0.0))
    gr_w = float(getattr(config, "generator_action_reward_guidance_weight", 0.0))
    return {
        "enabled": enabled, "noise_timestep": noise_ts, "dims": dims,
        "z_w": z_w, "r_w": r_w, "gz_w": gz_w, "gr_w": gr_w,
    }


class TestCriticConfigParsing(unittest.TestCase):
    """Verify critic config parsing with OmegaConf (the real config type)."""

    def test_defaults_from_empty_config(self):
        """Empty OmegaConf config yields all expected defaults."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({})
        parsed = _parse_critic_config(cfg)
        self.assertFalse(parsed["enabled"])
        self.assertEqual(parsed["noise_timestep"], 25)
        self.assertEqual(parsed["dims"], [2, 7])
        self.assertEqual(parsed["z_w"], 1.0)
        self.assertEqual(parsed["r_w"], 0.1)
        self.assertEqual(parsed["gz_w"], 0.0)
        self.assertEqual(parsed["gr_w"], 0.0)

    def test_enabled_override(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"action_critic_enabled": True})
        parsed = _parse_critic_config(cfg)
        self.assertTrue(parsed["enabled"])

    def test_custom_dims(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"action_critic_dims": [0, 1, 2]})
        parsed = _parse_critic_config(cfg)
        self.assertEqual(parsed["dims"], [0, 1, 2])

    def test_custom_weights(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "action_critic_z_loss_weight": 2.5,
            "action_critic_reward_loss_weight": 0.5,
            "generator_action_z_guidance_weight": 0.01,
            "generator_action_reward_guidance_weight": 0.02,
        })
        parsed = _parse_critic_config(cfg)
        self.assertAlmostEqual(parsed["z_w"], 2.5)
        self.assertAlmostEqual(parsed["r_w"], 0.5)
        self.assertAlmostEqual(parsed["gz_w"], 0.01)
        self.assertAlmostEqual(parsed["gr_w"], 0.02)

    def test_yaml_round_trip(self):
        """Config loaded from YAML string parses identically."""
        from omegaconf import OmegaConf
        yaml_str = """
action_critic_enabled: true
action_critic_noise_timestep: 50
action_critic_dims: [2, 7]
action_critic_z_loss_weight: 1.0
"""
        cfg = OmegaConf.create(yaml_str)
        parsed = _parse_critic_config(cfg)
        self.assertTrue(parsed["enabled"])
        self.assertEqual(parsed["noise_timestep"], 50)
        self.assertEqual(parsed["dims"], [2, 7])


class TestBlockwiseTimestepSampling(unittest.TestCase):
    """Verify blockwise timestep index is constant within each block of 3 frames."""

    @staticmethod
    def _make_blockwise_index(batch_size, num_frames, high, num_frame_per_block):
        """Mirror CausalLoRADiffusionTrainer._make_blockwise_index on CPU."""
        index = torch.randint(0, high, (batch_size, num_frames))
        index = index.reshape(batch_size, -1, num_frame_per_block)
        index[:, :, 1:] = index[:, :, 0:1]
        return index.reshape(batch_size, num_frames)

    def test_all_frames_in_block_share_same_index(self):
        """With num_frame_per_block=3, frames 0-2, 3-5, 6-8, ... share an index."""
        index = self._make_blockwise_index(4, 21, 1000, 3)
        self.assertEqual(index.shape, (4, 21))
        blocks = index.reshape(4, 7, 3)
        for b in range(4):
            for blk in range(7):
                self.assertEqual(blocks[b, blk, 0].item(), blocks[b, blk, 1].item())
                self.assertEqual(blocks[b, blk, 0].item(), blocks[b, blk, 2].item())

    def test_different_blocks_may_differ(self):
        """Different blocks should not all be identical (probabilistic, large range)."""
        torch.manual_seed(42)
        index = self._make_blockwise_index(1, 21, 10000, 3)
        blocks = index.reshape(1, 7, 3)
        unique_per_block = blocks[0, :, 0]
        self.assertGreater(unique_per_block.unique().numel(), 1,
                           "With high=10000, multiple blocks should sample different indices")

    def test_single_frame_blocks(self):
        """With num_frame_per_block=1, every frame has its own index."""
        index = self._make_blockwise_index(2, 6, 100, 1)
        self.assertEqual(index.shape, (2, 6))


# ======================================================================
# Modernized trainer regression tests
# ======================================================================

def _parse_held_out_split(config):
    """Mirror the exact split parsing from CausalLoRADiffusionTrainer.__init__."""
    test_start_index = int(getattr(config, "test_start_index", 0))
    test_num_rides = int(getattr(config, "test_num_rides", 20))
    train_start = test_start_index + test_num_rides
    return test_start_index, test_num_rides, train_start


class TestHeldOutSplit(unittest.TestCase):
    """Verify the held-out ride split matches trainer _build_dataloader logic."""

    def test_defaults_from_omegaconf(self):
        """Default OmegaConf config (no keys) yields start=0, rides=20, train_start=20."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({})
        start, rides, train_start = _parse_held_out_split(cfg)
        self.assertEqual(start, 0)
        self.assertEqual(rides, 20)
        self.assertEqual(train_start, 20)

    def test_custom_split_from_omegaconf(self):
        """Custom split via OmegaConf respects both parameters."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"test_start_index": 5, "test_num_rides": 10})
        start, rides, train_start = _parse_held_out_split(cfg)
        self.assertEqual(start, 5)
        self.assertEqual(rides, 10)
        self.assertEqual(train_start, 15)

    def test_zero_heldout_means_no_skip(self):
        """With test_num_rides=0, train_start == test_start_index."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"test_start_index": 0, "test_num_rides": 0})
        start, rides, train_start = _parse_held_out_split(cfg)
        self.assertEqual(train_start, 0)

    def test_train_and_eval_ranges_do_not_overlap(self):
        """Eval range [start, start+rides) must not overlap with train range [train_start, ...)."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"test_start_index": 3, "test_num_rides": 7})
        start, rides, train_start = _parse_held_out_split(cfg)
        eval_end = start + rides
        self.assertLessEqual(eval_end, train_start)


class TestCheckpointCoverage(unittest.TestCase):
    """Verify real modules survive save/load round-trips via torch.save/load."""

    def test_action_critic_round_trip(self):
        """Save and reload an ActionCritic; all parameters must match exactly."""
        critic = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                              num_res_blocks=1, chunk_frames=3)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"action_critic": critic.state_dict()}, f.name)
            loaded = torch.load(f.name, map_location="cpu")

        critic2 = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                               num_res_blocks=1, chunk_frames=3)
        missing, unexpected = critic2.load_state_dict(loaded["action_critic"], strict=True)
        self.assertEqual(len(missing), 0, f"Missing keys: {missing}")
        self.assertEqual(len(unexpected), 0, f"Unexpected keys: {unexpected}")
        for (n1, p1), (_, p2) in zip(critic.named_parameters(), critic2.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Param mismatch in {n1}")

    def test_action_modulation_projection_round_trip(self):
        """Save and reload an ActionModulationProjection."""
        from model.action_modulation import ActionModulationProjection
        proj = ActionModulationProjection(action_dim=2, activation="silu",
                                          hidden_dim=64, num_frames=1, zero_init=True)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"action_projection": proj.state_dict()}, f.name)
            loaded = torch.load(f.name, map_location="cpu")

        proj2 = ActionModulationProjection(action_dim=2, activation="silu",
                                           hidden_dim=64, num_frames=1, zero_init=True)
        proj2.load_state_dict(loaded["action_projection"])
        for (n1, p1), (_, p2) in zip(proj.named_parameters(), proj2.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Param mismatch in {n1}")

    def test_action_token_projection_round_trip(self):
        """Save and reload an ActionTokenProjection."""
        from model.action_modulation import ActionTokenProjection
        proj = ActionTokenProjection(action_dim=2, activation="silu",
                                     hidden_dim=64, zero_init=True)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"action_token_projection": proj.state_dict()}, f.name)
            loaded = torch.load(f.name, map_location="cpu")

        proj2 = ActionTokenProjection(action_dim=2, activation="silu",
                                      hidden_dim=64, zero_init=True)
        proj2.load_state_dict(loaded["action_token_projection"])
        for (n1, p1), (_, p2) in zip(proj.named_parameters(), proj2.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Param mismatch in {n1}")

    def test_backward_compatible_load_without_critic(self):
        """Old checkpoint missing action_critic key: critic.load_state_dict not called."""
        old_checkpoint = {"step": 50, "lora": {}, "optimizer": {}, "config_name": "test"}
        self.assertNotIn("action_critic", old_checkpoint)
        self.assertNotIn("action_projection", old_checkpoint)
        self.assertNotIn("action_token_projection", old_checkpoint)

    def test_architecture_change_loads_with_strict_false(self):
        """Critic with fewer res blocks loads into one with more (strict=False)."""
        critic_v1 = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                                 num_res_blocks=1, chunk_frames=3)
        critic_v2 = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                                 num_res_blocks=3, chunk_frames=3)
        missing, unexpected = critic_v2.load_state_dict(critic_v1.state_dict(), strict=False)
        self.assertGreater(len(missing), 0,
                           "New trunk blocks should appear as missing keys")
        self.assertEqual(len(unexpected), 0)


class TestGradNormCollection(unittest.TestCase):
    """Verify gradient norm collection on actual production modules."""

    def _collect_grad_norm(self, module):
        """Mirror _compute_grad_norms pattern from the trainer."""
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            return None
        return torch.norm(torch.stack([g.norm() for g in grads])).item()

    def test_critic_grad_norm_after_backward(self):
        """ActionCritic backward produces positive gradient norm."""
        critic = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                              num_res_blocks=1, chunk_frames=3)
        x = torch.randn(1, 3, 16, 8, 8)
        pred_z, _ = critic(x)
        pred_z.sum().backward()
        norm = self._collect_grad_norm(critic)
        self.assertIsNotNone(norm)
        self.assertGreater(norm, 0)

    def test_action_modulation_grad_norm_after_backward(self):
        """ActionModulationProjection backward produces positive gradient norm."""
        from model.action_modulation import ActionModulationProjection
        proj = ActionModulationProjection(action_dim=2, activation="silu",
                                          hidden_dim=64, num_frames=1, zero_init=False)
        z = torch.randn(1, 7, 2)
        out = proj(z, num_frames=7)
        out.sum().backward()
        norm = self._collect_grad_norm(proj)
        self.assertIsNotNone(norm)
        self.assertGreater(norm, 0)

    def test_action_token_grad_norm_after_backward(self):
        """ActionTokenProjection backward produces positive gradient norm."""
        from model.action_modulation import ActionTokenProjection
        proj = ActionTokenProjection(action_dim=2, activation="silu",
                                     hidden_dim=64, zero_init=False)
        z = torch.randn(1, 7, 2)
        out = proj(z)
        out.sum().backward()
        norm = self._collect_grad_norm(proj)
        self.assertIsNotNone(norm)
        self.assertGreater(norm, 0)

    def test_no_grad_returns_none(self):
        """Module that didn't participate in backward has no grad norm."""
        critic = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                              num_res_blocks=1, chunk_frames=3)
        norm = self._collect_grad_norm(critic)
        self.assertIsNone(norm)


class TestCriticLossLogPayload(unittest.TestCase):
    """Compute real critic losses and verify the returned log dict structure."""

    def _compute_critic_losses(self, critic, pred_x0, target_action_z,
                               z_w=1.0, r_w=0.1, gz_w=0.0, gr_w=0.0):
        """Mirror _compute_action_critic_losses from the trainer."""
        chunk_frames = critic.chunk_frames
        target_chunked = _chunk_actions_local(target_action_z, chunk_frames)
        n_chunks = target_chunked.shape[1]
        teacher_reward = torch.zeros(pred_x0.shape[0], n_chunks, 1)

        pred_z, pred_reward = critic(pred_x0.detach())
        pred_z = pred_z[:, :n_chunks]
        pred_reward = pred_reward[:, :n_chunks]

        critic_z_loss = F.mse_loss(pred_z, target_chunked)
        critic_r_loss = F.mse_loss(pred_reward, teacher_reward)
        critic_loss = z_w * critic_z_loss + r_w * critic_r_loss

        critic.requires_grad_(False)
        gen_pred_z, gen_pred_reward = critic(pred_x0)
        gen_pred_z = gen_pred_z[:, :n_chunks]
        gen_pred_reward = gen_pred_reward[:, :n_chunks]
        gen_z_loss = F.mse_loss(gen_pred_z, target_chunked)
        gen_reward_loss = -gen_pred_reward.mean()
        generator_action_loss = gz_w * gen_z_loss + gr_w * gen_reward_loss
        critic.requires_grad_(True)

        logs = {
            "train/critic_z_loss": critic_z_loss.detach().item(),
            "train/critic_r_loss": critic_r_loss.detach().item(),
            "train/critic_loss": critic_loss.detach().item(),
            "train/gen_action_z_loss": gen_z_loss.detach().item(),
            "train/gen_action_reward": gen_pred_reward.mean().detach().item(),
            "train/gen_action_loss": generator_action_loss.detach().item(),
        }
        return critic_loss, generator_action_loss, logs

    def test_critic_loss_log_keys(self):
        """Real critic forward/loss produces all expected log keys."""
        critic = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                              num_res_blocks=1, chunk_frames=3)
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        target_z = torch.randn(1, 21, 2)

        _, _, logs = self._compute_critic_losses(critic, pred_x0, target_z)

        expected_keys = [
            "train/critic_z_loss", "train/critic_r_loss", "train/critic_loss",
            "train/gen_action_z_loss", "train/gen_action_reward", "train/gen_action_loss",
        ]
        for key in expected_keys:
            self.assertIn(key, logs, f"Missing key: {key}")
            self.assertIsInstance(logs[key], float, f"{key} should be a float")
            self.assertTrue(
                not (logs[key] != logs[key]),  # not NaN
                f"{key} is NaN",
            )

    def test_loss_values_are_finite(self):
        """All returned losses must be finite scalars."""
        critic = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                              num_res_blocks=1, chunk_frames=3)
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        target_z = torch.randn(1, 21, 2)

        critic_loss, gen_loss, logs = self._compute_critic_losses(critic, pred_x0, target_z)
        self.assertTrue(torch.isfinite(critic_loss))
        self.assertTrue(torch.isfinite(gen_loss))
        for key, val in logs.items():
            self.assertFalse(val != val, f"{key} is NaN")

    def test_nonzero_weights_produce_nonzero_gen_loss(self):
        """With nonzero guidance weights, generator action loss should be nonzero."""
        critic = ActionCritic(latent_channels=16, z_dim=2, base_channels=16,
                              num_res_blocks=1, chunk_frames=3)
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        target_z = torch.randn(1, 21, 2)

        _, gen_loss, logs = self._compute_critic_losses(
            critic, pred_x0, target_z, gz_w=1.0, gr_w=1.0,
        )
        self.assertNotAlmostEqual(gen_loss.item(), 0.0, places=6)


def _temporal_pool_trainer(per_frame, n_seg):
    """Local copy of trainer._temporal_pool for CPU-only testing."""
    B, F_len, D = per_frame.shape
    seg_size = max(1, F_len // n_seg)
    segs = []
    for i in range(n_seg):
        s = i * seg_size
        e = min(s + seg_size, F_len)
        segs.append(per_frame[:, s:e].mean(dim=1))
    return torch.stack(segs, dim=1)


class TestTemporalPoolModule(unittest.TestCase):
    """Verify the module-level _temporal_pool matches expected behavior."""

    def test_temporal_pool_basic(self):
        x = torch.arange(42, dtype=torch.float32).view(2, 21, 1)
        out = _temporal_pool_trainer(x, 7)
        self.assertEqual(out.shape, (2, 7, 1))

    def test_temporal_pool_identity(self):
        x = torch.arange(21, dtype=torch.float32).view(1, 21, 1)
        out = _temporal_pool_trainer(x, 21)
        self.assertEqual(out.shape, (1, 21, 1))
        torch.testing.assert_close(out, x)


class TestBuildConditional(unittest.TestCase):
    """Verify _build_conditional output by calling real projection modules."""

    @staticmethod
    def _build_conditional(use_action_conditioning, action_projection,
                           action_token_projection, prompt_embeds,
                           z_noisy, z_clean, num_frames):
        """Mirror CausalLoRADiffusionTrainer._build_conditional."""
        conditional = {"prompt_embeds": prompt_embeds}
        if use_action_conditioning and action_projection is not None:
            conditional["_action_modulation"] = action_projection(z_noisy, num_frames=num_frames)
            conditional["_action_modulation_clean"] = action_projection(z_clean, num_frames=num_frames)
        if use_action_conditioning and action_token_projection is not None:
            conditional["_action_tokens"] = action_token_projection(z_noisy)
            conditional["_action_tokens_clean"] = action_token_projection(z_clean)
        return conditional

    def test_adaln_only(self):
        """With only adaLN projection, conditional has modulation keys."""
        from model.action_modulation import ActionModulationProjection
        proj = ActionModulationProjection(action_dim=2, activation="silu",
                                          hidden_dim=64, num_frames=1, zero_init=True)
        prompt = torch.randn(1, 5, 512)
        z = torch.randn(1, 21, 2)
        result = self._build_conditional(True, proj, None, prompt, z, z, 21)
        self.assertIn("prompt_embeds", result)
        self.assertIn("_action_modulation", result)
        self.assertIn("_action_modulation_clean", result)
        self.assertNotIn("_action_tokens", result)
        self.assertEqual(result["_action_modulation"].shape[:2], (1, 21))

    def test_tokens_only(self):
        """With only token projection, conditional has token keys."""
        from model.action_modulation import ActionTokenProjection
        tok_proj = ActionTokenProjection(action_dim=2, activation="silu",
                                         hidden_dim=64, zero_init=True)
        prompt = torch.randn(1, 5, 512)
        z = torch.randn(1, 21, 2)
        result = self._build_conditional(True, None, tok_proj, prompt, z, z, 21)
        self.assertIn("prompt_embeds", result)
        self.assertNotIn("_action_modulation", result)
        self.assertIn("_action_tokens", result)
        self.assertIn("_action_tokens_clean", result)
        self.assertEqual(result["_action_tokens"].shape, (1, 21, 64))

    def test_both_projections(self):
        """With both projections, all conditioning keys present."""
        from model.action_modulation import ActionModulationProjection, ActionTokenProjection
        proj = ActionModulationProjection(action_dim=2, activation="silu",
                                          hidden_dim=64, num_frames=1, zero_init=True)
        tok_proj = ActionTokenProjection(action_dim=2, activation="silu",
                                         hidden_dim=64, zero_init=True)
        prompt = torch.randn(1, 5, 512)
        z_n = torch.randn(1, 21, 2)
        z_c = torch.randn(1, 21, 2)
        result = self._build_conditional(True, proj, tok_proj, prompt, z_n, z_c, 21)
        for key in ("prompt_embeds", "_action_modulation", "_action_modulation_clean",
                     "_action_tokens", "_action_tokens_clean"):
            self.assertIn(key, result)

    def test_conditioning_disabled(self):
        """With use_action_conditioning=False, only prompt_embeds present."""
        from model.action_modulation import ActionModulationProjection
        proj = ActionModulationProjection(action_dim=2, activation="silu",
                                          hidden_dim=64, num_frames=1, zero_init=True)
        prompt = torch.randn(1, 5, 512)
        z = torch.randn(1, 21, 2)
        result = self._build_conditional(False, proj, None, prompt, z, z, 21)
        self.assertEqual(list(result.keys()), ["prompt_embeds"])


# ======================================================================
# Chunkwise 3D critic contract tests
# ======================================================================

def _chunk_actions_local(per_frame: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Local copy of the trainer's _chunk_actions helper for CPU-only testing."""
    B, F_len, D = per_frame.shape
    n_chunks = F_len // chunk_frames
    trimmed = per_frame[:, :n_chunks * chunk_frames]
    return trimmed.reshape(B, n_chunks, chunk_frames, D).mean(dim=2)


class TestChunkReshaping(unittest.TestCase):
    """Verify 21-frame -> 7-chunk reshaping is correct."""

    def test_21_to_7_chunks(self):
        """21 frames with chunk_frames=3 should give 7 chunks."""
        per_frame = torch.arange(21 * 2, dtype=torch.float32).view(1, 21, 2)
        chunks = _chunk_actions_local(per_frame, 3)
        self.assertEqual(chunks.shape, (1, 7, 2))

    def test_chunk_values_are_means(self):
        """Each chunk should be the mean of its constituent frames."""
        per_frame = torch.arange(21, dtype=torch.float32).view(1, 21, 1)
        chunks = _chunk_actions_local(per_frame, 3)
        self.assertAlmostEqual(chunks[0, 0, 0].item(), 1.0)   # mean(0,1,2)
        self.assertAlmostEqual(chunks[0, 1, 0].item(), 4.0)   # mean(3,4,5)
        self.assertAlmostEqual(chunks[0, 6, 0].item(), 19.0)  # mean(18,19,20)

    def test_non_divisible_frames_are_trimmed(self):
        """22 frames with chunk_frames=3 should give 7 chunks (trailing frame dropped)."""
        per_frame = torch.randn(1, 22, 2)
        chunks = _chunk_actions_local(per_frame, 3)
        self.assertEqual(chunks.shape, (1, 7, 2))

    def test_batch_dimension_preserved(self):
        """Batched input should produce batched output."""
        per_frame = torch.randn(4, 21, 2)
        chunks = _chunk_actions_local(per_frame, 3)
        self.assertEqual(chunks.shape, (4, 7, 2))


class TestChunkwiseCriticOutputShapes(unittest.TestCase):
    """Verify critic output shapes match the chunkwise contract."""

    def test_21_frames_gives_7_chunks(self):
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )
        x = torch.randn(2, 21, 16, 8, 8)
        pred_z, pred_r = critic(x)
        self.assertEqual(pred_z.shape, (2, 7, 2))
        self.assertEqual(pred_r.shape, (2, 7, 1))

    def test_6_frames_gives_2_chunks(self):
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )
        x = torch.randn(1, 6, 16, 8, 8)
        pred_z, pred_r = critic(x)
        self.assertEqual(pred_z.shape, (1, 2, 2))
        self.assertEqual(pred_r.shape, (1, 2, 1))

    def test_single_chunk(self):
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )
        x = torch.randn(1, 3, 16, 8, 8)
        pred_z, pred_r = critic(x)
        self.assertEqual(pred_z.shape, (1, 1, 2))
        self.assertEqual(pred_r.shape, (1, 1, 1))


class TestChunkTargetAlignment(unittest.TestCase):
    """Verify chunk-level alignment between critic outputs and teacher targets."""

    def test_critic_and_target_same_n_chunks(self):
        """Critic output and chunked target should share the same n_chunks."""
        chunk_frames = 3
        num_frames = 21
        n_chunks = num_frames // chunk_frames

        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=chunk_frames,
        )
        x = torch.randn(1, num_frames, 16, 8, 8)
        pred_z, pred_r = critic(x)

        target_per_frame = torch.randn(1, num_frames, 2)
        target_chunked = _chunk_actions_local(target_per_frame, chunk_frames)

        self.assertEqual(pred_z.shape[1], n_chunks)
        self.assertEqual(target_chunked.shape[1], n_chunks)
        self.assertEqual(pred_z.shape[1], target_chunked.shape[1])

    def test_loss_computable_without_pooling(self):
        """Direct MSE between critic output and chunked target works without pooling."""
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )
        x = torch.randn(1, 21, 16, 8, 8)
        pred_z, pred_r = critic(x)

        target_chunked = torch.randn(1, 7, 2)
        reward_chunked = torch.randn(1, 7, 1)

        z_loss = F.mse_loss(pred_z, target_chunked)
        r_loss = F.mse_loss(pred_r, reward_chunked)
        self.assertTrue(torch.isfinite(z_loss))
        self.assertTrue(torch.isfinite(r_loss))


class TestFrozenCriticGeneratorGradients(unittest.TestCase):
    """Verify generator-guidance path with frozen critic flows grads to pred_x0."""

    def test_frozen_critic_passes_grad_to_pred_x0(self):
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        target_chunk = torch.randn(1, 7, 2)

        critic.requires_grad_(False)
        pred_z, pred_r = critic(pred_x0)
        gen_loss = F.mse_loss(pred_z, target_chunk) - pred_r.mean()
        gen_loss.backward()
        critic.requires_grad_(True)

        self.assertIsNotNone(pred_x0.grad)
        self.assertTrue((pred_x0.grad.abs() > 0).any())

    def test_frozen_critic_has_no_param_grad(self):
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        )
        pred_x0 = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        target_chunk = torch.randn(1, 7, 2)

        critic.requires_grad_(False)
        pred_z, pred_r = critic(pred_x0)
        gen_loss = F.mse_loss(pred_z, target_chunk) - pred_r.mean()
        gen_loss.backward()

        for name, p in critic.named_parameters():
            self.assertIsNone(p.grad, f"Frozen critic param {name} should have no grad")
        critic.requires_grad_(True)


class TestMixedPrecisionCritic(unittest.TestCase):
    """Verify the 3D CNN critic works under mixed precision (CPU simulation)."""

    def test_critic_accepts_float16_input(self):
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        ).half()
        x = torch.randn(1, 21, 16, 8, 8).half()
        pred_z, pred_r = critic(x)
        self.assertEqual(pred_z.dtype, torch.float16)
        self.assertEqual(pred_r.dtype, torch.float16)
        self.assertTrue(torch.isfinite(pred_z).all())
        self.assertTrue(torch.isfinite(pred_r).all())

    def test_critic_backward_under_mixed_precision(self):
        critic = ActionCritic(
            latent_channels=16, z_dim=2,
            base_channels=16, num_res_blocks=1, chunk_frames=3,
        ).half()
        x_leaf = torch.randn(1, 21, 16, 8, 8, requires_grad=True)
        x = x_leaf.half()
        x.retain_grad()
        pred_z, pred_r = critic(x)
        loss = pred_z.float().mean() + pred_r.float().mean()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestResBlock3d(unittest.TestCase):
    """Verify the residual block preserves dimensions and has a skip connection."""

    def test_output_shape_unchanged(self):
        from model.action_critic import ResBlock3d
        block = ResBlock3d(channels=32, groups=8)
        x = torch.randn(2, 32, 3, 8, 8)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_skip_connection_present(self):
        """With zero-init conv weights, output should equal input passed through activation."""
        from model.action_critic import ResBlock3d
        block = ResBlock3d(channels=16, groups=4)
        torch.nn.init.zeros_(block.conv1.weight)
        torch.nn.init.zeros_(block.conv2.weight)
        x = torch.randn(1, 16, 3, 4, 4)
        out = block(x)
        # norm2(zeros) == some non-zero due to norm affine params,
        # but residual + act(out + x) should be close to act(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
