#!/usr/bin/env python3
"""Verify the DMD scorers' joint-TF RoPE convention matches the v14
LoRA's training contract.

v14 was trained with:
    Joint-TF input: [clean_half (F frames), noisy_half (F frames)]
    Clean half at RoPE [0, F)
    Noisy half at RoPE [npb, npb + F)
    Total RoPE span = F + npb = 8 chunks (for F=21, npb=3).
    Same chunk index occupies the SAME RoPE position in both halves
    (in the overlap region).

Regression-catches:
  * Reintroducing the ``DIAG_TF_ROPE_OFFSET`` env override or any
    other path that lets ``tf_rope_offset`` drift from
    ``num_frame_per_block`` for the action-forcing scorers.
  * Reintroducing the dual-source ``rope_offset = context_shift *
    num_frame_per_block`` derivation in CausalWanModel.forward when
    ``tf_rope_offset_frames`` is explicitly set.
  * Mis-wiring ``tf_rope_offset_frames`` propagation from model →
    per-block ``self_attn.tf_rope_offset``.

CPU-only. No CUDA, no model weights, no DDP.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _import_rope_helpers():
    with patch("torch.cuda.current_device", return_value=0):
        from wan.modules.model import rope_apply, rope_params

    return rope_apply, rope_params


class TestV14RopeConvention(unittest.TestCase):
    """Lock the v14 LoRA's training-time RoPE convention into the
    DMD scorers and catch any drift via static + behavioural tests.
    """

    F = 21        # num_training_frames
    NPB = 3       # num_frame_per_block (= tf_rope_offset)
    H = 4         # latent grid H
    W = 6         # latent grid W
    NUM_HEADS = 4
    HEAD_DIM = 16    # must be divisible by 6 (RoPE splits 3 ways, even per chunk)

    # ------------------------------------------------------------------
    # Static checks: the trainer hardcodes the right value.
    #
    # NOTE: these static checks use literal substring matches against
    # production source. They are brittle — a clarity-only reword of
    # the asserted comments / strings will fail this test even though
    # the semantics are unchanged. That's intentional for a regression
    # guard against intentional reverts of the cleanup, not subtle
    # bugs. Update both source AND test together when rewording.
    # ------------------------------------------------------------------
    def test_trainer_hardcodes_tf_rope_offset_to_npb(self):
        """``model/dmd_action_forcing.py`` no longer reads any env var
        for ``_tf_rope_off``; the value comes from
        ``self.num_frame_per_block`` and is asserted == 3.
        """
        path = Path(__file__).resolve().parents[1] / "model" / "dmd_action_forcing.py"
        src = path.read_text()
        # Old env-override path is gone.
        self.assertNotIn(
            "DIAG_TF_ROPE_OFFSET", src,
            "DIAG_TF_ROPE_OFFSET env override must not be reintroduced — "
            "v14's tf_rope_offset is hardcoded to num_frame_per_block.",
        )
        # New path hardcodes from npb and asserts the v14 contract.
        self.assertIn(
            "_tf_rope_off = int(self.num_frame_per_block)", src,
            "Trainer must derive _tf_rope_off from self.num_frame_per_block.",
        )
        self.assertIn(
            "v14 LoRA was trained with num_frame_per_block=3", src,
            "Trainer must assert num_frame_per_block == 3 (v14 contract).",
        )

    def test_action_patch_comment_describes_npb_offset(self):
        """The action-aware bidir patch's comment must describe the
        actual offset (= num_frame_per_block, = 1-chunk shift) — not
        the obsolete ``dmd_context_clean_frames`` (= cf, = 9 frames).
        """
        path = Path(__file__).resolve().parents[1] / "model" / "action_model_patch.py"
        src = path.read_text()
        self.assertNotIn(
            "tf_rope_offset_frames = dmd_context_clean_frames", src,
            "action_model_patch.py comment must not describe the obsolete "
            "cf-frame offset; v14's actual offset = num_frame_per_block.",
        )
        self.assertIn(
            "tf_rope_offset_frames = num_frame_per_block", src,
            "action_model_patch.py comment must reflect the actual offset.",
        )

    # ------------------------------------------------------------------
    # Behavioural checks: rope_apply + temporal_offset semantics.
    # ------------------------------------------------------------------
    def _build_freqs(self):
        """Build the same RoPE freqs tensor the model uses, for a
        head_dim that splits cleanly 3 ways.

        Total split: ``[head_dim_half - 2*(head_dim_half//3),
                       head_dim_half//3, head_dim_half//3]``.
        """
        _, rope_params = _import_rope_helpers()
        d = self.HEAD_DIM
        # Match the model's split: ``d - 4*(d//6)``, ``2*(d//6)``,
        # ``2*(d//6)``. With HEAD_DIM=16: 16 - 4*2 = 8, 4, 4. Sum = 16.
        # rope_params expects an even dim; each piece must be even.
        max_seq = 64  # > F + npb
        return torch.cat([
            rope_params(max_seq, d - 4 * (d // 6)),
            rope_params(max_seq, 2 * (d // 6)),
            rope_params(max_seq, 2 * (d // 6)),
        ], dim=1)  # [max_seq, head_dim_half = HEAD_DIM // 2 ... wait]
        # rope_params returns [max_seq, dim/2] complex. Concat across last → [max_seq, head_dim_half_total].

    def _make_x_per_frame(self, content_per_frame):
        """Build x = [B=1, seq_len = F*H*W, num_heads, head_dim].

        ``content_per_frame``: list of length F, each is the (real)
        scalar value to fill that frame's tokens with. Returns x with
        identifiable per-frame content so we can index back after
        rotation.
        """
        F_, H_, W_ = self.F, self.H, self.W
        nh, hd = self.NUM_HEADS, self.HEAD_DIM
        # Two reals per complex → fill imaginary part = 0, real part = content.
        x = torch.zeros(1, F_ * H_ * W_, nh, hd)
        for f_idx, val in enumerate(content_per_frame):
            start = f_idx * H_ * W_
            end = start + H_ * W_
            # Set the REAL part of every complex pair (= even indices)
            x[0, start:end, :, 0::2] = float(val)
        return x

    def test_temporal_offset_shifts_rope_position(self):
        """``rope_apply(x, gs, freqs, temporal_offset=N)`` rotates
        frame i at RoPE position i+N.

        Concretely: clone identical content into clean half (offset=0)
        and noisy half (offset=npb). The noisy half's frame i sees
        the same RoPE rotation as the clean half's frame i+npb.
        """
        rope_apply, _ = _import_rope_helpers()
        freqs = self._build_freqs()

        F_, H_, W_, npb = self.F, self.H, self.W, self.NPB

        # IDENTICAL uniform content in both halves so the only thing
        # that can differ between clean[frame=i+npb] and noisy[frame=i]
        # is the RoPE rotation applied. v14's contract: those positions
        # share the absolute RoPE index (i+npb), so the rotated outputs
        # must be bit-equal.
        content = [1.0] * F_
        x_clean = self._make_x_per_frame(content)
        x_noisy = self._make_x_per_frame(content)

        gs = torch.tensor([[F_, H_, W_]], dtype=torch.long)

        clean_rotated = rope_apply(x_clean, gs, freqs, temporal_offset=0)
        noisy_rotated = rope_apply(x_noisy, gs, freqs, temporal_offset=npb)

        # For each chunk K in the overlap region (k=0..F/npb-1 chunks
        # of the noisy half map to k+1..F/npb in absolute RoPE units;
        # they overlap with the clean-half chunks at index k+1):
        #   noisy_chunk[K] (RoPE pos K+npb) == clean_chunk[K+1] (RoPE pos K+1·npb if K+1 chunks fit)
        # More cleanly per FRAME: for any frame i in [0, F-npb):
        #   noisy[frame=i] (RoPE pos i+npb) == clean[frame=i+npb] (RoPE pos i+npb)
        for i in range(F_ - npb):
            clean_frame_at_i_plus_npb = clean_rotated[
                0, (i + npb) * H_ * W_ : (i + npb + 1) * H_ * W_
            ]
            noisy_frame_at_i = noisy_rotated[
                0, i * H_ * W_ : (i + 1) * H_ * W_
            ]
            torch.testing.assert_close(
                clean_frame_at_i_plus_npb,
                noisy_frame_at_i,
                rtol=1e-5, atol=1e-5,
                msg=(
                    f"v14 RoPE convention violated: noisy[frame={i}] "
                    f"and clean[frame={i + npb}] should share RoPE "
                    f"position {i + npb} and produce identical rotations "
                    f"on identical content."
                ),
            )

    def test_total_rope_span_is_F_plus_npb(self):
        """Joint-TF input spans RoPE [0, F+npb) frames. Verify the
        clean half ends at F-1 and the noisy half ends at F+npb-1.
        """
        rope_apply, _ = _import_rope_helpers()
        freqs = self._build_freqs()
        F_, H_, W_, npb = self.F, self.H, self.W, self.NPB

        # Clean half last frame is at RoPE position F-1.
        # Noisy half last frame is at RoPE position F-1 + npb = F+npb-1.
        # Build single-frame inputs at those positions and verify they
        # rotate the same value differently — i.e. RoPE position F-1
        # ≠ F+npb-1 (so the span is non-degenerate).
        single = torch.zeros(1, H_ * W_, self.NUM_HEADS, self.HEAD_DIM)
        single[..., 0::2] = 1.0
        gs1 = torch.tensor([[1, H_, W_]], dtype=torch.long)

        rot_at_F_minus_1 = rope_apply(single, gs1, freqs, temporal_offset=F_ - 1)
        rot_at_F_plus_npb_minus_1 = rope_apply(single, gs1, freqs, temporal_offset=F_ + npb - 1)

        # The two rotations must differ — otherwise the noisy half's
        # last chunk collides onto the clean half's last chunk.
        diff = (rot_at_F_minus_1 - rot_at_F_plus_npb_minus_1).abs().max().item()
        self.assertGreater(
            diff, 1e-3,
            f"RoPE position F-1 and F+npb-1 produce nearly identical "
            f"rotations (max-abs-diff={diff:.2e}) — the F+npb-frame "
            f"span has collapsed.",
        )

        # And npb must be exactly 3 in the v14 contract.
        self.assertEqual(npb, 3, "v14 LoRA training contract: npb=3.")

    # ------------------------------------------------------------------
    # Behavioural check: causal_model.py respects tf_rope_offset_frames.
    # ------------------------------------------------------------------
    def test_causal_model_reads_tf_rope_offset_frames(self):
        """The single-source-of-truth read in
        ``CausalWanModel.forward`` should use
        ``self.tf_rope_offset_frames`` when explicitly set, falling
        back to ``context_shift * num_frame_per_block`` only when
        unset (= 0).

        Static check: ensure the dual-source ``rope_offset =
        self.context_shift * self.num_frame_per_block`` derivation
        without the explicit-attr branch is gone.
        """
        path = (
            Path(__file__).resolve().parents[1] / "wan" / "modules" / "causal_model.py"
        )
        src = path.read_text()
        # Old single-line dual-source must be gone — the new version
        # branches on tf_rope_offset_frames first.
        self.assertNotIn(
            "rope_offset = self.context_shift * self.num_frame_per_block "
            "if clean_x is not None else 0",
            src,
            "causal_model.py must not derive rope_offset solely from "
            "context_shift * num_frame_per_block; the trainer-set "
            "tf_rope_offset_frames is the single source of truth.",
        )
        # New version must reference tf_rope_offset_frames with a
        # ``None`` sentinel so an explicit ``0`` (no shift) is
        # distinguishable from "not set" and isn't silently
        # overridden by the derivation.
        self.assertIn(
            'getattr(self, "tf_rope_offset_frames", None)',
            src,
            "causal_model.py must read tf_rope_offset_frames with a "
            "``None`` sentinel — using ``0`` would conflate "
            "'unset' with 'explicitly zero'.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
