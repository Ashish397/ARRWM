import pytest
import torch

from utils.eval_causal_AR import _apply_carn_seam_affine


def test_carn_seam_affine_matches_blended_target_statistics():
    torch.manual_seed(0)
    chunk = torch.randn(2, 3, 4, 5, 6, dtype=torch.float16)
    target_mean = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    target_std = torch.tensor([0.4, 0.7, 1.1, 1.4])

    result = _apply_carn_seam_affine(
        chunk, target_mean, target_std, strength=0.5
    ).float()
    chunk_float = chunk.float()
    expected_mean = 0.5 * (
        target_mean + chunk_float.mean(dim=(0, 1, 3, 4))
    )
    expected_std = 0.5 * (
        target_std + chunk_float.std(dim=(0, 1, 3, 4))
    )

    torch.testing.assert_close(
        result.mean(dim=(0, 1, 3, 4)), expected_mean, atol=5e-4, rtol=5e-4
    )
    torch.testing.assert_close(
        result.std(dim=(0, 1, 3, 4)), expected_std, atol=5e-4, rtol=5e-4
    )


def test_carn_seam_affine_zero_is_identity():
    chunk = torch.randn(1, 3, 4, 2, 2)
    result = _apply_carn_seam_affine(
        chunk, torch.zeros(4), torch.ones(4), strength=0.0
    )

    assert result is chunk


def test_carn_seam_affine_rejects_invalid_strength():
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _apply_carn_seam_affine(
            torch.randn(1, 3, 4, 2, 2),
            torch.zeros(4),
            torch.ones(4),
            strength=1.1,
        )
