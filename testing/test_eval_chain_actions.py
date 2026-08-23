import pytest
import torch

from utils.eval_chain import frame_actions_to_chunk_actions


def test_frame_actions_to_chunk_actions_supports_long_rollouts():
    actions = torch.arange(1 * 300 * 2, dtype=torch.float32).reshape(1, 300, 2)

    chunked = frame_actions_to_chunk_actions(actions)

    assert chunked.shape == (1, 100, 2)
    torch.testing.assert_close(chunked[:, 0], actions[:, :3].mean(dim=1))
    torch.testing.assert_close(chunked[:, -1], actions[:, -3:].mean(dim=1))


def test_frame_actions_to_chunk_actions_rejects_partial_chunk():
    with pytest.raises(ValueError, match="not divisible"):
        frame_actions_to_chunk_actions(torch.zeros(1, 22, 2))
