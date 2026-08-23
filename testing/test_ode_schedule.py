import pytest

from af_utils.schedule import resolve_denoising_step_list


def test_14e_20_step_snapshot_vocabulary_resolves_real_rungs():
    timesteps = resolve_denoising_step_list(
        usable_stored_indices=[0, 1, 2, 3],
        num_inference_steps=20,
        shift=5.0,
        snapshot_steps=[0, 15, 18, 19, -1],
    )

    assert timesteps.tolist() == pytest.approx(
        [1000.0, 625.0, 357.142857, 208.333333], rel=1e-5,
    )


def test_schedule_rejects_snapshot_from_wrong_grid():
    with pytest.raises(ValueError, match="outside the 20-step scheduler"):
        resolve_denoising_step_list(
            usable_stored_indices=[0, 1],
            num_inference_steps=20,
            shift=5.0,
            snapshot_steps=[0, 36, -1],
        )


def test_schedule_rejects_implicit_negative_index():
    with pytest.raises(ValueError, match="only -1"):
        resolve_denoising_step_list(
            usable_stored_indices=[0],
            num_inference_steps=20,
            snapshot_steps=[-2],
        )
