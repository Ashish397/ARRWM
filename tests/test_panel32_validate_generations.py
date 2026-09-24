import pytest

from grids.eval.panel32_validate_generations import (
    ValidationError,
    sidecar_fields,
    validate_native_contract,
)


def test_sidecar_fields_accepts_yume_scene_prompt_names():
    payload = {
        "window": "ego4d-bristol-01",
        "action": "F",
        "sampling_seed": 17,
        "panel_manifest_sha256": "manifest",
        "source_boundary_normalized_rgb24_sha256": "boundary",
        "scene_prompt": "A first-person view of an outdoor environment.",
        "scene_prompt_sha256": "prompt-sha",
    }

    fields = sidecar_fields(payload)

    assert fields["prompt"] == payload["scene_prompt"]
    assert fields["prompt_sha"] == payload["scene_prompt_sha256"]


def test_panel32_nested_provenance_takes_precedence():
    payload = {
        "window": "wrong-context",
        "prompt": "wrong prompt",
        "panel32": {
            "context_id": "sekai-walk-01",
            "action": "L",
            "sampling_seed": 23,
            "panel_manifest_sha256": "manifest",
            "canonical_boundary_rgb24_sha256": "boundary",
            "neutral_prompt": "A first-person view of an outdoor environment.",
            "neutral_prompt_sha256": "prompt-sha",
        },
    }

    fields = sidecar_fields(payload)

    assert fields["context_id"] == "sekai-walk-01"
    assert fields["prompt"] == payload["panel32"]["neutral_prompt"]
    assert fields["prompt_sha"] == payload["panel32"]["neutral_prompt_sha256"]


@pytest.mark.parametrize(
    ("model", "context_frames", "disk_action", "payload"),
    (
        ("minwm", 29, "L", {
            "stage": "dmd", "config": "causal_forcing_dmd_camera.yaml",
            "seed_start_frame": 4, "seed_frames": 29, "seed_latents": 8,
            "total_latents": 128, "generated_latents": 120,
            "generation_boundary_real_frame": 32, "direction": "L",
            "yaw_adapter": "released_minwm_yaw_negated_to_common_convention",
        }),
        ("minwm_ode", 13, "NOOP", {
            "stage": "ode", "config": "causal_ode_camera.yaml",
            "seed_start_frame": 20, "seed_frames": 13, "seed_latents": 4,
            "total_latents": 124, "generated_latents": 120,
            "generation_boundary_real_frame": 32, "direction": "NOOP",
            "yaw_adapter": "released_minwm_yaw_negated_to_common_convention",
        }),
    ),
)
def test_minwm_native_contracts(model, context_frames, disk_action, payload):
    validate_native_contract(model, payload, context_frames, disk_action)


def test_minwm_native_contract_rejects_unadapted_yaw():
    payload = {
        "stage": "dmd", "config": "causal_forcing_dmd_camera.yaml",
        "seed_start_frame": 4, "seed_frames": 29, "seed_latents": 8,
        "total_latents": 128, "generated_latents": 120,
        "generation_boundary_real_frame": 32, "direction": "R",
        "yaw_adapter": "released_native_unadapted",
    }
    with pytest.raises(ValidationError, match="corrected generation-time yaw adapter"):
        validate_native_contract("minwm", payload, 29, "R")


def test_minwm_native_contract_rejects_stale_diagonal_pose_order():
    payload = {
        "stage": "dmd", "config": "causal_forcing_dmd_camera.yaml",
        "seed_start_frame": 4, "seed_frames": 29, "seed_latents": 8,
        "total_latents": 128, "generated_latents": 120,
        "generation_boundary_real_frame": 32, "direction": "FL",
        "yaw_adapter": "released_minwm_yaw_negated_to_common_convention",
    }
    with pytest.raises(ValidationError, match="diagonal pose update"):
        validate_native_contract("minwm", payload, 29, "FL")


def matrix_payload(direction="R"):
    return {
        "model": "Matrix-Game-2.0",
        "inference_pipeline": "vendor CausalInferencePipeline",
        "mode": "universal",
        "native_latent_frames_per_block": 3,
        "keyboard_dims": 4,
        "camera_yaw_per_frame": 0.1,
        "latent_frames": 189,
        "frames": 753,
        "fps": 25,
        "generation_boundary_real_frame": 32,
        "sampling_seed": 0,
        "held_action_equivalence": (
            "constant pixel-frame tensors equal repeated vendor streaming "
            "actions for every three-latent-frame block"
        ),
        "direction": direction,
        "checkpoint_path": "/vendor/base_distilled_model/base_distill.safetensors",
    }


def test_matrixgame_native_contract_accepts_released_seed_zero_runner():
    validate_native_contract("matrixgame2", matrix_payload(), 1, "R")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("sampling_seed", 1234),
        ("mode", "interactive"),
        ("native_latent_frames_per_block", 4),
        ("keyboard_dims", 3),
        ("camera_yaw_per_frame", -0.1),
    ),
)
def test_matrixgame_native_contract_rejects_mixed_legacy_outputs(field, bad_value):
    payload = matrix_payload()
    payload[field] = bad_value
    with pytest.raises(ValidationError, match=field):
        validate_native_contract("matrixgame2", payload, 1, "R")
