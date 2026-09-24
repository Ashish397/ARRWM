from argparse import Namespace
import hashlib
import json
from pathlib import Path

import torch

from grids.eval import ode_panel32_runner as runner
from interactive.engine_api import PHYSICAL_NULL_STEER, PHYSICAL_NULL_THROTTLE


ROOT = Path(__file__).resolve().parents[1]


def _dry_run_contract(tmp_path: Path) -> Namespace:
    datasets = (
        ("FrodoBots", "frodobots"),
        ("Ego4D", "ego4d"),
        ("Sekai", "sekai"),
        ("SpatialVID", "spatialvid"),
    )
    contexts = []
    for dataset, prefix in datasets:
        for index in range(8):
            context_id = f"{prefix}-route_alpha_{index:03d}"
            contexts.append({
                "context_id": context_id,
                "dataset": dataset,
                "source_id": f"{prefix}-source-{index:03d}",
                "source_video": f"not-needed/{context_id}.mp4",
                "source_sha256": hashlib.sha256(context_id.encode()).hexdigest(),
                "source_uri": f"https://example.invalid/{context_id}.mp4",
                "license": "synthetic-test",
                "outdoor_verified": True,
                "outdoor_verification_note": "synthetic outdoor scene",
                "boundary": {"frame_index": 32},
            })
    manifest = tmp_path / "panel.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "panel_id": "synthetic-ode-panel32",
        "selection_protocol": "unit test",
        "created_utc": "2026-09-23T00:00:00Z",
        "dataset_counts": {name: 8 for name, _ in datasets},
        "actions": list(runner.ACTION_ORDER),
        "canonical_source": {
            "frame_count": 33, "fps": 20, "width": 832, "height": 480,
            "resize_mode": "center_crop", "boundary_frame_index": 32,
            "wan_latent_frames": 9,
        },
        "contexts": contexts,
    }))
    manifest_hash = runner.sha256_file(manifest)
    provenance = tmp_path / "source.json"
    provenance.write_text(json.dumps({
        "status": "pass", "complete_panel": True,
        "panel_id": "synthetic-ode-panel32", "manifest_sha256": manifest_hash,
        "rows": [{"context_id": row["context_id"]} for row in contexts],
    }))
    prompt = tmp_path / "neutral_prompt.pt"
    torch.save({
        "prompt": runner.PROMPT,
        "prompt_embeds": torch.zeros(1, 512, 4096, dtype=torch.float16),
    }, prompt)
    bundles = tmp_path / "bundles.json"
    bundles.write_text(json.dumps({
        "status": "pass", "contexts": 32,
        "panel_id": "synthetic-ode-panel32",
        "panel_manifest_sha256": manifest_hash,
        "source_provenance_sha256": runner.sha256_file(provenance),
        "seed_pixel_frames": 33, "seed_latent_frames": 9,
        "seed_action_spans_half_open": [[0, 9], [9, 21], [21, 33]],
        "neutral_prompt": runner.PROMPT,
        "neutral_prompt_sha256": runner.prompt_sha256(),
        "neutral_prompt_embedding": str(prompt),
        "neutral_prompt_embedding_sha256": runner.sha256_file(prompt),
        "rows": [
            {"context_id": row["context_id"], "dataset": row["dataset"]}
            for row in contexts
        ],
    }))
    checkpoint = tmp_path / "student.pt"
    teacher = tmp_path / "teacher.pt"
    config = tmp_path / "eval.yaml"
    checkpoint.write_bytes(b"student")
    teacher.write_bytes(b"teacher")
    config.write_text("config_name: test\n")
    wan_root = tmp_path / "wan"
    (wan_root / "Wan2.1-T2V-1.3B").mkdir(parents=True)
    return Namespace(
        panel_manifest=manifest,
        source_provenance=provenance,
        bundle_index=bundles,
        checkpoint=checkpoint,
        teacher_checkpoint=teacher,
        config=config,
        wan_model_root=wan_root,
        output=tmp_path / "must-not-exist",
        model_id="ode_local_kl",
        objective="local_kl",
        action="F",
        shard_index=0,
        shard_count=4,
        base_seed=1234,
        dry_run=True,
    )


def test_locked_action_order_geometry_and_physical_noop() -> None:
    assert runner.ACTION_ORDER == ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
    assert tuple(runner.ACTION_VECTORS) == runner.ACTION_ORDER
    assert runner.ACTION_VECTORS["F"] == (0.5, 0.0)
    assert runner.ACTION_VECTORS["R"] == (0.0, 0.5)
    assert runner.ACTION_VECTORS["L"] == (0.0, -0.5)
    assert runner.ACTION_VECTORS["N"] == (
        PHYSICAL_NULL_THROTTLE,
        PHYSICAL_NULL_STEER,
    )
    assert runner.ACTION_VECTORS["N"] != (0.0, 0.0)


def test_action_tensor_retains_three_seed_actions_then_holds_command() -> None:
    seed_actions = torch.tensor([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]])
    actions = runner.build_action_tensor(seed_actions, "FL")
    assert tuple(actions.shape) == (1, 129, 2)
    for chunk in range(3):
        expected = seed_actions[chunk].expand(3, 2)
        assert torch.equal(actions[0, chunk * 3:(chunk + 1) * 3], expected)
    held = torch.tensor(runner.ACTION_VECTORS["FL"])
    assert torch.equal(actions[0, 9:], held.expand(120, 2))
    assert runner.GENERATED_CHUNKS == 40
    assert runner.GENERATED_PIXEL_FRAMES == 480
    assert runner.TOTAL_PIXEL_FRAMES == 513


def test_seed_is_full_context_only_and_filename_preserves_identity() -> None:
    alpha = "ego4d-route_alpha_007"
    beta = "ego4d-route_beta_007"
    assert runner.sampling_seed(alpha) == runner.sampling_seed(alpha)
    assert runner.sampling_seed(alpha) != runner.sampling_seed(beta)
    # No action or model is accepted by sampling_seed, so every branch for a
    # context necessarily starts from the same stochastic stream.
    assert runner.stable_path(Path("out"), "ode_local_kl", alpha, "FR") == Path(
        "out/ode_local_kl_ego4d-route_alpha_007_FR.mp4"
    )


def test_prompt_and_temporal_contract_are_locked() -> None:
    assert runner.PROMPT == "A first-person view of an outdoor environment."
    assert runner.SEED_PIXEL_FRAMES == 33
    assert runner.SEED_LATENT_FRAMES == 9
    assert runner.TOTAL_LATENT_FRAMES == 129
    assert runner.DENOISING_RUNGS == (1000.0, 625.0, 357.142857, 208.333333)
    assert runner.CACHE_CHUNKS == 6


def test_isambard_payload_maps_both_exact_step400_checkpoints_and_four_workers() -> None:
    payload = (ROOT / "sbatch/u6qf/ode_panel32_holder_action.sh").read_text()
    assert (
        "logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt"
        in payload
    )
    assert (
        "logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000400.pt"
        in payload
    )
    assert "for gpu in 0 1 2 3" in payload
    assert 'CONTEXT_SHARD_COUNT=${PANEL32_CONTEXT_SHARD_COUNT:-4}' in payload
    assert 'export CUDA_VISIBLE_DEVICES="$gpu"' in payload
    assert '--shard-count "$CONTEXT_SHARD_COUNT"' in payload
    assert "torch.distributed.run" not in payload


def test_ode_pipeline_has_portable_generation_only_build_hooks() -> None:
    source = (ROOT / "utils/eval_causal_AR.py").read_text()
    assert "teacher_checkpoint: Optional[str] = None" in source
    assert "wan_model_root: Optional[str] = None" in source
    assert "build_evaluators: bool = True" in source
    assert "if build_evaluators:" in source


def test_dry_run_loads_exact_bundle_prompt_without_writing(tmp_path: Path) -> None:
    args = _dry_run_contract(tmp_path)
    runner.run(args)
    assert not args.output.exists()

    payload = json.loads(args.bundle_index.read_text())
    payload["neutral_prompt"] = "A different prompt."
    args.bundle_index.write_text(json.dumps(payload))
    try:
        runner.run(args)
    except runner.ContractError as exc:
        assert "prompt text mismatch" in str(exc)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("changed bundle prompt was accepted")
