from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from grids.eval import yume5b_panel32_runner as runner
from grids.eval import panel32_model_inputs as shared
from grids.eval.panel32_manifest import sha256_file


DATASETS = ("FrodoBots", "Ego4D", "Sekai", "SpatialVID")
PREFIX = {
    "FrodoBots": "frodobots",
    "Ego4D": "ego4d",
    "Sekai": "sekai",
    "SpatialVID": "spatialvid",
}


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


def _rgb(path: Path, frame: int, width: int, height: int) -> bytes:
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
        "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])


def _make_contract(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    canonical = tmp_path / "source33.avi"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "color=c=0x42688f:size=832x480:rate=20", "-frames:v", "33",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", str(canonical),
    ], check=True)
    boundary = tmp_path / "source33_f32.png"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", str(canonical),
        "-vf", "select=eq(n\\,32)", "-frames:v", "1", str(boundary),
    ], check=True)
    canonical_rgb = subprocess.check_output([
        "ffmpeg", "-v", "error", "-threads", "1", "-i", str(canonical),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    boundary_rgb = _rgb(boundary, 0, 832, 480)

    contexts = []
    for dataset in DATASETS:
        for index in range(8):
            # Underscores and numeric suffixes catch the historical split-on-
            # suffix bug. The full dataset-prefixed ID is the only identity.
            context_id = f"{PREFIX[dataset]}-route_alpha_{index:03d}"
            contexts.append({
                "context_id": context_id,
                "dataset": dataset,
                "source_id": f"{PREFIX[dataset]}-source-{index:03d}",
                "source_video": f"unneeded/{context_id}.mp4",
                "source_sha256": hashlib.sha256(context_id.encode()).hexdigest(),
                "source_uri": f"https://example.invalid/{context_id}.mp4",
                "license": "synthetic-test-only",
                "outdoor_verified": True,
                "outdoor_verification_note": "synthetic outdoor test scene",
                "boundary": {"frame_index": 64},
            })
    manifest_payload = {
        "schema_version": 1,
        "panel_id": "synthetic-mixed-panel32-v1",
        "selection_protocol": "synthetic unit-test panel",
        "created_utc": "2026-09-23T00:00:00Z",
        "dataset_counts": {dataset: 8 for dataset in DATASETS},
        "actions": list(runner.ACTIONS),
        "canonical_source": {
            "frame_count": 33, "fps": 20, "width": 832, "height": 480,
            "resize_mode": "center_crop", "boundary_frame_index": 32,
            "wan_latent_frames": 9,
        },
        "contexts": contexts,
    }
    manifest = tmp_path / "panel32.json"
    manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    manifest_sha = sha256_file(manifest)
    rows = []
    for context in contexts:
        rows.append({
            "schema_version": 1,
            "panel_id": manifest_payload["panel_id"],
            "panel_manifest": str(manifest),
            "panel_manifest_sha256": manifest_sha,
            "context_id": context["context_id"],
            "dataset": context["dataset"],
            "source_id": context["source_id"],
            "source": {"sha256": context["source_sha256"]},
            "boundary": {
                "source_frame_index": 64,
                "source_timestamp_s": 3.2,
                "original_rgb24_sha256": "1" * 64,
                "normalized_rgb24_sha256": hashlib.sha256(boundary_rgb).hexdigest(),
                "canonical_frame_index": 32,
            },
            "canonical": {
                "stream": str(canonical),
                "stream_sha256": sha256_file(canonical),
                "decoded_rgb24_sha256": hashlib.sha256(canonical_rgb).hexdigest(),
                "boundary_png": str(boundary),
                "boundary_png_sha256": sha256_file(boundary),
                "frame_count": 33, "fps": 20.0, "width": 832, "height": 480,
            },
            "wan_interface": {"pixel_frames": 33, "latent_frames": 9},
        })
    provenance_payload = {
        "status": "pass",
        "panel_id": manifest_payload["panel_id"],
        "manifest_sha256": manifest_sha,
        "complete_panel": True,
        "extracted_contexts": 32,
        "rows": rows,
    }
    provenance = tmp_path / "panel32_source_provenance.json"
    provenance.write_text(json.dumps(provenance_payload, indent=2) + "\n")

    yume_root = tmp_path / "YUME"
    (yume_root / "fastvideo/sample").mkdir(parents=True)
    (yume_root / "fastvideo/utils").mkdir(parents=True)
    (yume_root / "fastvideo/sample/sample_5b.py").write_text("# test\n")
    (yume_root / "fastvideo/utils/video_reader.py").write_text("# test\n")
    model = tmp_path / "Yume-5B-720P"
    model.mkdir()
    for name in (
        "diffusion_pytorch_model.safetensors", "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth", "config.json",
    ):
        (model / name).write_bytes((name + "\n").encode())
    return manifest, provenance, yume_root, model


def _args(
    tmp_path: Path, manifest: Path, provenance: Path,
    yume_root: Path, model: Path,
) -> Namespace:
    return Namespace(
        action="FL", panel_manifest=manifest, source_provenance=provenance,
        repo_root=Path(__file__).resolve().parents[1], remote_root=tmp_path / "remote",
        yume_root=yume_root, model_dir=model, stage=tmp_path / "work",
        output=tmp_path / "outputs", scene_prompt=runner.DEFAULT_PROMPT,
        base_seed=runner.BASE_SEED,
    )


def test_action_contract_and_full_context_identity() -> None:
    assert runner.DEFAULT_PROMPT == "A first-person view of an outdoor environment."
    assert tuple(runner.ACTION_SPEC) == runner.ACTIONS
    assert runner.ACTION_SPEC["F"] == {
        "keys": "W", "mouse": "·", "distance": 4, "turn": 0, "rotation": 0,
    }
    assert runner.ACTION_SPEC["L"]["mouse"] == "←"
    assert runner.ACTION_SPEC["R"]["mouse"] == "→"
    first = "ego4d-route_alpha_007"
    second = "ego4d-route_beta_007"
    assert runner.context_token(first) != runner.context_token(second)
    assert runner.effective_seed(first) == runner.effective_seed(first)
    assert runner.stable_path(
        Namespace(output=Path("out"), action="FL"), first
    ) == Path("out/yume5b_ego4d-route_alpha_007_FL.mp4")


def test_dry_run_is_read_only_and_prepare_stages_explicit_map(tmp_path: Path) -> None:
    manifest, provenance, yume_root, model = _make_contract(tmp_path)
    args = _args(tmp_path, manifest, provenance, yume_root, model)
    runner.dry_run(args)
    assert not args.stage.exists()
    assert not args.output.exists()

    runner.prepare(args)
    staged = json.loads((args.stage / "manifest_FL.json").read_text())
    assert len(staged["contexts"]) == 32
    assert len(staged["work_items"]) == 32
    assert all(not item["padding"] for item in staged["work_items"])
    mapping = {item["vendor_token"]: item["context_id"] for item in staged["work_items"]}
    assert set(mapping.values()) == {
        row["context_id"] for row in staged["contexts"]
    }
    assert "ego4d-route_alpha_007" in mapping.values()
    for item in staged["work_items"]:
        link = Path(staged["input_root"]) / f"{item['vendor_token']}.png"
        assert link.is_symlink()
        assert link.resolve() == Path(
            next(row for row in staged["contexts"]
                 if row["context_id"] == item["context_id"])["canonical_boundary_png"]
        )
    seeds = {row["context_id"]: row["effective_seed"] for row in staged["contexts"]}
    assert len(set(seeds.values())) == 32

    changed = _args(tmp_path, manifest, provenance, yume_root, model)
    changed.scene_prompt = "A different prompt that must be rejected."
    with pytest.raises(runner.ContractError, match="scene prompt must be exactly"):
        runner.dry_run(changed)


def test_package_has_one_context_plus_exactly_thirty_seconds(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    generated = tmp_path / "generated.mp4"
    destination = tmp_path / "stable.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "color=c=0x335577:size=96x64:rate=1", "-frames:v", "1", str(source),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=96x64:rate=16", "-frames:v", "493", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(generated),
    ], check=True)
    runner.package_video(source, generated, destination, "synthetic-panel")
    info = runner.probe(destination)
    assert info["frames"] == 481
    assert info["fps"] == 16
    assert "canonical_frame=32" in info["tags"]["comment"]
    assert "generation_start_video_frame=1" in info["tags"]["comment"]
    runner.read_rgb(destination, 0, width=96, height=64)
    runner.read_rgb(destination, 1, width=96, height=64)
    runner.read_rgb(destination, 480, width=96, height=64)


def test_vendor_caption_has_no_hidden_city_or_person_prefix() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor = (root / "third_party/YUME/fastvideo/sample/sample_5b.py").read_text()
    holder = (root / "sbatch/u6qf/yume5b_panel32_holder_action.sh").read_text()
    assert "This video depicts a city walk" not in vendor
    assert "Person moves" not in vendor
    assert "Person stands" not in vendor
    assert 'os.environ.get("YUME_ACTION_PREFIX", "")' in vendor
    assert 'export YUME_ACTION_PREFIX=""' in holder
    assert (
        "PROMPT=${YUME_PANEL_SCENE_PROMPT:-A first-person view of an outdoor environment.}"
        in holder
    )


def test_shared_external_model_plans_preserve_ids_and_provenance(tmp_path: Path) -> None:
    manifest, provenance, _, _ = _make_contract(tmp_path)
    stage = tmp_path / "shared-inputs"
    output = tmp_path / "outputs"
    expected = {
        "lingbot": ([32, 32], 1, None),
        "dreamx": ([32, 32], 1, None),
        "matrixgame2": ([32, 32], 1, None),
        "minwm": ([4, 32], 29, 8),
        "minwm_ode": ([20, 32], 13, 4),
        "yume5b": ([32, 32], 1, None),
    }
    plans = {}
    for model, contract in expected.items():
        plan = shared.build_plan(
            manifest, provenance, model, "N", stage, output
        )
        plans[model] = plan
        assert len(plan["items"]) == 32
        assert plan["neutral_prompt"] == runner.DEFAULT_PROMPT
        assert (
            plan["input_contract"]["source_frames_inclusive"],
            plan["input_contract"]["pixel_frames"],
            plan["input_contract"]["latent_frames"],
        ) == contract
        row = next(
            item for item in plan["items"]
            if item["context_id"] == "ego4d-route_alpha_007"
        )
        assert row["seed_match_group"] == "ego4d-route_alpha_007"
        assert row["source_video_sha256"]
        assert row["canonical_stream_sha256"]
        assert row["canonical_boundary_png_sha256"]
        assert "ego4d-route_alpha_007" in row["stable_output"]
        assert row["disk_action"] == (
            "NOOP" if model in {"matrixgame2", "minwm", "minwm_ode"} else "N"
        )
        env = plan["runner_environment"]
        assert env["EVAL_REAL_FRAME"] == "32"
        if model == "yume5b":
            assert env["YUME_PANEL_ACTION"] == "N"
        else:
            windows = next(
                value for key, value in env.items() if key.endswith("_WINDOWS")
            )
            assert "ego4d-route_alpha_007" in windows
        if model != "matrixgame2":
            prompt_values = [value for key, value in env.items() if key.endswith("_CAP")]
            if model == "yume5b":
                prompt_values = [env["YUME_PANEL_SCENE_PROMPT"]]
            assert prompt_values == [runner.DEFAULT_PROMPT]

    minwm_env = plans["minwm"]["runner_environment"]
    assert minwm_env["MW_SEED_START"] == "4"
    assert minwm_env["MW_SEED_LAT"] == "8"
    assert minwm_env["MW_NUMLAT"] == "128"
    assert minwm_env["MW_TAG"] == "_aligned32"
    assert minwm_env["MW_CPU_T5"] == "1"
    assert minwm_env["MW_CHUNK_DECODE"] == "8"

    minwm_ode_env = plans["minwm_ode"]["runner_environment"]
    assert minwm_ode_env["MW_SEED_START"] == "20"
    assert minwm_ode_env["MW_SEED_LAT"] == "4"
    assert minwm_ode_env["MW_NUMLAT"] == "124"
    assert minwm_ode_env["MW_TAG"] == "_ode"
    assert minwm_ode_env["MW_STAGE"] == "ode"

    yume_row = next(
        item for item in plans["yume5b"]["items"]
        if item["context_id"] == "ego4d-route_alpha_007"
    )
    assert yume_row["sampling_seed"] == runner.effective_seed(
        "ego4d-route_alpha_007"
    )
    forward = shared.build_plan(
        manifest, provenance, "dreamx", "F", stage, output
    )
    forward_seeds = {item["context_id"]: item["sampling_seed"] for item in forward["items"]}
    noop_seeds = {
        item["context_id"]: item["sampling_seed"]
        for item in plans["dreamx"]["items"]
    }
    assert forward_seeds == noop_seeds

    shared.materialize(plans["minwm"], stage)
    assert (stage / "frame32/seed65_ego4d-route_alpha_007_f0.png").is_symlink()
    assert (stage / "streams/seed65_ego4d-route_alpha_007.avi").is_symlink()
    written = json.loads((stage / "plan_minwm_N.json").read_text())
    assert written["items"] == plans["minwm"]["items"]

    sidecar = tmp_path / "candidate.mp4.json"
    sidecar.write_text(json.dumps({"model": "test-model"}) + "\n")
    minwm_row = next(
        item for item in plans["minwm"]["items"]
        if item["context_id"] == "spatialvid-route_alpha_007"
    )
    shared.attach_sidecar_provenance(plans["minwm"], minwm_row, sidecar)
    attached = json.loads(sidecar.read_text())
    assert attached["model"] == "test-model"
    assert attached["panel32"] == shared.sidecar_provenance(
        plans["minwm"], minwm_row
    )
    assert attached["panel32"]["context_id"] == (
        "spatialvid-route_alpha_007"
    )
    assert attached["panel32"]["model_input_source_frames_inclusive"] == [4, 32]
    assert attached["panel32"]["model_input_latent_frames"] == 8
    shared.attach_sidecar_provenance(plans["minwm"], minwm_row, sidecar)

    attached["panel32"]["generation_boundary_canonical_frame"] = 31
    sidecar.write_text(json.dumps(attached) + "\n")
    with pytest.raises(shared.PlanError, match="conflicting existing panel32"):
        shared.attach_sidecar_provenance(plans["minwm"], minwm_row, sidecar)

    with pytest.raises(shared.PlanError, match="prompt must be exactly"):
        shared.build_plan(
            manifest, provenance, "lingbot", "F", stage, output,
            prompt="A city street.",
        )
