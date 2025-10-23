import json
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ARRWM.utils import pre_encode_text


class DummyEncoder:
    instances = []

    def __init__(self):
        self.called = False
        DummyEncoder.instances.append(self)

    def __call__(self, text_prompts):
        self.called = True
        raise AssertionError("WanTextEncoder should not be invoked during dry run")


class RecordingEncoder:
    instances = []

    def __init__(self):
        self.calls = []
        RecordingEncoder.instances.append(self)

    def __call__(self, text_prompts):
        self.calls.append(list(text_prompts))
        embedding = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]]
        ])
        return {"prompt_embeds": embedding}


def test_dry_run_reports_and_skips_encoding(monkeypatch, tmp_path, caplog):
    caption_dir = tmp_path / "captions"
    caption_dir.mkdir()
    caption_file = caption_dir / "sample.json"
    caption_file.write_text(
        json.dumps(
            {
                "combined_analysis": "A sample caption that should be augmented.",
                "chunks": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(pre_encode_text, "WanTextEncoder", DummyEncoder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pre_encode_text.py",
            "--source-dir",
            str(tmp_path),
            "--pattern",
            "*.json",
            "--dry-run",
            "--num-workers",
            "1",
            "--workers-per-device",
            "1",
        ],
    )

    DummyEncoder.instances.clear()

    with caplog.at_level(logging.INFO):
        pre_encode_text.main()

    assert DummyEncoder.instances, "The text encoder should have been instantiated"
    assert not DummyEncoder.instances[0].called, "Dry run must skip actual encoding"
    assert not list(caption_dir.glob("*_encoded.json")), "Dry run should not write outputs"

    messages = [record.getMessage() for record in caplog.records]
    assert any("Would encode" in msg for msg in messages), "Dry run activity was not reported"
    assert any("Processed 1 file(s)" in msg for msg in messages), "Summary log missing processed count"


def test_encodes_and_writes_expected_embedding(monkeypatch, tmp_path):
    caption_dir = tmp_path / "captions"
    caption_dir.mkdir()
    caption_file = caption_dir / "sample.json"
    caption_file.write_text(
        json.dumps(
            {
                "combined_analysis": "A second caption.",
                "chunks": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(pre_encode_text, "WanTextEncoder", RecordingEncoder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pre_encode_text.py",
            "--source-dir",
            str(tmp_path),
            "--pattern",
            "*.json",
            "--num-workers",
            "1",
            "--workers-per-device",
            "1",
        ],
    )

    RecordingEncoder.instances.clear()

    pre_encode_text.main()

    assert RecordingEncoder.instances, "The encoder should be created"
    encoder_instance = RecordingEncoder.instances[0]
    assert encoder_instance.calls == [
        ["A second caption. " + pre_encode_text.STYLE_SENTENCE]
    ], "Caption text did not include the style suffix"

    output_files = list(caption_dir.glob("*sample_encoded.json"))
    assert output_files, "Encoded output JSON was not created"
    encoded_payload = json.loads(output_files[0].read_text())
    assert list(encoded_payload.keys()) == ["caption_encoded"], "Unexpected JSON fields"
    assert encoded_payload["caption_encoded"] == [[1.0, 2.0], [3.0, 4.0]], "Embedding content mismatch"


def test_negative_prompt_encoding(monkeypatch, tmp_path):
    configs_dir = tmp_path / "configs"
    negative_dir = tmp_path / "negative"
    configs_dir.mkdir()
    negative_dir.mkdir()

    (configs_dir / "alpha.yaml").write_text("negative_prompt: first negative\n")
    (configs_dir / "beta.yaml").write_text("negative_prompt: second negative\n")

    monkeypatch.setattr(pre_encode_text, "CONFIG_DIR", configs_dir)
    monkeypatch.setattr(pre_encode_text, "DEFAULT_NEGATIVE_ROOT", negative_dir)
    monkeypatch.setattr(pre_encode_text, "WanTextEncoder", RecordingEncoder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pre_encode_text.py",
            "--negative",
        ],
    )

    RecordingEncoder.instances.clear()

    pre_encode_text.main()

    assert RecordingEncoder.instances, "Negative prompt should instantiate the encoder"
    encoder_instance = RecordingEncoder.instances[0]
    assert encoder_instance.calls == [["first negative"], ["second negative"]], "Negative prompt calls mismatch"

    expected_files = {
        negative_dir / "alpha_negative_1.json",
        negative_dir / "beta_negative_1.json",
    }
    for file_path in expected_files:
        assert file_path.is_file(), f"Negative prompt embedding missing at {file_path}"
        payload = json.loads(file_path.read_text())
        assert payload["caption_encoded"] == [[1.0, 2.0], [3.0, 4.0]], "Negative prompt embedding mismatch"
