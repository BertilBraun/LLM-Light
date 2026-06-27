import json
from pathlib import Path

import torch

from llm_lite.pipeline.runner import run_pipeline
from tests.artifact_helpers import stage_artifact_directory


def test_pipeline_creates_new_pretraining_artifact_for_longer_training(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "extend_training"
    first_configuration_path = tmp_path / "extend_training_first.yaml"
    second_configuration_path = tmp_path / "extend_training_second.yaml"
    base_configuration_text = Path("configs/verify_one_sentence.yaml").read_text(
        encoding="utf-8",
    )
    first_configuration_text = _training_extension_configuration_text(
        configuration_text=base_configuration_text,
        run_directory=run_directory,
        maximum_steps=4,
    )
    second_configuration_text = _training_extension_configuration_text(
        configuration_text=base_configuration_text,
        run_directory=run_directory,
        maximum_steps=6,
    )
    first_configuration_path.write_text(first_configuration_text, encoding="utf-8")
    second_configuration_path.write_text(second_configuration_text, encoding="utf-8")

    first_exit_code = run_pipeline(
        configuration_path=first_configuration_path,
        dry_run=False,
        force_stages=(),
    )
    first_pretraining_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="pretraining",
    )
    first_pretraining_manifest = json.loads(
        (first_pretraining_artifact_directory / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    second_exit_code = run_pipeline(
        configuration_path=second_configuration_path,
        dry_run=False,
        force_stages=(),
    )
    second_pretraining_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="pretraining",
    )
    second_pretraining_manifest = json.loads(
        (second_pretraining_artifact_directory / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_pretraining_artifact_directory != second_pretraining_artifact_directory
    assert first_pretraining_manifest["fingerprint"] != second_pretraining_manifest["fingerprint"]
    assert first_pretraining_manifest["metrics"]["final_step"] == 4
    assert second_pretraining_manifest["metrics"]["final_step"] == 6
    assert second_pretraining_manifest["metrics"]["resumed_from_step"] == 0
    assert second_pretraining_manifest["metrics"]["requested_maximum_steps"] == 6


def test_pipeline_creates_new_pretraining_artifact_for_training_configuration_changes(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "resume_training_configuration"
    first_configuration_path = tmp_path / "resume_training_first.yaml"
    second_configuration_path = tmp_path / "resume_training_second.yaml"
    base_configuration_text = Path("configs/verify_one_sentence.yaml").read_text(
        encoding="utf-8",
    )
    first_configuration_text = _training_extension_configuration_text(
        configuration_text=base_configuration_text,
        run_directory=run_directory,
        maximum_steps=4,
    )
    second_configuration_text = (
        _training_extension_configuration_text(
            configuration_text=base_configuration_text,
            run_directory=run_directory,
            maximum_steps=6,
        )
        .replace("learning_rate: 0.05", "learning_rate: 0.01")
        .replace("weight_decay: 0.0", "weight_decay: 0.001")
        .replace("gradient_clip_norm: 1.0", "gradient_clip_norm: 0.5")
        .replace("pin_memory: false", "pin_memory: true")
    )
    first_configuration_path.write_text(first_configuration_text, encoding="utf-8")
    second_configuration_path.write_text(second_configuration_text, encoding="utf-8")

    first_exit_code = run_pipeline(
        configuration_path=first_configuration_path,
        dry_run=False,
        force_stages=(),
    )
    first_pretraining_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="pretraining",
    )
    first_pretraining_manifest = json.loads(
        (first_pretraining_artifact_directory / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    second_exit_code = run_pipeline(
        configuration_path=second_configuration_path,
        dry_run=False,
        force_stages=(),
    )
    second_pretraining_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="pretraining",
    )
    second_pretraining_manifest = json.loads(
        (second_pretraining_artifact_directory / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    checkpoint_data = torch.load(
        second_pretraining_artifact_directory / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_pretraining_artifact_directory != second_pretraining_artifact_directory
    assert first_pretraining_manifest["fingerprint"] != second_pretraining_manifest["fingerprint"]
    assert first_pretraining_manifest["metrics"]["final_step"] == 4
    assert second_pretraining_manifest["metrics"]["final_step"] == 6
    assert second_pretraining_manifest["metrics"]["resumed_from_step"] == 0
    assert checkpoint_data["optimizer"]["param_groups"][0]["lr"] == 0.01
    assert checkpoint_data["optimizer"]["param_groups"][0]["weight_decay"] == 0.001


def test_pipeline_creates_new_artifact_when_interrupted_pretraining_configuration_changes(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "resume_interrupted_pretraining"
    first_configuration_path = tmp_path / "resume_interrupted_first.yaml"
    second_configuration_path = tmp_path / "resume_interrupted_second.yaml"
    base_configuration_text = Path("configs/verify_one_sentence.yaml").read_text(
        encoding="utf-8",
    )
    first_configuration_text = _training_extension_configuration_text(
        configuration_text=base_configuration_text,
        run_directory=run_directory,
        maximum_steps=4,
    )
    second_configuration_text = _training_extension_configuration_text(
        configuration_text=base_configuration_text,
        run_directory=run_directory,
        maximum_steps=6,
    ).replace("batch_size_sequences: 1", "batch_size_sequences: 2")
    first_configuration_path.write_text(first_configuration_text, encoding="utf-8")
    second_configuration_path.write_text(second_configuration_text, encoding="utf-8")

    first_exit_code = run_pipeline(
        configuration_path=first_configuration_path,
        dry_run=False,
        force_stages=(),
    )
    first_pretraining_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="pretraining",
    )
    first_pretraining_manifest_path = first_pretraining_artifact_directory / "manifest.json"
    first_pretraining_manifest = json.loads(
        first_pretraining_manifest_path.read_text(encoding="utf-8"),
    )
    first_pretraining_manifest["status"] = "running"
    first_pretraining_manifest_path.write_text(
        json.dumps(first_pretraining_manifest, indent=2),
        encoding="utf-8",
    )
    second_exit_code = run_pipeline(
        configuration_path=second_configuration_path,
        dry_run=False,
        force_stages=(),
    )
    preserved_interrupted_manifest = json.loads(
        first_pretraining_manifest_path.read_text(encoding="utf-8"),
    )
    second_pretraining_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="pretraining",
    )
    second_pretraining_manifest = json.loads(
        (second_pretraining_artifact_directory / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_pretraining_artifact_directory != second_pretraining_artifact_directory
    assert preserved_interrupted_manifest["status"] == "running"
    assert second_pretraining_manifest["metrics"]["final_step"] == 6
    assert second_pretraining_manifest["metrics"]["resumed_from_step"] == 0


def test_pipeline_recovers_interrupted_pretraining_at_requested_step(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "recover_interrupted_pretraining"
    configuration_path = tmp_path / "recover_interrupted.yaml"
    base_configuration_text = Path("configs/verify_one_sentence.yaml").read_text(
        encoding="utf-8",
    )
    configuration_text = _training_extension_configuration_text(
        configuration_text=base_configuration_text,
        run_directory=run_directory,
        maximum_steps=4,
    )
    configuration_path.write_text(configuration_text, encoding="utf-8")

    first_exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
    )
    pretraining_artifact_directory = stage_artifact_directory(
        run_directory=run_directory,
        stage_name="pretraining",
    )
    pretraining_manifest_path = pretraining_artifact_directory / "manifest.json"
    pretraining_manifest = json.loads(pretraining_manifest_path.read_text(encoding="utf-8"))
    pretraining_manifest["status"] = "running"
    pretraining_manifest_path.write_text(
        json.dumps(pretraining_manifest, indent=2),
        encoding="utf-8",
    )
    second_exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
    )
    recovered_manifest = json.loads(pretraining_manifest_path.read_text(encoding="utf-8"))

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert recovered_manifest["status"] == "complete"
    assert recovered_manifest["metrics"]["final_step"] == 4
    assert recovered_manifest["metrics"]["resumed_from_step"] == 4
    assert (
        stage_artifact_directory(
            run_directory=run_directory,
            stage_name="pretraining",
        )
        == pretraining_artifact_directory
    )


def _training_extension_configuration_text(
    configuration_text: str,
    run_directory: Path,
    maximum_steps: int,
) -> str:
    exact_reproduction_evaluation = (
        "evaluation:\n"
        "  exact_reproduction:\n"
        '    prompt: ""\n'
        '    expected_completion: "hello world\\n"'
    )
    return (
        configuration_text.replace(
            "output_dir: runs/verify_one_sentence",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        )
        .replace("maximum_steps: 60", f"maximum_steps: {maximum_steps}")
        .replace("checkpoint_interval_steps: 10", "checkpoint_interval_steps: 2")
        .replace("log_interval_steps: 5", "log_interval_steps: 2")
        .replace(exact_reproduction_evaluation, "evaluation: {}")
    )
