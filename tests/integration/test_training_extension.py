import json
from pathlib import Path

from llm_lite.pipeline.runner import run_pipeline


def test_pipeline_extends_compatible_pretraining(tmp_path: Path) -> None:
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
    second_exit_code = run_pipeline(
        configuration_path=second_configuration_path,
        dry_run=False,
        force_stages=(),
    )
    pretraining_manifest = json.loads(
        (run_directory / "artifacts" / "pretraining" / "manifest.json").read_text(
            encoding="utf-8",
        ),
    )

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert pretraining_manifest["metrics"]["final_step"] == 6
    assert pretraining_manifest["metrics"]["resumed_from_step"] == 4
    assert pretraining_manifest["metrics"]["requested_maximum_steps"] == 6


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
