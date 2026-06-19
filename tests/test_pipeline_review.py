from pathlib import Path

import pytest

from llm_lite.pipeline.runner import build_argument_parser, run_pipeline


def test_pipeline_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_pipeline(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
        dry_run=True,
        force_stages=(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "raw_dataset" in captured.out
    assert "pretraining" in captured.out


def test_force_without_stage_defaults_to_raw_dataset() -> None:
    argument_parser = build_argument_parser()

    arguments = argument_parser.parse_args(
        ["--config", "configs/verify_one_sentence.yaml", "--force"],
    )

    assert arguments.force == ["raw_dataset"]


def test_force_with_stage_preserves_stage_choice() -> None:
    argument_parser = build_argument_parser()

    arguments = argument_parser.parse_args(
        ["--config", "configs/verify_one_sentence.yaml", "--force", "pretraining"],
    )

    assert arguments.force == ["pretraining"]
