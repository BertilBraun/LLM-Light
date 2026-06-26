from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from llm_lite.evaluation.tensorboard import write_evaluation_metrics_to_tensorboard
from llm_lite.pipeline.tensorboard import RUN_TENSORBOARD_DIRECTORY_ENVIRONMENT


def test_evaluation_metrics_write_tensorboard_scalars(tmp_path: Path) -> None:
    write_evaluation_metrics_to_tensorboard(
        tensorboard_directory=tmp_path / "tensorboard",
        metrics={
            "perplexity": 1.5,
            "perplexity_loss": 0.4,
            "python_completion_pass_rate": 0.75,
            "python_completion_passed_checks": 3,
            "python_completion_family_string_parsing_concrete_pass_rate": 0.5,
            "non_numeric": "skip",
        },
        step=42,
    )

    tensorboard_events = EventAccumulator(str(tmp_path / "tensorboard"))
    tensorboard_events.Reload()

    assert set(tensorboard_events.Tags()["scalars"]) == {
        "eval/perplexity",
        "eval/perplexity_loss",
        "eval/python_completion/pass_rate",
        "eval/python_completion/passed_checks",
        "eval/python_completion/family/string_parsing_concrete/pass_rate",
    }
    assert tensorboard_events.Scalars("eval/python_completion/pass_rate")[0].step == 42
    assert tensorboard_events.Scalars("eval/python_completion/pass_rate")[0].value == 0.75


def test_evaluation_metrics_write_run_view_tensorboard_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tensorboard_directory = tmp_path / "run" / "tensorboard" / "evaluation"
    monkeypatch.setenv(RUN_TENSORBOARD_DIRECTORY_ENVIRONMENT, str(run_tensorboard_directory))

    write_evaluation_metrics_to_tensorboard(
        tensorboard_directory=tmp_path / "artifact" / "tensorboard",
        metrics={"perplexity": 1.5},
        step=3,
    )

    tensorboard_events = EventAccumulator(str(run_tensorboard_directory))
    tensorboard_events.Reload()

    assert tensorboard_events.Scalars("eval/perplexity")[0].step == 3
