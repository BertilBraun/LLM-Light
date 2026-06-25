from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from llm_lite.evaluation.tensorboard import write_evaluation_metrics_to_tensorboard


def test_evaluation_metrics_write_tensorboard_scalars(tmp_path: Path) -> None:
    write_evaluation_metrics_to_tensorboard(
        tensorboard_directory=tmp_path / "tensorboard",
        metrics={
            "perplexity": 1.5,
            "perplexity_loss": 0.4,
            "python_completion_pass_rate": 0.75,
            "python_completion_passed_checks": 3,
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
    }
    assert tensorboard_events.Scalars("eval/python_completion/pass_rate")[0].step == 42
    assert tensorboard_events.Scalars("eval/python_completion/pass_rate")[0].value == 0.75
