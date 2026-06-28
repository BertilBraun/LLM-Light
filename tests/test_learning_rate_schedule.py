from pathlib import Path

import pytest

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    CosineWarmupDecayLearningRateScheduleConfiguration,
    LearningRateScheduleType,
    LinearWarmupDecayLearningRateScheduleConfiguration,
    OptimizerConfiguration,
    TrainingConfiguration,
)
from llm_lite.training.lr_schedule import learning_rate_for_step


@pytest.mark.parametrize(
    ("step", "expected_learning_rate"),
    (
        (1, 0.1),
        (5, 0.1),
        (10, 0.1),
    ),
)
def test_fixed_learning_rate_schedule(
    step: int,
    expected_learning_rate: float,
) -> None:
    training_configuration = TrainingConfiguration(
        maximum_steps=10,
        batch_size_sequences=1,
    )

    learning_rate = learning_rate_for_step(
        base_learning_rate=0.1,
        schedule_configuration=training_configuration.optimizer.learning_rate_schedule,
        step=step,
        maximum_steps=training_configuration.maximum_steps,
    )

    assert learning_rate == pytest.approx(expected_learning_rate)


@pytest.mark.parametrize(
    ("step", "expected_learning_rate"),
    (
        (1, 0.05),
        (2, 0.1),
        (3, 0.1),
        (6, 0.06142857142857142),
        (10, 0.01),
    ),
)
def test_linear_warmup_decay_learning_rate_schedule(
    step: int,
    expected_learning_rate: float,
) -> None:
    learning_rate = learning_rate_for_step(
        base_learning_rate=0.1,
        schedule_configuration=LinearWarmupDecayLearningRateScheduleConfiguration(
            type=LearningRateScheduleType.LINEAR_WARMUP_DECAY,
            warmup_steps=2,
            minimum_learning_rate_ratio=0.1,
        ),
        step=step,
        maximum_steps=10,
    )

    assert learning_rate == pytest.approx(expected_learning_rate)


@pytest.mark.parametrize(
    ("step", "expected_learning_rate"),
    (
        (1, 0.05),
        (2, 0.1),
        (3, 0.1),
        (6, 0.06501344202803415),
        (10, 0.01),
    ),
)
def test_cosine_warmup_decay_learning_rate_schedule(
    step: int,
    expected_learning_rate: float,
) -> None:
    learning_rate = learning_rate_for_step(
        base_learning_rate=0.1,
        schedule_configuration=CosineWarmupDecayLearningRateScheduleConfiguration(
            type=LearningRateScheduleType.COSINE_WARMUP_DECAY,
            warmup_steps=2,
            minimum_learning_rate_ratio=0.1,
        ),
        step=step,
        maximum_steps=10,
    )

    assert learning_rate == pytest.approx(expected_learning_rate)


def test_learning_rate_schedule_loads_from_yaml(tmp_path: Path) -> None:
    configuration_path = tmp_path / "schedule_config.yaml"
    configuration_path.write_text(
        "\n".join(
            (
                "experiment:",
                "  name: schedule_config",
                "  output_dir: runs/schedule_config",
                "dataset:",
                "  type: inline_text",
                "  documents: ['hello world']",
                "tokenizer:",
                "  type: character",
                "packing:",
                "  context_length: 8",
                "model:",
                "  type: dense_gpt",
                "  dimension: 8",
                "  layers: 1",
                "  attention_heads: 1",
                "  feed_forward_dimension: 16",
                "training:",
                "  maximum_steps: 10",
                "  batch_size_sequences: 1",
                "  optimizer:",
                "    learning_rate: 0.1",
                "    learning_rate_schedule:",
                "      type: cosine_warmup_decay",
                "      warmup_steps: 2",
                "      minimum_learning_rate_ratio: 0.2",
            ),
        ),
        encoding="utf-8",
    )

    experiment_configuration = load_experiment_configuration(configuration_path=configuration_path)

    assert (
        experiment_configuration.training.optimizer.learning_rate_schedule.type
        is LearningRateScheduleType.COSINE_WARMUP_DECAY
    )


def test_learning_rate_warmup_must_fit_training_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfiguration(
            maximum_steps=2,
            batch_size_sequences=1,
            optimizer=OptimizerConfiguration(
                learning_rate=0.1,
                learning_rate_schedule=LinearWarmupDecayLearningRateScheduleConfiguration(
                    type=LearningRateScheduleType.LINEAR_WARMUP_DECAY,
                    warmup_steps=2,
                ),
            ),
        )
