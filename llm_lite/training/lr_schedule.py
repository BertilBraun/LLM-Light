from enum import Enum
from math import cos, pi

from llm_lite.config.models import (
    CosineWarmupDecayLearningRateScheduleConfiguration,
    FixedLearningRateScheduleConfiguration,
    LearningRateScheduleConfiguration,
    LinearWarmupDecayLearningRateScheduleConfiguration,
)


class DecayShape(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"


def learning_rate_for_step(
    base_learning_rate: float,
    schedule_configuration: LearningRateScheduleConfiguration,
    step: int,
    maximum_steps: int,
) -> float:
    if step < 1:
        raise ValueError("Learning rate schedule step must be at least 1.")
    if maximum_steps < 1:
        raise ValueError("Learning rate schedule maximum_steps must be at least 1.")
    if step > maximum_steps:
        raise ValueError("Learning rate schedule step must not exceed maximum_steps.")

    match schedule_configuration:
        case FixedLearningRateScheduleConfiguration():
            return base_learning_rate
        case LinearWarmupDecayLearningRateScheduleConfiguration(
            warmup_steps=warmup_steps,
            minimum_learning_rate_ratio=minimum_learning_rate_ratio,
        ):
            schedule_ratio = _warmup_decay_ratio(
                step=step,
                maximum_steps=maximum_steps,
                warmup_steps=warmup_steps,
                minimum_learning_rate_ratio=minimum_learning_rate_ratio,
                decay_shape=DecayShape.LINEAR,
            )
            return base_learning_rate * schedule_ratio
        case CosineWarmupDecayLearningRateScheduleConfiguration(
            warmup_steps=warmup_steps,
            minimum_learning_rate_ratio=minimum_learning_rate_ratio,
        ):
            schedule_ratio = _warmup_decay_ratio(
                step=step,
                maximum_steps=maximum_steps,
                warmup_steps=warmup_steps,
                minimum_learning_rate_ratio=minimum_learning_rate_ratio,
                decay_shape=DecayShape.COSINE,
            )
            return base_learning_rate * schedule_ratio


def _warmup_decay_ratio(
    step: int,
    maximum_steps: int,
    warmup_steps: int,
    minimum_learning_rate_ratio: float,
    decay_shape: DecayShape,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps

    decay_steps = maximum_steps - warmup_steps - 1
    if decay_steps <= 0:
        return 1.0

    decay_step = step - warmup_steps - 1
    decay_progress = decay_step / decay_steps
    match decay_shape:
        case DecayShape.LINEAR:
            decay_multiplier = 1.0 - decay_progress
        case DecayShape.COSINE:
            decay_multiplier = 0.5 * (1.0 + cos(pi * decay_progress))
    return minimum_learning_rate_ratio + (1.0 - minimum_learning_rate_ratio) * decay_multiplier
