from dataclasses import dataclass

from torch import nn

from llm_lite.model.gpt import DenseGpt
from llm_lite.model.modern import ModernDenseGpt, ModernMoeGpt
from llm_lite.model.moe import MoeGpt


@dataclass(frozen=True)
class ModelParameterSummary:
    total_parameters: int
    trainable_parameters: int
    active_parameters: int
    trainable_active_parameters: int


def model_parameter_summary(model: nn.Module) -> ModelParameterSummary:
    match model:
        case DenseGpt():
            total_parameters = _parameter_count(model=model)
            trainable_parameters = _trainable_parameter_count(model=model)
            return ModelParameterSummary(
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                active_parameters=total_parameters,
                trainable_active_parameters=trainable_parameters,
            )
        case MoeGpt():
            return ModelParameterSummary(
                total_parameters=_parameter_count(model=model),
                trainable_parameters=_trainable_parameter_count(model=model),
                active_parameters=model.active_parameter_count(),
                trainable_active_parameters=model.trainable_active_parameter_count(),
            )
        case ModernDenseGpt():
            total_parameters = _parameter_count(model=model)
            trainable_parameters = _trainable_parameter_count(model=model)
            return ModelParameterSummary(
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                active_parameters=total_parameters,
                trainable_active_parameters=trainable_parameters,
            )
        case ModernMoeGpt():
            return ModelParameterSummary(
                total_parameters=_parameter_count(model=model),
                trainable_parameters=_trainable_parameter_count(model=model),
                active_parameters=model.active_parameter_count(),
                trainable_active_parameters=model.trainable_active_parameter_count(),
            )
        case _:
            total_parameters = _parameter_count(model=model)
            trainable_parameters = _trainable_parameter_count(model=model)
            return ModelParameterSummary(
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                active_parameters=total_parameters,
                trainable_active_parameters=trainable_parameters,
            )


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
