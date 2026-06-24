from torch import nn

from llm_lite.config.models import DenseGptConfiguration, ModelConfiguration, MoeGptConfiguration
from llm_lite.model.gpt import DenseGpt
from llm_lite.model.moe import MoeGpt


def build_model(
    model_configuration: ModelConfiguration,
    vocabulary_size: int,
) -> nn.Module:
    match model_configuration:
        case DenseGptConfiguration():
            return DenseGpt(
                model_configuration=model_configuration,
                vocabulary_size=vocabulary_size,
            )
        case MoeGptConfiguration():
            return MoeGpt(
                model_configuration=model_configuration,
                vocabulary_size=vocabulary_size,
            )
