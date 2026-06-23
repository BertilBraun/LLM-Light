from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import (
    ExactReproductionEvaluationConfiguration,
    InferenceConfiguration,
)
from llm_lite.inference.naive import generate_greedy
from llm_lite.tokenizer.loading import TextTokenizer


class ExactReproductionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    generated_text: str
    expected_text: str


def evaluate_exact_reproduction(
    model: nn.Module,
    tokenizer: TextTokenizer,
    evaluation_configuration: ExactReproductionEvaluationConfiguration,
    inference_configuration: InferenceConfiguration,
) -> ExactReproductionResult:
    generated_text = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        prompt=evaluation_configuration.prompt,
        maximum_new_tokens=inference_configuration.maximum_new_tokens,
    )
    expected_text = evaluation_configuration.prompt + evaluation_configuration.expected_completion
    return ExactReproductionResult(
        passed=generated_text == expected_text,
        generated_text=generated_text,
        expected_text=expected_text,
    )
