from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import (
    FixedPromptGenerationEvaluationConfiguration,
    InferenceConfiguration,
)
from llm_lite.inference.engine import generate_text
from llm_lite.tokenizer.loading import TextTokenizer


class FixedPromptGenerationSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    generated_text: str


class FixedPromptGenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    samples: tuple[FixedPromptGenerationSample, ...]


def evaluate_fixed_prompt_generation(
    model: nn.Module,
    tokenizer: TextTokenizer,
    evaluation_configuration: FixedPromptGenerationEvaluationConfiguration,
    inference_configuration: InferenceConfiguration,
) -> FixedPromptGenerationResult:
    samples = tuple(
        FixedPromptGenerationSample(
            prompt=prompt,
            generated_text=generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                inference_configuration=InferenceConfiguration(
                    engine=inference_configuration.engine,
                    precision=inference_configuration.precision,
                    quantization=inference_configuration.quantization,
                    decoding=inference_configuration.decoding,
                    maximum_new_tokens=evaluation_configuration.maximum_new_tokens,
                ),
            ),
        )
        for prompt in evaluation_configuration.prompts
    )
    return FixedPromptGenerationResult(samples=samples)
