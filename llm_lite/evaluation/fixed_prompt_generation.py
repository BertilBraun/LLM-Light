from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import (
    FixedPromptGenerationEvaluationConfiguration,
    InferenceConfiguration,
)
from llm_lite.inference.engine import generate_batch
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
    generation_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=evaluation_configuration.prompts,
        inference_configuration=InferenceConfiguration(
            engine=inference_configuration.engine,
            precision=inference_configuration.precision,
            quantization=inference_configuration.quantization,
            decoding=inference_configuration.decoding,
            maximum_new_tokens=evaluation_configuration.maximum_new_tokens,
            batch_size=inference_configuration.batch_size,
            stop_sequences=inference_configuration.stop_sequences,
        ),
    )
    samples = tuple(
        FixedPromptGenerationSample(
            prompt=generation_result.prompt,
            generated_text=generation_result.full_text,
        )
        for generation_result in generation_results
    )
    return FixedPromptGenerationResult(samples=samples)
