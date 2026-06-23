from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import FixedPromptGenerationEvaluationConfiguration
from llm_lite.inference.naive import generate_greedy
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
) -> FixedPromptGenerationResult:
    samples = tuple(
        FixedPromptGenerationSample(
            prompt=prompt,
            generated_text=generate_greedy(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                maximum_new_tokens=evaluation_configuration.maximum_new_tokens,
            ),
        )
        for prompt in evaluation_configuration.prompts
    )
    return FixedPromptGenerationResult(samples=samples)
