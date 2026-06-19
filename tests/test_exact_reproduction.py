import torch
from torch import nn

from llm_lite.config.models import (
    ExactReproductionEvaluationConfiguration,
    InferenceConfiguration,
    InferenceEngine,
    Precision,
    QuantizationType,
)
from llm_lite.evaluation.exact_reproduction import evaluate_exact_reproduction
from llm_lite.model.output import ModelOutput
from llm_lite.tokenizer.character import train_character_tokenizer


class DeterministicNextTokenModel(nn.Module):
    def __init__(self, target_token_ids: list[int], vocabulary_size: int) -> None:
        super().__init__()
        self.target_token_ids = target_token_ids
        self.vocabulary_size = vocabulary_size

    def forward(self, token_ids: torch.Tensor) -> ModelOutput:
        batch_size, sequence_length = token_ids.shape
        logits = torch.zeros(batch_size, sequence_length, self.vocabulary_size)
        target_index = min(sequence_length - 1, len(self.target_token_ids) - 1)
        logits[:, -1, self.target_token_ids[target_index]] = 1.0
        return ModelOutput(logits=logits)


def test_exact_reproduction_runs_generation_and_comparison() -> None:
    tokenizer = train_character_tokenizer(
        texts=["hello\n"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    target_token_ids = tokenizer.encode(text="hello\n", add_bos=False, add_eos=True)
    model = DeterministicNextTokenModel(
        target_token_ids=target_token_ids,
        vocabulary_size=tokenizer.vocabulary_size,
    )

    result = evaluate_exact_reproduction(
        model=model,
        tokenizer=tokenizer,
        evaluation_configuration=ExactReproductionEvaluationConfiguration(
            prompt="",
            expected_completion="hello\n",
        ),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            maximum_new_tokens=10,
        ),
    )

    assert result.passed is True
    assert result.generated_text == "hello\n"
    assert result.expected_text == "hello\n"
