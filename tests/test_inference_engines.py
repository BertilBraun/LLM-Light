from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from llm_lite.config.models import (
    DecodingStrategy,
    DenseGptConfiguration,
    GenerationStopReason,
    GreedyDecodingConfiguration,
    InferenceConfiguration,
    InferenceEngine,
    ModelType,
    MoeGptConfiguration,
    Precision,
    QuantizationType,
    SamplingDecodingConfiguration,
)
from llm_lite.inference.candidates import (
    CandidateGenerationResult,
    CandidatePrompt,
    GeneratedCandidateRecord,
    generate_candidates,
    write_candidate_jsonl,
)
from llm_lite.inference.decoding import select_next_token_id
from llm_lite.inference.engine import generate_batch, generate_text
from llm_lite.inference.kv_cache import generate_greedy as generate_greedy_with_cache
from llm_lite.inference.naive import generate_greedy as generate_greedy_naively
from llm_lite.inference.runtime import (
    MutableGenerationState,
    append_next_token,
    prepare_model_for_inference,
)
from llm_lite.model.gpt import DenseGpt
from llm_lite.model.moe import MoeGpt
from llm_lite.tokenizer.character import train_character_tokenizer


def test_cached_forward_matches_full_sequence_forward() -> None:
    torch.manual_seed(7)
    tokenizer = train_character_tokenizer(
        texts=["hello world"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)
    prompt_token_ids = tokenizer.encode(text="hello", add_bos=True, add_eos=False)
    continuation_token_ids = tokenizer.encode(text=" world", add_bos=False, add_eos=False)

    full_token_ids = torch.tensor([prompt_token_ids + continuation_token_ids], dtype=torch.long)
    full_model_output = model(full_token_ids)
    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long)
    cached_model_output = model.forward_with_cache(
        token_ids=prompt_tensor,
        inference_cache=model.empty_inference_cache(
            batch_size=prompt_tensor.shape[0],
            device=prompt_tensor.device,
        ),
    )
    for continuation_token_id in continuation_token_ids:
        cached_model_output = model.forward_with_cache(
            token_ids=torch.tensor([[continuation_token_id]], dtype=torch.long),
            inference_cache=cached_model_output.inference_cache,
        )

    torch.testing.assert_close(
        cached_model_output.logits[:, -1, :],
        full_model_output.logits[:, -1, :],
        rtol=0.00001,
        atol=0.00001,
    )


def test_moe_cached_forward_matches_full_sequence_forward() -> None:
    torch.manual_seed(8)
    tokenizer = train_character_tokenizer(
        texts=["hello world"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_moe_model(vocabulary_size=tokenizer.vocabulary_size)
    prompt_token_ids = tokenizer.encode(text="hello", add_bos=True, add_eos=False)
    continuation_token_ids = tokenizer.encode(text=" world", add_bos=False, add_eos=False)

    full_token_ids = torch.tensor([prompt_token_ids + continuation_token_ids], dtype=torch.long)
    full_model_output = model(full_token_ids)
    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long)
    cached_model_output = model.forward_with_cache(
        token_ids=prompt_tensor,
        inference_cache=model.empty_inference_cache(
            batch_size=prompt_tensor.shape[0],
            device=prompt_tensor.device,
        ),
    )
    for continuation_token_id in continuation_token_ids:
        cached_model_output = model.forward_with_cache(
            token_ids=torch.tensor([[continuation_token_id]], dtype=torch.long),
            inference_cache=cached_model_output.inference_cache,
        )

    torch.testing.assert_close(
        cached_model_output.logits[:, -1, :],
        full_model_output.logits[:, -1, :],
        rtol=0.00001,
        atol=0.00001,
    )


def test_kv_cache_generation_matches_naive_generation() -> None:
    torch.manual_seed(11)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)

    naive_text = generate_greedy_naively(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        maximum_new_tokens=5,
    )
    cached_text = generate_greedy_with_cache(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        maximum_new_tokens=5,
    )

    assert cached_text == naive_text


def test_moe_kv_cache_generation_matches_naive_generation() -> None:
    torch.manual_seed(12)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_moe_model(vocabulary_size=tokenizer.vocabulary_size)

    naive_text = generate_greedy_naively(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        maximum_new_tokens=5,
    )
    cached_text = generate_greedy_with_cache(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        maximum_new_tokens=5,
    )

    assert cached_text == naive_text


def test_generate_text_uses_configured_kv_cache_engine() -> None:
    torch.manual_seed(13)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.KV_CACHE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=5,
        ),
    )

    assert generated_text == generate_greedy_naively(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        maximum_new_tokens=5,
    )


def test_generate_text_uses_configured_kv_cache_engine_for_moe() -> None:
    torch.manual_seed(14)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_moe_model(vocabulary_size=tokenizer.vocabulary_size)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.KV_CACHE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=5,
        ),
    )

    assert generated_text == generate_greedy_naively(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        maximum_new_tokens=5,
    )


def test_moe_kv_cache_generation_supports_bfloat16_precision() -> None:
    torch.manual_seed(15)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_moe_model(vocabulary_size=tokenizer.vocabulary_size)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.KV_CACHE,
            precision=Precision.BF16,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=2,
        ),
    )
    inference_cache = model.empty_inference_cache(batch_size=1, device=torch.device("cpu"))

    assert isinstance(generated_text, str)
    assert next(model.parameters()).dtype == torch.bfloat16
    assert inference_cache.layers[0].key_states.dtype == torch.bfloat16
    assert inference_cache.layers[0].value_states.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_moe_kv_cache_generation_uses_model_device_on_cuda() -> None:
    torch.manual_seed(16)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_moe_model(vocabulary_size=tokenizer.vocabulary_size).to("cuda")

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.KV_CACHE,
            precision=Precision.BF16,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=2,
        ),
    )

    assert isinstance(generated_text, str)


def test_generate_text_samples_with_configured_decoding() -> None:
    torch.manual_seed(17)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)
    with torch.no_grad():
        model.output_projection.weight.zero_()

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=SamplingDecodingConfiguration(
                strategy=DecodingStrategy.SAMPLE,
                temperature=1.0,
                top_k=None,
            ),
            maximum_new_tokens=3,
        ),
    )

    assert isinstance(generated_text, str)


def test_batched_greedy_generation_matches_single_prompt_generation() -> None:
    torch.manual_seed(23)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)
    inference_configuration = InferenceConfiguration(
        engine=InferenceEngine.NAIVE,
        precision=Precision.FP32,
        quantization=QuantizationType.NONE,
        decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
        maximum_new_tokens=4,
        batch_size=2,
    )

    batch_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=("a", "b"),
        inference_configuration=inference_configuration,
    )

    assert tuple(result.full_text for result in batch_results) == tuple(
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            inference_configuration=inference_configuration,
        )
        for prompt in ("a", "b")
    )


def test_batched_sampling_returns_output_contract() -> None:
    torch.manual_seed(29)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)
    with torch.no_grad():
        model.output_projection.weight.zero_()

    batch_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=("a", "b", "c"),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=SamplingDecodingConfiguration(
                strategy=DecodingStrategy.SAMPLE,
                temperature=1.0,
                top_k=None,
            ),
            maximum_new_tokens=2,
            batch_size=3,
        ),
    )

    assert len(batch_results) == 3
    assert all(result.prompt_length > 0 for result in batch_results)
    assert all(result.generated_token_count <= 2 for result in batch_results)


def test_variable_length_prompt_batching_is_supported() -> None:
    torch.manual_seed(31)
    tokenizer = train_character_tokenizer(
        texts=["abcd"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)

    batch_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=("a", "abc"),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.KV_CACHE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=3,
            batch_size=2,
        ),
    )

    assert len(batch_results) == 2
    assert batch_results[0].prompt_length != batch_results[1].prompt_length
    assert all(isinstance(result.full_text, str) for result in batch_results)


def test_batched_kv_cache_generation_matches_naive_generation() -> None:
    torch.manual_seed(37)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)

    naive_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=("a", "b"),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=4,
            batch_size=2,
        ),
    )
    cached_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=("a", "b"),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.KV_CACHE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=4,
            batch_size=2,
        ),
    )

    assert tuple(result.full_text for result in cached_results) == tuple(
        result.full_text for result in naive_results
    )


def test_stop_sequences_end_batched_generation() -> None:
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = StepTokenModel(
        vocabulary_size=tokenizer.vocabulary_size,
        token_ids=(
            tokenizer.encode(text="b", add_bos=False, add_eos=False)[0],
            tokenizer.encode(text="c", add_bos=False, add_eos=False)[0],
        ),
    )

    batch_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=("a",),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=5,
            batch_size=1,
            stop_sequences=("bc",),
        ),
    )

    assert batch_results[0].generated_text == ""
    assert batch_results[0].stop_reason is GenerationStopReason.STOP_SEQUENCE


def test_append_next_token_skips_decode_when_no_stop_sequences() -> None:
    tokenizer = CountingTokenizer()
    state = MutableGenerationState(
        prompt="a",
        prompt_token_ids=(1,),
        generated_token_ids=[],
        stopped=False,
        stop_reason=None,
    )

    next_state = append_next_token(
        state=state,
        next_token_id=2,
        tokenizer=tokenizer,
        stop_sequences=(),
    )

    assert next_state.generated_token_ids == [2]
    assert not next_state.stopped
    assert tokenizer.decode_calls == 0


def test_inference_metrics_are_recorded() -> None:
    torch.manual_seed(41)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)

    generation_result = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=("a",),
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.KV_CACHE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=2,
            batch_size=1,
        ),
    )[0]

    assert generation_result.timing.total_seconds >= 0.0
    assert generation_result.throughput.tokens_per_second >= 0.0
    assert generation_result.throughput.sequences_per_second >= 0.0


def test_candidate_generation_writes_jsonl_artifact(tmp_path) -> None:
    torch.manual_seed(43)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)
    candidate_generation_result = generate_candidates(
        model=model,
        tokenizer=tokenizer,
        candidate_prompts=(
            CandidatePrompt(task_id="task-1", prompt="a"),
            CandidatePrompt(task_id="task-2", prompt="b"),
        ),
        samples_per_prompt=2,
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=2,
            batch_size=2,
        ),
    )
    output_path = tmp_path / "candidates.jsonl"

    write_candidate_jsonl(
        candidate_generation_result=candidate_generation_result,
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    parsed_result = CandidateGenerationResult(
        candidates=tuple(GeneratedCandidateRecord.model_validate_json(line) for line in lines),
    )
    first_record = json.loads(lines[0])
    assert len(parsed_result.candidates) == 4
    assert first_record["task_id"] == "task-1"
    assert first_record["sample_index"] == 0
    assert first_record["decoding"]["strategy"] == "greedy"


def test_generate_text_applies_configured_precision() -> None:
    torch.manual_seed(19)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _build_model(vocabulary_size=tokenizer.vocabulary_size)
    model.double()

    generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        inference_configuration=InferenceConfiguration(
            engine=InferenceEngine.NAIVE,
            precision=Precision.FP32,
            quantization=QuantizationType.NONE,
            decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
            maximum_new_tokens=1,
        ),
    )

    assert next(model.parameters()).dtype == torch.float32


def test_prepare_model_for_inference_applies_half_precision() -> None:
    model = torch.nn.Linear(2, 2)

    prepare_model_for_inference(
        model=model,
        inference_configuration=InferenceConfiguration(
            precision=Precision.FP16,
            maximum_new_tokens=1,
        ),
    )

    assert next(model.parameters()).dtype == torch.float16


def test_prepare_model_for_inference_applies_bfloat16_precision() -> None:
    model = torch.nn.Linear(2, 2)

    prepare_model_for_inference(
        model=model,
        inference_configuration=InferenceConfiguration(
            precision=Precision.BF16,
            maximum_new_tokens=1,
        ),
    )

    assert next(model.parameters()).dtype == torch.bfloat16


def test_prepare_model_for_inference_rejects_unimplemented_quantization() -> None:
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="not implemented"):
        prepare_model_for_inference(
            model=model,
            inference_configuration=InferenceConfiguration(
                quantization=QuantizationType.INT8_DYNAMIC,
                maximum_new_tokens=1,
            ),
        )


def test_sample_decoding_draws_from_distribution() -> None:
    torch.manual_seed(3)
    next_token_id = select_next_token_id(
        logits=torch.tensor([-10.0, 10.0, -10.0]),
        decoding_configuration=SamplingDecodingConfiguration(
            strategy=DecodingStrategy.SAMPLE,
            temperature=1.0,
            top_k=None,
        ),
    )

    assert next_token_id == 1


def test_sample_decoding_can_limit_candidates_with_top_k() -> None:
    torch.manual_seed(3)
    next_token_id = select_next_token_id(
        logits=torch.tensor([10.0, 9.0, 8.0]),
        decoding_configuration=SamplingDecodingConfiguration(
            strategy=DecodingStrategy.SAMPLE,
            temperature=1.0,
            top_k=1,
        ),
    )

    assert next_token_id == 0


def _build_model(vocabulary_size: int) -> DenseGpt:
    model = DenseGpt(
        model_configuration=DenseGptConfiguration(
            type=ModelType.DENSE_GPT,
            dimension=16,
            layers=2,
            attention_heads=4,
            feed_forward_dimension=32,
            dropout=0.0,
            tie_embeddings=False,
        ),
        vocabulary_size=vocabulary_size,
    )
    model.eval()
    return model


def _build_moe_model(vocabulary_size: int) -> MoeGpt:
    model = MoeGpt(
        model_configuration=MoeGptConfiguration(
            type=ModelType.MOE_GPT,
            dimension=16,
            layers=2,
            attention_heads=4,
            expert_feed_forward_dimension=32,
            expert_count=4,
            router_top_k=2,
            dropout=0.0,
            tie_embeddings=False,
        ),
        vocabulary_size=vocabulary_size,
    )
    model.eval()
    return model


class StepTokenModel(nn.Module):
    def __init__(self, vocabulary_size: int, token_ids: tuple[int, ...]) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.token_ids = token_ids

    def forward(self, token_ids: torch.Tensor) -> FakeModelOutput:
        batch_size, sequence_length = token_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocabulary_size),
            -1000.0,
        )
        generated_index = max(0, sequence_length - 2)
        selected_token_id = self.token_ids[min(generated_index, len(self.token_ids) - 1)]
        logits[:, -1, selected_token_id] = 1000.0
        return FakeModelOutput(logits=logits)


@dataclass(frozen=True)
class FakeModelOutput:
    logits: torch.Tensor


class CountingTokenizer:
    vocabulary_size = 3
    pad_token_id = None
    eos_token_id = 0

    def __init__(self) -> None:
        self.decode_calls = 0

    def encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]:
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        self.decode_calls += 1
        return "x" * len(token_ids)

    def save(self, directory) -> None:  # noqa: ANN001
        return None
