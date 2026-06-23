import pytest
import torch

from llm_lite.config.models import (
    DecodingStrategy,
    DenseGptConfiguration,
    GreedyDecodingConfiguration,
    InferenceConfiguration,
    InferenceEngine,
    ModelType,
    Precision,
    QuantizationType,
    SamplingDecodingConfiguration,
)
from llm_lite.inference.decoding import select_next_token_id
from llm_lite.inference.engine import generate_text
from llm_lite.inference.kv_cache import generate_greedy as generate_greedy_with_cache
from llm_lite.inference.naive import generate_greedy as generate_greedy_naively
from llm_lite.inference.runtime import prepare_model_for_inference
from llm_lite.model.gpt import DenseGpt
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
