import torch

from llm_lite.config.models import (
    ModelType,
    ModernDenseGptConfiguration,
    ModernMoeGptConfiguration,
)
from llm_lite.inference.kv_cache import generate_greedy as generate_greedy_with_cache
from llm_lite.inference.naive import generate_greedy as generate_greedy_naively
from llm_lite.model.factory import build_model
from llm_lite.model.modern import ModernDenseGpt, ModernMoeGpt
from llm_lite.model.parameters import model_parameter_summary
from llm_lite.model.router_usage import collect_router_usage_summaries, reset_router_usage
from llm_lite.tokenizer.character import train_character_tokenizer


def test_model_factory_returns_modern_models() -> None:
    dense_model = build_model(
        model_configuration=_modern_dense_configuration(),
        vocabulary_size=16,
    )
    moe_model = build_model(
        model_configuration=_modern_moe_configuration(),
        vocabulary_size=16,
    )

    assert isinstance(dense_model, ModernDenseGpt)
    assert isinstance(moe_model, ModernMoeGpt)


def test_modern_moe_forward_output_shape_and_auxiliary_loss() -> None:
    torch.manual_seed(101)
    model = ModernMoeGpt(model_configuration=_modern_moe_configuration(), vocabulary_size=19)
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    model_output = model(token_ids)

    assert model_output.logits.shape == (2, 4, 19)
    assert model_output.auxiliary_loss is not None
    assert torch.isfinite(model_output.auxiliary_loss)


def test_modern_moe_parameter_summary_reports_active_parameters() -> None:
    model = ModernMoeGpt(model_configuration=_modern_moe_configuration(), vocabulary_size=19)

    parameter_summary = model_parameter_summary(model=model)

    assert parameter_summary.total_parameters > parameter_summary.active_parameters
    assert parameter_summary.trainable_parameters > parameter_summary.trainable_active_parameters


def test_modern_moe_router_usage_helpers_report_layers() -> None:
    model = ModernMoeGpt(model_configuration=_modern_moe_configuration(), vocabulary_size=19)
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    model(token_ids)
    router_usage_summaries = collect_router_usage_summaries(model=model)
    reset_router_usage(model=model)

    assert len(router_usage_summaries) == model.model_configuration.layers


def test_modern_dense_parameter_summary_reports_all_parameters_active() -> None:
    model = ModernDenseGpt(model_configuration=_modern_dense_configuration(), vocabulary_size=19)

    parameter_summary = model_parameter_summary(model=model)

    assert parameter_summary.total_parameters == parameter_summary.active_parameters
    assert parameter_summary.trainable_parameters == parameter_summary.trainable_active_parameters


def test_modern_dense_qk_normalization_forward_output_shape() -> None:
    torch.manual_seed(102)
    model = build_model(
        model_configuration=_modern_dense_configuration(query_key_normalization=True),
        vocabulary_size=19,
    )
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    model_output = model(token_ids)

    assert model_output.logits.shape == (2, 4, 19)


def test_modern_dense_cached_forward_matches_full_sequence_forward() -> None:
    torch.manual_seed(103)
    tokenizer = train_character_tokenizer(
        texts=["hello world"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = ModernDenseGpt(
        model_configuration=_modern_dense_configuration(),
        vocabulary_size=tokenizer.vocabulary_size,
    )
    model.eval()

    _assert_cached_forward_matches_full_forward(model=model, tokenizer_text="hello world")


def test_modern_dense_qk_normalization_cached_forward_matches_full_sequence_forward() -> None:
    torch.manual_seed(104)
    tokenizer = train_character_tokenizer(
        texts=["hello world"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = ModernDenseGpt(
        model_configuration=_modern_dense_configuration(query_key_normalization=True),
        vocabulary_size=tokenizer.vocabulary_size,
    )
    model.eval()

    _assert_cached_forward_matches_full_forward(model=model, tokenizer_text="hello world")


def test_modern_moe_cached_forward_matches_full_sequence_forward() -> None:
    torch.manual_seed(107)
    tokenizer = train_character_tokenizer(
        texts=["hello world"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = ModernMoeGpt(
        model_configuration=_modern_moe_configuration(),
        vocabulary_size=tokenizer.vocabulary_size,
    )
    model.eval()

    _assert_cached_forward_matches_full_forward(model=model, tokenizer_text="hello world")


def test_modern_moe_kv_cache_generation_matches_naive_generation() -> None:
    torch.manual_seed(109)
    tokenizer = train_character_tokenizer(
        texts=["abc"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = ModernMoeGpt(
        model_configuration=_modern_moe_configuration(),
        vocabulary_size=tokenizer.vocabulary_size,
    )
    model.eval()

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


def _assert_cached_forward_matches_full_forward(
    model: ModernDenseGpt | ModernMoeGpt,
    tokenizer_text: str,
) -> None:
    tokenizer = train_character_tokenizer(
        texts=[tokenizer_text],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
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


def _modern_dense_configuration(
    query_key_normalization: bool = False,
) -> ModernDenseGptConfiguration:
    return ModernDenseGptConfiguration(
        type=ModelType.MODERN_DENSE_GPT,
        dimension=16,
        layers=2,
        attention_heads=4,
        feed_forward_dimension=32,
        query_key_normalization=query_key_normalization,
        dropout=0.0,
        tie_embeddings=False,
    )


def _modern_moe_configuration() -> ModernMoeGptConfiguration:
    return ModernMoeGptConfiguration(
        type=ModelType.MODERN_MOE_GPT,
        dimension=16,
        layers=2,
        attention_heads=4,
        expert_feed_forward_dimension=32,
        expert_count=4,
        router_top_k=2,
        dropout=0.0,
        tie_embeddings=False,
    )
