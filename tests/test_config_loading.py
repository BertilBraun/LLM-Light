from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    DataLoaderConfiguration,
    DecodingStrategy,
    EvaluationConfiguration,
    InferenceConfiguration,
    InferenceEngine,
    Precision,
    QuantizationType,
)


def test_load_verify_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )

    assert experiment_configuration.experiment.name == "verify_one_sentence"
    assert experiment_configuration.dataset.documents == ("hello world\n",)
    assert len(experiment_configuration.preprocessing.transforms) == 1
    assert experiment_configuration.training.dataloader.num_workers == 0
    exact_reproduction_configuration = experiment_configuration.evaluation.exact_reproduction

    assert exact_reproduction_configuration is not None
    assert exact_reproduction_configuration.expected_completion == "hello world\n"


def test_load_byte_bpe_verification_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("tests/configs/verify_byte_bpe.yaml"),
    )

    assert experiment_configuration.experiment.name == "verify_byte_bpe"
    assert experiment_configuration.tokenizer.type.value == "byte_bpe"


def test_load_tinystories_huggingface_smoke_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/tinystories_hf_smoke.yaml"),
    )

    assert experiment_configuration.dataset.type.value == "huggingface"
    assert experiment_configuration.training.evaluation is not None


def test_evaluation_configuration_allows_no_configured_evaluator() -> None:
    evaluation_configuration = EvaluationConfiguration.model_validate({})

    assert evaluation_configuration.exact_reproduction is None
    assert evaluation_configuration.perplexity is None
    assert evaluation_configuration.fixed_prompt_generation is None


def test_evaluation_configuration_rejects_unknown_evaluator() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfiguration.model_validate(
            {"other_evaluation_type": {"parameters_for_that": "here"}},
        )


def test_inference_configuration_accepts_kv_cache_engine() -> None:
    inference_configuration = InferenceConfiguration.model_validate(
        {
            "engine": "kv_cache",
            "precision": "fp32",
            "quantization": "none",
            "decoding": {
                "strategy": "sample",
                "temperature": 0.7,
                "top_k": 5,
            },
            "maximum_new_tokens": 8,
        },
    )

    assert inference_configuration.engine is InferenceEngine.KV_CACHE
    assert inference_configuration.decoding.strategy is DecodingStrategy.SAMPLE


def test_inference_configuration_defaults_common_runtime_options() -> None:
    inference_configuration = InferenceConfiguration.model_validate(
        {
            "maximum_new_tokens": 8,
        },
    )

    assert inference_configuration.engine is InferenceEngine.KV_CACHE
    assert inference_configuration.precision is Precision.FP32
    assert inference_configuration.quantization is QuantizationType.NONE
    assert inference_configuration.decoding.strategy is DecodingStrategy.GREEDY


def test_dataloader_configuration_rejects_worker_options_without_workers() -> None:
    with pytest.raises(ValidationError, match="persistent_workers"):
        DataLoaderConfiguration.model_validate(
            {
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": True,
                "prefetch_factor": None,
            },
        )

    with pytest.raises(ValidationError, match="prefetch_factor"):
        DataLoaderConfiguration.model_validate(
            {
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
        )
