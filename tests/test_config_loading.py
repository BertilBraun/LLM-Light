from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    DataLoaderConfiguration,
    DecodingStrategy,
    DirectPreferenceOptimizationConfiguration,
    EvaluationConfiguration,
    InferenceConfiguration,
    InferenceEngine,
    PostTrainingType,
    Precision,
    PythonGeneratedDirectPreferenceOptimizationConfiguration,
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


def test_load_distributed_data_parallel_smoke_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("tests/configs/distributed_data_parallel_smoke.yaml"),
    )

    assert experiment_configuration.experiment.name == "distributed_data_parallel_smoke"
    assert experiment_configuration.training.maximum_steps == 3
    assert experiment_configuration.distributed.enabled is True
    assert experiment_configuration.distributed.backend.value == "gloo"
    assert experiment_configuration.distributed.strategy.value == "data_parallel"
    assert experiment_configuration.distributed.world_size == 2
    assert experiment_configuration.distributed.parallelism.data == 2
    assert experiment_configuration.distributed.checkpoint.type.value == "sharded"


def test_load_tinystories_huggingface_smoke_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/tinystories_hf_smoke.yaml"),
    )

    assert experiment_configuration.dataset.type.value == "huggingface"
    assert experiment_configuration.training.evaluation is not None


def test_load_tinystories_moe_full_configuration_uses_fast_tokenizer() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/tinystories_moe_full.yaml"),
    )

    assert experiment_configuration.experiment.name == "tinystories_moe_full"
    assert experiment_configuration.tokenizer.type.value == "rust_byte_bpe"


def test_packing_fill_in_middle_configuration_loads_from_yaml(tmp_path: Path) -> None:
    configuration_path = tmp_path / "fim_config.yaml"
    configuration_path.write_text(
        "\n".join(
            (
                "experiment:",
                "  name: fim_config",
                "  output_dir: runs/fim_config",
                "dataset:",
                "  type: inline_text",
                "  documents: ['abcdefghijklmnopqrstuvwxyz']",
                "tokenizer:",
                "  type: character",
                "  additional_special_tokens:",
                "    - <fim_prefix>",
                "    - <fim_suffix>",
                "    - <fim_middle>",
                "packing:",
                "  context_length: 8",
                "  fill_in_middle:",
                "    enabled: true",
                "    probability: 0.25",
                "    minimum_segment_characters: 3",
                "model:",
                "  type: dense_gpt",
                "  dimension: 8",
                "  layers: 1",
                "  attention_heads: 1",
                "  feed_forward_dimension: 16",
                "training:",
                "  maximum_steps: 1",
                "  batch_size_sequences: 1",
            ),
        ),
        encoding="utf-8",
    )

    experiment_configuration = load_experiment_configuration(configuration_path=configuration_path)

    assert experiment_configuration.packing.fill_in_middle.enabled
    assert experiment_configuration.packing.fill_in_middle.probability == 0.25
    assert experiment_configuration.packing.fill_in_middle.minimum_segment_characters == 3


def test_packing_fill_in_middle_requires_tokenizer_special_tokens(tmp_path: Path) -> None:
    configuration_path = tmp_path / "invalid_fim_config.yaml"
    configuration_path.write_text(
        "\n".join(
            (
                "experiment:",
                "  name: invalid_fim_config",
                "  output_dir: runs/invalid_fim_config",
                "dataset:",
                "  type: inline_text",
                "  documents: ['abcdefghijklmnopqrstuvwxyz']",
                "tokenizer:",
                "  type: character",
                "packing:",
                "  context_length: 8",
                "  fill_in_middle:",
                "    enabled: true",
                "    probability: 0.25",
                "model:",
                "  type: dense_gpt",
                "  dimension: 8",
                "  layers: 1",
                "  attention_heads: 1",
                "  feed_forward_dimension: 16",
                "training:",
                "  maximum_steps: 1",
                "  batch_size_sequences: 1",
            ),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="additional_special_tokens"):
        load_experiment_configuration(configuration_path=configuration_path)


def test_load_python_moe_full_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/python_moe_full.yaml"),
    )

    assert experiment_configuration.experiment.name == "python_moe_full"
    assert experiment_configuration.dataset.type.value == "huggingface"
    assert experiment_configuration.dataset.name == "BertilBraun/TinyPython"
    assert experiment_configuration.model.type.value == "moe_gpt"
    assert experiment_configuration.model.dimension == 320
    assert experiment_configuration.tokenizer.vocabulary_size == 6000
    assert experiment_configuration.packing.context_length == 256
    assert experiment_configuration.packing.pack_documents is True


def test_evaluation_configuration_allows_no_configured_evaluator() -> None:
    evaluation_configuration = EvaluationConfiguration.model_validate({})

    assert evaluation_configuration.exact_reproduction is None
    assert evaluation_configuration.perplexity is None
    assert evaluation_configuration.fixed_prompt_generation is None
    assert evaluation_configuration.python_completion is None


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


def test_direct_preference_optimization_configuration_loads() -> None:
    post_training_configuration = DirectPreferenceOptimizationConfiguration.model_validate(
        {
            "type": "direct_preference_optimization",
            "preference_dataset_path": "preferences.jsonl",
            "beta": 0.2,
            "maximum_steps": 3,
            "batch_size_pairs": 2,
        },
    )

    assert post_training_configuration.type is PostTrainingType.DIRECT_PREFERENCE_OPTIMIZATION
    assert post_training_configuration.beta == 0.2


def test_python_generated_dpo_configuration_loads() -> None:
    post_training_configuration = (
        PythonGeneratedDirectPreferenceOptimizationConfiguration.model_validate(
            {
                "type": "python_generated_direct_preference_optimization",
                "tasks_path": "tasks.jsonl",
                "samples_per_prompt": 4,
                "beta": 0.1,
                "maximum_steps": 3,
                "batch_size_pairs": 2,
            },
        )
    )

    assert (
        post_training_configuration.type
        is PostTrainingType.PYTHON_GENERATED_DIRECT_PREFERENCE_OPTIMIZATION
    )
    assert post_training_configuration.samples_per_prompt == 4


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
