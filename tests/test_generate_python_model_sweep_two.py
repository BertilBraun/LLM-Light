from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    DenseGptConfiguration,
    LearningRateScheduleType,
    ModelConfiguration,
    ModelType,
    ModernDenseGptConfiguration,
    ModernMoeGptConfiguration,
    MoeGptConfiguration,
)
from llm_lite.model.factory import build_model
from scripts.generate_python_model_sweep_two import generate_python_model_sweep_two

EXPECTED_EXPERIMENT_NAMES = (
    "python_modern_dense_vocab4000_baseline",
    "python_modern_dense_vocab2000_param_match",
    "python_modern_dense_qknorm_vocab4000",
    "python_modern_dense_qknorm_lr2e3",
    "python_modern_dense_cosine_lr2e3_qknorm",
    "python_modern_dense_heads4",
    "python_modern_dense_heads8",
    "python_modern_dense_heads12",
    "python_modern_dense_ffn_small",
    "python_modern_dense_ffn_medium",
    "python_modern_dense_ffn_large",
    "python_modern_dense_deep_qknorm",
    "python_modern_moe_vocab4000_aux010",
    "python_moe_vocab4000_aux010",
)
QK_NORMALIZATION_EXPERIMENT_NAMES = {
    "python_modern_dense_qknorm_vocab4000",
    "python_modern_dense_qknorm_lr2e3",
    "python_modern_dense_cosine_lr2e3_qknorm",
    "python_modern_dense_deep_qknorm",
}
EXPECTED_TRAINING_BATCH_SIZE_SEQUENCES = {
    "python_modern_dense_ffn_large": 128,
}


def test_generate_python_model_sweep_two_writes_expected_configs(
    tmp_path: Path,
) -> None:
    stale_configuration_path = tmp_path / "stale.yaml"
    stale_configuration_path.write_text("stale: true\n", encoding="utf-8")
    non_yaml_path = tmp_path / "notes.txt"
    non_yaml_path.write_text("keep\n", encoding="utf-8")

    generated_configurations = generate_python_model_sweep_two(
        base_configuration_path=Path("configs/python_moe_full.yaml"),
        output_directory=tmp_path,
    )

    assert tuple(
        generated_configuration.path.name for generated_configuration in generated_configurations
    ) == tuple(f"{name}.yaml" for name in EXPECTED_EXPERIMENT_NAMES)
    assert tuple(
        generated_configuration.parameter_summary.active_parameters
        for generated_configuration in generated_configurations
    ) == (
        938624,
        977536,
        938624,
        938624,
        938624,
        989760,
        989760,
        989760,
        963792,
        938624,
        1036928,
        1049952,
        976224,
        931392,
    )
    assert not stale_configuration_path.exists()
    assert non_yaml_path.exists()

    for generated_configuration in generated_configurations:
        experiment_configuration = load_experiment_configuration(
            configuration_path=generated_configuration.path,
        )
        build_model(
            model_configuration=experiment_configuration.model,
            vocabulary_size=experiment_configuration.tokenizer.vocabulary_size,
        )

        assert experiment_configuration.experiment.name in EXPECTED_EXPERIMENT_NAMES
        assert not experiment_configuration.experiment.name.startswith("python2_")
        assert not experiment_configuration.packing.fill_in_middle.enabled
        assert experiment_configuration.training.batch_size_sequences == (
            EXPECTED_TRAINING_BATCH_SIZE_SEQUENCES.get(
                experiment_configuration.experiment.name,
                256,
            )
        )
        assert experiment_configuration.training.maximum_steps == 15000
        assert experiment_configuration.training.max_checkpoints == 2
        assert experiment_configuration.training.optimizer.weight_decay == 0.1
        assert 500_000 <= generated_configuration.parameter_summary.active_parameters <= 1_800_000
        assert (
            500_000
            <= generated_configuration.parameter_summary.trainable_active_parameters
            <= 1_800_000
        )
        assert query_key_normalization_enabled(
            model_configuration=experiment_configuration.model,
        ) is (experiment_configuration.experiment.name in QK_NORMALIZATION_EXPERIMENT_NAMES)


def test_generate_python_model_sweep_two_applies_requested_ablations(
    tmp_path: Path,
) -> None:
    generate_python_model_sweep_two(
        base_configuration_path=Path("configs/python_moe_full.yaml"),
        output_directory=tmp_path,
    )

    baseline_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_vocab4000_baseline.yaml",
    )
    assert baseline_configuration.model.type is ModelType.MODERN_DENSE_GPT
    assert baseline_configuration.tokenizer.vocabulary_size == 4000
    assert baseline_configuration.training.optimizer.learning_rate == 0.001
    assert (
        baseline_configuration.training.optimizer.learning_rate_schedule.type
        is LearningRateScheduleType.FIXED
    )

    vocab2000_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_vocab2000_param_match.yaml",
    )
    assert vocab2000_configuration.tokenizer.vocabulary_size == 2000
    assert vocab2000_configuration.model.feed_forward_dimension == 768

    cosine_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_cosine_lr2e3_qknorm.yaml",
    )
    assert cosine_configuration.training.optimizer.learning_rate == 0.002
    assert (
        cosine_configuration.training.optimizer.learning_rate_schedule.type
        is LearningRateScheduleType.COSINE_WARMUP_DECAY
    )
    assert cosine_configuration.training.optimizer.learning_rate_schedule.warmup_steps == 150
    assert (
        cosine_configuration.training.optimizer.learning_rate_schedule.minimum_learning_rate_ratio
        == 0.1
    )

    heads4_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_heads4.yaml",
    )
    heads8_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_heads8.yaml",
    )
    heads12_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_heads12.yaml",
    )
    assert heads4_configuration.model.attention_heads == 4
    assert heads8_configuration.model.attention_heads == 8
    assert heads12_configuration.model.attention_heads == 12
    assert heads4_configuration.model.dimension == 192
    assert heads8_configuration.model.dimension == 192
    assert heads12_configuration.model.dimension == 192
    assert heads4_configuration.tokenizer.vocabulary_size == 4000
    assert heads8_configuration.tokenizer.vocabulary_size == 4000
    assert heads12_configuration.tokenizer.vocabulary_size == 4000
    assert heads12_configuration.model.dimension != 256
    assert (
        heads12_configuration.model.dimension // heads12_configuration.model.attention_heads
    ) % 2 == 0

    ffn_small_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_ffn_small.yaml",
    )
    assert ffn_small_configuration.model.dimension == 144
    assert ffn_small_configuration.model.layers == 2
    assert ffn_small_configuration.model.feed_forward_dimension == 256

    deep_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_deep_qknorm.yaml",
    )
    assert deep_configuration.model.layers == 12
    assert deep_configuration.model.attention_heads == 6
    assert deep_configuration.model.feed_forward_dimension == 64
    assert deep_configuration.model.query_key_normalization

    modern_moe_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_moe_vocab4000_aux010.yaml",
    )
    classic_moe_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_moe_vocab4000_aux010.yaml",
    )
    assert modern_moe_configuration.model.type is ModelType.MODERN_MOE_GPT
    assert classic_moe_configuration.model.type is ModelType.MOE_GPT
    assert modern_moe_configuration.model.dimension == 96
    assert modern_moe_configuration.model.expert_feed_forward_dimension == 384
    assert modern_moe_configuration.model.router_top_k == 1
    assert classic_moe_configuration.model.router_top_k == 1
    assert modern_moe_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.1
    assert classic_moe_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.1


def query_key_normalization_enabled(model_configuration: ModelConfiguration) -> bool:
    match model_configuration:
        case (
            ModernDenseGptConfiguration(
                query_key_normalization=query_key_normalization,
            )
            | ModernMoeGptConfiguration(query_key_normalization=query_key_normalization)
        ):
            return query_key_normalization
        case DenseGptConfiguration() | MoeGptConfiguration():
            return False
