from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import LearningRateScheduleType, ModelType
from llm_lite.model.factory import build_model
from scripts.generate_python_model_sweep_three import generate_python_model_sweep_three

EXPECTED_EXPERIMENT_NAMES = (
    "python_modern_moe_vocab2000_aux010",
    "python_modern_moe_vocab2000_aux020",
    "python_modern_moe_vocab2000_topk2_aux010",
    "python_modern_moe_vocab2000_topk2_aux020",
    "python_modern_moe_deep10_vocab2000_aux020",
    "python_modern_dense_active10m_vocab2000",
)
EXPECTED_ACTIVE_PARAMETERS = (986608, 986608, 1001680, 1001680, 9784576, 9737280)


def test_generate_python_model_sweep_three_writes_expected_configs(
    tmp_path: Path,
) -> None:
    stale_configuration_path = tmp_path / "stale.yaml"
    stale_configuration_path.write_text("stale: true\n", encoding="utf-8")
    non_yaml_path = tmp_path / "notes.txt"
    non_yaml_path.write_text("keep\n", encoding="utf-8")

    generated_configurations = generate_python_model_sweep_three(
        base_configuration_path=Path("configs/python_moe_full.yaml"),
        output_directory=tmp_path,
    )

    assert tuple(
        generated_configuration.path.name for generated_configuration in generated_configurations
    ) == tuple(f"{name}.yaml" for name in EXPECTED_EXPERIMENT_NAMES)
    assert (
        tuple(
            generated_configuration.parameter_summary.active_parameters
            for generated_configuration in generated_configurations
        )
        == EXPECTED_ACTIVE_PARAMETERS
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
        assert experiment_configuration.model.type in {
            ModelType.MODERN_MOE_GPT,
            ModelType.MODERN_DENSE_GPT,
        }
        assert experiment_configuration.tokenizer.vocabulary_size == 2000
        assert not experiment_configuration.packing.fill_in_middle.enabled
        assert experiment_configuration.training.batch_size_sequences == 256
        assert experiment_configuration.training.maximum_steps == 15000
        assert experiment_configuration.training.max_checkpoints == 2
        assert experiment_configuration.training.optimizer.learning_rate == 0.001
        assert (
            experiment_configuration.training.optimizer.learning_rate_schedule.type
            is LearningRateScheduleType.FIXED
        )
        assert experiment_configuration.training.optimizer.weight_decay == 0.1
        assert experiment_configuration.export.bundle_path == Path("export/bundle.zip")


def test_generate_python_model_sweep_three_applies_requested_moe_variants(
    tmp_path: Path,
) -> None:
    generate_python_model_sweep_three(
        base_configuration_path=Path("configs/python_moe_full.yaml"),
        output_directory=tmp_path,
    )

    small_aux010_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_moe_vocab2000_aux010.yaml",
    )
    assert small_aux010_configuration.model.dimension == 112
    assert small_aux010_configuration.model.layers == 4
    assert small_aux010_configuration.model.attention_heads == 4
    assert small_aux010_configuration.model.expert_feed_forward_dimension == 416
    assert small_aux010_configuration.model.expert_count == 4
    assert small_aux010_configuration.model.router_top_k == 1
    assert small_aux010_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.1

    small_aux020_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_moe_vocab2000_aux020.yaml",
    )
    assert small_aux020_configuration.model.dimension == 112
    assert small_aux020_configuration.model.layers == 4
    assert small_aux020_configuration.model.attention_heads == 4
    assert small_aux020_configuration.model.expert_feed_forward_dimension == 416
    assert small_aux020_configuration.model.expert_count == 4
    assert small_aux020_configuration.model.router_top_k == 1
    assert small_aux020_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.2

    topk2_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_moe_vocab2000_topk2_aux010.yaml",
    )
    assert topk2_configuration.model.dimension == 80
    assert topk2_configuration.model.layers == 4
    assert topk2_configuration.model.attention_heads == 4
    assert topk2_configuration.model.expert_feed_forward_dimension == 384
    assert topk2_configuration.model.router_top_k == 2
    assert topk2_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.1

    topk2_aux020_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_moe_vocab2000_topk2_aux020.yaml",
    )
    assert topk2_aux020_configuration.model.dimension == 80
    assert topk2_aux020_configuration.model.layers == 4
    assert topk2_aux020_configuration.model.attention_heads == 4
    assert topk2_aux020_configuration.model.expert_feed_forward_dimension == 384
    assert topk2_aux020_configuration.model.router_top_k == 2
    assert topk2_aux020_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.2

    deep10_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_moe_deep10_vocab2000_aux020.yaml",
    )
    assert deep10_configuration.model.dimension == 256
    assert deep10_configuration.model.layers == 10
    assert deep10_configuration.model.attention_heads == 8
    assert deep10_configuration.model.expert_feed_forward_dimension == 864
    assert deep10_configuration.model.expert_count == 4
    assert deep10_configuration.model.router_top_k == 1
    assert deep10_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.2

    dense10_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_dense_active10m_vocab2000.yaml",
    )
    assert dense10_configuration.model.type is ModelType.MODERN_DENSE_GPT
    assert dense10_configuration.model.dimension == 320
    assert dense10_configuration.model.layers == 6
    assert dense10_configuration.model.attention_heads == 8
    assert dense10_configuration.model.feed_forward_dimension == 1152
    assert dense10_configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.0
