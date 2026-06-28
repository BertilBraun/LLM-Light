from pathlib import Path

from llm_lite.config.export import (
    ExperimentOverrides,
    dense_model,
    export_experiment_configs,
    export_experiment_variant,
    modern_moe_model,
    moe_model,
    training,
)
from llm_lite.config.loading import load_experiment_configuration


def test_export_experiment_configs_writes_dense_config(tmp_path: Path) -> None:
    output_paths = export_experiment_configs(
        base_configuration_path=Path("configs/verify_one_sentence.yaml"),
        output_directory=tmp_path,
        overrides=(
            ExperimentOverrides(
                name="dense_deep_narrow",
                model=dense_model(
                    dimension=16,
                    layers=4,
                    attention_heads=1,
                    feed_forward_dimension=32,
                    dropout=0.05,
                ),
            ),
        ),
    )

    configuration = load_experiment_configuration(configuration_path=output_paths[0])

    assert output_paths == (tmp_path / "dense_deep_narrow.yaml",)
    assert configuration.experiment.name == "dense_deep_narrow"
    assert configuration.experiment.output_dir == Path("runs/dense_deep_narrow")
    assert configuration.model.type.value == "dense_gpt"
    assert configuration.model.dimension == 16
    assert configuration.model.layers == 4


def test_export_experiment_configs_writes_moe_config_with_training_override(
    tmp_path: Path,
) -> None:
    base_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    output_paths = export_experiment_configs(
        base_configuration_path=Path("configs/verify_one_sentence.yaml"),
        output_directory=tmp_path,
        overrides=(
            ExperimentOverrides(
                name="moe_wide",
                model=moe_model(
                    dimension=24,
                    layers=2,
                    attention_heads=1,
                    expert_feed_forward_dimension=48,
                    expert_count=2,
                    router_top_k=1,
                    dropout=0.05,
                ),
                training=training(
                    base_training=base_configuration.training,
                    maximum_steps=5,
                    batch_size_sequences=2,
                    learning_rate=0.0005,
                    auxiliary_loss_weight=0.01,
                ),
            ),
        ),
    )

    configuration = load_experiment_configuration(configuration_path=output_paths[0])

    assert configuration.experiment.name == "moe_wide"
    assert configuration.model.type.value == "moe_gpt"
    assert configuration.model.expert_count == 2
    assert configuration.training.maximum_steps == 5
    assert configuration.training.batch_size_sequences == 2
    assert configuration.training.optimizer.learning_rate == 0.0005
    assert configuration.training.causal_language_modeling.auxiliary_loss_weight == 0.01


def test_export_experiment_variant_can_be_called_repeatedly(tmp_path: Path) -> None:
    base_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    output_paths: list[Path] = []
    for dimension in (16, 24):
        output_paths.append(
            export_experiment_variant(
                base_configuration=base_configuration,
                output_directory=tmp_path,
                overrides=ExperimentOverrides(
                    name=f"dense_d{dimension}",
                    model=dense_model(
                        dimension=dimension,
                        layers=1,
                        attention_heads=1,
                        feed_forward_dimension=dimension * 2,
                        dropout=0.0,
                    ),
                ),
            ),
        )

    first_configuration = load_experiment_configuration(configuration_path=output_paths[0])
    second_configuration = load_experiment_configuration(configuration_path=output_paths[1])

    assert output_paths == [tmp_path / "dense_d16.yaml", tmp_path / "dense_d24.yaml"]
    assert first_configuration.model.dimension == 16
    assert second_configuration.model.dimension == 24


def test_export_experiment_configs_writes_modern_moe_config(tmp_path: Path) -> None:
    output_paths = export_experiment_configs(
        base_configuration_path=Path("configs/verify_one_sentence.yaml"),
        output_directory=tmp_path,
        overrides=(
            ExperimentOverrides(
                name="modern_moe",
                model=modern_moe_model(
                    dimension=16,
                    layers=2,
                    attention_heads=4,
                    expert_feed_forward_dimension=32,
                    expert_count=4,
                    router_top_k=2,
                    dropout=0.05,
                ),
            ),
        ),
    )

    configuration = load_experiment_configuration(configuration_path=output_paths[0])

    assert configuration.model.type.value == "modern_moe_gpt"
    assert configuration.model.dimension == 16
    assert configuration.model.expert_count == 4
