from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    DenseGptConfiguration,
    ExperimentFile,
    ExportConfiguration,
    LearningRateScheduleConfiguration,
    ModelType,
    ModernDenseGptConfiguration,
    ModernMoeGptConfiguration,
    MoeGptConfiguration,
    OptimizerConfiguration,
    PackingConfiguration,
    TokenizerConfiguration,
    TrainingConfiguration,
)


@dataclass(frozen=True)
class ExperimentOverrides:
    name: str
    model: (
        DenseGptConfiguration
        | MoeGptConfiguration
        | ModernDenseGptConfiguration
        | ModernMoeGptConfiguration
    )
    training: TrainingConfiguration | None = None
    tokenizer: TokenizerConfiguration | None = None
    packing: PackingConfiguration | None = None
    export: ExportConfiguration | None = None
    output_dir: Path | None = None


def dense_model(
    dimension: int,
    layers: int,
    attention_heads: int,
    feed_forward_dimension: int,
    dropout: float,
    tie_embeddings: bool = True,
) -> DenseGptConfiguration:
    return DenseGptConfiguration(
        type=ModelType.DENSE_GPT,
        dimension=dimension,
        layers=layers,
        attention_heads=attention_heads,
        feed_forward_dimension=feed_forward_dimension,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
    )


def moe_model(
    dimension: int,
    layers: int,
    attention_heads: int,
    expert_feed_forward_dimension: int,
    expert_count: int,
    router_top_k: int,
    dropout: float,
    tie_embeddings: bool = True,
) -> MoeGptConfiguration:
    return MoeGptConfiguration(
        type=ModelType.MOE_GPT,
        dimension=dimension,
        layers=layers,
        attention_heads=attention_heads,
        expert_feed_forward_dimension=expert_feed_forward_dimension,
        expert_count=expert_count,
        router_top_k=router_top_k,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
    )


def modern_dense_model(
    dimension: int,
    layers: int,
    attention_heads: int,
    feed_forward_dimension: int,
    dropout: float,
    rope_base: float = 10000.0,
    normalization_epsilon: float = 1e-5,
    query_key_normalization: bool = False,
    tie_embeddings: bool = True,
) -> ModernDenseGptConfiguration:
    return ModernDenseGptConfiguration(
        type=ModelType.MODERN_DENSE_GPT,
        dimension=dimension,
        layers=layers,
        attention_heads=attention_heads,
        feed_forward_dimension=feed_forward_dimension,
        rope_base=rope_base,
        normalization_epsilon=normalization_epsilon,
        query_key_normalization=query_key_normalization,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
    )


def modern_moe_model(
    dimension: int,
    layers: int,
    attention_heads: int,
    expert_feed_forward_dimension: int,
    expert_count: int,
    router_top_k: int,
    dropout: float,
    rope_base: float = 10000.0,
    normalization_epsilon: float = 1e-5,
    query_key_normalization: bool = False,
    tie_embeddings: bool = True,
) -> ModernMoeGptConfiguration:
    return ModernMoeGptConfiguration(
        type=ModelType.MODERN_MOE_GPT,
        dimension=dimension,
        layers=layers,
        attention_heads=attention_heads,
        expert_feed_forward_dimension=expert_feed_forward_dimension,
        expert_count=expert_count,
        router_top_k=router_top_k,
        rope_base=rope_base,
        normalization_epsilon=normalization_epsilon,
        query_key_normalization=query_key_normalization,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
    )


def training(
    base_training: TrainingConfiguration,
    maximum_steps: int,
    batch_size_sequences: int,
    learning_rate: float,
    weight_decay: float | None = None,
    auxiliary_loss_weight: float | None = None,
    learning_rate_schedule: LearningRateScheduleConfiguration | None = None,
    max_checkpoints: int | None = None,
) -> TrainingConfiguration:
    optimizer = base_training.optimizer.model_copy(
        update={
            "learning_rate": learning_rate,
            "weight_decay": (
                base_training.optimizer.weight_decay if weight_decay is None else weight_decay
            ),
            "learning_rate_schedule": (
                base_training.optimizer.learning_rate_schedule
                if learning_rate_schedule is None
                else learning_rate_schedule
            ),
        },
    )
    causal_language_modeling = base_training.causal_language_modeling
    if auxiliary_loss_weight is not None:
        causal_language_modeling = causal_language_modeling.model_copy(
            update={"auxiliary_loss_weight": auxiliary_loss_weight},
        )
    return base_training.model_copy(
        update={
            "maximum_steps": maximum_steps,
            "batch_size_sequences": batch_size_sequences,
            "optimizer": OptimizerConfiguration.model_validate(optimizer),
            "causal_language_modeling": causal_language_modeling,
            "max_checkpoints": max_checkpoints,
        },
    )


def experiment_config(
    base_configuration: ExperimentFile,
    overrides: ExperimentOverrides,
) -> ExperimentFile:
    output_dir = overrides.output_dir
    if output_dir is None:
        output_dir = Path("runs") / overrides.name
    experiment = base_configuration.experiment.model_copy(
        update={"name": overrides.name, "output_dir": output_dir},
    )
    tokenizer = base_configuration.tokenizer
    if overrides.tokenizer is not None:
        tokenizer = overrides.tokenizer
    packing = base_configuration.packing
    if overrides.packing is not None:
        packing = overrides.packing
    training_configuration = base_configuration.training
    if overrides.training is not None:
        training_configuration = overrides.training
    export_configuration = base_configuration.export
    if overrides.export is not None:
        export_configuration = overrides.export
    return base_configuration.model_copy(
        update={
            "experiment": experiment,
            "model": overrides.model,
            "tokenizer": tokenizer,
            "packing": packing,
            "training": training_configuration,
            "export": export_configuration,
        },
    )


def export_experiment_config(
    configuration: ExperimentFile,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(configuration.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def export_experiment_variant(
    base_configuration: ExperimentFile,
    output_directory: Path,
    overrides: ExperimentOverrides,
) -> Path:
    configuration = experiment_config(
        base_configuration=base_configuration,
        overrides=overrides,
    )
    return export_experiment_config(
        configuration=configuration,
        output_path=output_directory / f"{overrides.name}.yaml",
    )


def export_experiment_configs(
    base_configuration_path: Path,
    output_directory: Path,
    overrides: tuple[ExperimentOverrides, ...],
) -> tuple[Path, ...]:
    base_configuration = load_experiment_configuration(configuration_path=base_configuration_path)
    output_paths: list[Path] = []
    for experiment_overrides in overrides:
        output_paths.append(
            export_experiment_variant(
                base_configuration=base_configuration,
                output_directory=output_directory,
                overrides=experiment_overrides,
            ),
        )
    return tuple(output_paths)
