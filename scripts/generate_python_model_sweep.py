from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.export import (
    ExperimentOverrides,
    dense_model,
    export_experiment_variant,
    moe_model,
    training,
)
from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import ExperimentFile


@dataclass(frozen=True)
class DenseModelSpec:
    name: str
    dimension: int
    layers: int
    feed_forward_dimension: int


@dataclass(frozen=True)
class MoeModelSpec:
    name: str
    dimension: int
    layers: int
    expert_feed_forward_dimension: int
    expert_count: int


def main() -> int:
    base_configuration = load_experiment_configuration(
        configuration_path=Path("configs/python_moe_full.yaml"),
    )
    output_directory = Path("configs/generated/python_model_sweep")

    for dense_specification in dense_model_specifications():
        output_path = export_dense_config(
            base_configuration=base_configuration,
            output_directory=output_directory,
            specification=dense_specification,
        )
        print(output_path)

    for moe_specification in moe_model_specifications():
        output_path = export_moe_config(
            base_configuration=base_configuration,
            output_directory=output_directory,
            specification=moe_specification,
        )
        print(output_path)

    return 0


def dense_model_specifications() -> tuple[DenseModelSpec, ...]:
    return (
        DenseModelSpec(
            name="python_dense_deep_narrow",
            dimension=160,
            layers=8,
            feed_forward_dimension=640,
        ),
        DenseModelSpec(
            name="python_dense_shallow_wide",
            dimension=256,
            layers=4,
            feed_forward_dimension=1024,
        ),
    )


def moe_model_specifications() -> tuple[MoeModelSpec, ...]:
    return (
        MoeModelSpec(
            name="python_moe_small",
            dimension=160,
            layers=4,
            expert_feed_forward_dimension=640,
            expert_count=2,
        ),
        MoeModelSpec(
            name="python_moe_deeper",
            dimension=224,
            layers=8,
            expert_feed_forward_dimension=896,
            expert_count=4,
        ),
    )


def export_dense_config(
    base_configuration: ExperimentFile,
    output_directory: Path,
    specification: DenseModelSpec,
) -> Path:
    return export_experiment_variant(
        base_configuration=base_configuration,
        output_directory=output_directory,
        overrides=ExperimentOverrides(
            name=specification.name,
            model=dense_model(
                dimension=specification.dimension,
                layers=specification.layers,
                attention_heads=8,
                feed_forward_dimension=specification.feed_forward_dimension,
                dropout=0.05,
            ),
            training=training(
                base_training=base_configuration.training,
                maximum_steps=1500,
                batch_size_sequences=256,
                learning_rate=0.0008,
                auxiliary_loss_weight=0.0,
            ),
        ),
    )


def export_moe_config(
    base_configuration: ExperimentFile,
    output_directory: Path,
    specification: MoeModelSpec,
) -> Path:
    return export_experiment_variant(
        base_configuration=base_configuration,
        output_directory=output_directory,
        overrides=ExperimentOverrides(
            name=specification.name,
            model=moe_model(
                dimension=specification.dimension,
                layers=specification.layers,
                attention_heads=8,
                expert_feed_forward_dimension=specification.expert_feed_forward_dimension,
                expert_count=specification.expert_count,
                router_top_k=1,
                dropout=0.05,
            ),
            training=training(
                base_training=base_configuration.training,
                maximum_steps=1500,
                batch_size_sequences=256,
                learning_rate=0.0008,
                auxiliary_loss_weight=0.01,
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
