import argparse
from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.export import (
    ExperimentOverrides,
    export_experiment_variant,
    modern_dense_model,
    modern_moe_model,
    training,
)
from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    ByteBpeTokenizerConfiguration,
    ExperimentFile,
    ExportConfiguration,
    ModelConfiguration,
    RustByteBpeTokenizerConfiguration,
    TokenizerConfiguration,
)
from llm_lite.model.factory import build_model
from llm_lite.model.parameters import ModelParameterSummary, model_parameter_summary

DEFAULT_BASE_CONFIGURATION_PATH = Path("configs/python_moe_full.yaml")
DEFAULT_OUTPUT_DIRECTORY = Path("configs/generated/python_model_sweep_three")
MAXIMUM_STEPS = 15000
BATCH_SIZE_SEQUENCES = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
MAX_CHECKPOINTS = 2
VOCABULARY_SIZE = 2000


@dataclass(frozen=True)
class ExperimentSpecification:
    name: str
    model: ModelConfiguration
    auxiliary_loss_weight: float


@dataclass(frozen=True)
class GeneratedConfiguration:
    path: Path
    parameter_summary: ModelParameterSummary


@dataclass(frozen=True)
class ParsedArguments:
    base_configuration_path: Path
    output_directory: Path


def main() -> int:
    arguments = parse_arguments()
    generated_configurations = generate_python_model_sweep_three(
        base_configuration_path=arguments.base_configuration_path,
        output_directory=arguments.output_directory,
    )
    for generated_configuration in generated_configurations:
        summary = generated_configuration.parameter_summary
        print(
            f"{generated_configuration.path} "
            f"total={summary.total_parameters} "
            f"active={summary.active_parameters} "
            f"trainable_active={summary.trainable_active_parameters}",
        )
    return 0


def parse_arguments() -> ParsedArguments:
    parser = argparse.ArgumentParser(
        description="Generate the third Python model sweep experiment configurations.",
    )
    parser.add_argument(
        "--base-configuration-path",
        default=DEFAULT_BASE_CONFIGURATION_PATH,
        type=Path,
        help="Base experiment configuration to specialize.",
    )
    parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
        type=Path,
        help="Directory where generated YAML files are written.",
    )
    namespace = parser.parse_args()
    return ParsedArguments(
        base_configuration_path=namespace.base_configuration_path,
        output_directory=namespace.output_directory,
    )


def generate_python_model_sweep_three(
    base_configuration_path: Path,
    output_directory: Path,
) -> tuple[GeneratedConfiguration, ...]:
    base_configuration = load_experiment_configuration(
        configuration_path=base_configuration_path,
    )
    clear_existing_generated_configs(output_directory=output_directory)
    generated_configurations: list[GeneratedConfiguration] = []
    for experiment_specification in experiment_specifications():
        output_path = export_experiment_variant(
            base_configuration=base_configuration,
            output_directory=output_directory,
            overrides=overrides_for_experiment(
                base_configuration=base_configuration,
                experiment_specification=experiment_specification,
            ),
        )
        generated_configurations.append(
            GeneratedConfiguration(
                path=output_path,
                parameter_summary=parameter_summary(
                    model_configuration=experiment_specification.model,
                    vocabulary_size=VOCABULARY_SIZE,
                ),
            ),
        )
    return tuple(generated_configurations)


def clear_existing_generated_configs(output_directory: Path) -> None:
    if not output_directory.exists():
        return
    for configuration_path in output_directory.glob("*.yaml"):
        configuration_path.unlink()


def experiment_specifications() -> tuple[ExperimentSpecification, ...]:
    return (
        modern_moe_vocab2000_aux010(),
        modern_moe_vocab2000_aux020(),
        modern_moe_vocab2000_topk2_aux010(),
        modern_moe_vocab2000_topk2_aux020(),
        modern_moe_deep10_vocab2000_aux020(),
        modern_dense_active10m_vocab2000(),
    )


def modern_moe_vocab2000_aux010() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_moe_vocab2000_aux010",
        model=small_topk1_model(),
        auxiliary_loss_weight=0.1,
    )


def modern_moe_vocab2000_aux020() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_moe_vocab2000_aux020",
        model=small_topk1_model(),
        auxiliary_loss_weight=0.2,
    )


def modern_moe_vocab2000_topk2_aux010() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_moe_vocab2000_topk2_aux010",
        model=small_topk2_model(),
        auxiliary_loss_weight=0.1,
    )


def modern_moe_vocab2000_topk2_aux020() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_moe_vocab2000_topk2_aux020",
        model=small_topk2_model(),
        auxiliary_loss_weight=0.2,
    )


def modern_moe_deep10_vocab2000_aux020() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_moe_deep10_vocab2000_aux020",
        model=modern_moe_model(
            dimension=256,
            layers=10,
            attention_heads=8,
            expert_feed_forward_dimension=864,
            expert_count=4,
            router_top_k=1,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.2,
    )


def modern_dense_active10m_vocab2000() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_active10m_vocab2000",
        model=modern_dense_model(
            dimension=320,
            layers=6,
            attention_heads=8,
            feed_forward_dimension=1152,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.0,
    )


def small_topk1_model() -> ModelConfiguration:
    return modern_moe_model(
        dimension=112,
        layers=4,
        attention_heads=4,
        expert_feed_forward_dimension=416,
        expert_count=4,
        router_top_k=1,
        dropout=0.05,
    )


def small_topk2_model() -> ModelConfiguration:
    return modern_moe_model(
        dimension=80,
        layers=4,
        attention_heads=4,
        expert_feed_forward_dimension=384,
        expert_count=4,
        router_top_k=2,
        dropout=0.05,
    )


def overrides_for_experiment(
    base_configuration: ExperimentFile,
    experiment_specification: ExperimentSpecification,
) -> ExperimentOverrides:
    return ExperimentOverrides(
        name=experiment_specification.name,
        model=experiment_specification.model,
        training=training(
            base_training=base_configuration.training,
            maximum_steps=MAXIMUM_STEPS,
            batch_size_sequences=BATCH_SIZE_SEQUENCES,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            auxiliary_loss_weight=experiment_specification.auxiliary_loss_weight,
            max_checkpoints=MAX_CHECKPOINTS,
        ),
        tokenizer=tokenizer_configuration(
            base_tokenizer=base_configuration.tokenizer,
            vocabulary_size=VOCABULARY_SIZE,
        ),
        export=ExportConfiguration(
            bundle_path=Path("export/bundle.zip"),
            include_tensorboard=True,
            include_all_checkpoints=False,
        ),
    )


def tokenizer_configuration(
    base_tokenizer: TokenizerConfiguration,
    vocabulary_size: int,
) -> TokenizerConfiguration:
    match base_tokenizer:
        case ByteBpeTokenizerConfiguration():
            return ByteBpeTokenizerConfiguration(
                type=base_tokenizer.type,
                vocabulary_size=vocabulary_size,
                max_training_documents=base_tokenizer.max_training_documents,
                max_training_bytes=base_tokenizer.max_training_bytes,
                training_workers=base_tokenizer.training_workers,
                add_bos_token=base_tokenizer.add_bos_token,
                add_eos_token=base_tokenizer.add_eos_token,
                add_pad_token=base_tokenizer.add_pad_token,
                additional_special_tokens=base_tokenizer.additional_special_tokens,
            )
        case RustByteBpeTokenizerConfiguration():
            return RustByteBpeTokenizerConfiguration(
                type=base_tokenizer.type,
                vocabulary_size=vocabulary_size,
                max_training_documents=base_tokenizer.max_training_documents,
                max_training_bytes=base_tokenizer.max_training_bytes,
                training_workers=base_tokenizer.training_workers,
                add_bos_token=base_tokenizer.add_bos_token,
                add_eos_token=base_tokenizer.add_eos_token,
                add_pad_token=base_tokenizer.add_pad_token,
                additional_special_tokens=base_tokenizer.additional_special_tokens,
            )
        case _:
            raise ValueError("Sweep-three generation requires a BPE tokenizer configuration.")


def parameter_summary(
    model_configuration: ModelConfiguration,
    vocabulary_size: int,
) -> ModelParameterSummary:
    model = build_model(
        model_configuration=model_configuration,
        vocabulary_size=vocabulary_size,
    )
    return model_parameter_summary(model=model)


if __name__ == "__main__":
    raise SystemExit(main())
