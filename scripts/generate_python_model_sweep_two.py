import argparse
from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.export import (
    ExperimentOverrides,
    export_experiment_variant,
    modern_dense_model,
    modern_moe_model,
    moe_model,
    training,
)
from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    ByteBpeTokenizerConfiguration,
    CosineWarmupDecayLearningRateScheduleConfiguration,
    ExperimentFile,
    ExportConfiguration,
    LearningRateScheduleType,
    ModelConfiguration,
    RustByteBpeTokenizerConfiguration,
    TokenizerConfiguration,
)
from llm_lite.model.factory import build_model
from llm_lite.model.parameters import ModelParameterSummary, model_parameter_summary

DEFAULT_BASE_CONFIGURATION_PATH = Path("configs/python_moe_full.yaml")
DEFAULT_OUTPUT_DIRECTORY = Path("configs/generated/python_model_sweep_two")
MAXIMUM_STEPS = 15000
BATCH_SIZE_SEQUENCES = 256
DEFAULT_LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
MAX_CHECKPOINTS = 2
VOCABULARY_SIZE_2000 = 2000
VOCABULARY_SIZE_4000 = 4000


@dataclass(frozen=True)
class ExperimentSpecification:
    name: str
    model: ModelConfiguration
    vocabulary_size: int
    batch_size_sequences: int = BATCH_SIZE_SEQUENCES
    auxiliary_loss_weight: float = 0.0
    learning_rate: float = DEFAULT_LEARNING_RATE
    learning_rate_schedule: CosineWarmupDecayLearningRateScheduleConfiguration | None = None


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
    generated_configurations = generate_python_model_sweep_two(
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
        description="Generate the second Python model sweep experiment configurations.",
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


def generate_python_model_sweep_two(
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
                    vocabulary_size=experiment_specification.vocabulary_size,
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
        modern_dense_baseline(),
        modern_dense_vocab2000(),
        modern_dense_qknorm_vocab4000(),
        modern_dense_qknorm_lr2e3(),
        modern_dense_cosine_lr2e3_qknorm(),
        modern_dense_heads4(),
        modern_dense_heads8(),
        modern_dense_heads12(),
        modern_dense_ffn_small(),
        modern_dense_ffn_medium(),
        modern_dense_ffn_large(),
        modern_dense_deep_qknorm(),
        modern_moe_vocab4000_aux010(),
        moe_vocab4000_aux010(),
    )


def modern_dense_baseline() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_vocab4000_baseline",
        model=baseline_modern_dense_model(query_key_normalization=False),
        vocabulary_size=VOCABULARY_SIZE_4000,
    )


def modern_dense_vocab2000() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_vocab2000_param_match",
        model=modern_dense_model(
            dimension=128,
            layers=2,
            attention_heads=4,
            feed_forward_dimension=768,
            dropout=0.05,
        ),
        vocabulary_size=VOCABULARY_SIZE_2000,
    )


def modern_dense_qknorm_vocab4000() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_qknorm_vocab4000",
        model=baseline_modern_dense_model(query_key_normalization=True),
        vocabulary_size=VOCABULARY_SIZE_4000,
    )


def modern_dense_qknorm_lr2e3() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_qknorm_lr2e3",
        model=baseline_modern_dense_model(query_key_normalization=True),
        vocabulary_size=VOCABULARY_SIZE_4000,
        learning_rate=0.002,
    )


def modern_dense_cosine_lr2e3_qknorm() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_cosine_lr2e3_qknorm",
        model=baseline_modern_dense_model(query_key_normalization=True),
        vocabulary_size=VOCABULARY_SIZE_4000,
        learning_rate=0.002,
        learning_rate_schedule=CosineWarmupDecayLearningRateScheduleConfiguration(
            type=LearningRateScheduleType.COSINE_WARMUP_DECAY,
            warmup_steps=150,
            minimum_learning_rate_ratio=0.1,
        ),
    )


def modern_dense_heads4() -> ExperimentSpecification:
    return head_sweep_experiment(name="python_modern_dense_heads4", attention_heads=4)


def modern_dense_heads8() -> ExperimentSpecification:
    return head_sweep_experiment(name="python_modern_dense_heads8", attention_heads=8)


def modern_dense_heads12() -> ExperimentSpecification:
    return head_sweep_experiment(name="python_modern_dense_heads12", attention_heads=12)


def modern_dense_ffn_small() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_ffn_small",
        model=modern_dense_model(
            dimension=144,
            layers=2,
            attention_heads=4,
            feed_forward_dimension=256,
            dropout=0.05,
        ),
        vocabulary_size=VOCABULARY_SIZE_4000,
    )


def modern_dense_ffn_medium() -> ExperimentSpecification:
    return ffn_sweep_experiment(
        name="python_modern_dense_ffn_medium",
        feed_forward_dimension=384,
    )


def modern_dense_ffn_large() -> ExperimentSpecification:
    return ffn_sweep_experiment(
        name="python_modern_dense_ffn_large",
        feed_forward_dimension=512,
        batch_size_sequences=128,
    )


def modern_dense_deep_qknorm() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_dense_deep_qknorm",
        model=modern_dense_model(
            dimension=96,
            layers=12,
            attention_heads=6,
            feed_forward_dimension=64,
            dropout=0.05,
            query_key_normalization=True,
        ),
        vocabulary_size=VOCABULARY_SIZE_4000,
    )


def modern_moe_vocab4000_aux010() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_moe_vocab4000_aux010",
        model=modern_moe_model(
            dimension=96,
            layers=4,
            attention_heads=4,
            expert_feed_forward_dimension=384,
            expert_count=4,
            router_top_k=1,
            dropout=0.05,
        ),
        vocabulary_size=VOCABULARY_SIZE_4000,
        auxiliary_loss_weight=0.1,
    )


def moe_vocab4000_aux010() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_moe_vocab4000_aux010",
        model=moe_model(
            dimension=96,
            layers=4,
            attention_heads=4,
            expert_feed_forward_dimension=384,
            expert_count=4,
            router_top_k=1,
            dropout=0.05,
        ),
        vocabulary_size=VOCABULARY_SIZE_4000,
        auxiliary_loss_weight=0.1,
    )


def baseline_modern_dense_model(query_key_normalization: bool) -> ModelConfiguration:
    return modern_dense_model(
        dimension=128,
        layers=2,
        attention_heads=4,
        feed_forward_dimension=384,
        dropout=0.05,
        query_key_normalization=query_key_normalization,
    )


def head_sweep_experiment(name: str, attention_heads: int) -> ExperimentSpecification:
    return ExperimentSpecification(
        name=name,
        model=modern_dense_model(
            dimension=192,
            layers=1,
            attention_heads=attention_heads,
            feed_forward_dimension=128,
            dropout=0.05,
        ),
        vocabulary_size=VOCABULARY_SIZE_4000,
    )


def ffn_sweep_experiment(
    name: str,
    feed_forward_dimension: int,
    batch_size_sequences: int = BATCH_SIZE_SEQUENCES,
) -> ExperimentSpecification:
    return ExperimentSpecification(
        name=name,
        model=modern_dense_model(
            dimension=128,
            layers=2,
            attention_heads=4,
            feed_forward_dimension=feed_forward_dimension,
            dropout=0.05,
        ),
        vocabulary_size=VOCABULARY_SIZE_4000,
        batch_size_sequences=batch_size_sequences,
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
            batch_size_sequences=experiment_specification.batch_size_sequences,
            learning_rate=experiment_specification.learning_rate,
            weight_decay=WEIGHT_DECAY,
            auxiliary_loss_weight=experiment_specification.auxiliary_loss_weight,
            learning_rate_schedule=experiment_specification.learning_rate_schedule,
            max_checkpoints=MAX_CHECKPOINTS,
        ),
        tokenizer=tokenizer_configuration(
            base_tokenizer=base_configuration.tokenizer,
            vocabulary_size=experiment_specification.vocabulary_size,
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
            raise ValueError("Sweep-two generation requires a BPE tokenizer configuration.")


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
