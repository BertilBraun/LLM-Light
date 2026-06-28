import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from llm_lite.config.export import (
    ExperimentOverrides,
    dense_model,
    export_experiment_variant,
    modern_moe_model,
    moe_model,
    training,
)
from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    CosineWarmupDecayLearningRateScheduleConfiguration,
    ExperimentFile,
    ExportConfiguration,
    FillInMiddleConfiguration,
    LearningRateScheduleType,
    LinearWarmupDecayLearningRateScheduleConfiguration,
    ModelConfiguration,
    PackingConfiguration,
    TokenizerConfiguration,
)
from llm_lite.model.factory import build_model
from llm_lite.model.parameters import ModelParameterSummary, model_parameter_summary

DEFAULT_BASE_CONFIGURATION_PATH = Path("configs/python_moe_full.yaml")
DEFAULT_OUTPUT_DIRECTORY = Path("configs/generated/python_model_sweep")
SMALL_MAXIMUM_STEPS = 1500
LARGE_MAXIMUM_STEPS = 3750
BATCH_SIZE_SEQUENCES = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
MAX_CHECKPOINTS = 2
FIM_MARKERS = ("<fim_prefix>", "<fim_suffix>", "<fim_middle>")


class SweepMode(str, Enum):
    PILOT = "pilot"
    FULL = "full"


@dataclass(frozen=True)
class ExperimentSpecification:
    name: str
    model: ModelConfiguration
    auxiliary_loss_weight: float
    maximum_steps: int = SMALL_MAXIMUM_STEPS
    fim_enabled: bool = False
    learning_rate_schedule: (
        LinearWarmupDecayLearningRateScheduleConfiguration
        | CosineWarmupDecayLearningRateScheduleConfiguration
        | None
    ) = None


@dataclass(frozen=True)
class GeneratedConfiguration:
    path: Path
    parameter_summary: ModelParameterSummary


@dataclass(frozen=True)
class ParsedArguments:
    mode: SweepMode
    base_configuration_path: Path
    output_directory: Path


def main() -> int:
    arguments = parse_arguments()
    generated_configurations = generate_python_model_sweep(
        mode=arguments.mode,
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
        description="Generate Python model sweep experiment configurations.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in SweepMode),
        default=SweepMode.PILOT,
        type=SweepMode,
        help="Generate the four-config pilot set or the full minimal sweep.",
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
        mode=namespace.mode,
        base_configuration_path=namespace.base_configuration_path,
        output_directory=namespace.output_directory,
    )


def generate_python_model_sweep(
    mode: SweepMode,
    base_configuration_path: Path,
    output_directory: Path,
) -> tuple[GeneratedConfiguration, ...]:
    base_configuration = load_experiment_configuration(
        configuration_path=base_configuration_path,
    )
    clear_existing_generated_configs(output_directory=output_directory)
    generated_configurations: list[GeneratedConfiguration] = []
    for experiment_specification in experiment_specifications(mode=mode):
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
                    base_configuration=base_configuration,
                    model_configuration=experiment_specification.model,
                ),
            ),
        )
    return tuple(generated_configurations)


def clear_existing_generated_configs(output_directory: Path) -> None:
    if not output_directory.exists():
        return
    for configuration_path in output_directory.glob("*.yaml"):
        configuration_path.unlink()


def experiment_specifications(mode: SweepMode) -> tuple[ExperimentSpecification, ...]:
    pilot_specifications = (
        small_moe_deep_plain(),
        small_moe_wide_plain(),
        small_dense_deep_plain(),
        small_dense_wide_plain(),
    )
    if mode is SweepMode.PILOT:
        return pilot_specifications
    return (
        *pilot_specifications,
        small_moe_deep_fim(),
        small_moe_deep_linear_schedule(),
        small_moe_deep_cosine_schedule(),
        small_modern_moe_deep(),
        large_dense_active_parameter_match(),
    )


def small_moe_deep_plain() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_moe_small_deep_plain",
        model=moe_model(
            dimension=88,
            layers=4,
            attention_heads=4,
            expert_feed_forward_dimension=352,
            expert_count=4,
            router_top_k=1,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.01,
    )


def small_moe_wide_plain() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_moe_small_wide_plain",
        model=moe_model(
            dimension=104,
            layers=2,
            attention_heads=4,
            expert_feed_forward_dimension=416,
            expert_count=4,
            router_top_k=1,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.01,
    )


def small_dense_deep_plain() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_dense_small_deep_plain",
        model=dense_model(
            dimension=88,
            layers=4,
            attention_heads=4,
            feed_forward_dimension=352,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.0,
    )


def small_dense_wide_plain() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_dense_small_wide_plain",
        model=dense_model(
            dimension=104,
            layers=2,
            attention_heads=4,
            feed_forward_dimension=416,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.0,
    )


def small_moe_deep_fim() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_moe_small_deep_fim",
        model=small_moe_deep_plain().model,
        auxiliary_loss_weight=0.01,
        fim_enabled=True,
    )


def small_moe_deep_linear_schedule() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_moe_small_deep_linear_warmup_decay",
        model=small_moe_deep_plain().model,
        auxiliary_loss_weight=0.01,
        learning_rate_schedule=LinearWarmupDecayLearningRateScheduleConfiguration(
            type=LearningRateScheduleType.LINEAR_WARMUP_DECAY,
            warmup_steps=150,
            minimum_learning_rate_ratio=0.1,
        ),
    )


def small_moe_deep_cosine_schedule() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_moe_small_deep_cosine_warmup_decay",
        model=small_moe_deep_plain().model,
        auxiliary_loss_weight=0.01,
        learning_rate_schedule=CosineWarmupDecayLearningRateScheduleConfiguration(
            type=LearningRateScheduleType.COSINE_WARMUP_DECAY,
            warmup_steps=150,
            minimum_learning_rate_ratio=0.1,
        ),
    )


def small_modern_moe_deep() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_modern_moe_small_deep_plain",
        model=modern_moe_model(
            dimension=88,
            layers=4,
            attention_heads=4,
            expert_feed_forward_dimension=256,
            expert_count=4,
            router_top_k=1,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.01,
    )


def large_dense_active_parameter_match() -> ExperimentSpecification:
    return ExperimentSpecification(
        name="python_dense_active_9m6",
        model=dense_model(
            dimension=320,
            layers=6,
            attention_heads=8,
            feed_forward_dimension=1280,
            dropout=0.05,
        ),
        auxiliary_loss_weight=0.0,
        maximum_steps=LARGE_MAXIMUM_STEPS,
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
            maximum_steps=experiment_specification.maximum_steps,
            batch_size_sequences=BATCH_SIZE_SEQUENCES,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            auxiliary_loss_weight=experiment_specification.auxiliary_loss_weight,
            learning_rate_schedule=experiment_specification.learning_rate_schedule,
            max_checkpoints=MAX_CHECKPOINTS,
        ),
        tokenizer=tokenizer_configuration(
            base_configuration=base_configuration,
            fim_enabled=experiment_specification.fim_enabled,
        ),
        packing=packing_configuration(
            base_configuration=base_configuration,
            fim_enabled=experiment_specification.fim_enabled,
        ),
        export=ExportConfiguration(
            bundle_path=Path("export/bundle.zip"),
            include_tensorboard=True,
            include_all_checkpoints=False,
        ),
    )


def tokenizer_configuration(
    base_configuration: ExperimentFile,
    fim_enabled: bool,
) -> TokenizerConfiguration:
    if not fim_enabled:
        return base_configuration.tokenizer
    return base_configuration.tokenizer.model_copy(
        update={
            "additional_special_tokens": append_missing_tokens(
                configured_tokens=base_configuration.tokenizer.additional_special_tokens,
                required_tokens=FIM_MARKERS,
            ),
        },
    )


def packing_configuration(
    base_configuration: ExperimentFile,
    fim_enabled: bool,
) -> PackingConfiguration:
    if not fim_enabled:
        return base_configuration.packing
    return base_configuration.packing.model_copy(
        update={
            "fill_in_middle": FillInMiddleConfiguration(
                enabled=True,
                probability=0.5,
                minimum_segment_characters=8,
                prefix_marker=FIM_MARKERS[0],
                suffix_marker=FIM_MARKERS[1],
                middle_marker=FIM_MARKERS[2],
            ),
        },
    )


def append_missing_tokens(
    configured_tokens: tuple[str, ...],
    required_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    appended_tokens = list(configured_tokens)
    for required_token in required_tokens:
        if required_token not in appended_tokens:
            appended_tokens.append(required_token)
    return tuple(appended_tokens)


def parameter_summary(
    base_configuration: ExperimentFile,
    model_configuration: ModelConfiguration,
) -> ModelParameterSummary:
    model = build_model(
        model_configuration=model_configuration,
        vocabulary_size=base_configuration.tokenizer.vocabulary_size,
    )
    return model_parameter_summary(model=model)


if __name__ == "__main__":
    raise SystemExit(main())
