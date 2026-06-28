from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import LearningRateScheduleType, ModelType
from scripts.generate_python_model_sweep import (
    SweepMode,
    generate_python_model_sweep,
)


def test_generate_python_model_sweep_pilot_writes_four_small_configs(
    tmp_path: Path,
) -> None:
    stale_configuration_path = tmp_path / "stale.yaml"
    stale_configuration_path.write_text("stale: true\n", encoding="utf-8")

    generated_configurations = generate_python_model_sweep(
        mode=SweepMode.PILOT,
        base_configuration_path=Path("configs/python_moe_full.yaml"),
        output_directory=tmp_path,
    )

    assert tuple(
        generated_configuration.path.name for generated_configuration in generated_configurations
    ) == (
        "python_moe_small_deep_plain.yaml",
        "python_moe_small_wide_plain.yaml",
        "python_dense_small_deep_plain.yaml",
        "python_dense_small_wide_plain.yaml",
    )
    assert tuple(
        generated_configuration.parameter_summary.active_parameters
        for generated_configuration in generated_configurations
    ) == (995984, 993824, 994576, 992992)
    assert not stale_configuration_path.exists()

    for generated_configuration in generated_configurations:
        experiment_configuration = load_experiment_configuration(
            configuration_path=generated_configuration.path,
        )
        assert experiment_configuration.training.batch_size_sequences == 512
        assert experiment_configuration.training.max_checkpoints == 2
        assert experiment_configuration.export.bundle_path == Path("export/bundle.zip")
        assert generated_configuration.parameter_summary.active_parameters < 1_000_000


def test_generate_python_model_sweep_full_adds_minimal_ablation_set(
    tmp_path: Path,
) -> None:
    generated_configurations = generate_python_model_sweep(
        mode=SweepMode.FULL,
        base_configuration_path=Path("configs/python_moe_full.yaml"),
        output_directory=tmp_path,
    )

    assert tuple(
        generated_configuration.path.name for generated_configuration in generated_configurations
    ) == (
        "python_moe_small_deep_plain.yaml",
        "python_moe_small_wide_plain.yaml",
        "python_dense_small_deep_plain.yaml",
        "python_dense_small_wide_plain.yaml",
        "python_moe_small_deep_fim.yaml",
        "python_moe_small_deep_linear_warmup_decay.yaml",
        "python_moe_small_deep_cosine_warmup_decay.yaml",
        "python_modern_moe_small_deep_plain.yaml",
        "python_dense_active_9m6.yaml",
    )

    fim_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_moe_small_deep_fim.yaml",
    )
    assert fim_configuration.packing.fill_in_middle.enabled
    assert set(fim_configuration.tokenizer.additional_special_tokens) >= {
        "<fim_prefix>",
        "<fim_suffix>",
        "<fim_middle>",
    }

    linear_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_moe_small_deep_linear_warmup_decay.yaml",
    )
    assert (
        linear_configuration.training.optimizer.learning_rate_schedule.type
        is LearningRateScheduleType.LINEAR_WARMUP_DECAY
    )

    cosine_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_moe_small_deep_cosine_warmup_decay.yaml",
    )
    assert (
        cosine_configuration.training.optimizer.learning_rate_schedule.type
        is LearningRateScheduleType.COSINE_WARMUP_DECAY
    )

    modern_configuration = load_experiment_configuration(
        configuration_path=tmp_path / "python_modern_moe_small_deep_plain.yaml",
    )
    assert modern_configuration.model.type is ModelType.MODERN_MOE_GPT

    large_dense_summary = generated_configurations[-1].parameter_summary
    assert large_dense_summary.active_parameters == 9_646_080
