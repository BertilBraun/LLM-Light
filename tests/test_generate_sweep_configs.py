from pathlib import Path

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.scripts.generate_sweep_configs import generate_sweep_configs


def test_generate_sweep_configs_writes_loadable_experiment_configs(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    output_directory = tmp_path / "generated"
    sweep_path.write_text(
        "\n".join(
            (
                'name_template: "python_small_d{model.dimension}_l{model.layers}"',
                "grid:",
                "  model.dimension: [16, 24]",
                "  model.layers: [1]",
            ),
        ),
        encoding="utf-8",
    )

    output_paths = generate_sweep_configs(
        base_configuration_path=Path("configs/verify_one_sentence.yaml"),
        sweep_configuration_path=sweep_path,
        output_directory=output_directory,
    )

    first_configuration = load_experiment_configuration(configuration_path=output_paths[0])
    second_configuration = load_experiment_configuration(configuration_path=output_paths[1])

    assert tuple(output_path.name for output_path in output_paths) == (
        "python_small_d16_l1.yaml",
        "python_small_d24_l1.yaml",
    )
    assert first_configuration.experiment.name == "python_small_d16_l1"
    assert first_configuration.experiment.output_dir == Path("runs/python_small_d16_l1")
    assert first_configuration.model.dimension == 16
    assert second_configuration.model.dimension == 24
