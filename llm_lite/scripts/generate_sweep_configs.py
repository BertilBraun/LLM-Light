import argparse
import itertools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SweepConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    name_template: str
    grid: dict[str, tuple[str | int | float | bool, ...]] = Field(min_length=1)


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--base-config", required=True, type=Path)
    argument_parser.add_argument("--sweep", required=True, type=Path)
    argument_parser.add_argument("--output-dir", required=True, type=Path)
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    generate_sweep_configs(
        base_configuration_path=arguments.base_config,
        sweep_configuration_path=arguments.sweep,
        output_directory=arguments.output_dir,
    )
    return 0


def generate_sweep_configs(
    base_configuration_path: Path,
    sweep_configuration_path: Path,
    output_directory: Path,
) -> tuple[Path, ...]:
    base_configuration = _load_yaml_mapping(path=base_configuration_path)
    sweep_configuration = SweepConfiguration.model_validate(
        _load_yaml_mapping(path=sweep_configuration_path),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for assignment in _grid_assignments(sweep_configuration=sweep_configuration):
        generated_configuration = _deep_copy_json_mapping(value=base_configuration)
        for path, assigned_value in assignment.items():
            _set_nested_value(
                configuration=generated_configuration,
                dotted_path=path,
                assigned_value=assigned_value,
            )
        experiment_name = _render_name(
            name_template=sweep_configuration.name_template,
            assignment=assignment,
        )
        _set_nested_value(
            configuration=generated_configuration,
            dotted_path="experiment.name",
            assigned_value=experiment_name,
        )
        _set_nested_value(
            configuration=generated_configuration,
            dotted_path="experiment.output_dir",
            assigned_value=f"runs/{experiment_name}",
        )
        output_path = output_directory / f"{experiment_name}.yaml"
        output_path.write_text(
            yaml.safe_dump(generated_configuration, sort_keys=False),
            encoding="utf-8",
        )
        output_paths.append(output_path)
    return tuple(output_paths)


def _grid_assignments(sweep_configuration: SweepConfiguration) -> tuple[dict[str, Any], ...]:
    grid_paths = tuple(sweep_configuration.grid.keys())
    grid_values = tuple(sweep_configuration.grid[path] for path in grid_paths)
    return tuple(
        dict(zip(grid_paths, value_combination, strict=True))
        for value_combination in itertools.product(*grid_values)
    )


def _render_name(name_template: str, assignment: dict[str, Any]) -> str:
    rendered_name = name_template
    for path, assigned_value in assignment.items():
        rendered_name = rendered_name.replace("{" + path + "}", str(assigned_value))
    return rendered_name


def _set_nested_value(
    configuration: dict[str, Any],
    dotted_path: str,
    assigned_value: Any,
) -> None:
    path_parts = dotted_path.split(".")
    current_mapping = configuration
    for path_part in path_parts[:-1]:
        child_value = current_mapping[path_part]
        if not isinstance(child_value, dict):
            raise ValueError(f"Configuration path {dotted_path} does not resolve to a mapping.")
        current_mapping = child_value
    current_mapping[path_parts[-1]] = assigned_value


def _deep_copy_json_mapping(value: dict[str, Any]) -> dict[str, Any]:
    copied_value = yaml.safe_load(yaml.safe_dump(value))
    if not isinstance(copied_value, dict):
        raise ValueError("Copied configuration must be a mapping.")
    return copied_value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded_value, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return loaded_value


if __name__ == "__main__":
    raise SystemExit(main())
