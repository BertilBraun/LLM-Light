from pathlib import Path
from typing import Any

import yaml

from llm_lite.config.models import ExperimentFile


def load_experiment_configuration(configuration_path: Path) -> ExperimentFile:
    configuration_data = _load_yaml_mapping(configuration_path=configuration_path)
    return ExperimentFile.model_validate(configuration_data)


def _load_yaml_mapping(configuration_path: Path) -> dict[str, Any]:
    with configuration_path.open("r", encoding="utf-8") as configuration_file:
        configuration_data = yaml.safe_load(configuration_file)
    match configuration_data:
        case dict():
            return configuration_data
        case _:
            raise ValueError("Experiment configuration must be a YAML mapping.")
