from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def write_json_model(model: BaseModel, output_path: Path) -> None:
    output_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def write_yaml_mapping(mapping: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
