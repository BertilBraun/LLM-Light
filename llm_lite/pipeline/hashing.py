import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def hash_model(model: BaseModel) -> str:
    dumped_model = model.model_dump(mode="json")
    return hash_json_value(value=dumped_model)


def hash_json_value(value: Any) -> str:
    encoded_value = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _format_digest(digest=hashlib.sha256(encoded_value).hexdigest())


def hash_file(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            sha256_hash.update(chunk)
    return _format_digest(digest=sha256_hash.hexdigest())


def _format_digest(digest: str) -> str:
    return f"sha256:{digest}"
