import os
from pathlib import Path

RUN_TENSORBOARD_DIRECTORY_ENVIRONMENT = "LLM_LITE_RUN_TENSORBOARD_DIRECTORY"


def configured_run_tensorboard_directory() -> Path | None:
    tensorboard_directory = os.environ.get(RUN_TENSORBOARD_DIRECTORY_ENVIRONMENT)
    if tensorboard_directory is None:
        return None
    return Path(tensorboard_directory)
