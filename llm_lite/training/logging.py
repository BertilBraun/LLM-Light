import json
from pathlib import Path


class JsonlMetricLogger:
    def __init__(self, metrics_path: Path) -> None:
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, step: int, loss: float) -> None:
        record = {"step": step, "loss": loss}
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
