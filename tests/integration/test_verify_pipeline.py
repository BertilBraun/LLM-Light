import json
import shutil
from pathlib import Path

from llm_lite.pipeline.runner import run_pipeline


def test_verify_pipeline_reproduces_sentence() -> None:
    run_directory = Path("runs/verify_one_sentence")
    if run_directory.exists():
        shutil.rmtree(run_directory)

    exit_code = run_pipeline(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
        dry_run=False,
        force_stages=(),
    )
    report_path = run_directory / "artifacts" / "evaluation" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["passed"] is True
