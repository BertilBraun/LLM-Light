import json
from pathlib import Path

from llm_lite.pipeline.runner import run_pipeline


def test_verify_pipeline_reproduces_sentence(tmp_path: Path) -> None:
    run_directory = tmp_path / "verify_one_sentence"
    configuration_path = tmp_path / "verify_one_sentence.yaml"
    configuration_text = Path("configs/verify_one_sentence.yaml").read_text(encoding="utf-8")
    configuration_path.write_text(
        configuration_text.replace(
            "output_dir: runs/verify_one_sentence",
            f"output_dir: {str(run_directory).replace(chr(92), '/')}",
        ),
        encoding="utf-8",
    )

    exit_code = run_pipeline(
        configuration_path=configuration_path,
        dry_run=False,
        force_stages=(),
    )
    report_path = run_directory / "artifacts" / "evaluation" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pretraining_artifact_directory = run_directory / "artifacts" / "pretraining"
    pretraining_manifest_path = pretraining_artifact_directory / "manifest.json"
    pretraining_manifest = json.loads(pretraining_manifest_path.read_text(encoding="utf-8"))
    pipeline_events_path = run_directory / "pipeline.jsonl"
    pipeline_events = [
        json.loads(event_line)
        for event_line in pipeline_events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert exit_code == 0
    assert report["exact_reproduction"]["passed"] is True
    assert report["exact_reproduction"]["expected_text"] == "hello world\n"
    assert pretraining_manifest["files"]["tensorboard"] == "tensorboard"
    assert list(
        (pretraining_artifact_directory / "tensorboard").glob(
            "events.out.tfevents.*",
        ),
    )
    assert pipeline_events[-1]["event_type"] == "stage_complete"
    assert pipeline_events[-1]["stage_name"] == "evaluation"
