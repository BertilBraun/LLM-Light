import argparse
from pathlib import Path

from llm_lite.orchestration.checkpoint_evaluation import (
    checkpoint_evaluation_artifact,
    checkpoint_evaluation_target_from_manifest,
)
from llm_lite.orchestration.runtime import load_resolved_run, run_stage_job
from llm_lite.pipeline.stage import StageName


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--resolved-config", required=True, type=Path)
    argument_parser.add_argument(
        "--stage",
        required=True,
        choices=[stage_name.value for stage_name in StageName],
    )
    argument_parser.add_argument("--fingerprint", required=True)
    argument_parser.add_argument("--checkpoint-manifest", type=Path)
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    resolved_run = load_resolved_run(resolved_configuration_path=arguments.resolved_config)
    checkpoint_evaluation_target = (
        None
        if arguments.checkpoint_manifest is None
        else checkpoint_evaluation_target_from_manifest(
            resolved_run=resolved_run,
            checkpoint_manifest_path=arguments.checkpoint_manifest,
        )
    )
    planned_artifact_override = (
        None
        if checkpoint_evaluation_target is None
        else checkpoint_evaluation_artifact(
            resolved_run=resolved_run,
            target=checkpoint_evaluation_target,
        )
    )
    run_stage_job(
        resolved_run=resolved_run,
        stage_name=StageName(arguments.stage),
        expected_fingerprint=arguments.fingerprint,
        planned_artifact_override=planned_artifact_override,
        checkpoint_evaluation_target=checkpoint_evaluation_target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
