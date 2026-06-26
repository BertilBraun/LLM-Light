import argparse
from pathlib import Path

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
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    resolved_run = load_resolved_run(resolved_configuration_path=arguments.resolved_config)
    run_stage_job(
        resolved_run=resolved_run,
        stage_name=StageName(arguments.stage),
        expected_fingerprint=arguments.fingerprint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
