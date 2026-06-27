from pathlib import Path

import pytest

HEAVY_TEST_PATH_PARTS = (Path("tests") / "integration",)
HEAVY_TEST_IDS = frozenset(
    {
        "tests/test_moe_model.py::test_tiny_pipeline_config_trains_moe_for_a_few_steps",
        "tests/test_training_checkpoint.py::test_training_checkpoint_resume",
        "tests/test_distributed_training.py::test_two_process_gloo_data_parallel_tiny_training_and_resume",
        "tests/test_orchestration_models.py::test_pipeline_writes_resolved_run_and_semantic_manifest",
        "tests/test_orchestration_models.py::test_run_plan_writes_raw_dataset_to_artifact_store",
        "tests/test_orchestration_models.py::test_run_plan_accepts_parallel_sweep_configs",
    },
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="Run pipeline, subprocess, and training-heavy tests.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    for item in items:
        if _is_heavy_test(item=item):
            item.add_marker(pytest.mark.heavy)
    if config.getoption("--run-heavy"):
        return
    skip_heavy = pytest.mark.skip(reason="requires --run-heavy")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)


def _is_heavy_test(item: pytest.Item) -> bool:
    item_path = Path(str(item.path)).as_posix()
    if any(path_part.as_posix() in item_path for path_part in HEAVY_TEST_PATH_PARTS):
        return True
    return item.nodeid in HEAVY_TEST_IDS
