from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from llm_lite.pipeline.tensorboard import configured_run_tensorboard_directory

EVALUATION_TENSORBOARD_DIRECTORY_NAME = "tensorboard"

EVALUATION_SCALAR_TAGS = {
    "perplexity": "eval/perplexity",
    "perplexity_loss": "eval/perplexity_loss",
    "perplexity_documents": "eval/perplexity/documents",
    "perplexity_sequences": "eval/perplexity/sequences",
    "python_completion_tasks": "eval/python_completion/tasks",
    "python_completion_parsed_tasks": "eval/python_completion/parsed_tasks",
    "python_completion_executed_tasks": "eval/python_completion/executed_tasks",
    "python_completion_passed_checks": "eval/python_completion/passed_checks",
    "python_completion_total_checks": "eval/python_completion/total_checks",
    "python_completion_pass_rate": "eval/python_completion/pass_rate",
    "exact_reproduction_passed": "eval/exact_reproduction/passed",
    "fixed_prompt_generation_samples": "eval/fixed_prompt_generation/samples",
}


def write_evaluation_metrics_to_tensorboard(
    tensorboard_directory: Path,
    metrics: dict[str, int | float | str | bool],
    step: int,
) -> None:
    tensorboard_directory.mkdir(parents=True, exist_ok=True)
    run_tensorboard_directory = configured_run_tensorboard_directory()
    if run_tensorboard_directory is not None:
        run_tensorboard_directory.mkdir(parents=True, exist_ok=True)
        summary_writers = (
            SummaryWriter(log_dir=str(tensorboard_directory)),
            SummaryWriter(log_dir=str(run_tensorboard_directory)),
        )
    else:
        summary_writers = (SummaryWriter(log_dir=str(tensorboard_directory)),)
    try:
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_value, int | float | bool):
                continue
            for summary_writer in summary_writers:
                summary_writer.add_scalar(
                    _tensorboard_tag(metric_name=metric_name),
                    float(metric_value),
                    step,
                )
        for summary_writer in summary_writers:
            summary_writer.flush()
    finally:
        for summary_writer in summary_writers:
            summary_writer.close()


def _tensorboard_tag(metric_name: str) -> str:
    known_tag = EVALUATION_SCALAR_TAGS.get(metric_name)
    if known_tag is not None:
        return known_tag
    family_prefix = "python_completion_family_"
    family_suffix = "_pass_rate"
    if metric_name.startswith(family_prefix) and metric_name.endswith(family_suffix):
        family_name = metric_name.removeprefix(family_prefix).removesuffix(family_suffix)
        return f"eval/python_completion/family/{family_name}/pass_rate"
    return f"eval/{metric_name}"
