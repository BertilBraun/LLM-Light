from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


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
    summary_writer = SummaryWriter(log_dir=str(tensorboard_directory))
    try:
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_value, int | float | bool):
                continue
            summary_writer.add_scalar(
                EVALUATION_SCALAR_TAGS.get(metric_name, f"eval/{metric_name}"),
                float(metric_value),
                step,
            )
        summary_writer.flush()
    finally:
        summary_writer.close()
