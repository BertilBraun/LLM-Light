from __future__ import annotations

import argparse
import html
import io
import json
import math
import statistics
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TypeVar

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from llm_lite.config.models import (
    ByteBpeTokenizerConfiguration,
    CharacterTokenizerConfiguration,
    DenseGptConfiguration,
    ExperimentFile,
    ModernDenseGptConfiguration,
    ModernMoeGptConfiguration,
    MoeGptConfiguration,
    RustByteBpeTokenizerConfiguration,
)
from llm_lite.model.factory import build_model
from llm_lite.model.parameters import ModelParameterSummary, model_parameter_summary

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = Mapping[str, JsonValue]
SummaryType = TypeVar("SummaryType")
ObjectMetric = Callable[[SummaryType], float | None]

EXPECTED_EXPERIMENTS = (
    "python_dense_active_9m6",
    "python_dense_small_deep_plain",
    "python_dense_small_wide_plain",
    "python_modern_moe_small_deep_plain",
    "python_moe_small_deep_cosine_warmup_decay",
    "python_moe_small_deep_fim",
    "python_moe_small_deep_linear_warmup_decay",
    "python_moe_small_deep_plain",
    "python_moe_small_wide_plain",
)


@dataclass(frozen=True)
class BundleCandidate:
    artifact_id: str
    outer_bundle_path: str
    experiment: str
    created_at: str
    file_count: int
    report_count: int
    event_count: int
    checkpoint_count: int
    bundle_size_bytes: int
    nested_test_bad_file: str | None


@dataclass(frozen=True)
class RequirementStatus:
    name: str
    present: bool
    detail: str


@dataclass(frozen=True)
class TrainingSummary:
    final_step: int
    final_loss: float
    final_learning_rate: float
    final_gradient_norm: float
    throughput_field: str
    final_tokens_per_second: float
    mean_last_100_tokens_per_second: float
    median_tokens_per_second: float
    minimum_tokens_per_second: float
    maximum_tokens_per_second: float
    world_size: int | None
    strategy: str | None


@dataclass(frozen=True)
class EvaluationMetrics:
    step: int | None
    validation_documents: int | None
    validation_sequences: int | None
    validation_loss: float | None
    perplexity: float | None
    completion_tasks: int | None
    parsed_tasks: int | None
    executed_tasks: int | None
    passed_checks: int | None
    total_checks: int | None
    pass_rate: float | None
    fixed_prompt_samples: int | None
    report_path: str


@dataclass(frozen=True)
class CheckpointSummary:
    evaluations: tuple[EvaluationMetrics, ...]
    best_validation: EvaluationMetrics | None
    best_pass_rate: EvaluationMetrics | None
    final_checkpoint: EvaluationMetrics | None


@dataclass(frozen=True)
class TensorBoardSummary:
    pretraining_event_files: int
    evaluation_event_files: int
    pretraining_scalar_tags: tuple[str, ...]
    evaluation_scalar_tags: tuple[str, ...]
    router_final_worst_dominance: float | None
    router_final_worst_imbalance: float | None
    router_final_worst_entropy: float | None
    router_max_worst_dominance: float | None
    router_max_worst_imbalance: float | None
    router_min_worst_entropy: float | None


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    candidate: BundleCandidate
    requirements: tuple[RequirementStatus, ...]
    configuration: ExperimentFile
    parameter_summary: ModelParameterSummary
    training_summary: TrainingSummary
    final_evaluation: EvaluationMetrics
    checkpoint_summary: CheckpointSummary
    tensorboard_summary: TensorBoardSummary
    analyzed_files: tuple[str, ...]


@dataclass(frozen=True)
class PlotRecord:
    title: str
    path: Path
    caption: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clean-export", required=False, type=Path)
    parser.add_argument("--plots-directory", required=False, type=Path)
    args = parser.parse_args()

    outer_zip_path = args.export
    output_path = args.output
    clean_export_path = args.clean_export
    plots_directory = args.plots_directory or output_path.parent / "images" / "python_model_sweep"

    with zipfile.ZipFile(outer_zip_path) as outer_archive:
        source_bad_file = outer_archive.testzip()
        if source_bad_file is not None:
            raise ValueError(f"Source export ZIP failed integrity check at {source_bad_file}.")
        candidates = discover_bundle_candidates(outer_archive=outer_archive)
        selected_candidates = select_candidates(candidates=candidates)
        if clean_export_path is not None:
            write_clean_export(
                source_archive=outer_archive,
                output_path=clean_export_path,
                selected_candidates=selected_candidates,
            )

    analysis_zip_path = clean_export_path if clean_export_path is not None else outer_zip_path
    with zipfile.ZipFile(analysis_zip_path) as outer_archive:
        outer_bad_file = outer_archive.testzip()
        candidates = discover_bundle_candidates(outer_archive=outer_archive)
        selected_candidates = select_candidates(candidates=candidates)
        summaries = tuple(
            analyze_experiment(outer_archive=outer_archive, candidate=candidate)
            for candidate in selected_candidates
        )

    plot_records = write_plots(summaries=summaries, plots_directory=plots_directory)
    markdown = render_markdown(
        outer_zip_path=analysis_zip_path,
        outer_bad_file=outer_bad_file,
        candidates=candidates,
        summaries=summaries,
        plot_records=plot_records,
        output_path=output_path,
    )
    output_path.write_text(markdown, encoding="utf-8")


def discover_bundle_candidates(outer_archive: zipfile.ZipFile) -> tuple[BundleCandidate, ...]:
    candidates: list[BundleCandidate] = []
    for bundle_manifest_path in sorted(
        path for path in outer_archive.namelist() if path.endswith("bundle_manifest.json")
    ):
        manifest = json_object(json.loads(outer_archive.read(bundle_manifest_path)))
        files = tuple(json_objects(array_field(manifest, "files")))
        archive_paths = tuple(string_field(file_record, "archive_path") for file_record in files)
        outer_directory = bundle_manifest_path.rsplit("/", maxsplit=1)[0]
        outer_bundle_path = f"{outer_directory}/bundle.zip"
        artifact_id = outer_directory.split("/")[-1]
        nested_bad_file = test_nested_bundle(
            outer_archive=outer_archive,
            outer_bundle_path=outer_bundle_path,
        )
        candidates.append(
            BundleCandidate(
                artifact_id=artifact_id,
                outer_bundle_path=outer_bundle_path,
                experiment=string_field(manifest, "experiment"),
                created_at=string_field(manifest, "created_at"),
                file_count=integer_field(manifest, "file_count"),
                report_count=count_suffix(paths=archive_paths, suffix="report.json"),
                event_count=count_contains(paths=archive_paths, needle="events.out.tfevents"),
                checkpoint_count=count_contains(paths=archive_paths, needle="/checkpoints/"),
                bundle_size_bytes=outer_archive.getinfo(outer_bundle_path).file_size,
                nested_test_bad_file=nested_bad_file,
            ),
        )
    return tuple(candidates)


def test_nested_bundle(outer_archive: zipfile.ZipFile, outer_bundle_path: str) -> str | None:
    with zipfile.ZipFile(io.BytesIO(outer_archive.read(outer_bundle_path))) as nested_archive:
        return nested_archive.testzip()


def select_candidates(candidates: Sequence[BundleCandidate]) -> tuple[BundleCandidate, ...]:
    expected_experiments = tuple(
        experiment
        for experiment in EXPECTED_EXPERIMENTS
        if any(candidate.experiment == experiment for candidate in candidates)
    )
    if expected_experiments:
        return select_candidates_for_experiments(
            candidates=candidates,
            experiments=expected_experiments,
        )
    discovered_experiments = tuple(sorted({candidate.experiment for candidate in candidates}))
    return select_candidates_for_experiments(
        candidates=candidates,
        experiments=discovered_experiments,
    )


def select_candidates_for_experiments(
    candidates: Sequence[BundleCandidate],
    experiments: Sequence[str],
) -> tuple[BundleCandidate, ...]:
    selected: list[BundleCandidate] = []
    for experiment in experiments:
        experiment_candidates = tuple(
            candidate for candidate in candidates if candidate.experiment == experiment
        )
        if not experiment_candidates:
            continue
        selected.append(
            max(
                experiment_candidates,
                key=lambda candidate: (
                    candidate.file_count,
                    candidate.report_count,
                    candidate.event_count,
                    candidate.checkpoint_count,
                    candidate.bundle_size_bytes,
                    candidate.created_at,
                ),
            ),
        )
    return tuple(selected)


def write_clean_export(
    source_archive: zipfile.ZipFile,
    output_path: Path,
    selected_candidates: Sequence[BundleCandidate],
) -> None:
    if not selected_candidates:
        raise ValueError("Cannot write clean export; no experiment bundles were selected.")

    selected_prefixes = tuple(
        f"{candidate.outer_bundle_path.rsplit('/', maxsplit=1)[0]}/"
        for candidate in selected_candidates
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as clean_archive:
        for source_info in source_archive.infolist():
            if source_info.is_dir():
                continue
            if not source_info.filename.startswith(selected_prefixes):
                continue
            clean_archive.writestr(source_info, source_archive.read(source_info.filename))


def analyze_experiment(
    outer_archive: zipfile.ZipFile,
    candidate: BundleCandidate,
) -> ExperimentSummary:
    with zipfile.ZipFile(io.BytesIO(outer_archive.read(candidate.outer_bundle_path))) as archive:
        names = tuple(archive.namelist())
        configuration = ExperimentFile.model_validate_json(archive.read("resolved_config.json"))
        vocabulary_size = vocabulary_size_from_configuration(configuration=configuration)
        parameter_summary = model_parameter_summary(
            build_model(
                model_configuration=configuration.model,
                vocabulary_size=vocabulary_size,
            ),
        )
        training_rows = tuple(
            read_jsonl_objects(archive=archive, path="artifacts/pretraining/metrics.jsonl"),
        )
        training_summary = build_training_summary(rows=training_rows)
        final_evaluation = evaluation_metrics_from_report(
            report=json_object(json.loads(archive.read("artifacts/evaluation/report.json"))),
            report_path="artifacts/evaluation/report.json",
            step=event_step_for_directory(
                archive=archive,
                directory="artifacts/evaluation/tensorboard",
            ),
        )
        checkpoint_summary = build_checkpoint_summary(archive=archive, names=names)
        tensorboard_summary = build_tensorboard_summary(archive=archive, names=names)
        requirements = validate_requirements(names=names)
        analyzed_files = analyzed_file_list(names=names)
        return ExperimentSummary(
            name=candidate.experiment,
            candidate=candidate,
            requirements=requirements,
            configuration=configuration,
            parameter_summary=parameter_summary,
            training_summary=training_summary,
            final_evaluation=final_evaluation,
            checkpoint_summary=checkpoint_summary,
            tensorboard_summary=tensorboard_summary,
            analyzed_files=analyzed_files,
        )


def vocabulary_size_from_configuration(configuration: ExperimentFile) -> int:
    tokenizer_configuration = configuration.tokenizer
    match tokenizer_configuration:
        case ByteBpeTokenizerConfiguration() | RustByteBpeTokenizerConfiguration():
            return tokenizer_configuration.vocabulary_size
        case CharacterTokenizerConfiguration():
            raise ValueError("Character tokenizers do not expose configured vocabulary_size.")


def read_jsonl_objects(archive: zipfile.ZipFile, path: str) -> Iterable[JsonObject]:
    with archive.open(path) as file:
        for raw_line in file:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json_object(json.loads(line))


def build_training_summary(rows: Sequence[JsonObject]) -> TrainingSummary:
    if not rows:
        raise ValueError("Training metrics are missing.")
    final_row = rows[-1]
    throughput_field = (
        "distributed_global_tokens_per_second"
        if "distributed_global_tokens_per_second" in final_row
        else "tokens_per_second"
    )
    throughputs = tuple(float_field(row, throughput_field) for row in rows)
    last_100 = throughputs[-100:]
    return TrainingSummary(
        final_step=integer_field(final_row, "step"),
        final_loss=float_field(final_row, "loss"),
        final_learning_rate=float_field(final_row, "learning_rate"),
        final_gradient_norm=float_field(final_row, "gradient_norm"),
        throughput_field=throughput_field,
        final_tokens_per_second=float_field(final_row, throughput_field),
        mean_last_100_tokens_per_second=statistics.fmean(last_100),
        median_tokens_per_second=statistics.median(throughputs),
        minimum_tokens_per_second=min(throughputs),
        maximum_tokens_per_second=max(throughputs),
        world_size=optional_integer_field(final_row, "distributed_world_size"),
        strategy=optional_string_field(final_row, "distributed_strategy"),
    )


def build_checkpoint_summary(archive: zipfile.ZipFile, names: Sequence[str]) -> CheckpointSummary:
    evaluation_reports: list[EvaluationMetrics] = []
    for report_path in sorted(
        name
        for name in names
        if name.startswith("artifacts/evaluation/sha256_") and name.endswith("/report.json")
    ):
        directory = report_path.rsplit("/", maxsplit=1)[0]
        report = json_object(json.loads(archive.read(report_path)))
        evaluation_reports.append(
            evaluation_metrics_from_report(
                report=report,
                report_path=report_path,
                step=event_step_for_directory(
                    archive=archive,
                    directory=f"{directory}/tensorboard",
                ),
            ),
        )
    ordered = tuple(sorted(evaluation_reports, key=evaluation_step_sort_key))
    return CheckpointSummary(
        evaluations=ordered,
        best_validation=min_optional(
            values=ordered,
            metric=lambda evaluation: evaluation.validation_loss,
        ),
        best_pass_rate=max_optional(
            values=ordered,
            metric=lambda evaluation: evaluation.pass_rate,
        ),
        final_checkpoint=max(
            ordered,
            key=evaluation_step_sort_key,
            default=None,
        ),
    )


def evaluation_metrics_from_report(
    report: JsonObject,
    report_path: str,
    step: int | None,
) -> EvaluationMetrics:
    perplexity = optional_object_field(report, "perplexity")
    python_completion = optional_object_field(report, "python_completion")
    fixed_prompt_generation = optional_object_field(report, "fixed_prompt_generation")
    return EvaluationMetrics(
        step=step,
        validation_documents=optional_integer_field(perplexity, "documents")
        if perplexity is not None
        else None,
        validation_sequences=optional_integer_field(perplexity, "sequences")
        if perplexity is not None
        else None,
        validation_loss=optional_float_field(perplexity, "loss")
        if perplexity is not None
        else None,
        perplexity=optional_float_field(perplexity, "perplexity")
        if perplexity is not None
        else None,
        completion_tasks=python_completion_task_count(python_completion=python_completion),
        parsed_tasks=optional_integer_field(python_completion, "parsed_tasks")
        if python_completion is not None
        else None,
        executed_tasks=optional_integer_field(python_completion, "executed_tasks")
        if python_completion is not None
        else None,
        passed_checks=optional_integer_field(python_completion, "passed_checks")
        if python_completion is not None
        else None,
        total_checks=optional_integer_field(python_completion, "total_checks")
        if python_completion is not None
        else None,
        pass_rate=optional_float_field(python_completion, "pass_rate")
        if python_completion is not None
        else None,
        fixed_prompt_samples=fixed_prompt_sample_count(
            fixed_prompt_generation=fixed_prompt_generation,
        ),
        report_path=report_path,
    )


def fixed_prompt_sample_count(fixed_prompt_generation: JsonObject | None) -> int | None:
    if fixed_prompt_generation is None:
        return None
    samples = optional_array_field(fixed_prompt_generation, "samples")
    if samples is None:
        return None
    return len(samples)


def python_completion_task_count(python_completion: JsonObject | None) -> int | None:
    if python_completion is None:
        return None
    tasks = python_completion.get("tasks")
    if tasks is None:
        return None
    if isinstance(tasks, list):
        return len(tasks)
    if isinstance(tasks, int):
        return tasks
    raise ValueError("Expected python_completion.tasks to be an array or integer.")


def build_tensorboard_summary(
    archive: zipfile.ZipFile,
    names: Sequence[str],
) -> TensorBoardSummary:
    pretraining_event_paths = tuple(
        name
        for name in names
        if name.startswith("artifacts/pretraining/") and "events.out.tfevents" in name
    )
    evaluation_event_paths = tuple(
        name
        for name in names
        if name.startswith("artifacts/evaluation/") and "events.out.tfevents" in name
    )
    pretraining_scalars = read_tensorboard_scalars(
        archive=archive,
        event_paths=pretraining_event_paths,
    )
    evaluation_scalars = read_tensorboard_scalars(
        archive=archive,
        event_paths=evaluation_event_paths,
    )
    return TensorBoardSummary(
        pretraining_event_files=len(pretraining_event_paths),
        evaluation_event_files=len(evaluation_event_paths),
        pretraining_scalar_tags=tuple(sorted(pretraining_scalars)),
        evaluation_scalar_tags=tuple(sorted(evaluation_scalars)),
        router_final_worst_dominance=final_scalar_value(
            scalar_events=pretraining_scalars,
            tag="moe/summary/worst_layer_dominance",
        ),
        router_final_worst_imbalance=final_scalar_value(
            scalar_events=pretraining_scalars,
            tag="moe/summary/worst_layer_imbalance",
        ),
        router_final_worst_entropy=final_scalar_value(
            scalar_events=pretraining_scalars,
            tag="moe/summary/worst_layer_entropy",
        ),
        router_max_worst_dominance=max_scalar_value(
            scalar_events=pretraining_scalars,
            tag="moe/summary/worst_layer_dominance",
        ),
        router_max_worst_imbalance=max_scalar_value(
            scalar_events=pretraining_scalars,
            tag="moe/summary/worst_layer_imbalance",
        ),
        router_min_worst_entropy=min_scalar_value(
            scalar_events=pretraining_scalars,
            tag="moe/summary/worst_layer_entropy",
        ),
    )


def read_tensorboard_scalars(
    archive: zipfile.ZipFile,
    event_paths: Sequence[str],
) -> dict[str, tuple[tuple[int, float], ...]]:
    scalar_events: dict[str, list[tuple[int, float]]] = {}
    if not event_paths:
        return {}
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        for index, event_path in enumerate(event_paths):
            event_file_path = directory / f"events_{index}.tfevents"
            event_file_path.write_bytes(archive.read(event_path))
        accumulator = EventAccumulator(str(directory))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = scalar_events.setdefault(tag, [])
            values.extend((event.step, float(event.value)) for event in accumulator.Scalars(tag))
    return {tag: tuple(sorted(values)) for tag, values in scalar_events.items()}


def event_step_for_directory(archive: zipfile.ZipFile, directory: str) -> int | None:
    event_paths = tuple(
        name
        for name in archive.namelist()
        if name.startswith(f"{directory}/") and "events.out.tfevents" in name
    )
    scalar_events = read_tensorboard_scalars(archive=archive, event_paths=event_paths)
    steps = tuple(
        event_step for values in scalar_events.values() for event_step, _event_value in values
    )
    if not steps:
        return None
    unique_steps = set(steps)
    if len(unique_steps) != 1:
        return max(unique_steps)
    return steps[0]


def validate_requirements(names: Sequence[str]) -> tuple[RequirementStatus, ...]:
    return (
        path_status(names=names, requirement="resolved_config.json", path="resolved_config.json"),
        path_status(names=names, requirement="run_manifest.json", path="run_manifest.json"),
        path_status(
            names=names,
            requirement="pretraining metrics.jsonl",
            path="artifacts/pretraining/metrics.jsonl",
        ),
        path_status(
            names=names,
            requirement="final evaluation/report.json",
            path="artifacts/evaluation/report.json",
        ),
        pattern_status(
            names=names,
            requirement="checkpoint-evaluation report.json files",
            prefix="artifacts/evaluation/sha256_",
            suffix="/report.json",
        ),
        contains_status(
            names=names,
            requirement="TensorBoard event files",
            needle="events.out.tfevents",
        ),
        pattern_status(
            names=names,
            requirement="tokenizer files",
            prefix="artifacts/tokenizer/",
            suffix=".json",
        ),
        contains_status(
            names=names,
            requirement="latest checkpoint",
            needle="/checkpoints/latest.",
        ),
    )


def path_status(names: Sequence[str], requirement: str, path: str) -> RequirementStatus:
    return RequirementStatus(
        name=requirement,
        present=path in names,
        detail=path if path in names else "missing",
    )


def pattern_status(
    names: Sequence[str],
    requirement: str,
    prefix: str,
    suffix: str,
) -> RequirementStatus:
    matches = tuple(name for name in names if name.startswith(prefix) and name.endswith(suffix))
    return RequirementStatus(
        name=requirement,
        present=bool(matches),
        detail=f"{len(matches)} files",
    )


def contains_status(names: Sequence[str], requirement: str, needle: str) -> RequirementStatus:
    matches = tuple(name for name in names if needle in name)
    return RequirementStatus(
        name=requirement,
        present=bool(matches),
        detail=f"{len(matches)} files",
    )


def analyzed_file_list(names: Sequence[str]) -> tuple[str, ...]:
    selected_prefixes = (
        "artifacts/evaluation/",
        "artifacts/pretraining/checkpoints/",
        "artifacts/pretraining/metrics.jsonl",
        "artifacts/pretraining/tensorboard/",
        "artifacts/tokenizer/",
    )
    selected_names = [
        name
        for name in names
        if name in {"bundle_manifest.json", "resolved_config.json", "run_manifest.json"}
        or name.startswith(selected_prefixes)
    ]
    return tuple(sorted(selected_names))


def write_plots(
    summaries: Sequence[ExperimentSummary],
    plots_directory: Path,
) -> tuple[PlotRecord, ...]:
    plots_directory.mkdir(parents=True, exist_ok=True)
    plot_records: list[PlotRecord] = []
    pass_rate_series = checkpoint_line_series(
        summaries=summaries,
        metric_name="pass_rate",
        scale=100.0,
    )
    ast_parse_series = parse_rate_series(summaries=summaries)
    passed_checks_series = checkpoint_line_series(
        summaries=summaries,
        metric_name="passed_checks",
        scale=1.0,
    )
    checkpoint_plot_records = (
        (
            pass_rate_series,
            PlotRecord(
                title="Checkpoint Python Completion Pass Rate",
                path=plots_directory / "checkpoint_pass_rate.svg",
                caption=("Percentage of unit-test checks passed during checkpoint evaluation."),
            ),
            "Pass rate (%)",
        ),
        (
            ast_parse_series,
            PlotRecord(
                title="Checkpoint AST Parse Rate",
                path=plots_directory / "checkpoint_parse_rate.svg",
                caption=(
                    "Percentage of completion tasks that produced parseable Python ASTs. "
                    "This isolates syntax validity from semantic correctness."
                ),
            ),
            "AST parse rate (%)",
        ),
        (
            passed_checks_series,
            PlotRecord(
                title="Checkpoint Passed Checks",
                path=plots_directory / "checkpoint_passed_checks.svg",
                caption="Raw number of passed unit-test checks over training.",
            ),
            "Passed checks",
        ),
    )
    for series, plot_record, y_label in checkpoint_plot_records:
        if not line_series_has_points(series=series):
            continue
        write_line_plot(
            path=plot_record.path,
            title=plot_record.title,
            y_label=y_label,
            series=series,
        )
        plot_records.append(plot_record)

    final_plot_records = (
        (
            PlotRecord(
                title="Active Parameters vs Final Pass Rate",
                path=plots_directory / "size_vs_pass_rate.svg",
                caption=(
                    "Final full-evaluation pass rate against active parameter count. The x-axis is "
                    "log-scaled so small and larger models are visible together."
                ),
            ),
            "parameter_pass",
        ),
        (
            PlotRecord(
                title="Depth vs Final Pass Rate",
                path=plots_directory / "depth_vs_pass_rate.svg",
                caption=(
                    "Final full-evaluation pass rate by layer count. Point radius scales with "
                    "active parameter count."
                ),
            ),
            "depth_pass",
        ),
        (
            PlotRecord(
                title="Final Pass Rate and Perplexity",
                path=plots_directory / "final_pass_perplexity.svg",
                caption=(
                    "Final full-evaluation pass rate and validation perplexity. Lower perplexity "
                    "generally tracks better completion performance here."
                ),
            ),
            "final_pass_perplexity",
        ),
    )
    for plot_record, plot_kind in final_plot_records:
        match plot_kind:
            case "parameter_pass":
                write_scatter_plot(
                    path=plot_record.path,
                    title=plot_record.title,
                    x_label="Active parameters (log10)",
                    y_label="Final pass rate (%)",
                    points=parameter_pass_points(summaries=summaries),
                )
            case "depth_pass":
                write_scatter_plot(
                    path=plot_record.path,
                    title=plot_record.title,
                    x_label="Layers",
                    y_label="Final pass rate (%)",
                    points=depth_pass_points(summaries=summaries),
                )
            case "final_pass_perplexity":
                write_dual_bar_plot(
                    path=plot_record.path,
                    title=plot_record.title,
                    summaries=summaries,
                )
            case _:
                raise ValueError(f"Unsupported plot kind: {plot_kind}.")
        plot_records.append(plot_record)
    return tuple(plot_records)


def line_series_has_points(
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
) -> bool:
    return any(points for _name, points in series)


def checkpoint_line_series(
    summaries: Sequence[ExperimentSummary],
    metric_name: str,
    scale: float,
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    series: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for summary in summaries:
        points: list[tuple[float, float]] = []
        for evaluation in summary.checkpoint_summary.evaluations:
            if evaluation.step is None:
                continue
            metric_value = evaluation_metric_value(
                evaluation=evaluation,
                metric_name=metric_name,
            )
            if metric_value is None:
                continue
            points.append((float(evaluation.step), metric_value * scale))
        series.append((short_experiment_name(summary.name), tuple(points)))
    return tuple(series)


def parse_rate_series(
    summaries: Sequence[ExperimentSummary],
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    series: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for summary in summaries:
        points: list[tuple[float, float]] = []
        for evaluation in summary.checkpoint_summary.evaluations:
            if (
                evaluation.step is None
                or evaluation.completion_tasks is None
                or evaluation.parsed_tasks is None
            ):
                continue
            points.append(
                (
                    float(evaluation.step),
                    (float(evaluation.parsed_tasks) / float(evaluation.completion_tasks)) * 100.0,
                ),
            )
        series.append((short_experiment_name(summary.name), tuple(points)))
    return tuple(series)


def parameter_pass_points(
    summaries: Sequence[ExperimentSummary],
) -> tuple[tuple[str, float, float, float], ...]:
    points: list[tuple[str, float, float, float]] = []
    for summary in summaries:
        if summary.final_evaluation.pass_rate is None:
            continue
        active_parameters = float(summary.parameter_summary.active_parameters)
        radius = 4.0 + (math.log10(active_parameters) - 5.9) * 4.0
        points.append(
            (
                short_experiment_name(summary.name),
                math.log10(active_parameters),
                summary.final_evaluation.pass_rate * 100.0,
                max(4.0, radius),
            ),
        )
    return tuple(points)


def depth_pass_points(
    summaries: Sequence[ExperimentSummary],
) -> tuple[tuple[str, float, float, float], ...]:
    points: list[tuple[str, float, float, float]] = []
    for summary in summaries:
        if summary.final_evaluation.pass_rate is None:
            continue
        active_parameters = float(summary.parameter_summary.active_parameters)
        radius = 4.0 + (active_parameters / 9_646_080.0) * 8.0
        points.append(
            (
                short_experiment_name(summary.name),
                float(summary.configuration.model.layers),
                summary.final_evaluation.pass_rate * 100.0,
                radius,
            ),
        )
    return tuple(points)


def evaluation_metric_value(
    evaluation: EvaluationMetrics,
    metric_name: str,
) -> float | None:
    match metric_name:
        case "pass_rate":
            return evaluation.pass_rate
        case "passed_checks":
            return float(evaluation.passed_checks) if evaluation.passed_checks is not None else None
        case _:
            raise ValueError(f"Unsupported checkpoint metric: {metric_name}")


def write_line_plot(
    path: Path,
    title: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
) -> None:
    width = 1120
    height = 560
    margin_left = 76
    margin_right = 250
    margin_top = 54
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_points = tuple(point for _name, points in series for point in points)
    x_values = tuple(point[0] for point in all_points)
    y_values = tuple(point[1] for point in all_points)
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(0.0, min(y_values))
    y_max = max(y_values)
    y_padding = max((y_max - y_min) * 0.08, 1.0)
    y_max += y_padding

    def x_position(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_position(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    elements = svg_base_elements(
        width=width,
        height=height,
        title=title,
        x_label="Training step",
        y_label=y_label,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
    )
    elements.extend(
        svg_grid(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_position=x_position,
            y_position=y_position,
            margin_left=margin_left,
            margin_top=margin_top,
            plot_width=plot_width,
            plot_height=plot_height,
        ),
    )
    for index, (name, points) in enumerate(series):
        color = plot_color(index)
        path_points = " ".join(f"{x_position(x):.1f},{y_position(y):.1f}" for x, y in points)
        elements.append(
            f'<polyline points="{path_points}" fill="none" stroke="{color}" '
            'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" />'
        )
        for x_value, y_value in points[-1:]:
            elements.append(
                f'<circle cx="{x_position(x_value):.1f}" cy="{y_position(y_value):.1f}" '
                f'r="3.5" fill="{color}" />'
            )
        legend_y = margin_top + 12 + index * 22
        legend_x = width - margin_right + 32
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3" />'
        )
        elements.append(
            f'<text x="{legend_x + 26}" y="{legend_y + 4}" class="legend">{escape_svg(name)}</text>'
        )
    write_svg(path=path, width=width, height=height, elements=elements)


def write_scatter_plot(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    points: Sequence[tuple[str, float, float, float]],
) -> None:
    width = 980
    height = 560
    margin_left = 76
    margin_right = 36
    margin_top = 54
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_values = tuple(point[1] for point in points)
    y_values = tuple(point[2] for point in points)
    x_min = min(x_values) - 0.05
    x_max = max(x_values) + 0.05
    y_min = 0.0
    y_max = max(y_values) + 8.0

    def x_position(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_position(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    elements = svg_base_elements(
        width=width,
        height=height,
        title=title,
        x_label=x_label,
        y_label=y_label,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
    )
    elements.extend(
        svg_grid(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_position=x_position,
            y_position=y_position,
            margin_left=margin_left,
            margin_top=margin_top,
            plot_width=plot_width,
            plot_height=plot_height,
        ),
    )
    for index, (name, x_value, y_value, radius) in enumerate(points):
        color = plot_color(index)
        x_coordinate = x_position(x_value)
        y_coordinate = y_position(y_value)
        elements.append(
            f'<circle cx="{x_coordinate:.1f}" cy="{y_coordinate:.1f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="0.82" stroke="#20242c" stroke-width="1" />'
        )
        label_y = y_coordinate - radius - 5.0
        elements.append(
            f'<text x="{x_coordinate:.1f}" y="{label_y:.1f}" class="point-label" '
            f'text-anchor="middle">{escape_svg(name)}</text>'
        )
    write_svg(path=path, width=width, height=height, elements=elements)


def write_dual_bar_plot(
    path: Path,
    title: str,
    summaries: Sequence[ExperimentSummary],
) -> None:
    width = 1160
    height = 620
    margin_left = 72
    margin_right = 76
    margin_top = 58
    margin_bottom = 146
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    ordered = tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.final_evaluation.pass_rate
                if summary.final_evaluation.pass_rate is not None
                else -1.0
            ),
            reverse=True,
        ),
    )
    max_perplexity = max(summary.final_evaluation.perplexity or 0.0 for summary in ordered)
    group_width = plot_width / len(ordered)
    bar_width = min(34.0, group_width * 0.34)

    def y_pass(value: float) -> float:
        return margin_top + plot_height - (value / 90.0) * plot_height

    def y_perplexity(value: float) -> float:
        return margin_top + plot_height - (value / (max_perplexity * 1.12)) * plot_height

    elements = svg_base_elements(
        width=width,
        height=height,
        title=title,
        x_label="Experiment",
        y_label="Pass rate (%) / perplexity",
        margin_left=margin_left,
        margin_right=margin_right,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
    )
    elements.extend(
        svg_horizontal_grid(
            y_min=0.0,
            y_max=90.0,
            y_position=y_pass,
            margin_left=margin_left,
            margin_top=margin_top,
            plot_width=plot_width,
            plot_height=plot_height,
        ),
    )
    for index, summary in enumerate(ordered):
        center_x = margin_left + group_width * index + group_width / 2.0
        pass_value = (summary.final_evaluation.pass_rate or 0.0) * 100.0
        perplexity_value = summary.final_evaluation.perplexity or 0.0
        pass_top = y_pass(pass_value)
        perplexity_top = y_perplexity(perplexity_value)
        elements.append(
            f'<rect x="{center_x - bar_width - 2:.1f}" y="{pass_top:.1f}" '
            f'width="{bar_width:.1f}" height="{margin_top + plot_height - pass_top:.1f}" '
            'fill="#2f6fbb" />'
        )
        elements.append(
            f'<rect x="{center_x + 2:.1f}" y="{perplexity_top:.1f}" '
            f'width="{bar_width:.1f}" height="{margin_top + plot_height - perplexity_top:.1f}" '
            'fill="#d98634" />'
        )
        label = short_experiment_name(summary.name)
        elements.append(
            f'<text x="{center_x:.1f}" y="{height - 36}" class="x-label" '
            f'text-anchor="end" transform="rotate(-38 {center_x:.1f} {height - 36})">'
            f"{escape_svg(label)}</text>"
        )
    elements.append('<rect x="884" y="72" width="16" height="12" fill="#2f6fbb" />')
    elements.append('<text x="908" y="83" class="legend">Pass rate (%)</text>')
    elements.append('<rect x="884" y="94" width="16" height="12" fill="#d98634" />')
    elements.append('<text x="908" y="105" class="legend">Perplexity (scaled)</text>')
    write_svg(path=path, width=width, height=height, elements=elements)


def svg_base_elements(
    width: int,
    height: int,
    title: str,
    x_label: str,
    y_label: str,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
) -> list[str]:
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    return [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="title">'
        f"{escape_svg(title)}</text>",
        (
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
            f'height="{plot_height}" fill="#fbfbfd" stroke="#d5d9e2" />'
        ),
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 18}" '
        f'text-anchor="middle" class="axis-label">{escape_svg(x_label)}</text>',
        f'<text x="20" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" '
        f'class="axis-label" transform="rotate(-90 20 {margin_top + plot_height / 2:.1f})">'
        f"{escape_svg(y_label)}</text>",
    ]


def svg_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_position: Callable[[float], float],
    y_position: Callable[[float], float],
    margin_left: int,
    margin_top: int,
    plot_width: int,
    plot_height: int,
) -> list[str]:
    elements: list[str] = []
    for tick_index in range(6):
        fraction = tick_index / 5.0
        y_value = y_min + (y_max - y_min) * fraction
        y_coordinate = y_position(y_value)
        elements.append(
            f'<line x1="{margin_left}" y1="{y_coordinate:.1f}" '
            f'x2="{margin_left + plot_width}" y2="{y_coordinate:.1f}" '
            'stroke="#e6e9ef" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{margin_left - 10}" y="{y_coordinate + 4:.1f}" '
            f'text-anchor="end" class="tick">{y_value:.1f}</text>'
        )
    for tick_index in range(6):
        fraction = tick_index / 5.0
        x_value = x_min + (x_max - x_min) * fraction
        x_coordinate = x_position(x_value)
        elements.append(
            f'<line x1="{x_coordinate:.1f}" y1="{margin_top}" '
            f'x2="{x_coordinate:.1f}" y2="{margin_top + plot_height}" '
            'stroke="#eef0f4" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{x_coordinate:.1f}" y="{margin_top + plot_height + 20}" '
            f'text-anchor="middle" class="tick">{x_value:.0f}</text>'
        )
    return elements


def svg_horizontal_grid(
    y_min: float,
    y_max: float,
    y_position: Callable[[float], float],
    margin_left: int,
    margin_top: int,
    plot_width: int,
    plot_height: int,
) -> list[str]:
    elements: list[str] = []
    for tick_index in range(6):
        fraction = tick_index / 5.0
        y_value = y_min + (y_max - y_min) * fraction
        y_coordinate = y_position(y_value)
        elements.append(
            f'<line x1="{margin_left}" y1="{y_coordinate:.1f}" '
            f'x2="{margin_left + plot_width}" y2="{y_coordinate:.1f}" '
            'stroke="#e6e9ef" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{margin_left - 10}" y="{y_coordinate + 4:.1f}" '
            f'text-anchor="end" class="tick">{y_value:.1f}</text>'
        )
    elements.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
        'stroke="#ccd2dc" stroke-width="1" />'
    )
    return elements


def write_svg(path: Path, width: int, height: int, elements: Sequence[str]) -> None:
    style = """
<style>
.title { font: 700 18px Arial, sans-serif; fill: #171a20; }
.axis-label { font: 12px Arial, sans-serif; fill: #333843; }
.tick { font: 11px Arial, sans-serif; fill: #586070; }
.legend { font: 12px Arial, sans-serif; fill: #232832; }
.point-label { font: 10px Arial, sans-serif; fill: #20242c; }
.x-label { font: 10px Arial, sans-serif; fill: #20242c; }
</style>
""".strip()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"{style}\n" + "\n".join(elements) + "\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def plot_color(index: int) -> str:
    colors = (
        "#2f6fbb",
        "#d98634",
        "#3a8f5c",
        "#9b59b6",
        "#c44e52",
        "#5f7f8f",
        "#b39c2f",
        "#2a9d8f",
        "#7f6d5f",
    )
    return colors[index % len(colors)]


def short_experiment_name(name: str) -> str:
    replacements = {
        "python_": "",
        "_small": "",
        "_plain": "",
        "_warmup_decay": "",
        "_deep": "-deep",
        "_wide": "-wide",
        "_active": "-active",
        "_linear": "-linear",
        "_cosine": "-cosine",
        "_modern": "-modern",
    }
    label = name
    for old_text, new_text in replacements.items():
        label = label.replace(old_text, new_text)
    return label.replace("_", "-")


def escape_svg(value: str) -> str:
    return html.escape(value, quote=True)


def render_markdown(
    outer_zip_path: Path,
    outer_bad_file: str | None,
    candidates: Sequence[BundleCandidate],
    summaries: Sequence[ExperimentSummary],
    plot_records: Sequence[PlotRecord],
    output_path: Path,
) -> str:
    selected_count = len(summaries)
    stale_count = len(candidates) - selected_count
    lines: list[str] = [
        "# Python Model Sweep Results",
        "",
        "## Overview",
        "",
        (
            f"Analyzed `{outer_zip_path}` on {date.today().isoformat()}. "
            f"{integrity_sentence(outer_bad_file=outer_bad_file, summaries=summaries)} "
            f"The selected export view contains "
            f"{selected_count} current experiment bundles"
            f"{f'; {stale_count} stale duplicate bundles were removed' if stale_count else ''}."
        ),
        "",
        (
            "Bundle selection is deterministic: for each experiment, choose the candidate with "
            "the most files, checkpoint-evaluation reports, TensorBoard event files, checkpoint "
            "files, compressed size, and creation time. "
            f"{selected_bundle_profile(summaries=summaries)}"
        ),
        "",
        (
            "Model parameter counts were computed by validating each `resolved_config.json`, "
            "building the configured model with the configured tokenizer vocabulary size, "
            "and using `llm_lite.model.parameters.model_parameter_summary`. "
            "Checkpoint-evaluation steps were read from TensorBoard event files inside each "
            "checkpoint-evaluation artifact. Fixed prompt generation samples were present in "
            "checkpoint-evaluation reports and absent from the final full evaluation reports."
        ),
        "",
        hardware_notes(summaries=summaries),
        "",
    ]
    lines.extend(render_key_findings(summaries=summaries))
    lines.extend(render_summary_table(summaries=summaries))
    lines.extend(render_plots_section(plot_records=plot_records, output_path=output_path))
    lines.extend(render_model_training_table(summaries=summaries))
    lines.extend(render_checkpoint_peak_table(summaries=summaries))
    lines.extend(render_anomalies(summaries=summaries))
    lines.extend(render_recommendations())
    return "\n".join(lines) + "\n"


def integrity_sentence(
    outer_bad_file: str | None,
    summaries: Sequence[ExperimentSummary],
) -> str:
    nested_bad_files = tuple(
        summary.candidate.nested_test_bad_file
        for summary in summaries
        if summary.candidate.nested_test_bad_file is not None
    )
    if outer_bad_file is None and not nested_bad_files:
        return "ZIP integrity checks passed for the outer export and all selected nested bundles."
    if outer_bad_file is not None:
        return f"Outer ZIP integrity failed at `{outer_bad_file}`."
    return f"Nested ZIP integrity failed for {len(nested_bad_files)} selected bundle(s)."


def selected_bundle_profile(summaries: Sequence[ExperimentSummary]) -> str:
    file_counts = tuple(summary.candidate.file_count for summary in summaries)
    report_counts = tuple(summary.candidate.report_count for summary in summaries)
    event_counts = tuple(summary.candidate.event_count for summary in summaries)
    checkpoint_counts = tuple(summary.candidate.checkpoint_count for summary in summaries)
    return (
        f"Selected bundle ranges: {format_int_range(file_counts)} files, "
        f"{format_int_range(report_counts)} reports, {format_int_range(event_counts)} "
        "TensorBoard event files, and "
        f"{format_int_range(checkpoint_counts)} {pluralize('checkpoint file', checkpoint_counts)}."
    )


def hardware_notes(summaries: Sequence[ExperimentSummary]) -> str:
    world_sizes = sorted(
        {
            summary.training_summary.world_size
            for summary in summaries
            if summary.training_summary.world_size is not None
        }
    )
    strategies = sorted(
        {
            summary.training_summary.strategy
            for summary in summaries
            if summary.training_summary.strategy is not None
        }
    )
    precisions = sorted(
        {str(summary.configuration.training.precision.value) for summary in summaries}
    )
    return (
        "Hardware/run notes: the export records "
        f"`distributed_world_size={format_joined_values(world_sizes)}`, "
        f"`distributed_strategy={format_joined_values(strategies)}`, and "
        f"`precision={format_joined_values(precisions)}` for the analyzed training logs. "
        "The exact GPU model, host CPU, and wall-clock environment are not present in the export."
    )


def render_key_findings(summaries: Sequence[ExperimentSummary]) -> list[str]:
    best_pass = max_optional(
        values=summaries,
        metric=lambda summary: summary.final_evaluation.pass_rate,
    )
    best_small = max_optional(
        values=tuple(
            summary
            for summary in summaries
            if summary.parameter_summary.active_parameters < 2_000_000
        ),
        metric=lambda summary: summary.final_evaluation.pass_rate,
    )
    best_small_pass_rate = best_small.final_evaluation.pass_rate if best_small is not None else None
    fastest = max(
        summaries,
        key=lambda summary: summary.training_summary.mean_last_100_tokens_per_second,
    )
    lowest_perplexity = min_optional(
        values=summaries,
        metric=lambda summary: summary.final_evaluation.perplexity,
    )
    router_dominant = tuple(
        summary
        for summary in summaries
        if summary.tensorboard_summary.router_final_worst_dominance is not None
        and summary.tensorboard_summary.router_final_worst_dominance >= 0.90
    )
    checkpoint_regressions = tuple(
        summary
        for summary in summaries
        if pass_delta(
            summary.checkpoint_summary.final_checkpoint,
            summary.checkpoint_summary.best_pass_rate,
        )
        is not None
        and pass_delta(
            summary.checkpoint_summary.final_checkpoint,
            summary.checkpoint_summary.best_pass_rate,
        )
        < -0.01
    )
    findings = [
        "## Key Findings",
        "",
        (
            f"- `{best_pass.name if best_pass else 'missing'}` is the strongest overall run by "
            f"final Python completion pass rate "
            f"({format_percent(best_pass.final_evaluation.pass_rate if best_pass else None)})."
        ),
        (
            f"- `{best_small.name if best_small else 'missing'}` is the strongest sub-2M-active-"
            f"parameter model ({format_percent(best_small_pass_rate)} final pass rate)."
        ),
        (
            f"- `{fastest.name}` is the highest-throughput run at "
            f"{format_int(round(fastest.training_summary.mean_last_100_tokens_per_second))} "
            "tokens/s over the last 100 training log points, but its final pass rate is only "
            f"{format_percent(fastest.final_evaluation.pass_rate)}."
        ),
    ]
    if lowest_perplexity is not None:
        findings.append(
            f"- `{lowest_perplexity.name}` has the lowest final validation perplexity "
            f"({format_optional_float(lowest_perplexity.final_evaluation.perplexity)})."
        )
    if router_dominant:
        findings.append(
            f"- {len(router_dominant)} MoE run(s) show severe final expert dominance "
            "(worst-layer dominance >= 0.90)."
        )
    if checkpoint_regressions:
        findings.append(
            f"- {len(checkpoint_regressions)} run(s) ended more than 1 percentage point below "
            "their best checkpoint pass rate."
        )
    findings.append("")
    return findings


def render_plots_section(
    plot_records: Sequence[PlotRecord],
    output_path: Path,
) -> list[str]:
    lines = [
        "## Plots",
        "",
        (
            "These are generated SVGs linked below as static Markdown images. For interactive "
            "drill-down, use the TensorBoard event files inside the export bundles."
        ),
        "",
    ]
    for plot_record in plot_records:
        relative_path = plot_record.path.relative_to(output_path.parent).as_posix()
        lines.extend(
            [
                f"### {plot_record.title}",
                "",
                f"![{plot_record.title}]({relative_path})",
                "",
                plot_record.caption,
                "",
            ],
        )
    return lines


def render_checkpoint_peak_table(summaries: Sequence[ExperimentSummary]) -> list[str]:
    lines = [
        "## Checkpoint Peaks",
        "",
        (
            "Checkpoint evaluations used the smaller validation slice in the export. This table "
            "keeps only the peak and final checkpoint signals; full trajectories are shown in the "
            "plots above."
        ),
        "",
        (
            "| Experiment | Best val step/ppl | Best pass step/rate | Final ckpt pass | "
            "Final vs best pass |"
        ),
        "| --- | --- | --- | ---: | ---: |",
    ]
    for summary in summaries:
        best_validation = summary.checkpoint_summary.best_validation
        best_pass_rate = summary.checkpoint_summary.best_pass_rate
        final_checkpoint = summary.checkpoint_summary.final_checkpoint
        lines.append(
            "| "
            f"`{summary.name}` | "
            f"{best_validation.step if best_validation else 'missing'}/"
            f"{format_optional_float(best_validation.perplexity if best_validation else None)} | "
            f"{best_pass_rate.step if best_pass_rate else 'missing'}/"
            f"{format_percent(best_pass_rate.pass_rate if best_pass_rate else None)} | "
            f"{format_percent(final_checkpoint.pass_rate if final_checkpoint else None)} | "
            f"{format_percent_delta(pass_delta(final_checkpoint, best_pass_rate))} |",
        )
    lines.append("")
    return lines


def render_validation_section(
    candidates: Sequence[BundleCandidate],
    summaries: Sequence[ExperimentSummary],
) -> list[str]:
    selected_artifact_ids = {summary.candidate.artifact_id for summary in summaries}
    lines = [
        "## Export Validation",
        "",
        (
            "| Artifact | Experiment | Selected | Files | Reports | Events | "
            "Checkpoint files | Nested ZIP test |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in sorted(candidates, key=lambda item: (item.experiment, item.created_at)):
        lines.append(
            "| "
            f"`{candidate.artifact_id}` | `{candidate.experiment}` | "
            f"{'yes' if candidate.artifact_id in selected_artifact_ids else 'no'} | "
            f"{candidate.file_count} | {candidate.report_count} | {candidate.event_count} | "
            f"{candidate.checkpoint_count} | `{candidate.nested_test_bad_file or 'OK'}` |",
        )
    lines.extend(["", "Selected bundle required-file checks:", ""])
    lines.append("| Experiment | Required files present | Notes |")
    lines.append("| --- | --- | --- |")
    for summary in summaries:
        missing = tuple(status for status in summary.requirements if not status.present)
        detail = "; ".join(f"{status.name}: {status.detail}" for status in summary.requirements)
        lines.append(
            f"| `{summary.name}` | {'yes' if not missing else 'no'} | {detail} |",
        )
    lines.append("")
    return lines


def render_summary_table(summaries: Sequence[ExperimentSummary]) -> list[str]:
    lines = [
        "## Summary Results",
        "",
        (
            "| Experiment | Type | Active params | Final train loss | Final eval loss | "
            "Final ppl | Final pass rate | Best ckpt pass | Mean TPS last 100 |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        best_pass_rate = optional_metric(summary.checkpoint_summary.best_pass_rate, "pass_rate")
        lines.append(
            "| "
            f"`{summary.name}` | {model_type_label(summary)} | "
            f"{format_int(summary.parameter_summary.active_parameters)} | "
            f"{format_float(summary.training_summary.final_loss)} | "
            f"{format_optional_float(summary.final_evaluation.validation_loss)} | "
            f"{format_optional_float(summary.final_evaluation.perplexity)} | "
            f"{format_percent(summary.final_evaluation.pass_rate)} | "
            f"{format_percent(best_pass_rate)} | "
            f"{format_int(round(summary.training_summary.mean_last_100_tokens_per_second))} |",
        )
    lines.append("")
    return lines


def render_model_training_table(summaries: Sequence[ExperimentSummary]) -> list[str]:
    lines = [
        "## Model And Training Details",
        "",
        (
            "| Experiment | Dim | Layers | Heads | FFN/expert FFN | Experts | Top-k | "
            "Total params | Active params | Steps | Batch seqs | LR schedule | LR final | "
            "Weight decay | Max ckpts | FIM |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | --- | ---: | ---: | ---: | --- |"
        ),
    ]
    for summary in summaries:
        model = summary.configuration.model
        training = summary.configuration.training
        optimizer = training.optimizer
        lines.append(
            "| "
            f"`{summary.name}` | {model.dimension} | {model.layers} | {model.attention_heads} | "
            f"{feed_forward_dimension(summary=summary)} | {expert_count(summary=summary)} | "
            f"{router_top_k(summary=summary)} | "
            f"{format_int(summary.parameter_summary.total_parameters)} | "
            f"{format_int(summary.parameter_summary.active_parameters)} | "
            f"{training.maximum_steps} | {training.batch_size_sequences} | "
            f"{optimizer.learning_rate_schedule.type.value} | "
            f"{format_float(summary.training_summary.final_learning_rate)} | "
            f"{format_float(optimizer.weight_decay)} | "
            f"{training.max_checkpoints or 'missing'} | "
            f"{'yes' if summary.configuration.packing.fill_in_middle.enabled else 'no'} |",
        )
    lines.append("")
    return lines


def render_checkpoint_table(summaries: Sequence[ExperimentSummary]) -> list[str]:
    lines = [
        "## Checkpoint Evaluation",
        "",
        (
            "Checkpoint-evaluation reports used the smaller validation slice recorded in "
            "those artifacts. The final full evaluation is reported separately in the "
            "summary table."
        ),
        "",
        (
            "| Experiment | Eval steps | Fixed prompt samples | Best val step/loss/ppl | "
            "Best pass step/rate | Final ckpt loss/ppl/pass | Final vs best val | "
            "Final vs best pass |"
        ),
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for summary in summaries:
        checkpoint_summary = summary.checkpoint_summary
        final_checkpoint = checkpoint_summary.final_checkpoint
        best_validation = checkpoint_summary.best_validation
        best_pass_rate = checkpoint_summary.best_pass_rate
        fixed_prompt_samples = (
            final_checkpoint.fixed_prompt_samples if final_checkpoint is not None else None
        )
        lines.append(
            "| "
            f"`{summary.name}` | {step_list(checkpoint_summary.evaluations)} | "
            f"{format_optional_int(fixed_prompt_samples)} | "
            f"{evaluation_loss_cell(best_validation)} | "
            f"{evaluation_pass_cell(best_pass_rate)} | "
            f"{evaluation_final_cell(final_checkpoint)} | "
            f"{format_delta(loss_delta(final_checkpoint, best_validation))} | "
            f"{format_percent_delta(pass_delta(final_checkpoint, best_pass_rate))} |",
        )
    lines.extend(["", "Checkpoint trajectory details:", ""])
    for summary in summaries:
        lines.append(f"- `{summary.name}`: {trajectory(summary.checkpoint_summary.evaluations)}")
    lines.append("")
    return lines


def render_comparison_sections(summaries: Sequence[ExperimentSummary]) -> list[str]:
    by_name = {summary.name: summary for summary in summaries}
    comparisons = (
        (
            "Small Dense vs Small MoE deep/wide",
            (
                "python_dense_small_deep_plain",
                "python_dense_small_wide_plain",
                "python_moe_small_deep_plain",
                "python_moe_small_wide_plain",
            ),
        ),
        (
            "MoE plain vs FIM",
            ("python_moe_small_deep_plain", "python_moe_small_deep_fim"),
        ),
        (
            "MoE fixed LR vs linear warmup decay vs cosine warmup decay",
            (
                "python_moe_small_deep_plain",
                "python_moe_small_deep_linear_warmup_decay",
                "python_moe_small_deep_cosine_warmup_decay",
            ),
        ),
        (
            "Classic MoE vs modern MoE",
            ("python_moe_small_deep_plain", "python_modern_moe_small_deep_plain"),
        ),
        (
            "Small models vs dense_active_9m6",
            (
                "python_dense_small_deep_plain",
                "python_moe_small_deep_plain",
                "python_dense_active_9m6",
            ),
        ),
    )
    lines = [
        "## Comparisons",
        "",
    ]
    for title, names in comparisons:
        selected = tuple(by_name[name] for name in names if name in by_name)
        lines.extend(render_comparison_table(title=title, summaries=selected))
    lines.extend(render_speed_tradeoffs(summaries=summaries))
    return lines


def render_comparison_table(title: str, summaries: Sequence[ExperimentSummary]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        (
            "| Experiment | Active params | Train loss | Final eval ppl | Final pass | "
            "Best ckpt ppl | Best ckpt pass | Mean TPS last 100 |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        best_validation = summary.checkpoint_summary.best_validation
        best_pass_rate = summary.checkpoint_summary.best_pass_rate
        lines.append(
            "| "
            f"`{summary.name}` | {format_int(summary.parameter_summary.active_parameters)} | "
            f"{format_float(summary.training_summary.final_loss)} | "
            f"{format_optional_float(summary.final_evaluation.perplexity)} | "
            f"{format_percent(summary.final_evaluation.pass_rate)} | "
            f"{format_optional_float(best_validation.perplexity if best_validation else None)} | "
            f"{format_percent(best_pass_rate.pass_rate if best_pass_rate else None)} | "
            f"{format_int(round(summary.training_summary.mean_last_100_tokens_per_second))} |",
        )
    lines.extend(["", comparison_interpretation(title=title, summaries=summaries), ""])
    return lines


def render_speed_tradeoffs(summaries: Sequence[ExperimentSummary]) -> list[str]:
    ordered = sorted(
        summaries,
        key=lambda summary: summary.training_summary.mean_last_100_tokens_per_second,
        reverse=True,
    )
    lines = [
        "### Speed/Throughput Tradeoffs",
        "",
        "| Rank | Experiment | Mean TPS last 100 | Final eval ppl | Final pass |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for index, summary in enumerate(ordered, start=1):
        lines.append(
            "| "
            f"{index} | `{summary.name}` | "
            f"{format_int(round(summary.training_summary.mean_last_100_tokens_per_second))} | "
            f"{format_optional_float(summary.final_evaluation.perplexity)} | "
            f"{format_percent(summary.final_evaluation.pass_rate)} |",
        )
    lines.append("")
    return lines


def render_anomalies(summaries: Sequence[ExperimentSummary]) -> list[str]:
    lines = [
        "## Anomalies And Observations",
        "",
    ]
    for summary in summaries:
        notes = anomaly_notes(summary=summary)
        lines.append(
            f"- `{summary.name}`: {' '.join(notes) if notes else 'No export-level anomaly found.'}"
        )
    lines.append("")
    return lines


def anomaly_notes(summary: ExperimentSummary) -> tuple[str, ...]:
    notes: list[str] = []
    if any(not requirement.present for requirement in summary.requirements):
        missing = ", ".join(
            requirement.name for requirement in summary.requirements if not requirement.present
        )
        notes.append(f"Missing required artifacts: {missing}.")
    final_checkpoint = summary.checkpoint_summary.final_checkpoint
    best_validation = summary.checkpoint_summary.best_validation
    best_pass_rate = summary.checkpoint_summary.best_pass_rate
    validation_delta = loss_delta(final_checkpoint, best_validation)
    if validation_delta is not None and validation_delta > 0.01:
        notes.append(
            f"Checkpoint validation loss degraded by {format_delta(validation_delta)} versus best.",
        )
    pass_rate_delta = pass_delta(final_checkpoint, best_pass_rate)
    if pass_rate_delta is not None and pass_rate_delta < -0.01:
        notes.append(
            f"Checkpoint pass rate ended {format_percent_delta(pass_rate_delta)} below best.",
        )
    tensorboard_summary = summary.tensorboard_summary
    if (
        tensorboard_summary.router_final_worst_dominance is not None
        and tensorboard_summary.router_final_worst_dominance >= 0.90
    ):
        notes.append(
            "MoE router shows severe final expert dominance "
            f"({format_float(tensorboard_summary.router_final_worst_dominance)}).",
        )
    if (
        tensorboard_summary.router_final_worst_entropy is not None
        and tensorboard_summary.router_final_worst_entropy <= 0.20
    ):
        notes.append(
            "MoE router final entropy is very low "
            f"({format_float(tensorboard_summary.router_final_worst_entropy)}).",
        )
    if summary.training_summary.maximum_tokens_per_second > (
        summary.training_summary.median_tokens_per_second * 1.5
    ):
        notes.append("Throughput has large positive spikes relative to median.")
    return tuple(notes)


def render_recommendations() -> list[str]:
    return [
        "## Interpretation And Recommendations",
        "",
        (
            "Prefer the model family that improves Python completion pass rate without a large "
            "validation-perplexity or throughput penalty. Treat checkpoint-evaluation pass rate "
            "as noisy because it is derived from a fixed held-out task set, but use it to decide "
            "whether a run peaked before the final checkpoint."
        ),
        "",
        "Concrete next experiments:",
        "",
        (
            "- Re-run the strongest small configuration with multiple seeds to separate "
            "architecture signal from task-sampling variance."
        ),
        (
            "- For MoE runs with router dominance above 0.90, test stronger auxiliary balancing, "
            "router noise/jitter, or top-k=2 before increasing expert count."
        ),
        (
            "- If FIM improves completion metrics, run a probability sweep rather than a binary "
            "plain/FIM comparison."
        ),
        (
            "- For learning-rate schedules, sweep warmup length and minimum LR ratio around the "
            "best schedule instead of only comparing schedule families."
        ),
        (
            "- Evaluate the best checkpoint, not only the final checkpoint, on the full validation "
            "set and Python completion suite."
        ),
        (
            "- Record hardware metadata in future exports: GPU model/count, CPU, RAM, "
            "driver/CUDA/PyTorch versions, and wall-clock start/end timestamps."
        ),
        "",
        "## Limitations",
        "",
        "- Hardware details beyond distributed world size and strategy are absent from the export.",
        (
            "- Checkpoint-evaluation validation metrics use 200 validation documents, while final "
            "evaluation uses 464 documents; those losses/perplexities are related but not "
            "strictly identical measurements."
        ),
        (
            "- TensorBoard scalar extraction covers scalar tags and does not decode histogram "
            "distributions beyond the scalar router summaries."
        ),
        (
            "- The export contains only the latest checkpoint file for each selected run, "
            "not all checkpoint weights."
        ),
        "",
    ]


def render_analyzed_files(summaries: Sequence[ExperimentSummary]) -> list[str]:
    lines = [
        "## Files Analyzed",
        "",
        (
            "Outer export entries analyzed: every `manifest.json`, `bundle_manifest.json`, "
            "and `bundle.zip` in the outer export was opened or tested. Selected experiment "
            "files analyzed below are listed by path inside each selected nested bundle."
        ),
        "",
    ]
    for summary in summaries:
        lines.append(f"### `{summary.name}`")
        lines.append("")
        for analyzed_file in summary.analyzed_files:
            lines.append(f"- `{analyzed_file}`")
        lines.append("")
    return lines


def comparison_interpretation(title: str, summaries: Sequence[ExperimentSummary]) -> str:
    if not summaries:
        return "No selected summaries were available."
    best_final_pass = max_optional(
        values=summaries,
        metric=lambda summary: summary.final_evaluation.pass_rate,
    )
    best_final_ppl = min_optional(
        values=summaries,
        metric=lambda summary: summary.final_evaluation.perplexity,
    )
    fastest = max(
        summaries,
        key=lambda summary: summary.training_summary.mean_last_100_tokens_per_second,
    )
    pass_name = best_final_pass.name if best_final_pass is not None else "missing"
    ppl_name = best_final_ppl.name if best_final_ppl is not None else "missing"
    return (
        f"Interpretation: in this slice, `{pass_name}` has the best final pass rate, "
        f"`{ppl_name}` has the lowest final perplexity, and `{fastest.name}` has the highest "
        f"late-training throughput. Comparison basis: {title}."
    )


def model_type_label(summary: ExperimentSummary) -> str:
    return str(summary.configuration.model.type.value)


def feed_forward_dimension(summary: ExperimentSummary) -> int:
    model = summary.configuration.model
    match model:
        case DenseGptConfiguration() | ModernDenseGptConfiguration():
            return model.feed_forward_dimension
        case MoeGptConfiguration() | ModernMoeGptConfiguration():
            return model.expert_feed_forward_dimension


def expert_count(summary: ExperimentSummary) -> str:
    model = summary.configuration.model
    match model:
        case DenseGptConfiguration() | ModernDenseGptConfiguration():
            return "n/a"
        case MoeGptConfiguration() | ModernMoeGptConfiguration():
            return str(model.expert_count)


def router_top_k(summary: ExperimentSummary) -> str:
    model = summary.configuration.model
    match model:
        case DenseGptConfiguration() | ModernDenseGptConfiguration():
            return "n/a"
        case MoeGptConfiguration() | ModernMoeGptConfiguration():
            return str(model.router_top_k)


def step_list(evaluations: Sequence[EvaluationMetrics]) -> str:
    return ", ".join(
        str(evaluation.step) for evaluation in evaluations if evaluation.step is not None
    )


def evaluation_loss_cell(evaluation: EvaluationMetrics | None) -> str:
    if evaluation is None:
        return "missing"
    return (
        f"{evaluation.step or 'missing'}/"
        f"{format_optional_float(evaluation.validation_loss)}/"
        f"{format_optional_float(evaluation.perplexity)}"
    )


def evaluation_pass_cell(evaluation: EvaluationMetrics | None) -> str:
    if evaluation is None:
        return "missing"
    return f"{evaluation.step or 'missing'}/{format_percent(evaluation.pass_rate)}"


def evaluation_final_cell(evaluation: EvaluationMetrics | None) -> str:
    if evaluation is None:
        return "missing"
    return (
        f"{format_optional_float(evaluation.validation_loss)}/"
        f"{format_optional_float(evaluation.perplexity)}/"
        f"{format_percent(evaluation.pass_rate)}"
    )


def trajectory(evaluations: Sequence[EvaluationMetrics]) -> str:
    return "; ".join(
        f"{evaluation.step}: loss {format_optional_float(evaluation.validation_loss)}, "
        f"ppl {format_optional_float(evaluation.perplexity)}, "
        f"pass {format_percent(evaluation.pass_rate)}"
        for evaluation in evaluations
    )


def evaluation_step_sort_key(evaluation: EvaluationMetrics) -> int:
    return evaluation.step if evaluation.step is not None else -1


def optional_metric(evaluation: EvaluationMetrics | None, field_name: str) -> float | None:
    if evaluation is None:
        return None
    if field_name == "pass_rate":
        return evaluation.pass_rate
    if field_name == "validation_loss":
        return evaluation.validation_loss
    raise ValueError(f"Unsupported metric field: {field_name}")


def loss_delta(
    final_checkpoint: EvaluationMetrics | None,
    best_validation: EvaluationMetrics | None,
) -> float | None:
    if (
        final_checkpoint is None
        or best_validation is None
        or final_checkpoint.validation_loss is None
        or best_validation.validation_loss is None
    ):
        return None
    return final_checkpoint.validation_loss - best_validation.validation_loss


def pass_delta(
    final_checkpoint: EvaluationMetrics | None,
    best_pass_rate: EvaluationMetrics | None,
) -> float | None:
    if (
        final_checkpoint is None
        or best_pass_rate is None
        or final_checkpoint.pass_rate is None
        or best_pass_rate.pass_rate is None
    ):
        return None
    return final_checkpoint.pass_rate - best_pass_rate.pass_rate


def min_optional(
    values: Sequence[SummaryType],
    metric: ObjectMetric[SummaryType],
) -> SummaryType | None:
    metric_values = tuple((value, metric(value)) for value in values)
    present_values = tuple(
        (value, metric_value) for value, metric_value in metric_values if metric_value is not None
    )
    if not present_values:
        return None
    return min(present_values, key=lambda item: item[1])[0]


def max_optional(
    values: Sequence[SummaryType],
    metric: ObjectMetric[SummaryType],
) -> SummaryType | None:
    metric_values = tuple((value, metric(value)) for value in values)
    present_values = tuple(
        (value, metric_value) for value, metric_value in metric_values if metric_value is not None
    )
    if not present_values:
        return None
    return max(present_values, key=lambda item: item[1])[0]


def final_scalar_value(
    scalar_events: Mapping[str, Sequence[tuple[int, float]]],
    tag: str,
) -> float | None:
    values = scalar_events.get(tag)
    if not values:
        return None
    return values[-1][1]


def max_scalar_value(
    scalar_events: Mapping[str, Sequence[tuple[int, float]]],
    tag: str,
) -> float | None:
    values = scalar_events.get(tag)
    if not values:
        return None
    return max(value for _step, value in values)


def min_scalar_value(
    scalar_events: Mapping[str, Sequence[tuple[int, float]]],
    tag: str,
) -> float | None:
    values = scalar_events.get(tag)
    if not values:
        return None
    return min(value for _step, value in values)


def count_suffix(paths: Sequence[str], suffix: str) -> int:
    return sum(1 for path in paths if path.endswith(suffix))


def count_contains(paths: Sequence[str], needle: str) -> int:
    return sum(1 for path in paths if needle in path)


def format_int(value: int) -> str:
    return f"{value:,}"


def format_int_range(values: Sequence[int]) -> str:
    if not values:
        return "missing"
    minimum_value = min(values)
    maximum_value = max(values)
    if minimum_value == maximum_value:
        return format_int(minimum_value)
    return f"{format_int(minimum_value)}-{format_int(maximum_value)}"


def format_joined_values(values: Sequence[int | str]) -> str:
    if not values:
        return "missing"
    return ", ".join(str(value) for value in values)


def pluralize(singular_label: str, values: Sequence[int]) -> str:
    if values and all(value == 1 for value in values):
        return singular_label
    return f"{singular_label}s"


def format_optional_int(value: int | None) -> str:
    if value is None:
        return "missing"
    return format_int(value)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "missing"
    return format_float(value)


def format_percent(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value * 100:.2f}%"


def format_delta(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:+.4f}"


def format_percent_delta(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value * 100:+.2f} pp"


def json_object(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object.")
    return value


def json_objects(values: Sequence[JsonValue]) -> Iterable[JsonObject]:
    for value in values:
        yield json_object(value)


def array_field(data: JsonObject, field_name: str) -> Sequence[JsonValue]:
    value = data.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array field {field_name}.")
    return value


def optional_array_field(data: JsonObject, field_name: str) -> Sequence[JsonValue] | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array field {field_name}.")
    return value


def optional_object_field(data: JsonObject, field_name: str) -> JsonObject | None:
    value = data.get(field_name)
    if value is None:
        return None
    return json_object(value)


def string_field(data: JsonObject, field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"Expected JSON string field {field_name}.")
    return value


def optional_string_field(data: JsonObject, field_name: str) -> str | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected JSON string field {field_name}.")
    return value


def integer_field(data: JsonObject, field_name: str) -> int:
    value = data.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"Expected JSON integer field {field_name}.")
    return value


def optional_integer_field(data: JsonObject, field_name: str) -> int | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Expected JSON integer field {field_name}.")
    return value


def float_field(data: JsonObject, field_name: str) -> float:
    value = data.get(field_name)
    if not isinstance(value, int | float):
        raise ValueError(f"Expected JSON numeric field {field_name}.")
    return float(value)


def optional_float_field(data: JsonObject, field_name: str) -> float | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"Expected JSON numeric field {field_name}.")
    return float(value)


if __name__ == "__main__":
    main()
