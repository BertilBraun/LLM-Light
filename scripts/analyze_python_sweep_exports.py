from __future__ import annotations

import argparse
import math
import statistics
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from analyze_python_sweep_export import (
    BundleCandidate,
    EvaluationMetrics,
    ExperimentSummary,
    analyze_experiment,
    discover_bundle_candidates,
    escape_svg,
    expert_count,
    feed_forward_dimension,
    format_int,
    model_type_label,
    optional_metric,
    plot_color,
    router_top_k,
    select_candidates,
    short_experiment_name,
    write_line_plot,
    write_scatter_plot,
    write_svg,
)


@dataclass(frozen=True)
class ExportAnalysis:
    candidates: tuple[BundleCandidate, ...]
    summaries: tuple[ExperimentSummary, ...]


@dataclass(frozen=True)
class MetricSpread:
    mean: float
    deviation: float
    count: int


@dataclass(frozen=True)
class ExperimentGroup:
    name: str
    summaries: tuple[ExperimentSummary, ...]


@dataclass(frozen=True)
class AggregatedCheckpointPoint:
    step: int
    mean: float


@dataclass(frozen=True)
class AggregatedPlotRecord:
    title: str
    path: Path
    caption: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plots-directory", required=False, type=Path)
    args = parser.parse_args()

    output_path = args.output
    plots_directory = args.plots_directory or output_path.parent / "images" / "python_model_sweep"
    analyses = tuple(analyze_export(export_path=export_path) for export_path in args.export)
    groups = group_experiments(summaries=all_summaries(analyses=analyses))
    plot_records = write_aggregated_plots(groups=groups, plots_directory=plots_directory)
    markdown = render_markdown(
        groups=groups,
        plot_records=plot_records,
        output_path=output_path,
        export_count=len(args.export),
        candidate_count=sum(len(analysis.candidates) for analysis in analyses),
    )
    output_path.write_text(markdown, encoding="utf-8")


def analyze_export(export_path: Path) -> ExportAnalysis:
    with zipfile.ZipFile(export_path) as outer_archive:
        source_bad_file = outer_archive.testzip()
        if source_bad_file is not None:
            raise ValueError(f"Source export ZIP failed integrity check at {source_bad_file}.")
        candidates = discover_bundle_candidates(outer_archive=outer_archive)
        selected_candidates = select_candidates(candidates=candidates)
        summaries = tuple(
            analyze_experiment(outer_archive=outer_archive, candidate=candidate)
            for candidate in selected_candidates
        )
    return ExportAnalysis(candidates=candidates, summaries=summaries)


def all_summaries(analyses: tuple[ExportAnalysis, ...]) -> tuple[ExperimentSummary, ...]:
    summaries: list[ExperimentSummary] = []
    for analysis in analyses:
        summaries.extend(analysis.summaries)
    return tuple(summaries)


def group_experiments(summaries: tuple[ExperimentSummary, ...]) -> tuple[ExperimentGroup, ...]:
    groups: list[ExperimentGroup] = []
    for name in sorted({summary.name for summary in summaries}):
        grouped_summaries = tuple(summary for summary in summaries if summary.name == name)
        groups.append(ExperimentGroup(name=name, summaries=grouped_summaries))
    return tuple(groups)


def write_aggregated_plots(
    groups: tuple[ExperimentGroup, ...],
    plots_directory: Path,
) -> tuple[AggregatedPlotRecord, ...]:
    plots_directory.mkdir(parents=True, exist_ok=True)
    plot_records = (
        AggregatedPlotRecord(
            title="Checkpoint Python Completion Pass Rate",
            path=plots_directory / "checkpoint_pass_rate.svg",
            caption="Mean checkpoint pass rate by experiment. Repeated runs are averaged per step.",
        ),
        AggregatedPlotRecord(
            title="Checkpoint AST Parse Rate",
            path=plots_directory / "checkpoint_parse_rate.svg",
            caption="Mean parseable-completion rate by experiment and checkpoint step.",
        ),
        AggregatedPlotRecord(
            title="Checkpoint Passed Checks",
            path=plots_directory / "checkpoint_passed_checks.svg",
            caption="Mean number of unit-test checks passed by experiment and checkpoint step.",
        ),
        AggregatedPlotRecord(
            title="Mean Final Pass Rate",
            path=plots_directory / "final_pass_rate.svg",
            caption="Mean final pass rate with +/- one population standard deviation.",
        ),
        AggregatedPlotRecord(
            title="Mean Final Perplexity",
            path=plots_directory / "final_perplexity.svg",
            caption="Mean final validation perplexity with +/- one population standard deviation.",
        ),
        AggregatedPlotRecord(
            title="Active Parameters vs Mean Final Pass Rate",
            path=plots_directory / "size_vs_pass_rate.svg",
            caption="Mean final pass rate against active parameter count.",
        ),
    )
    write_line_plot(
        path=plot_records[0].path,
        title=plot_records[0].title,
        y_label="Pass rate (%)",
        series=checkpoint_series(groups=groups, metric_name="pass_rate", scale=100.0),
    )
    write_line_plot(
        path=plot_records[1].path,
        title=plot_records[1].title,
        y_label="AST parse rate (%)",
        series=parse_rate_series(groups=groups),
    )
    write_line_plot(
        path=plot_records[2].path,
        title=plot_records[2].title,
        y_label="Passed checks",
        series=checkpoint_series(groups=groups, metric_name="passed_checks", scale=1.0),
    )
    write_metric_bar_plot(
        path=plot_records[3].path,
        title=plot_records[3].title,
        y_label="Final pass rate (%)",
        groups=groups,
        metric_name="final_pass_rate",
        scale=100.0,
        lower_is_better=False,
    )
    write_metric_bar_plot(
        path=plot_records[4].path,
        title=plot_records[4].title,
        y_label="Final perplexity",
        groups=groups,
        metric_name="final_perplexity",
        scale=1.0,
        lower_is_better=True,
    )
    write_scatter_plot(
        path=plot_records[5].path,
        title=plot_records[5].title,
        x_label="Active parameters (log10)",
        y_label="Mean final pass rate (%)",
        points=size_pass_points(groups=groups),
    )
    return plot_records


def checkpoint_series(
    groups: tuple[ExperimentGroup, ...],
    metric_name: str,
    scale: float,
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    series: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for group in groups:
        points: list[tuple[float, float]] = []
        for step in checkpoint_steps(group=group):
            values = tuple(
                checkpoint_metric_value(evaluation=evaluation, metric_name=metric_name) * scale
                for summary in group.summaries
                for evaluation in summary.checkpoint_summary.evaluations
                if evaluation.step == step
                and checkpoint_metric_value(evaluation=evaluation, metric_name=metric_name)
                is not None
            )
            if values:
                points.append((float(step), statistics.fmean(values)))
        if points:
            series.append((short_experiment_name(group.name), tuple(points)))
    return tuple(series)


def parse_rate_series(
    groups: tuple[ExperimentGroup, ...],
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    series: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for group in groups:
        points: list[tuple[float, float]] = []
        for step in checkpoint_steps(group=group):
            values = tuple(
                (float(evaluation.parsed_tasks) / float(evaluation.completion_tasks)) * 100.0
                for summary in group.summaries
                for evaluation in summary.checkpoint_summary.evaluations
                if evaluation.step == step
                and evaluation.parsed_tasks is not None
                and evaluation.completion_tasks is not None
            )
            if values:
                points.append((float(step), statistics.fmean(values)))
        if points:
            series.append((short_experiment_name(group.name), tuple(points)))
    return tuple(series)


def checkpoint_steps(group: ExperimentGroup) -> tuple[int, ...]:
    steps = tuple(
        evaluation.step
        for summary in group.summaries
        for evaluation in summary.checkpoint_summary.evaluations
        if evaluation.step is not None
    )
    return tuple(sorted(set(steps)))


def checkpoint_metric_value(
    evaluation: EvaluationMetrics,
    metric_name: str,
) -> float | None:
    match metric_name:
        case "pass_rate":
            return evaluation.pass_rate
        case "passed_checks":
            return float(evaluation.passed_checks) if evaluation.passed_checks is not None else None
        case _:
            raise ValueError(f"Unsupported checkpoint metric: {metric_name}.")


def size_pass_points(
    groups: tuple[ExperimentGroup, ...],
) -> tuple[tuple[str, float, float, float], ...]:
    points: list[tuple[str, float, float, float]] = []
    for group in groups:
        pass_spread = metric_spread(values=final_pass_rates(group=group))
        active_parameters = metric_spread(values=active_parameter_values(group=group)).mean
        radius = 4.0 + (math.log10(active_parameters) - 5.9) * 4.0
        points.append(
            (
                short_experiment_name(group.name),
                math.log10(active_parameters),
                pass_spread.mean * 100.0,
                max(4.0, radius),
            ),
        )
    return tuple(points)


def write_metric_bar_plot(
    path: Path,
    title: str,
    y_label: str,
    groups: tuple[ExperimentGroup, ...],
    metric_name: str,
    scale: float,
    lower_is_better: bool,
) -> None:
    width = 1500
    height = 720
    margin_left = 78
    margin_right = 46
    margin_top = 58
    margin_bottom = 220
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    records = tuple(
        sorted(
            (
                (
                    group,
                    metric_spread(values=group_metric_values(group=group, metric_name=metric_name)),
                )
                for group in groups
            ),
            key=lambda item: item[1].mean,
            reverse=not lower_is_better,
        )
    )
    maximum_value = max((spread.mean + spread.deviation) * scale for _group, spread in records)
    y_max = maximum_value * 1.12
    group_width = plot_width / len(records)
    bar_width = min(34.0, group_width * 0.58)

    def y_position(value: float) -> float:
        return margin_top + plot_height - (value / y_max) * plot_height

    elements = svg_base_elements(
        width=width,
        height=height,
        title=title,
        x_label="Experiment",
        y_label=y_label,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    elements.extend(
        svg_horizontal_grid(
            y_max=y_max,
            y_position=y_position,
            margin_left=margin_left,
            margin_top=margin_top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
    )
    for index, (group, spread) in enumerate(records):
        center_x = margin_left + group_width * index + group_width / 2.0
        mean_value = spread.mean * scale
        deviation = spread.deviation * scale
        top_y = y_position(mean_value)
        zero_y = margin_top + plot_height
        color = plot_color(index)
        elements.append(
            f'<rect x="{center_x - bar_width / 2.0:.1f}" y="{top_y:.1f}" '
            f'width="{bar_width:.1f}" height="{zero_y - top_y:.1f}" fill="{color}" />'
        )
        if deviation > 0.0:
            error_top = y_position(mean_value + deviation)
            error_bottom = y_position(max(0.0, mean_value - deviation))
            elements.append(
                f'<line x1="{center_x:.1f}" y1="{error_top:.1f}" '
                f'x2="{center_x:.1f}" y2="{error_bottom:.1f}" '
                'stroke="#20242c" stroke-width="1.4" />'
            )
            elements.append(
                f'<line x1="{center_x - 5:.1f}" y1="{error_top:.1f}" '
                f'x2="{center_x + 5:.1f}" y2="{error_top:.1f}" '
                'stroke="#20242c" stroke-width="1.4" />'
            )
            elements.append(
                f'<line x1="{center_x - 5:.1f}" y1="{error_bottom:.1f}" '
                f'x2="{center_x + 5:.1f}" y2="{error_bottom:.1f}" '
                'stroke="#20242c" stroke-width="1.4" />'
            )
        label = short_experiment_name(group.name)
        elements.append(
            f'<text x="{center_x:.1f}" y="{height - 42}" class="x-label" '
            f'text-anchor="end" transform="rotate(-42 {center_x:.1f} {height - 42})">'
            f"{escape_svg(label)}</text>"
        )
    write_svg(path=path, width=width, height=height, elements=elements)


def svg_base_elements(
    width: int,
    height: int,
    title: str,
    x_label: str,
    y_label: str,
    margin_left: int,
    margin_top: int,
    margin_bottom: int,
    plot_width: int,
    plot_height: int,
) -> list[str]:
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


def svg_horizontal_grid(
    y_max: float,
    y_position: Callable[[float], float],
    margin_left: int,
    margin_top: int,
    plot_width: int,
    plot_height: int,
) -> list[str]:
    elements: list[str] = []
    assert callable(y_position)
    for tick_index in range(6):
        y_value = y_max * (tick_index / 5.0)
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


def render_markdown(
    groups: tuple[ExperimentGroup, ...],
    plot_records: tuple[AggregatedPlotRecord, ...],
    output_path: Path,
    export_count: int,
    candidate_count: int,
) -> str:
    total_runs = sum(len(group.summaries) for group in groups)
    lines = [
        "# Python Model Sweep Results",
        "",
        (
            f"Aggregated {total_runs} selected runs into {len(groups)} experiment groups "
            f"on {date.today().isoformat()}."
        ),
        "",
        (
            f"Source identity is intentionally ignored. The input was {export_count} ZIP exports "
            f"with {candidate_count} candidate bundles; duplicate experiment names are grouped "
            "before reporting."
        ),
        "",
        (
            "Numeric cells use `mean +/- population standard deviation` across runs with the "
            "same experiment name. Static configuration fields are shown once when they are "
            "identical across the group."
        ),
        "",
    ]
    lines.extend(render_key_findings(groups=groups))
    lines.extend(render_result_table(groups=groups))
    lines.extend(render_plots(plot_records=plot_records, output_path=output_path))
    lines.extend(render_router_observations(groups=groups))
    lines.extend(render_recommendations(groups=groups))
    return "\n".join(lines) + "\n"


def render_key_findings(groups: tuple[ExperimentGroup, ...]) -> list[str]:
    best_group = max(groups, key=lambda group: metric_spread(final_pass_rates(group=group)).mean)
    best_small_group = max(
        (
            group
            for group in groups
            if metric_spread(active_parameter_values(group=group)).mean < 2_000_000
        ),
        key=lambda group: metric_spread(final_pass_rates(group=group)).mean,
    )
    fastest_group = max(
        groups,
        key=lambda group: metric_spread(throughput_values(group=group)).mean,
    )
    most_variable_group = max(
        groups,
        key=lambda group: metric_spread(final_pass_rates(group=group)).deviation,
    )
    return [
        "## Key Findings",
        "",
        (
            f"- Best mean final pass rate: `{best_group.name}` at "
            f"{format_percent_spread(metric_spread(final_pass_rates(group=best_group)))}."
        ),
        (
            f"- Best sub-2M-active-parameter model: `{best_small_group.name}` at "
            f"{format_percent_spread(metric_spread(final_pass_rates(group=best_small_group)))}."
        ),
        (
            f"- Highest mean late-training throughput: `{fastest_group.name}` at "
            f"{format_number_spread(metric_spread(throughput_values(group=fastest_group)), 0)} "
            "tokens/s."
        ),
        (
            f"- Largest repeated-run final-pass variation: `{most_variable_group.name}` at "
            f"{format_percent_spread(metric_spread(final_pass_rates(group=most_variable_group)))}."
        ),
        "",
    ]


def render_result_table(groups: tuple[ExperimentGroup, ...]) -> list[str]:
    ordered_groups = tuple(
        sorted(
            groups,
            key=lambda group: metric_spread(final_pass_rates(group=group)).mean,
            reverse=True,
        )
    )
    lines = [
        "## Aggregated Results",
        "",
        (
            "| Experiment | Runs | Type | Active params | Total params | Dim | Layers | Heads | "
            "FFN/expert FFN | Experts/top-k | Final pass | Best ckpt pass | Final ppl | "
            "Final eval loss | Train loss | Mean TPS last 100 |"
        ),
        (
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
    ]
    for group in ordered_groups:
        representative = group.summaries[0]
        lines.append(
            "| "
            f"`{group.name}` | {len(group.summaries)} | {model_type_label(representative)} | "
            f"{format_integer_spread(active_parameter_values(group=group))} | "
            f"{format_integer_spread(total_parameter_values(group=group))} | "
            f"{format_integer_spread(model_dimension_values(group=group))} | "
            f"{format_integer_spread(model_layer_values(group=group))} | "
            f"{format_integer_spread(model_head_values(group=group))} | "
            f"{format_integer_spread(feed_forward_values(group=group))} | "
            f"{format_experts_top_k(group=group)} | "
            f"{format_percent_spread(metric_spread(final_pass_rates(group=group)))} | "
            f"{format_percent_spread(metric_spread(best_checkpoint_pass_rates(group=group)))} | "
            f"{format_number_spread(metric_spread(final_perplexities(group=group)), 4)} | "
            f"{format_number_spread(metric_spread(final_eval_losses(group=group)), 4)} | "
            f"{format_number_spread(metric_spread(train_losses(group=group)), 4)} | "
            f"{format_number_spread(metric_spread(throughput_values(group=group)), 0)} |",
        )
    lines.append("")
    return lines


def render_plots(
    plot_records: tuple[AggregatedPlotRecord, ...],
    output_path: Path,
) -> list[str]:
    lines = [
        "## Plots",
        "",
        (
            "Checkpoint evaluations are shown only as plots; the full per-step table is "
            "intentionally omitted."
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
            ]
        )
    return lines


def render_router_observations(groups: tuple[ExperimentGroup, ...]) -> list[str]:
    moe_groups = tuple(group for group in groups if router_dominance_values(group=group))
    if not moe_groups:
        return []
    lines = [
        "## Router Observations",
        "",
        (
            "| Experiment | Runs | Final worst-layer dominance | Final worst-layer entropy | "
            "Observation |"
        ),
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for group in moe_groups:
        dominance = metric_spread(router_dominance_values(group=group))
        entropy = metric_spread(router_entropy_values(group=group))
        observation = (
            "severe expert dominance" if dominance.mean >= 0.90 else "router usage less collapsed"
        )
        lines.append(
            "| "
            f"`{group.name}` | {len(group.summaries)} | "
            f"{format_number_spread(dominance, 4)} | "
            f"{format_number_spread(entropy, 4)} | {observation} |",
        )
    lines.append("")
    return lines


def render_recommendations(groups: tuple[ExperimentGroup, ...]) -> list[str]:
    best_small_group = max(
        (
            group
            for group in groups
            if metric_spread(active_parameter_values(group=group)).mean < 2_000_000
        ),
        key=lambda group: metric_spread(final_pass_rates(group=group)).mean,
    )
    return [
        "## Interpretation",
        "",
        (
            f"`{best_small_group.name}` is the best small-model direction by aggregate final pass "
            "rate. Its checkpoint curve still matters: use the checkpoint plots to pick the "
            "full-evaluation checkpoint rather than assuming the final checkpoint is optimal."
        ),
        "",
        (
            "Classic MoE runs with high router dominance should be treated as router-collapse "
            "experiments before scaling expert count. The modern MoE result is the stronger "
            "small-model branch in these runs."
        ),
        "",
    ]


def group_metric_values(group: ExperimentGroup, metric_name: str) -> tuple[float, ...]:
    match metric_name:
        case "final_pass_rate":
            return final_pass_rates(group=group)
        case "final_perplexity":
            return final_perplexities(group=group)
        case _:
            raise ValueError(f"Unsupported group metric: {metric_name}.")


def final_pass_rates(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(
        summary.final_evaluation.pass_rate
        for summary in group.summaries
        if summary.final_evaluation.pass_rate is not None
    )


def best_checkpoint_pass_rates(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(
        pass_rate
        for pass_rate in (
            optional_metric(summary.checkpoint_summary.best_pass_rate, "pass_rate")
            for summary in group.summaries
        )
        if pass_rate is not None
    )


def final_perplexities(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(
        summary.final_evaluation.perplexity
        for summary in group.summaries
        if summary.final_evaluation.perplexity is not None
    )


def final_eval_losses(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(
        summary.final_evaluation.validation_loss
        for summary in group.summaries
        if summary.final_evaluation.validation_loss is not None
    )


def train_losses(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(summary.training_summary.final_loss for summary in group.summaries)


def throughput_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(
        summary.training_summary.mean_last_100_tokens_per_second for summary in group.summaries
    )


def active_parameter_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(float(summary.parameter_summary.active_parameters) for summary in group.summaries)


def total_parameter_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(float(summary.parameter_summary.total_parameters) for summary in group.summaries)


def model_dimension_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(float(summary.configuration.model.dimension) for summary in group.summaries)


def model_layer_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(float(summary.configuration.model.layers) for summary in group.summaries)


def model_head_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(float(summary.configuration.model.attention_heads) for summary in group.summaries)


def feed_forward_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(float(feed_forward_dimension(summary)) for summary in group.summaries)


def router_dominance_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(
        summary.tensorboard_summary.router_final_worst_dominance
        for summary in group.summaries
        if summary.tensorboard_summary.router_final_worst_dominance is not None
    )


def router_entropy_values(group: ExperimentGroup) -> tuple[float, ...]:
    return tuple(
        summary.tensorboard_summary.router_final_worst_entropy
        for summary in group.summaries
        if summary.tensorboard_summary.router_final_worst_entropy is not None
    )


def format_experts_top_k(group: ExperimentGroup) -> str:
    values = tuple(experts_top_k(summary=summary) for summary in group.summaries)
    unique_values = tuple(sorted(set(values)))
    if len(unique_values) == 1:
        return unique_values[0]
    return ", ".join(unique_values)


def experts_top_k(summary: ExperimentSummary) -> str:
    experts = expert_count(summary)
    top_k = router_top_k(summary)
    if experts == "n/a" and top_k == "n/a":
        return "n/a"
    return f"{experts}/{top_k}"


def metric_spread(values: tuple[float, ...]) -> MetricSpread:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence.")
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    return MetricSpread(mean=mean, deviation=deviation, count=len(values))


def format_percent_spread(spread: MetricSpread) -> str:
    return f"{spread.mean * 100.0:.2f}% +/- {spread.deviation * 100.0:.2f} pp"


def format_number_spread(spread: MetricSpread, decimals: int) -> str:
    if decimals == 0:
        return f"{format_int(round(spread.mean))} +/- {format_int(round(spread.deviation))}"
    return f"{spread.mean:.{decimals}f} +/- {spread.deviation:.{decimals}f}"


def format_integer_spread(values: tuple[float, ...]) -> str:
    spread = metric_spread(values=values)
    if spread.deviation == 0.0:
        return format_int(round(spread.mean))
    return f"{format_int(round(spread.mean))} +/- {format_int(round(spread.deviation))}"


if __name__ == "__main__":
    main()
