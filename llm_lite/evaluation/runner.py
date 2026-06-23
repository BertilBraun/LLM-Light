from dataclasses import dataclass

from torch import nn

from llm_lite.config.models import (
    EvaluationConfiguration,
    InferenceConfiguration,
    PackingConfiguration,
)
from llm_lite.evaluation.exact_reproduction import evaluate_exact_reproduction
from llm_lite.evaluation.fixed_prompt_generation import (
    FixedPromptGenerationSample,
    evaluate_fixed_prompt_generation,
)
from llm_lite.evaluation.perplexity import evaluate_perplexity
from llm_lite.evaluation.python_completion import evaluate_python_completion
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stages.io import iter_processed_document_texts
from llm_lite.tokenizer.loading import TextTokenizer


@dataclass(frozen=True)
class EvaluationRunResult:
    report: dict[str, object]
    metrics: dict[str, int | float | str | bool]


def run_configured_evaluators(
    model: nn.Module,
    tokenizer: TextTokenizer,
    registry: ArtifactRegistry,
    evaluation_configuration: EvaluationConfiguration,
    inference_configuration: InferenceConfiguration,
    packing_configuration: PackingConfiguration,
) -> EvaluationRunResult:
    report: dict[str, object] = {}
    metrics: dict[str, int | float | str | bool] = {}
    exact_reproduction_configuration = evaluation_configuration.exact_reproduction
    if exact_reproduction_configuration is not None:
        exact_reproduction_result = evaluate_exact_reproduction(
            model=model,
            tokenizer=tokenizer,
            evaluation_configuration=exact_reproduction_configuration,
            inference_configuration=inference_configuration,
        )
        report['exact_reproduction'] = exact_reproduction_result.model_dump()
        metrics['exact_reproduction_passed'] = exact_reproduction_result.passed
        print(
            f'[eval] exact_reproduction passed={exact_reproduction_result.passed}',
            flush=True,
        )
        if not exact_reproduction_result.passed:
            raise ValueError('Exact reproduction evaluation failed.')
    perplexity_configuration = evaluation_configuration.perplexity
    if perplexity_configuration is not None:
        perplexity_result = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=iter_processed_document_texts(
                registry=registry,
                split=perplexity_configuration.split,
            ),
            evaluation_configuration=perplexity_configuration,
            packing_configuration=packing_configuration,
        )
        report['perplexity'] = perplexity_result.model_dump()
        metrics['perplexity_loss'] = perplexity_result.loss
        metrics['perplexity'] = perplexity_result.perplexity
        metrics['perplexity_sequences'] = perplexity_result.sequences
        metrics['perplexity_documents'] = perplexity_result.documents
        print(
            '[eval] perplexity '
            f'split={perplexity_result.split} '
            f'documents={perplexity_result.documents} '
            f'sequences={perplexity_result.sequences} '
            f'loss={perplexity_result.loss:.6f} '
            f'perplexity={perplexity_result.perplexity:.4f}',
            flush=True,
        )
    fixed_prompt_generation_configuration = evaluation_configuration.fixed_prompt_generation
    if fixed_prompt_generation_configuration is not None:
        fixed_prompt_generation_result = evaluate_fixed_prompt_generation(
            model=model,
            tokenizer=tokenizer,
            evaluation_configuration=fixed_prompt_generation_configuration,
            inference_configuration=inference_configuration,
        )
        report['fixed_prompt_generation'] = fixed_prompt_generation_result.model_dump()
        metrics['fixed_prompt_generation_samples'] = len(
            fixed_prompt_generation_result.samples,
        )
        print(
            f'[eval] fixed_prompt_generation samples={len(fixed_prompt_generation_result.samples)}',
            flush=True,
        )
        _print_fixed_prompt_generation_samples(
            samples=fixed_prompt_generation_result.samples,
        )
    python_completion_configuration = evaluation_configuration.python_completion
    if python_completion_configuration is not None:
        python_completion_result = evaluate_python_completion(
            model=model,
            tokenizer=tokenizer,
            evaluation_configuration=python_completion_configuration,
            inference_configuration=inference_configuration,
        )
        report['python_completion'] = python_completion_result.model_dump()
        metrics['python_completion_tasks'] = len(python_completion_result.tasks)
        metrics['python_completion_parsed_tasks'] = python_completion_result.parsed_tasks
        metrics['python_completion_executed_tasks'] = python_completion_result.executed_tasks
        metrics['python_completion_passed_checks'] = python_completion_result.passed_checks
        metrics['python_completion_total_checks'] = python_completion_result.total_checks
        metrics['python_completion_pass_rate'] = python_completion_result.pass_rate
        print(
            '[eval] python_completion '
            f'tasks={len(python_completion_result.tasks)} '
            f'parsed={python_completion_result.parsed_tasks} '
            f'executed={python_completion_result.executed_tasks} '
            f'passed_checks={python_completion_result.passed_checks} '
            f'total_checks={python_completion_result.total_checks} '
            f'pass_rate={python_completion_result.pass_rate:.4f}',
            flush=True,
        )
    return EvaluationRunResult(report=report, metrics=metrics)


def _print_fixed_prompt_generation_samples(
    samples: tuple[FixedPromptGenerationSample, ...],
) -> None:
    for sample_index, sample in enumerate(samples):
        print(f'[eval-sample {sample_index}] {sample.generated_text}', flush=True)
