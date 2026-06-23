from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import InferenceConfiguration
from llm_lite.evaluation.python_completion import PythonCompletionTaskRecord
from llm_lite.inference.candidates import (
    CandidateGenerationResult,
    CandidatePrompt,
    generate_candidates,
    write_candidate_jsonl,
)
from llm_lite.post_training.preference import (
    DpoPreferenceDatasetRecord,
    build_dpo_preferences_from_scores,
    write_dpo_preferences_jsonl,
)
from llm_lite.post_training.scoring import (
    PythonCandidateScoringResult,
    score_python_candidates,
)
from llm_lite.tokenizer.loading import TextTokenizer


class PythonGeneratedDpoDataResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: CandidateGenerationResult
    scores: PythonCandidateScoringResult
    preferences: DpoPreferenceDatasetRecord


def generate_python_dpo_data(
    model: nn.Module,
    tokenizer: TextTokenizer,
    tasks: tuple[PythonCompletionTaskRecord, ...],
    samples_per_prompt: int,
    inference_configuration: InferenceConfiguration,
    execution_timeout_seconds: float,
) -> PythonGeneratedDpoDataResult:
    candidate_prompts = tuple(
        CandidatePrompt(task_id=task.task_id, prompt=task.prompt)
        for task in tasks
    )
    candidate_generation_result = generate_candidates(
        model=model,
        tokenizer=tokenizer,
        candidate_prompts=candidate_prompts,
        samples_per_prompt=samples_per_prompt,
        inference_configuration=inference_configuration,
    )
    scoring_result = score_python_candidates(
        candidates=candidate_generation_result.candidates,
        tasks=tasks,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    preference_dataset = build_dpo_preferences_from_scores(scores=scoring_result.scores)
    return PythonGeneratedDpoDataResult(
        candidates=candidate_generation_result,
        scores=scoring_result,
        preferences=preference_dataset,
    )


def write_python_generated_dpo_data(
    result: PythonGeneratedDpoDataResult,
    artifact_directory: Path,
) -> None:
    write_candidate_jsonl(
        candidate_generation_result=result.candidates,
        output_path=artifact_directory / "candidates.jsonl",
    )
    _write_scores_jsonl(
        scoring_result=result.scores,
        output_path=artifact_directory / "scores.jsonl",
    )
    write_dpo_preferences_jsonl(
        preference_dataset=result.preferences,
        output_path=artifact_directory / "preferences.jsonl",
    )


def _write_scores_jsonl(
    scoring_result: PythonCandidateScoringResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [score.model_dump_json() for score in scoring_result.scores]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
