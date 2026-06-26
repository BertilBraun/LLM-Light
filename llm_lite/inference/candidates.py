from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import (
    DecodingConfiguration,
    GenerationStopReason,
    InferenceConfiguration,
)
from llm_lite.inference.engine import generate_batch
from llm_lite.tokenizer.loading import TextTokenizer


class CandidatePrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    prompt: str


class CandidateTimingRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    tokens_per_second: float
    sequences_per_second: float


class GeneratedCandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    prompt: str
    sample_index: int
    generated_text: str
    full_text: str
    token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    prompt_length: int
    generated_token_count: int
    stop_reason: GenerationStopReason
    decoding: DecodingConfiguration
    timing: CandidateTimingRecord


class CandidateGenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: tuple[GeneratedCandidateRecord, ...]


class CandidateScoreRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    sample_index: int
    score: float
    parsed: bool
    passed_checks: int
    total_checks: int
    error: str | None


class CandidateScoringResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    scores: tuple[CandidateScoreRecord, ...]


def generate_candidates(
    model: nn.Module,
    tokenizer: TextTokenizer,
    candidate_prompts: tuple[CandidatePrompt, ...],
    samples_per_prompt: int,
    inference_configuration: InferenceConfiguration,
) -> CandidateGenerationResult:
    expanded_prompts = tuple(
        candidate_prompt.prompt
        for candidate_prompt in candidate_prompts
        for _sample_index in range(samples_per_prompt)
    )
    generation_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=expanded_prompts,
        inference_configuration=inference_configuration,
    )
    records: list[GeneratedCandidateRecord] = []
    generation_index = 0
    for candidate_prompt in candidate_prompts:
        for sample_index in range(samples_per_prompt):
            generation_result = generation_results[generation_index]
            records.append(
                GeneratedCandidateRecord(
                    task_id=candidate_prompt.task_id,
                    prompt=candidate_prompt.prompt,
                    sample_index=sample_index,
                    generated_text=generation_result.generated_text,
                    full_text=generation_result.full_text,
                    token_ids=generation_result.token_ids,
                    generated_token_ids=generation_result.generated_token_ids,
                    prompt_length=generation_result.prompt_length,
                    generated_token_count=generation_result.generated_token_count,
                    stop_reason=generation_result.stop_reason,
                    decoding=inference_configuration.decoding,
                    timing=CandidateTimingRecord(
                        prefill_seconds=generation_result.timing.prefill_seconds,
                        decode_seconds=generation_result.timing.decode_seconds,
                        total_seconds=generation_result.timing.total_seconds,
                        tokens_per_second=generation_result.throughput.tokens_per_second,
                        sequences_per_second=generation_result.throughput.sequences_per_second,
                    ),
                ),
            )
            generation_index += 1
    return CandidateGenerationResult(candidates=tuple(records))


def write_candidate_jsonl(
    candidate_generation_result: CandidateGenerationResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [candidate.model_dump_json() for candidate in candidate_generation_result.candidates]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
