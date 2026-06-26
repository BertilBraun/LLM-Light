from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llm_lite.post_training.scoring import PythonCandidateScoreRecord


class DpoPreferenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    prompt: str
    chosen_completion: str
    rejected_completion: str
    chosen_score: float
    rejected_score: float
    score_margin: float
    chosen_sample_index: int
    rejected_sample_index: int


class DpoPreferenceDatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    preferences: tuple[DpoPreferenceRecord, ...]


def build_dpo_preferences_from_scores(
    scores: tuple[PythonCandidateScoreRecord, ...],
) -> DpoPreferenceDatasetRecord:
    scores_by_task_id: dict[str, list[PythonCandidateScoreRecord]] = defaultdict(list)
    for score in scores:
        scores_by_task_id[score.task_id].append(score)
    preferences: list[DpoPreferenceRecord] = []
    for task_scores in scores_by_task_id.values():
        preferences.extend(_preferences_for_task(scores=tuple(task_scores)))
    return DpoPreferenceDatasetRecord(preferences=tuple(preferences))


def write_dpo_preferences_jsonl(
    preference_dataset: DpoPreferenceDatasetRecord,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [preference.model_dump_json() for preference in preference_dataset.preferences]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dpo_preferences_jsonl(input_path: Path) -> DpoPreferenceDatasetRecord:
    preferences = tuple(
        DpoPreferenceRecord.model_validate_json(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip() != ""
    )
    return DpoPreferenceDatasetRecord(preferences=preferences)


def _preferences_for_task(
    scores: tuple[PythonCandidateScoreRecord, ...],
) -> tuple[DpoPreferenceRecord, ...]:
    preferences: list[DpoPreferenceRecord] = []
    for chosen_index, chosen_score in enumerate(scores):
        for rejected_score in scores[chosen_index + 1 :]:
            preference = _preference_from_pair(
                first_score=chosen_score,
                second_score=rejected_score,
            )
            if preference is not None:
                preferences.append(preference)
    return tuple(preferences)


def _preference_from_pair(
    first_score: PythonCandidateScoreRecord,
    second_score: PythonCandidateScoreRecord,
) -> DpoPreferenceRecord | None:
    if first_score.score == second_score.score:
        return None
    if first_score.perfect and second_score.perfect:
        return None
    if first_score.score > second_score.score:
        return _preference_record(chosen_score=first_score, rejected_score=second_score)
    return _preference_record(chosen_score=second_score, rejected_score=first_score)


def _preference_record(
    chosen_score: PythonCandidateScoreRecord,
    rejected_score: PythonCandidateScoreRecord,
) -> DpoPreferenceRecord:
    return DpoPreferenceRecord(
        task_id=chosen_score.task_id,
        prompt=chosen_score.prompt,
        chosen_completion=chosen_score.generated_text,
        rejected_completion=rejected_score.generated_text,
        chosen_score=chosen_score.score,
        rejected_score=rejected_score.score,
        score_margin=chosen_score.score - rejected_score.score,
        chosen_sample_index=chosen_score.sample_index,
        rejected_sample_index=rejected_score.sample_index,
    )
