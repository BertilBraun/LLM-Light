import copy
from pathlib import Path

import torch

from llm_lite.config.models import (
    DataLoaderConfiguration,
    DecodingStrategy,
    DenseGptConfiguration,
    GenerationStopReason,
    GreedyDecodingConfiguration,
    ModelType,
    TrainingConfiguration,
)
from llm_lite.evaluation.python_completion import PythonCompletionTaskRecord
from llm_lite.inference.candidates import (
    CandidateTimingRecord,
    GeneratedCandidateRecord,
)
from llm_lite.model.gpt import DenseGpt
from llm_lite.post_training.dpo import build_dpo_preference_dataset
from llm_lite.post_training.preference import (
    DpoPreferenceRecord,
    build_dpo_preferences_from_scores,
    load_dpo_preferences_jsonl,
    write_dpo_preferences_jsonl,
)
from llm_lite.post_training.scoring import (
    PythonCandidateScoreRecord,
    score_python_candidates,
)
from llm_lite.tokenizer.character import train_character_tokenizer
from llm_lite.training.objectives import (
    DirectPreferenceOptimizationObjectiveRunner,
)
from llm_lite.training.trainer import train_model


def test_python_candidate_scoring_rewards_parse_and_passed_checks() -> None:
    task = PythonCompletionTaskRecord(
        task_id="add",
        prompt="def add(a, b):\n    return ",
        checks=("add(1, 2) == 3", "add(2, 5) == 7"),
    )
    scoring_result = score_python_candidates(
        candidates=(
            _candidate(task_id="add", sample_index=0, generated_text="a + b"),
            _candidate(task_id="add", sample_index=1, generated_text="a - b"),
            _candidate(task_id="add", sample_index=2, generated_text="@"),
        ),
        tasks=(task,),
        execution_timeout_seconds=2.0,
    )

    assert scoring_result.scores[0].score == 3.0
    assert scoring_result.scores[0].perfect is True
    assert scoring_result.scores[1].score == 1.0
    assert scoring_result.scores[1].perfect is False
    assert scoring_result.scores[2].score == 0.0
    assert scoring_result.scores[2].parsed is False


def test_preference_creation_skips_ties_and_both_perfect_pairs(tmp_path: Path) -> None:
    preference_dataset = build_dpo_preferences_from_scores(
        scores=(
            _score(task_id="task-a", sample_index=0, generated_text="best", score=2.0),
            _score(task_id="task-a", sample_index=1, generated_text="worse", score=1.0),
            _score(task_id="task-a", sample_index=2, generated_text="tie", score=1.0),
            _score(
                task_id="task-b",
                sample_index=0,
                generated_text="perfect-a",
                score=3.0,
                perfect=True,
            ),
            _score(
                task_id="task-b",
                sample_index=1,
                generated_text="perfect-b",
                score=3.0,
                perfect=True,
            ),
        ),
    )
    output_path = tmp_path / "preferences.jsonl"

    write_dpo_preferences_jsonl(
        preference_dataset=preference_dataset,
        output_path=output_path,
    )
    loaded_dataset = load_dpo_preferences_jsonl(input_path=output_path)

    assert len(loaded_dataset.preferences) == 2
    assert loaded_dataset.preferences[0].chosen_completion == "best"
    assert loaded_dataset.preferences[0].rejected_completion == "worse"
    assert loaded_dataset.preferences[1].chosen_completion == "best"
    assert loaded_dataset.preferences[1].rejected_completion == "tie"


def test_dpo_loss_is_finite_for_tiny_preference_batch() -> None:
    torch.manual_seed(5)
    tokenizer = train_character_tokenizer(
        texts=["abxy"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _tiny_model(vocabulary_size=tokenizer.vocabulary_size)
    reference_model = copy.deepcopy(model)
    dataset = build_dpo_preference_dataset(
        preferences=(
            DpoPreferenceRecord(
                task_id="task",
                prompt="a",
                chosen_completion="x",
                rejected_completion="y",
                chosen_score=2.0,
                rejected_score=1.0,
                score_margin=1.0,
                chosen_sample_index=0,
                rejected_sample_index=1,
            ),
        ),
        tokenizer=tokenizer,
    )
    batch = dataset[0]
    batched_batch = batch._replace(
        chosen_token_ids=batch.chosen_token_ids.unsqueeze(dim=0),
        rejected_token_ids=batch.rejected_token_ids.unsqueeze(dim=0),
        chosen_completion_mask=batch.chosen_completion_mask.unsqueeze(dim=0),
        rejected_completion_mask=batch.rejected_completion_mask.unsqueeze(dim=0),
    )

    loss = DirectPreferenceOptimizationObjectiveRunner(
        reference_model=reference_model,
        beta=0.1,
    ).loss(model=model, batch=batched_batch)

    assert torch.isfinite(loss)


def test_dpo_training_step_updates_policy_not_reference(tmp_path: Path) -> None:
    torch.manual_seed(7)
    tokenizer = train_character_tokenizer(
        texts=["abxy"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    model = _tiny_model(vocabulary_size=tokenizer.vocabulary_size)
    reference_model = copy.deepcopy(model)
    reference_parameters_before = tuple(
        parameter.detach().clone()
        for parameter in reference_model.parameters()
    )
    model_parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())
    dataset = build_dpo_preference_dataset(
        preferences=(
            DpoPreferenceRecord(
                task_id="task",
                prompt="a",
                chosen_completion="x",
                rejected_completion="y",
                chosen_score=2.0,
                rejected_score=1.0,
                score_margin=1.0,
                chosen_sample_index=0,
                rejected_sample_index=1,
            ),
        ),
        tokenizer=tokenizer,
    )

    train_model(
        model=model,
        dataset=dataset,
        training_configuration=TrainingConfiguration(
            maximum_steps=1,
            batch_size_sequences=1,
            dataloader=DataLoaderConfiguration(
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=None,
            ),
            checkpoint_interval_steps=1,
            log_interval_steps=1,
        ),
        artifact_directory=tmp_path,
        seed=11,
        evaluation_callback=None,
        objective_runner=DirectPreferenceOptimizationObjectiveRunner(
            reference_model=reference_model,
            beta=0.1,
        ),
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(model_parameters_before, model.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            reference_parameters_before,
            reference_model.parameters(),
            strict=True,
        )
    )


def _candidate(
    task_id: str,
    sample_index: int,
    generated_text: str,
) -> GeneratedCandidateRecord:
    return GeneratedCandidateRecord(
        task_id=task_id,
        prompt="",
        sample_index=sample_index,
        generated_text=generated_text,
        full_text=generated_text,
        token_ids=(),
        generated_token_ids=(),
        prompt_length=0,
        generated_token_count=0,
        stop_reason=GenerationStopReason.MAXIMUM_NEW_TOKENS,
        decoding=GreedyDecodingConfiguration(strategy=DecodingStrategy.GREEDY),
        timing=CandidateTimingRecord(
            prefill_seconds=0.0,
            decode_seconds=0.0,
            total_seconds=0.0,
            tokens_per_second=0.0,
            sequences_per_second=0.0,
        ),
    )


def _score(
    task_id: str,
    sample_index: int,
    generated_text: str,
    score: float,
    perfect: bool = False,
) -> PythonCandidateScoreRecord:
    return PythonCandidateScoreRecord(
        task_id=task_id,
        prompt=f"{task_id}: ",
        sample_index=sample_index,
        generated_text=generated_text,
        score=score,
        parsed=True,
        executed=True,
        passed_checks=int(score),
        total_checks=3,
        perfect=perfect,
        timed_out=False,
        error=None,
        stdout="",
        stderr="",
    )


def _tiny_model(vocabulary_size: int) -> DenseGpt:
    model = DenseGpt(
        model_configuration=DenseGptConfiguration(
            type=ModelType.DENSE_GPT,
            dimension=8,
            layers=1,
            attention_heads=2,
            feed_forward_dimension=16,
            dropout=0.0,
            tie_embeddings=False,
        ),
        vocabulary_size=vocabulary_size,
    )
    model.eval()
    return model
