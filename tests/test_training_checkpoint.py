import json
from pathlib import Path

import pytest
from torch import nn
from torch.optim import AdamW

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import (
    LearningRateScheduleType,
    LinearWarmupDecayLearningRateScheduleConfiguration,
    OptimizerConfiguration,
)
from llm_lite.data.datasets import (
    PackedSequence,
    load_packed_sequence_dataset,
    write_packed_sequence_stream,
)
from llm_lite.model.gpt import DenseGpt
from llm_lite.pipeline.artifact import ArtifactManifest, ArtifactStatus
from llm_lite.tokenizer.character import train_character_tokenizer
from llm_lite.training.checkpoint import (
    load_latest_checkpoint,
    retain_recent_checkpoints,
    save_checkpoint,
    save_rank_zero_full_checkpoint_bridge,
)
from llm_lite.training.objectives import CausalLanguageModelingObjectiveRunner
from llm_lite.training.trainer import train_model


def test_training_checkpoint_resume(tmp_path: Path) -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    tokenizer = train_character_tokenizer(
        texts=["hello world\n"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    token_ids = tokenizer.encode(text="hello world\n", add_bos=True, add_eos=True)
    packed_artifact_directory = tmp_path / "packed"
    packed_artifact_directory.mkdir()
    write_packed_sequence_stream(
        sequences=[PackedSequence(token_ids=tuple(token_ids))],
        artifact_directory=packed_artifact_directory,
        row_length=len(token_ids),
        maximum_shard_tokens=1024,
    )
    dataset = load_packed_sequence_dataset(artifact_directory=packed_artifact_directory)
    model = DenseGpt(
        model_configuration=experiment_configuration.model,
        vocabulary_size=tokenizer.vocabulary_size,
    )

    first_result = train_model(
        model=model,
        dataset=dataset,
        training_configuration=experiment_configuration.training,
        artifact_directory=tmp_path,
        seed=experiment_configuration.experiment.seed,
        evaluation_callback=None,
        objective_runner=CausalLanguageModelingObjectiveRunner(auxiliary_loss_weight=0.0),
    )
    second_result = train_model(
        model=model,
        dataset=dataset,
        training_configuration=experiment_configuration.training,
        artifact_directory=tmp_path,
        seed=experiment_configuration.experiment.seed,
        evaluation_callback=None,
        objective_runner=CausalLanguageModelingObjectiveRunner(auxiliary_loss_weight=0.0),
    )

    assert first_result.final_step == experiment_configuration.training.maximum_steps
    assert second_result.resumed_from_step == experiment_configuration.training.maximum_steps
    metric_records = [
        json.loads(metric_line)
        for metric_line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert metric_records[-1]["step"] == experiment_configuration.training.maximum_steps
    assert "learning_rate" in metric_records[-1]
    assert "gradient_norm" in metric_records[-1]
    assert "tokens_per_second" in metric_records[-1]
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))


def test_training_metrics_log_scheduled_learning_rate(tmp_path: Path) -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )
    optimizer_configuration = experiment_configuration.training.optimizer.model_copy(
        update={
            "learning_rate": 0.1,
            "learning_rate_schedule": LinearWarmupDecayLearningRateScheduleConfiguration(
                type=LearningRateScheduleType.LINEAR_WARMUP_DECAY,
                warmup_steps=1,
                minimum_learning_rate_ratio=0.1,
            ),
        },
    )
    training_configuration = experiment_configuration.training.model_copy(
        update={
            "maximum_steps": 4,
            "checkpoint_interval_steps": 10,
            "log_interval_steps": 1,
            "optimizer": OptimizerConfiguration.model_validate(optimizer_configuration),
        },
    )
    tokenizer = train_character_tokenizer(
        texts=["hello world\n"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    token_ids = tokenizer.encode(text="hello world\n", add_bos=True, add_eos=True)
    packed_artifact_directory = tmp_path / "packed"
    packed_artifact_directory.mkdir()
    write_packed_sequence_stream(
        sequences=[PackedSequence(token_ids=tuple(token_ids))],
        artifact_directory=packed_artifact_directory,
        row_length=len(token_ids),
        maximum_shard_tokens=1024,
    )
    dataset = load_packed_sequence_dataset(artifact_directory=packed_artifact_directory)
    model = DenseGpt(
        model_configuration=experiment_configuration.model,
        vocabulary_size=tokenizer.vocabulary_size,
    )

    train_model(
        model=model,
        dataset=dataset,
        training_configuration=training_configuration,
        artifact_directory=tmp_path,
        seed=experiment_configuration.experiment.seed,
        evaluation_callback=None,
        objective_runner=CausalLanguageModelingObjectiveRunner(auxiliary_loss_weight=0.0),
    )

    metric_records = [
        json.loads(metric_line)
        for metric_line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [record["learning_rate"] for record in metric_records] == pytest.approx(
        [
            0.1,
            0.1,
            0.055,
            0.010000000000000002,
        ],
    )


def test_load_latest_checkpoint_returns_none_when_missing(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)

    loaded_step = load_latest_checkpoint(
        checkpoint_directory=tmp_path,
        model=model,
        optimizer=None,
    )

    assert loaded_step is None


def test_load_latest_checkpoint_loads_model_and_returns_step(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=0.1)
    checkpoint_directory = tmp_path / "checkpoints"
    save_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        optimizer=optimizer,
        step=7,
    )
    loaded_model = nn.Linear(1, 1)
    loaded_optimizer = AdamW(loaded_model.parameters(), lr=0.1)

    loaded_step = load_latest_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=loaded_model,
        optimizer=loaded_optimizer,
    )

    assert loaded_step == 7
    assert (checkpoint_directory / "latest.pt").exists()
    assert not (checkpoint_directory / "latest.pt.pending").exists()


def test_save_checkpoint_writes_checkpoint_manifest_and_event(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=0.1)
    artifact_directory = tmp_path / "artifact"
    artifact_directory.mkdir()
    (artifact_directory / "manifest.json").write_text(
        ArtifactManifest(
            stage_name="pretraining",
            fingerprint="sha256:training",
            artifact_version=1,
            status=ArtifactStatus.RUNNING,
            created_at="2026-06-26T00:00:00Z",
            configuration_hash="sha256:configuration",
            contract_version=1,
            parents={},
            files={},
            metrics={},
        ).model_dump_json(),
        encoding="utf-8",
    )

    save_checkpoint(
        checkpoint_directory=artifact_directory / "checkpoints",
        model=model,
        optimizer=optimizer,
        step=7,
    )

    checkpoint_manifest = json.loads(
        (artifact_directory / "checkpoints" / "step_00000007" / "manifest.json").read_text(
            encoding="utf-8"
        ),
    )
    checkpoint_event = json.loads(
        (artifact_directory / "events" / "checkpoint_00000007.json").read_text(
            encoding="utf-8",
        ),
    )

    assert checkpoint_manifest["producing_artifact_fingerprint"] == "sha256:training"
    assert checkpoint_manifest["checkpoint_kind"] == "full"
    assert checkpoint_manifest["completion_status"] == "complete"
    assert checkpoint_event["checkpoint_step"] == 7
    assert checkpoint_event["checkpoint_manifest_path"] == (
        "checkpoints/step_00000007/manifest.json"
    )


def test_retain_recent_checkpoints_keeps_latest_plus_interval_count(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=0.1)
    checkpoint_directory = tmp_path / "artifact" / "checkpoints"
    checkpoint_directory.parent.mkdir()

    for step in range(1, 5):
        save_checkpoint(
            checkpoint_directory=checkpoint_directory,
            model=model,
            optimizer=optimizer,
            step=step,
        )

    retain_recent_checkpoints(checkpoint_directory=checkpoint_directory, max_checkpoints=2)

    assert not (checkpoint_directory / "step_00000001").exists()
    assert not (checkpoint_directory / "step_00000001.pt").exists()
    assert not (checkpoint_directory.parent / "events" / "checkpoint_00000001.json").exists()
    assert (checkpoint_directory / "step_00000002").exists()
    assert (checkpoint_directory / "step_00000002.pt").exists()
    assert (checkpoint_directory / "step_00000003").exists()
    assert (checkpoint_directory / "step_00000003.pt").exists()
    assert (checkpoint_directory / "step_00000004").exists()
    assert (checkpoint_directory / "step_00000004.pt").exists()
    assert (checkpoint_directory / "latest.pt").exists()


def test_rank_zero_full_checkpoint_bridge_writes_latest_atomically(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=0.1)
    checkpoint_directory = tmp_path / "checkpoints"

    bridge_path = save_rank_zero_full_checkpoint_bridge(
        checkpoint_directory=checkpoint_directory,
        model=model,
        optimizer=optimizer,
        step=7,
    )

    assert bridge_path == checkpoint_directory / "latest.pt"
    assert bridge_path.exists()
    assert not (checkpoint_directory / "latest.pt.pending").exists()
