import json
from pathlib import Path

from torch import nn
from torch.optim import AdamW

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.data.datasets import (
    PackedSequence,
    load_packed_sequence_dataset,
    write_packed_sequence_stream,
)
from llm_lite.model.gpt import DenseGpt
from llm_lite.tokenizer.character import train_character_tokenizer
from llm_lite.training.checkpoint import load_latest_checkpoint, save_checkpoint
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
    )
    second_result = train_model(
        model=model,
        dataset=dataset,
        training_configuration=experiment_configuration.training,
        artifact_directory=tmp_path,
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
