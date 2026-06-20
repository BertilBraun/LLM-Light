from pathlib import Path

import pytest

from llm_lite.data.datasets import (
    PackedSequence,
    load_packed_sequence_dataset,
    write_packed_sequence_stream,
)
from llm_lite.data.packing import pack_token_sequences


def test_pack_token_sequences_pads_to_context_plus_target() -> None:
    sequences = list(
        pack_token_sequences(
            tokenized_document_stream=[[1, 2, 3]],
            context_length=4,
            pad_token_id=0,
        ),
    )

    assert sequences[0].token_ids == (1, 2, 3, 0, 0)


def test_write_and_load_file_backed_packed_sequences(tmp_path: Path) -> None:
    write_packed_sequence_stream(
        sequences=[
            PackedSequence(token_ids=(1, 2, 3)),
            PackedSequence(token_ids=(4, 5, 6)),
        ],
        artifact_directory=tmp_path,
        row_length=3,
        maximum_shard_tokens=3,
    )

    dataset = load_packed_sequence_dataset(artifact_directory=tmp_path)

    assert len(dataset) == 2
    assert dataset[0].tolist() == [1, 2, 3]
    assert dataset[1].tolist() == [4, 5, 6]
    assert (tmp_path / "index.json").exists()
    assert (tmp_path / "shards" / "shard_000000.bin").exists()
    assert (tmp_path / "shards" / "shard_000001.bin").exists()


def test_file_backed_packed_sequences_reject_out_of_range_index(tmp_path: Path) -> None:
    write_packed_sequence_stream(
        sequences=[PackedSequence(token_ids=(1, 2, 3))],
        artifact_directory=tmp_path,
        row_length=3,
        maximum_shard_tokens=10,
    )
    dataset = load_packed_sequence_dataset(artifact_directory=tmp_path)

    with pytest.raises(IndexError):
        dataset[1]


def test_file_backed_packed_sequences_reject_uint16_overflow(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uint16"):
        write_packed_sequence_stream(
            sequences=[PackedSequence(token_ids=(1, 65536, 3))],
            artifact_directory=tmp_path,
            row_length=3,
            maximum_shard_tokens=10,
        )


def test_file_backed_packed_sequences_reject_empty_stream(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no training sequences"):
        write_packed_sequence_stream(
            sequences=[],
            artifact_directory=tmp_path,
            row_length=3,
            maximum_shard_tokens=10,
        )
