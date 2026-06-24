from pathlib import Path

import pytest

from llm_lite.data.datasets import (
    PackedSequence,
    load_iterable_packed_sequence_dataset,
    load_packed_sequence_dataset,
    write_packed_sequence_stream,
)
from llm_lite.data.packing import pack_document_token_sequences, pack_token_sequences


def test_pack_token_sequences_pads_to_context_plus_target() -> None:
    sequences = list(
        pack_token_sequences(
            tokenized_document_stream=[[1, 2, 3]],
            context_length=4,
            pad_token_id=0,
        ),
    )

    assert sequences[0].token_ids == (1, 2, 3, 0, 0)


def test_pack_token_sequences_chunks_long_documents_with_one_token_overlap() -> None:
    sequences = list(
        pack_token_sequences(
            tokenized_document_stream=[[1, 2, 3, 4, 5, 6]],
            context_length=3,
            pad_token_id=0,
        ),
    )

    assert [sequence.token_ids for sequence in sequences] == [
        (1, 2, 3, 4),
        (4, 5, 6, 0),
    ]


def test_pack_document_token_sequences_concatenates_short_documents() -> None:
    sequences = list(
        pack_document_token_sequences(
            tokenized_document_stream=[
                [1, 2, 3],
                [4, 5, 6],
                [7, 8],
            ],
            context_length=4,
            pad_token_id=0,
        ),
    )

    assert [sequence.token_ids for sequence in sequences] == [
        (1, 2, 3, 4, 5),
        (5, 6, 7, 8, 0),
    ]


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


def test_file_backed_packed_sequences_stream_all_rows(tmp_path: Path) -> None:
    write_packed_sequence_stream(
        sequences=[
            PackedSequence(token_ids=(1, 2, 3)),
            PackedSequence(token_ids=(4, 5, 6)),
            PackedSequence(token_ids=(7, 8, 9)),
        ],
        artifact_directory=tmp_path,
        row_length=3,
        maximum_shard_tokens=9,
    )
    dataset = load_packed_sequence_dataset(artifact_directory=tmp_path)

    rows = [dataset[index].tolist() for index in range(len(dataset))]

    assert sorted(rows) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_map_packed_sequences_reject_out_of_range_index(tmp_path: Path) -> None:
    write_packed_sequence_stream(
        sequences=[PackedSequence(token_ids=(1, 2, 3))],
        artifact_directory=tmp_path,
        row_length=3,
        maximum_shard_tokens=10,
    )
    dataset = load_packed_sequence_dataset(artifact_directory=tmp_path)

    with pytest.raises(IndexError):
        dataset[1]


def test_map_packed_sequences_read_shuffled_single_shard_batch(tmp_path: Path) -> None:
    write_packed_sequence_stream(
        sequences=[
            PackedSequence(token_ids=(1, 2, 3)),
            PackedSequence(token_ids=(4, 5, 6)),
            PackedSequence(token_ids=(7, 8, 9)),
        ],
        artifact_directory=tmp_path,
        row_length=3,
        maximum_shard_tokens=9,
    )
    dataset = load_packed_sequence_dataset(artifact_directory=tmp_path)

    batch = dataset.__getitems__([2, 0, 1])

    assert [row.tolist() for row in batch] == [[7, 8, 9], [1, 2, 3], [4, 5, 6]]


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


def test_iterable_dataset_partitions_shards_by_worker(tmp_path: Path) -> None:
    write_packed_sequence_stream(
        sequences=[
            PackedSequence(token_ids=(0, 0, 0)),
            PackedSequence(token_ids=(1, 1, 1)),
            PackedSequence(token_ids=(2, 2, 2)),
            PackedSequence(token_ids=(3, 3, 3)),
            PackedSequence(token_ids=(4, 4, 4)),
            PackedSequence(token_ids=(5, 5, 5)),
        ],
        artifact_directory=tmp_path,
        row_length=3,
        maximum_shard_tokens=6,
    )
    dataset = load_iterable_packed_sequence_dataset(artifact_directory=tmp_path, seed=0)

    worker_0_shards = dataset.shard_positions_for_worker(worker_id=0, worker_count=2)
    worker_1_shards = dataset.shard_positions_for_worker(worker_id=1, worker_count=2)

    assert worker_0_shards == (0, 2)
    assert worker_1_shards == (1,)
    assert set(worker_0_shards).isdisjoint(worker_1_shards)


def test_iterable_dataset_reshuffles_between_epochs(tmp_path: Path) -> None:
    write_packed_sequence_stream(
        sequences=[
            PackedSequence(token_ids=(0, 0, 0)),
            PackedSequence(token_ids=(1, 1, 1)),
            PackedSequence(token_ids=(2, 2, 2)),
            PackedSequence(token_ids=(3, 3, 3)),
            PackedSequence(token_ids=(4, 4, 4)),
            PackedSequence(token_ids=(5, 5, 5)),
        ],
        artifact_directory=tmp_path,
        row_length=3,
        maximum_shard_tokens=6,
    )
    dataset = load_iterable_packed_sequence_dataset(artifact_directory=tmp_path, seed=0)

    first_epoch_rows = [row.tolist() for row in dataset]
    second_epoch_rows = [row.tolist() for row in dataset]

    assert sorted(first_epoch_rows) == sorted(second_epoch_rows)
    assert first_epoch_rows != second_epoch_rows
