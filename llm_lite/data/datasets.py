from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PackedSequence:
    token_ids: tuple[int, ...]


class PackedShardIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    shard_index: int = Field(ge=0)
    path: str
    sequence_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    first_sequence_index: int = Field(ge=0)


class PackedDatasetIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    format_version: int
    dtype: str
    row_length: int = Field(gt=1)
    total_sequences: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    shards: tuple[PackedShardIndex, ...]


class PackedSequenceDataset(Dataset[torch.Tensor]):
    def __init__(self, artifact_directory: Path, index: PackedDatasetIndex) -> None:
        self.artifact_directory = artifact_directory
        self.index = index
        self.shard_start_indices = tuple(shard.first_sequence_index for shard in self.index.shards)
        self.mapped_shards: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.index.total_sequences

    def __getitem__(self, sequence_index: int) -> torch.Tensor:
        if sequence_index < 0 or sequence_index >= self.index.total_sequences:
            raise IndexError("Packed sequence index is out of range.")
        shard = self._shard_for_sequence(sequence_index=sequence_index)
        shard_tensor = self._mapped_shard(shard=shard)
        local_sequence_index = sequence_index - shard.first_sequence_index
        start_index = local_sequence_index * self.index.row_length
        end_index = start_index + self.index.row_length
        return shard_tensor[start_index:end_index].to(dtype=torch.long)

    def _shard_for_sequence(self, sequence_index: int) -> PackedShardIndex:
        shard_position = bisect_right(self.shard_start_indices, sequence_index) - 1
        return self.index.shards[shard_position]

    def _mapped_shard(self, shard: PackedShardIndex) -> torch.Tensor:
        mapped_shard = self.mapped_shards.get(shard.shard_index)
        if mapped_shard is not None:
            return mapped_shard
        mapped_shard = torch.from_file(
            str(self.artifact_directory / shard.path),
            shared=False,
            size=shard.token_count,
            dtype=torch.uint16,
        )
        self.mapped_shards[shard.shard_index] = mapped_shard
        return mapped_shard


class PackedShardWriter:
    def __init__(
        self,
        artifact_directory: Path,
        row_length: int,
        maximum_shard_tokens: int,
    ) -> None:
        self.artifact_directory = artifact_directory
        self.row_length = row_length
        self.maximum_shard_tokens = maximum_shard_tokens
        if self.maximum_shard_tokens < self.row_length:
            raise ValueError("Maximum shard tokens must fit at least one packed row.")
        self.shard_directory = artifact_directory / "shards"
        self.shard_directory.mkdir(parents=True, exist_ok=True)
        self.current_shard_file: BinaryIO | None = None
        self.current_shard_index = 0
        self.current_shard_sequences = 0
        self.total_sequences = 0
        self.shards: list[PackedShardIndex] = []

    def append(self, sequence: PackedSequence) -> None:
        self._validate_sequence(sequence=sequence)
        if self._should_open_next_shard():
            self._open_next_shard()
        assert self.current_shard_file is not None
        token_tensor = torch.tensor(sequence.token_ids, dtype=torch.uint16)
        token_tensor.numpy().tofile(self.current_shard_file)
        self.current_shard_sequences += 1
        self.total_sequences += 1

    def close(self) -> PackedDatasetIndex:
        self._close_current_shard()
        index = PackedDatasetIndex(
            format_version=1,
            dtype="uint16",
            row_length=self.row_length,
            total_sequences=self.total_sequences,
            total_tokens=self.total_sequences * self.row_length,
            shards=tuple(self.shards),
        )
        (self.artifact_directory / "index.json").write_text(
            index.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return index

    def _validate_sequence(self, sequence: PackedSequence) -> None:
        if len(sequence.token_ids) != self.row_length:
            raise ValueError("Packed dataset rows must have a stable length.")
        if any(token_id < 0 or token_id > 65535 for token_id in sequence.token_ids):
            raise ValueError("Packed token ids must fit in uint16.")

    def _should_open_next_shard(self) -> bool:
        if self.current_shard_file is None:
            return True
        next_shard_tokens = (self.current_shard_sequences + 1) * self.row_length
        return next_shard_tokens > self.maximum_shard_tokens

    def _open_next_shard(self) -> None:
        self._close_current_shard()
        shard_path = self.shard_directory / f"shard_{self.current_shard_index:06d}.bin"
        self.current_shard_file = shard_path.open("wb")

    def _close_current_shard(self) -> None:
        if self.current_shard_file is None:
            return
        self.current_shard_file.close()
        shard_path = f"shards/shard_{self.current_shard_index:06d}.bin"
        self.shards.append(
            PackedShardIndex(
                shard_index=self.current_shard_index,
                path=shard_path,
                sequence_count=self.current_shard_sequences,
                token_count=self.current_shard_sequences * self.row_length,
                first_sequence_index=self.total_sequences - self.current_shard_sequences,
            ),
        )
        self.current_shard_file = None
        self.current_shard_index += 1
        self.current_shard_sequences = 0


def write_packed_sequence_stream(
    sequences: Iterable[PackedSequence],
    artifact_directory: Path,
    row_length: int,
    maximum_shard_tokens: int,
) -> PackedDatasetIndex:
    writer = PackedShardWriter(
        artifact_directory=artifact_directory,
        row_length=row_length,
        maximum_shard_tokens=maximum_shard_tokens,
    )
    for sequence in sequences:
        writer.append(sequence=sequence)
    if writer.total_sequences == 0:
        raise ValueError("Packing produced no training sequences.")
    return writer.close()


def load_packed_sequence_dataset(artifact_directory: Path) -> PackedSequenceDataset:
    index = PackedDatasetIndex.model_validate_json(
        (artifact_directory / "index.json").read_text(encoding="utf-8"),
    )
    if index.dtype != "uint16":
        raise ValueError("Only uint16 packed datasets are supported.")
    return PackedSequenceDataset(artifact_directory=artifact_directory, index=index)
