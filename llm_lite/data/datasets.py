from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PackedSequence:
    token_ids: tuple[int, ...]


class PackedSequenceDataset(Dataset[torch.Tensor]):
    def __init__(self, sequences: list[PackedSequence]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, sequence_index: int) -> torch.Tensor:
        return torch.tensor(self.sequences[sequence_index].token_ids, dtype=torch.long)
