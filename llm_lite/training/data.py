from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from llm_lite.config.models import DataLoaderConfiguration


@runtime_checkable
class EpochAwareDataset(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass
class InfiniteDataIterator:
    data_loader: DataLoader[torch.Tensor]
    dataset: Dataset[torch.Tensor] | IterableDataset[torch.Tensor]
    epoch: int

    def __post_init__(self) -> None:
        self._set_dataset_epoch()
        self._iterator = iter(self.data_loader)

    def next_batch(self) -> torch.Tensor:
        try:
            return next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._set_dataset_epoch()
            self._iterator = iter(self.data_loader)
            return next(self._iterator)

    def _set_dataset_epoch(self) -> None:
        match self.dataset:
            case EpochAwareDataset():
                self.dataset.set_epoch(self.epoch)
            case _:
                return


def create_training_data_iterator(
    dataset: Dataset[torch.Tensor] | IterableDataset[torch.Tensor],
    batch_size_sequences: int,
    dataloader_configuration: DataLoaderConfiguration,
    seed: int,
) -> InfiniteDataIterator:
    is_iterable_dataset = isinstance(dataset, IterableDataset)
    if dataloader_configuration.num_workers > 0:
        if dataloader_configuration.prefetch_factor is None:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size_sequences,
                shuffle=not is_iterable_dataset,
                generator=None if is_iterable_dataset else torch.Generator().manual_seed(seed),
                num_workers=dataloader_configuration.num_workers,
                pin_memory=dataloader_configuration.pin_memory,
                persistent_workers=dataloader_configuration.persistent_workers,
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size_sequences,
                shuffle=not is_iterable_dataset,
                generator=None if is_iterable_dataset else torch.Generator().manual_seed(seed),
                num_workers=dataloader_configuration.num_workers,
                pin_memory=dataloader_configuration.pin_memory,
                persistent_workers=dataloader_configuration.persistent_workers,
                prefetch_factor=dataloader_configuration.prefetch_factor,
            )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size_sequences,
            shuffle=not is_iterable_dataset,
            generator=None if is_iterable_dataset else torch.Generator().manual_seed(seed),
            num_workers=dataloader_configuration.num_workers,
            pin_memory=dataloader_configuration.pin_memory,
        )
    return InfiniteDataIterator(data_loader=data_loader, dataset=dataset, epoch=0)
