from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from llm_lite.config.models import DataLoaderConfiguration
from llm_lite.training.objectives import TrainingBatch


@runtime_checkable
class EpochAwareDataset(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@runtime_checkable
class EpochAwareSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True)
class DistributedDataAssignment:
    rank: int
    world_size: int


@dataclass
class InfiniteDataIterator:
    data_loader: DataLoader[TrainingBatch]
    dataset: Dataset[TrainingBatch] | IterableDataset[TrainingBatch]
    sampler: EpochAwareSampler | None
    epoch: int
    batches_seen: int = 0

    def __post_init__(self) -> None:
        self._set_epoch()
        self._iterator = iter(self.data_loader)

    def next_batch(self) -> TrainingBatch:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._set_epoch()
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
        self.batches_seen += 1
        return batch

    @property
    def batches_per_epoch(self) -> int | None:
        try:
            return len(self.data_loader)
        except TypeError:
            return None

    @property
    def epoch_progress(self) -> float | None:
        batches_per_epoch = self.batches_per_epoch
        if batches_per_epoch is None or batches_per_epoch == 0:
            return None
        return self.batches_seen / batches_per_epoch

    def _set_epoch(self) -> None:
        match self.dataset:
            case EpochAwareDataset():
                self.dataset.set_epoch(self.epoch)
            case _:
                pass
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)


def create_training_data_iterator(
    dataset: Dataset[TrainingBatch] | IterableDataset[TrainingBatch],
    batch_size_sequences: int,
    dataloader_configuration: DataLoaderConfiguration,
    seed: int,
    distributed_data_assignment: DistributedDataAssignment | None = None,
) -> InfiniteDataIterator:
    is_iterable_dataset = isinstance(dataset, IterableDataset)
    sampler = _distributed_sampler(
        dataset=dataset,
        seed=seed,
        distributed_data_assignment=distributed_data_assignment,
    )
    if dataloader_configuration.num_workers > 0:
        if dataloader_configuration.prefetch_factor is None:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size_sequences,
                shuffle=not is_iterable_dataset and sampler is None,
                sampler=sampler,
                generator=None if is_iterable_dataset else torch.Generator().manual_seed(seed),
                num_workers=dataloader_configuration.num_workers,
                pin_memory=dataloader_configuration.pin_memory,
                persistent_workers=dataloader_configuration.persistent_workers,
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size_sequences,
                shuffle=not is_iterable_dataset and sampler is None,
                sampler=sampler,
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
            shuffle=not is_iterable_dataset and sampler is None,
            sampler=sampler,
            generator=None if is_iterable_dataset else torch.Generator().manual_seed(seed),
            num_workers=dataloader_configuration.num_workers,
            pin_memory=dataloader_configuration.pin_memory,
        )
    return InfiniteDataIterator(data_loader=data_loader, dataset=dataset, sampler=sampler, epoch=0)


def _distributed_sampler(
    dataset: Dataset[torch.Tensor] | IterableDataset[torch.Tensor],
    seed: int,
    distributed_data_assignment: DistributedDataAssignment | None,
) -> DistributedSampler[TrainingBatch] | None:
    if distributed_data_assignment is None:
        return None
    if isinstance(dataset, IterableDataset):
        return None
    return DistributedSampler(
        dataset=dataset,
        num_replicas=distributed_data_assignment.world_size,
        rank=distributed_data_assignment.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
