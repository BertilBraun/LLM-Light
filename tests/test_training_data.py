import torch
from torch.utils.data import TensorDataset

from llm_lite.config.models import DataLoaderConfiguration
from llm_lite.training.data import create_training_data_iterator


def test_infinite_data_iterator_wraps_finite_dataset() -> None:
    dataset = TensorDataset(torch.tensor([[1], [2]], dtype=torch.long))
    iterator = create_training_data_iterator(
        dataset=dataset,
        batch_size_sequences=1,
        dataloader_configuration=DataLoaderConfiguration(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        ),
        seed=0,
    )

    batches = [iterator.next_batch()[0].item() for _ in range(3)]

    assert sorted(batches[:2]) == [1, 2]
    assert batches[2] in {1, 2}
    assert iterator.epoch == 1
