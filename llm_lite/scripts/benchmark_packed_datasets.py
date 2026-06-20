from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_lite.data.datasets import (
    PackedSequence,
    load_iterable_packed_sequence_dataset,
    load_packed_sequence_dataset,
    write_packed_sequence_stream,
)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    seconds: float
    rows: int


def main() -> None:
    arguments = _parse_arguments()
    with TemporaryDirectory() as temporary_directory:
        artifact_directory = Path(temporary_directory)
        _write_dataset(
            artifact_directory=artifact_directory,
            sequence_count=arguments.sequences,
            row_length=arguments.row_length,
            shard_sequences=arguments.shard_sequences,
        )
        results = [
            _benchmark_map_dataset(
                artifact_directory=artifact_directory,
                batch_size=arguments.batch_size,
                passes=arguments.passes,
                num_workers=arguments.num_workers,
            ),
            _benchmark_iterable_dataset(
                artifact_directory=artifact_directory,
                batch_size=arguments.batch_size,
                passes=arguments.passes,
                num_workers=arguments.num_workers,
            ),
            _benchmark_in_memory_dataset(
                artifact_directory=artifact_directory,
                batch_size=arguments.batch_size,
                passes=arguments.passes,
                num_workers=arguments.num_workers,
            ),
        ]
    _print_results(arguments=arguments, results=results)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Benchmark packed dataset access modes.")
    parser.add_argument("--sequences", type=int, default=20_000)
    parser.add_argument("--row-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--shard-sequences", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _write_dataset(
    artifact_directory: Path,
    sequence_count: int,
    row_length: int,
    shard_sequences: int,
) -> None:
    sequences = (
        PackedSequence(
            token_ids=tuple(
                (sequence_index + token_index) % 512 for token_index in range(row_length)
            ),
        )
        for sequence_index in range(sequence_count)
    )
    write_packed_sequence_stream(
        sequences=sequences,
        artifact_directory=artifact_directory,
        row_length=row_length,
        maximum_shard_tokens=row_length * shard_sequences,
    )


def _benchmark_map_dataset(
    artifact_directory: Path,
    batch_size: int,
    passes: int,
    num_workers: int,
) -> BenchmarkResult:
    dataset = load_packed_sequence_dataset(artifact_directory=artifact_directory)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        num_workers=num_workers,
    )
    return _consume_loader(name="map_random", data_loader=data_loader, passes=passes)


def _benchmark_iterable_dataset(
    artifact_directory: Path,
    batch_size: int,
    passes: int,
    num_workers: int,
) -> BenchmarkResult:
    dataset = load_iterable_packed_sequence_dataset(artifact_directory=artifact_directory, seed=0)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    started = perf_counter()
    rows = 0
    for epoch in range(passes):
        dataset.set_epoch(epoch=epoch)
        for batch in data_loader:
            rows += int(batch.shape[0])
    return BenchmarkResult(name="iterable_sharded", seconds=perf_counter() - started, rows=rows)


def _benchmark_in_memory_dataset(
    artifact_directory: Path,
    batch_size: int,
    passes: int,
    num_workers: int,
) -> BenchmarkResult:
    source_dataset = load_packed_sequence_dataset(artifact_directory=artifact_directory)
    memory_tensor = torch.stack([source_dataset[index] for index in range(len(source_dataset))])
    data_loader = DataLoader(
        TensorDataset(memory_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        num_workers=num_workers,
    )
    started = perf_counter()
    rows = 0
    for _ in range(passes):
        for (batch,) in data_loader:
            rows += int(batch.shape[0])
    return BenchmarkResult(name="in_memory_random", seconds=perf_counter() - started, rows=rows)


def _consume_loader(
    name: str,
    data_loader: DataLoader[torch.Tensor],
    passes: int,
) -> BenchmarkResult:
    started = perf_counter()
    rows = 0
    for _ in range(passes):
        for batch in data_loader:
            rows += int(batch.shape[0])
    return BenchmarkResult(name=name, seconds=perf_counter() - started, rows=rows)


def _print_results(arguments: Namespace, results: list[BenchmarkResult]) -> None:
    baseline = next(result for result in results if result.name == "in_memory_random")
    print(f"rows_per_dataset_pass: {arguments.sequences}")
    print(f"passes: {arguments.passes}")
    print(f"row_length: {arguments.row_length}")
    print(f"batch_size: {arguments.batch_size}")
    print(f"shard_sequences: {arguments.shard_sequences}")
    print(f"num_workers: {arguments.num_workers}")
    for result in results:
        print(f"{result.name}_seconds: {result.seconds:.4f}")
        print(f"{result.name}_rows: {result.rows}")
        print(f"{result.name}_vs_memory_ratio: {result.seconds / baseline.seconds:.2f}x")
    map_result = next(result for result in results if result.name == "map_random")
    iterable_result = next(result for result in results if result.name == "iterable_sharded")
    print(f"map_vs_iterable_ratio: {map_result.seconds / iterable_result.seconds:.2f}x")


if __name__ == "__main__":
    main()
