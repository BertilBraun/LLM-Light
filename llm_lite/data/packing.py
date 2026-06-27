import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from tqdm.auto import tqdm

from llm_lite.config.models import FillInMiddleConfiguration, TokenizerConfiguration
from llm_lite.data.datasets import (
    PackedDatasetIndex,
    PackedSequence,
    PackedShardIndex,
    PackedShardWriter,
)
from llm_lite.data.text_shards import (
    TextShardReference,
    iter_text_shard_reference_documents,
    load_text_shard_corpus_manifest,
    text_shard_references,
)
from llm_lite.pipeline.progress import progress_bar
from llm_lite.tokenizer.loading import TextTokenizer, load_tokenizer


def pack_token_sequences(
    tokenized_document_stream: Iterable[list[int]],
    context_length: int,
    pad_token_id: int,
) -> Iterator[PackedSequence]:
    for token_ids in tokenized_document_stream:
        if len(token_ids) < 2:
            continue
        start_index = 0
        while start_index < len(token_ids) - 1:
            sequence = token_ids[start_index : start_index + context_length + 1]
            if len(sequence) < context_length + 1:
                sequence = sequence + [pad_token_id] * (context_length + 1 - len(sequence))
            yield PackedSequence(token_ids=tuple(sequence))
            start_index += context_length


def pack_document_token_sequences(
    tokenized_document_stream: Iterable[list[int]],
    context_length: int,
    pad_token_id: int,
) -> Iterator[PackedSequence]:
    row_length = context_length + 1
    buffer: list[int] = []
    for token_ids in tokenized_document_stream:
        if len(token_ids) < 2:
            continue
        buffer.extend(token_ids)
        while len(buffer) >= row_length:
            row = buffer[:row_length]
            del buffer[:context_length]
            yield PackedSequence(token_ids=tuple(row))
    if len(buffer) >= 2:
        yield PackedSequence(
            token_ids=tuple(buffer + [pad_token_id] * (row_length - len(buffer))),
        )


@dataclass(frozen=True)
class ParallelPackingResult:
    index: PackedDatasetIndex
    worker_count: int
    input_documents: int
    non_pad_tokens: int
    pad_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.non_pad_tokens + self.pad_tokens

    @property
    def pad_fraction(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.pad_tokens / self.total_tokens

    @property
    def average_sequence_fill(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.non_pad_tokens / self.total_tokens


@dataclass(frozen=True)
class PackingWorkerInput:
    worker_index: int
    shard_references: tuple[TextShardReference, ...]
    progress_total_documents: int | None
    progress_document_multiplier: int


@dataclass(frozen=True)
class PackingWorkerResult:
    worker_index: int
    sequence_count: int
    input_documents: int
    non_pad_tokens: int
    pad_tokens: int
    shards: tuple[PackedShardIndex, ...]


def pack_text_shards(
    input_artifact_directory: Path,
    output_artifact_directory: Path,
    tokenizer_directory: Path,
    tokenizer_configuration: TokenizerConfiguration,
    split: str | None,
    context_length: int,
    pad_token_id: int,
    add_bos: bool,
    add_eos: bool,
    pack_documents: bool,
    fill_in_middle_configuration: FillInMiddleConfiguration,
    maximum_shard_tokens: int,
    workers: int,
) -> ParallelPackingResult:
    shard_references = text_shard_references(
        artifact_directory=input_artifact_directory,
        split=split,
    )
    input_manifest = load_text_shard_corpus_manifest(
        artifact_directory=input_artifact_directory,
    )
    input_documents = sum(
        split_manifest.documents
        for split_manifest in input_manifest.splits
        if split is None or split_manifest.split == split
    )
    effective_workers = _effective_worker_count(
        requested_workers=workers,
        shard_count=len(shard_references),
    )
    worker_inputs = _contiguous_worker_inputs(
        shard_references=shard_references,
        worker_count=effective_workers,
        progress_total_documents=input_documents,
    )
    worker_arguments = [
        (
            worker_input,
            output_artifact_directory,
            tokenizer_directory,
            tokenizer_configuration,
            context_length,
            pad_token_id,
            add_bos,
            add_eos,
            pack_documents,
            fill_in_middle_configuration,
            maximum_shard_tokens,
        )
        for worker_input in worker_inputs
    ]
    if effective_workers == 1:
        worker_result = _packing_worker(*worker_arguments[0])
        worker_results = [worker_result]
    else:
        multiprocessing_context = get_context("spawn")
        with multiprocessing_context.Pool(processes=effective_workers) as pool:
            worker_results = [
                worker_result
                for worker_result in pool.imap_unordered(
                    _packing_worker_from_arguments,
                    worker_arguments,
                )
            ]
    index = _merge_packing_worker_results(
        worker_results=tuple(worker_results),
        output_artifact_directory=output_artifact_directory,
        row_length=context_length + 1,
    )
    if index.total_sequences == 0:
        raise ValueError("Packing produced no training sequences.")
    return ParallelPackingResult(
        index=index,
        worker_count=effective_workers,
        input_documents=sum(worker_result.input_documents for worker_result in worker_results),
        non_pad_tokens=sum(worker_result.non_pad_tokens for worker_result in worker_results),
        pad_tokens=sum(worker_result.pad_tokens for worker_result in worker_results),
    )


def _packing_worker(
    worker_input: PackingWorkerInput,
    output_artifact_directory: Path,
    tokenizer_directory: Path,
    tokenizer_configuration: TokenizerConfiguration,
    context_length: int,
    pad_token_id: int,
    add_bos: bool,
    add_eos: bool,
    pack_documents: bool,
    fill_in_middle_configuration: FillInMiddleConfiguration,
    maximum_shard_tokens: int,
) -> PackingWorkerResult:
    tokenizer = load_tokenizer(
        directory=tokenizer_directory,
        tokenizer_configuration=tokenizer_configuration,
    )
    writer = PackedShardWriter(
        artifact_directory=output_artifact_directory,
        row_length=context_length + 1,
        maximum_shard_tokens=maximum_shard_tokens,
        shard_name_prefix=f"part_{worker_input.worker_index:06d}_",
    )
    input_documents = _write_packed_worker_shards(
        worker_input=worker_input,
        writer=writer,
        tokenizer=tokenizer,
        context_length=context_length,
        pad_token_id=pad_token_id,
        add_bos=add_bos,
        add_eos=add_eos,
        pack_documents=pack_documents,
        fill_in_middle_configuration=fill_in_middle_configuration,
    )
    index = writer.close()
    return PackingWorkerResult(
        worker_index=worker_input.worker_index,
        sequence_count=index.total_sequences,
        input_documents=input_documents.input_documents,
        non_pad_tokens=input_documents.non_pad_tokens,
        pad_tokens=input_documents.pad_tokens,
        shards=index.shards,
    )


@dataclass(frozen=True)
class PackingProgress:
    progress_bar_instance: tqdm
    total_documents: int
    document_multiplier: int


@dataclass(frozen=True)
class PackingCounters:
    input_documents: int = 0
    non_pad_tokens: int = 0
    pad_tokens: int = 0

    def add(self, other: "PackingCounters") -> "PackingCounters":
        return PackingCounters(
            input_documents=self.input_documents + other.input_documents,
            non_pad_tokens=self.non_pad_tokens + other.non_pad_tokens,
            pad_tokens=self.pad_tokens + other.pad_tokens,
        )


def _write_packed_worker_shards(
    worker_input: PackingWorkerInput,
    writer: PackedShardWriter,
    tokenizer: TextTokenizer,
    context_length: int,
    pad_token_id: int,
    add_bos: bool,
    add_eos: bool,
    pack_documents: bool,
    fill_in_middle_configuration: FillInMiddleConfiguration,
) -> PackingCounters:
    if worker_input.progress_total_documents is None:
        return _write_packed_worker_documents(
            worker_input=worker_input,
            writer=writer,
            tokenizer=tokenizer,
            context_length=context_length,
            pad_token_id=pad_token_id,
            add_bos=add_bos,
            add_eos=add_eos,
            pack_documents=pack_documents,
            fill_in_middle_configuration=fill_in_middle_configuration,
            progress=None,
        )
    with progress_bar(
        description=f"pack/{worker_input.progress_document_multiplier} workers",
        total=worker_input.progress_total_documents,
        unit="doc",
    ) as progress_bar_instance:
        return _write_packed_worker_documents(
            worker_input=worker_input,
            writer=writer,
            tokenizer=tokenizer,
            context_length=context_length,
            pad_token_id=pad_token_id,
            add_bos=add_bos,
            add_eos=add_eos,
            pack_documents=pack_documents,
            fill_in_middle_configuration=fill_in_middle_configuration,
            progress=PackingProgress(
                progress_bar_instance=progress_bar_instance,
                total_documents=worker_input.progress_total_documents,
                document_multiplier=worker_input.progress_document_multiplier,
            ),
        )


def _write_packed_worker_documents(
    worker_input: PackingWorkerInput,
    writer: PackedShardWriter,
    tokenizer: TextTokenizer,
    context_length: int,
    pad_token_id: int,
    add_bos: bool,
    add_eos: bool,
    pack_documents: bool,
    fill_in_middle_configuration: FillInMiddleConfiguration,
    progress: PackingProgress | None,
) -> PackingCounters:
    if pack_documents:
        return _write_cross_document_packed_worker_documents(
            worker_input=worker_input,
            writer=writer,
            tokenizer=tokenizer,
            context_length=context_length,
            pad_token_id=pad_token_id,
            add_bos=add_bos,
            add_eos=add_eos,
            fill_in_middle_configuration=fill_in_middle_configuration,
            progress=progress,
        )
    input_documents = 0
    non_pad_tokens = 0
    pad_tokens = 0
    for shard_reference in worker_input.shard_references:
        for document in iter_text_shard_reference_documents(shard_reference=shard_reference):
            input_documents += 1
            document_text = fill_in_middle_text(
                text=document.text,
                document_id=document.document_id,
                configuration=fill_in_middle_configuration,
            )
            token_ids = tokenizer.encode(text=document_text, add_bos=add_bos, add_eos=add_eos)
            for sequence in pack_token_sequences(
                tokenized_document_stream=(token_ids,),
                context_length=context_length,
                pad_token_id=pad_token_id,
            ):
                writer.append(sequence=sequence)
                pad_count = sequence.token_ids.count(pad_token_id)
                non_pad_tokens += len(sequence.token_ids) - pad_count
                pad_tokens += pad_count
            if progress is not None:
                _update_scaled_progress(progress=progress)
    return PackingCounters(
        input_documents=input_documents,
        non_pad_tokens=non_pad_tokens,
        pad_tokens=pad_tokens,
    )


def _write_cross_document_packed_worker_documents(
    worker_input: PackingWorkerInput,
    writer: PackedShardWriter,
    tokenizer: TextTokenizer,
    context_length: int,
    pad_token_id: int,
    add_bos: bool,
    add_eos: bool,
    fill_in_middle_configuration: FillInMiddleConfiguration,
    progress: PackingProgress | None,
) -> PackingCounters:
    row_length = context_length + 1
    input_documents = 0
    non_pad_tokens = 0
    pad_tokens = 0

    def tokenized_documents() -> Iterator[list[int]]:
        nonlocal input_documents
        for shard_reference in worker_input.shard_references:
            for document in iter_text_shard_reference_documents(shard_reference=shard_reference):
                input_documents += 1
                document_text = fill_in_middle_text(
                    text=document.text,
                    document_id=document.document_id,
                    configuration=fill_in_middle_configuration,
                )
                token_ids = tokenizer.encode(
                    text=document_text,
                    add_bos=add_bos,
                    add_eos=add_eos,
                )
                if progress is not None:
                    _update_scaled_progress(progress=progress)
                yield token_ids

    for sequence in pack_document_token_sequences(
        tokenized_document_stream=tokenized_documents(),
        context_length=context_length,
        pad_token_id=pad_token_id,
    ):
        writer.append(sequence=sequence)
        pad_count = sequence.token_ids.count(pad_token_id)
        non_pad_tokens += row_length - pad_count
        pad_tokens += pad_count

    return PackingCounters(
        input_documents=input_documents,
        non_pad_tokens=non_pad_tokens,
        pad_tokens=pad_tokens,
    )


def _packing_worker_from_arguments(
    arguments: tuple[
        PackingWorkerInput,
        Path,
        Path,
        TokenizerConfiguration,
        int,
        int,
        bool,
        bool,
        bool,
        FillInMiddleConfiguration,
        int,
    ],
) -> PackingWorkerResult:
    return _packing_worker(*arguments)


def fill_in_middle_text(
    text: str,
    document_id: str,
    configuration: FillInMiddleConfiguration,
) -> str:
    if not configuration.enabled:
        return text
    if not _selected_for_fill_in_middle(
        document_id=document_id,
        probability=configuration.probability,
    ):
        return text
    minimum_total_characters = configuration.minimum_segment_characters * 3
    if len(text) < minimum_total_characters:
        return text
    prefix_end, middle_end = _fill_in_middle_boundaries(
        text=text,
        document_id=document_id,
        minimum_segment_characters=configuration.minimum_segment_characters,
    )
    prefix_text = text[:prefix_end]
    middle_text = text[prefix_end:middle_end]
    suffix_text = text[middle_end:]
    return (
        f"{configuration.prefix_marker}{prefix_text}"
        f"{configuration.suffix_marker}{suffix_text}"
        f"{configuration.middle_marker}{middle_text}"
    )


def _selected_for_fill_in_middle(document_id: str, probability: float) -> bool:
    if probability >= 1.0:
        return True
    selection_value = _normalized_hash_value(text=f"{document_id}:fill_in_middle")
    return selection_value < probability


def _fill_in_middle_boundaries(
    text: str,
    document_id: str,
    minimum_segment_characters: int,
) -> tuple[int, int]:
    first_available_span = len(text) - (minimum_segment_characters * 3)
    first_offset = _hash_modulo(
        text=f"{document_id}:fill_in_middle:first",
        divisor=first_available_span + 1,
    )
    prefix_end = minimum_segment_characters + first_offset
    second_minimum = prefix_end + minimum_segment_characters
    second_available_span = len(text) - second_minimum - minimum_segment_characters
    second_offset = _hash_modulo(
        text=f"{document_id}:fill_in_middle:second",
        divisor=second_available_span + 1,
    )
    middle_end = second_minimum + second_offset
    return prefix_end, middle_end


def _normalized_hash_value(text: str) -> float:
    hash_integer = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return hash_integer / float(16**16 - 1)


def _hash_modulo(text: str, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("Hash divisor must be positive.")
    hash_integer = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return hash_integer % divisor


def _merge_packing_worker_results(
    worker_results: tuple[PackingWorkerResult, ...],
    output_artifact_directory: Path,
    row_length: int,
) -> PackedDatasetIndex:
    total_sequences = 0
    shards: list[PackedShardIndex] = []
    for worker_result in sorted(worker_results, key=lambda result: result.worker_index):
        for worker_shard in worker_result.shards:
            shards.append(
                PackedShardIndex(
                    shard_index=len(shards),
                    path=worker_shard.path,
                    sequence_count=worker_shard.sequence_count,
                    token_count=worker_shard.token_count,
                    first_sequence_index=total_sequences,
                ),
            )
            total_sequences += worker_shard.sequence_count
    index = PackedDatasetIndex(
        format_version=1,
        dtype="uint16",
        row_length=row_length,
        total_sequences=total_sequences,
        total_tokens=total_sequences * row_length,
        shards=tuple(shards),
    )
    (output_artifact_directory / "index.json").write_text(
        index.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return index


def _effective_worker_count(requested_workers: int, shard_count: int) -> int:
    if shard_count == 0:
        return 1
    return min(requested_workers, shard_count)


def _contiguous_worker_inputs(
    shard_references: tuple[TextShardReference, ...],
    worker_count: int,
    progress_total_documents: int,
) -> tuple[PackingWorkerInput, ...]:
    worker_inputs: list[PackingWorkerInput] = []
    for worker_index in range(worker_count):
        start_index = worker_index * len(shard_references) // worker_count
        end_index = (worker_index + 1) * len(shard_references) // worker_count
        worker_inputs.append(
            PackingWorkerInput(
                worker_index=worker_index,
                shard_references=shard_references[start_index:end_index],
                progress_total_documents=(progress_total_documents if worker_index == 0 else None),
                progress_document_multiplier=worker_count,
            ),
        )
    return tuple(worker_inputs)


def _update_scaled_progress(progress: PackingProgress) -> None:
    remaining_documents = progress.total_documents - progress.progress_bar_instance.n
    if remaining_documents <= 0:
        return
    progress.progress_bar_instance.update(
        min(progress.document_multiplier, remaining_documents),
    )
