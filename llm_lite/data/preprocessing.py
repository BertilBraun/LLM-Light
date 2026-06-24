import ast
import hashlib
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from multiprocessing import get_context
from pathlib import Path

from tqdm.auto import tqdm

from llm_lite.config.models import (
    AssignSplitTransformConfiguration,
    ExactDeduplicationTransformConfiguration,
    ExtractPythonFunctionsTransformConfiguration,
    LowerCaseTransformConfiguration,
    MaxLengthTransformConfiguration,
    MinLengthTransformConfiguration,
    NormalizeLineEndingsTransformConfiguration,
    NormalizeUnicodeTransformConfiguration,
    PreprocessingTransformConfiguration,
)
from llm_lite.data.document import Document
from llm_lite.data.text_shards import (
    TextShardCorpusManifest,
    TextShardReference,
    TextShardSplitCounters,
    TextShardSplitManifest,
    TextShardWriter,
    iter_text_shard_reference_documents,
    load_text_shard_corpus_manifest,
    text_shard_references,
)
from llm_lite.pipeline.progress import progress_bar, track_progress


@dataclass(frozen=True)
class PreprocessingResult:
    documents: Iterator[Document]
    counters: "PreprocessingCounters"


@dataclass
class PreprocessingCounters:
    input_documents: int = 0
    output_documents: int = 0
    rejected_documents: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    input_characters: int = 0
    output_characters: int = 0
    unicode_normalized_documents: int = 0
    line_endings_normalized_documents: int = 0
    lower_cased_documents: int = 0
    deduplicated_documents: int = 0
    split_assigned_documents: int = 0
    python_extracted_functions: int = 0
    python_parse_failed_documents: int = 0

    def add(self, other: "PreprocessingCounters") -> None:
        self.input_documents += other.input_documents
        self.output_documents += other.output_documents
        self.rejected_documents += other.rejected_documents
        self.input_bytes += other.input_bytes
        self.output_bytes += other.output_bytes
        self.input_characters += other.input_characters
        self.output_characters += other.output_characters
        self.unicode_normalized_documents += other.unicode_normalized_documents
        self.line_endings_normalized_documents += other.line_endings_normalized_documents
        self.lower_cased_documents += other.lower_cased_documents
        self.deduplicated_documents += other.deduplicated_documents
        self.split_assigned_documents += other.split_assigned_documents
        self.python_extracted_functions += other.python_extracted_functions
        self.python_parse_failed_documents += other.python_parse_failed_documents


@dataclass(frozen=True)
class ParallelPreprocessingResult:
    corpus_manifest: TextShardCorpusManifest
    counters: PreprocessingCounters
    worker_count: int


def preprocess_documents(
    documents: Iterable[Document],
    transforms: tuple[PreprocessingTransformConfiguration, ...],
) -> PreprocessingResult:
    counters = PreprocessingCounters()
    seen_document_hashes: set[str] = set()

    def iter_processed_documents() -> Iterator[Document]:
        for document in documents:
            counters.input_documents += 1
            counters.input_bytes += len(document.text.encode("utf-8"))
            counters.input_characters += len(document.text)
            transformed_documents = _apply_transforms(
                document=document,
                transforms=transforms,
                counters=counters,
                seen_document_hashes=seen_document_hashes,
            )
            if not transformed_documents:
                counters.rejected_documents += 1
                continue
            for transformed_document in transformed_documents:
                counters.output_documents += 1
                counters.output_bytes += len(transformed_document.text.encode("utf-8"))
                counters.output_characters += len(transformed_document.text)
                yield transformed_document

    return PreprocessingResult(
        documents=iter_processed_documents(),
        counters=counters,
    )


def preprocess_text_shards(
    input_artifact_directory: Path,
    output_artifact_directory: Path,
    transforms: tuple[PreprocessingTransformConfiguration, ...],
    output_shard_documents: int,
    workers: int,
) -> ParallelPreprocessingResult:
    shard_references = text_shard_references(
        artifact_directory=input_artifact_directory,
        split=None,
    )
    input_manifest = load_text_shard_corpus_manifest(
        artifact_directory=input_artifact_directory,
    )
    input_documents = sum(split.documents for split in input_manifest.splits)
    effective_workers = _effective_worker_count(
        requested_workers=workers,
        shard_count=len(shard_references),
    )
    if effective_workers == 1 or _has_exact_deduplication(transforms=transforms):
        preprocessing_result = preprocess_documents(
            documents=chain.from_iterable(
                iter_text_shard_reference_documents(shard_reference=shard_reference)
                for shard_reference in shard_references
            ),
            transforms=transforms,
        )
        writer = TextShardWriter(
            artifact_directory=output_artifact_directory,
            shard_document_limit=output_shard_documents,
            shard_name_prefix="",
        )
        for document in track_progress(
            preprocessing_result.documents,
            description="preprocess",
            total=input_documents,
            unit="doc",
        ):
            writer.append(document=document)
        return ParallelPreprocessingResult(
            corpus_manifest=writer.close(),
            counters=preprocessing_result.counters,
            worker_count=1,
        )
    worker_inputs = _contiguous_worker_inputs(
        shard_references=shard_references,
        worker_count=effective_workers,
        progress_total_documents=input_documents,
    )
    worker_results: list[PreprocessingWorkerResult] = []
    multiprocessing_context = get_context("spawn")
    with multiprocessing_context.Pool(processes=effective_workers) as pool:
        worker_arguments = [
            (
                worker_input,
                output_artifact_directory,
                transforms,
                output_shard_documents,
            )
            for worker_input in worker_inputs
        ]
        worker_results.extend(
            pool.imap_unordered(
                _preprocess_worker_from_arguments,
                worker_arguments,
            ),
        )
    counters = PreprocessingCounters()
    for worker_result in sorted(worker_results, key=lambda result: result.worker_index):
        counters.add(other=worker_result.counters)
    corpus_manifest = _merge_worker_text_manifests(
        worker_results=tuple(worker_results),
        output_artifact_directory=output_artifact_directory,
        output_shard_documents=output_shard_documents,
    )
    return ParallelPreprocessingResult(
        corpus_manifest=corpus_manifest,
        counters=counters,
        worker_count=effective_workers,
    )


@dataclass(frozen=True)
class PreprocessingWorkerInput:
    worker_index: int
    shard_references: tuple[TextShardReference, ...]
    progress_total_documents: int | None
    progress_document_multiplier: int


@dataclass(frozen=True)
class PreprocessingWorkerResult:
    worker_index: int
    counters: PreprocessingCounters
    corpus_manifest: TextShardCorpusManifest


def _preprocess_worker(
    worker_input: PreprocessingWorkerInput,
    output_artifact_directory: Path,
    transforms: tuple[PreprocessingTransformConfiguration, ...],
    output_shard_documents: int,
) -> PreprocessingWorkerResult:
    preprocessing_result = preprocess_documents(
        documents=chain.from_iterable(
            iter_text_shard_reference_documents(shard_reference=shard_reference)
            for shard_reference in worker_input.shard_references
        ),
        transforms=transforms,
    )
    writer = TextShardWriter(
        artifact_directory=output_artifact_directory,
        shard_document_limit=output_shard_documents,
        shard_name_prefix=f"part_{worker_input.worker_index:06d}_",
    )
    _write_preprocessed_worker_documents(
        worker_input=worker_input,
        preprocessing_result=preprocessing_result,
        writer=writer,
    )
    return PreprocessingWorkerResult(
        worker_index=worker_input.worker_index,
        counters=preprocessing_result.counters,
        corpus_manifest=writer.close(),
    )


@dataclass(frozen=True)
class PreprocessingProgress:
    progress_bar_instance: tqdm
    total_documents: int
    document_multiplier: int
    reported_input_documents: int


def _write_preprocessed_worker_documents(
    worker_input: PreprocessingWorkerInput,
    preprocessing_result: PreprocessingResult,
    writer: TextShardWriter,
) -> None:
    if worker_input.progress_total_documents is None:
        for document in preprocessing_result.documents:
            writer.append(document=document)
        return
    with progress_bar(
        description=f"preprocess/{worker_input.progress_document_multiplier} workers",
        total=worker_input.progress_total_documents,
        unit="doc",
    ) as progress_bar_instance:
        progress = PreprocessingProgress(
            progress_bar_instance=progress_bar_instance,
            total_documents=worker_input.progress_total_documents,
            document_multiplier=worker_input.progress_document_multiplier,
            reported_input_documents=0,
        )
        for document in preprocessing_result.documents:
            writer.append(document=document)
            progress = _update_preprocessing_progress(
                progress=progress,
                input_documents=preprocessing_result.counters.input_documents,
            )
        _update_preprocessing_progress(
            progress=progress,
            input_documents=preprocessing_result.counters.input_documents,
        )


def _update_preprocessing_progress(
    progress: PreprocessingProgress,
    input_documents: int,
) -> PreprocessingProgress:
    unreported_input_documents = input_documents - progress.reported_input_documents
    scaled_increment = unreported_input_documents * progress.document_multiplier
    remaining_documents = progress.total_documents - progress.progress_bar_instance.n
    if remaining_documents > 0:
        progress.progress_bar_instance.update(min(scaled_increment, remaining_documents))
    return PreprocessingProgress(
        progress_bar_instance=progress.progress_bar_instance,
        total_documents=progress.total_documents,
        document_multiplier=progress.document_multiplier,
        reported_input_documents=input_documents,
    )


def _preprocess_worker_from_arguments(
    arguments: tuple[
        PreprocessingWorkerInput,
        Path,
        tuple[PreprocessingTransformConfiguration, ...],
        int,
    ],
) -> PreprocessingWorkerResult:
    return _preprocess_worker(*arguments)


def _apply_transforms(
    document: Document,
    transforms: tuple[PreprocessingTransformConfiguration, ...],
    counters: PreprocessingCounters,
    seen_document_hashes: set[str],
) -> tuple[Document, ...]:
    current_documents: tuple[Document, ...] = (document,)
    for transform in transforms:
        next_documents: list[Document] = []
        for current_document in current_documents:
            next_documents.extend(
                _apply_transform(
                    document=current_document,
                    transform=transform,
                    counters=counters,
                    seen_document_hashes=seen_document_hashes,
                ),
            )
        current_documents = tuple(next_documents)
        if not current_documents:
            return ()
    return current_documents


def _apply_transform(
    document: Document,
    transform: PreprocessingTransformConfiguration,
    counters: PreprocessingCounters,
    seen_document_hashes: set[str],
) -> tuple[Document, ...]:
    match transform:
        case NormalizeUnicodeTransformConfiguration(form=form):
            normalized_text = unicodedata.normalize(form, document.text)
            if normalized_text != document.text:
                counters.unicode_normalized_documents += 1
            return (
                Document(
                    document_id=document.document_id,
                    text=normalized_text,
                    split=document.split,
                ),
            )
        case NormalizeLineEndingsTransformConfiguration():
            normalized_text = document.text.replace("\r\n", "\n").replace("\r", "\n")
            if normalized_text != document.text:
                counters.line_endings_normalized_documents += 1
            return (
                Document(
                    document_id=document.document_id,
                    text=normalized_text,
                    split=document.split,
                ),
            )
        case LowerCaseTransformConfiguration():
            lowered_text = document.text.lower()
            if lowered_text != document.text:
                counters.lower_cased_documents += 1
            return (
                Document(
                    document_id=document.document_id,
                    text=lowered_text,
                    split=document.split,
                ),
            )
        case ExactDeduplicationTransformConfiguration():
            document_hash = _text_hash(text=document.text)
            if document_hash in seen_document_hashes:
                counters.deduplicated_documents += 1
                return ()
            seen_document_hashes.add(document_hash)
            return (
                Document(
                    document_id=document.document_id,
                    text=document.text,
                    split=document.split,
                ),
            )
        case MinLengthTransformConfiguration(min_characters=min_characters):
            if len(document.text) < min_characters:
                return ()
            return (document,)
        case MaxLengthTransformConfiguration(max_characters=max_characters):
            if len(document.text) > max_characters:
                return ()
            return (document,)
        case ExtractPythonFunctionsTransformConfiguration(
            include_async_functions=include_async_functions,
            include_private_functions=include_private_functions,
            include_methods=include_methods,
        ):
            return _extract_python_function_documents(
                document=document,
                counters=counters,
                include_async_functions=include_async_functions,
                include_private_functions=include_private_functions,
                include_methods=include_methods,
            )
        case AssignSplitTransformConfiguration(
            train_probability=train_probability,
            validation_probability=validation_probability,
        ):
            if document.split is not None:
                raise ValueError("Assign split cannot overwrite an existing document split.")
            counters.split_assigned_documents += 1
            split_name = _assigned_split(
                document_id=document.document_id,
                train_probability=train_probability,
                validation_probability=validation_probability,
            )
            return (
                Document(
                    document_id=document.document_id,
                    text=document.text,
                    split=split_name,
                ),
            )


def _extract_python_function_documents(
    document: Document,
    counters: PreprocessingCounters,
    include_async_functions: bool,
    include_private_functions: bool,
    include_methods: bool,
) -> tuple[Document, ...]:
    try:
        module = ast.parse(document.text)
    except SyntaxError:
        counters.python_parse_failed_documents += 1
        return ()
    extracted_documents: list[Document] = []
    function_index = 0
    for node in module.body:
        extracted_functions = _node_functions(
            node=node,
            include_async_functions=include_async_functions,
            include_methods=include_methods,
        )
        for function_node in extracted_functions:
            if not include_private_functions and function_node.name.startswith("_"):
                continue
            function_text = ast.unparse(function_node).strip() + "\n"
            extracted_documents.append(
                Document(
                    document_id=(
                        f"{document.document_id}__function_{function_index:04d}_"
                        f"{function_node.name}"
                    ),
                    text=function_text,
                    split=document.split,
                ),
            )
            function_index += 1
    counters.python_extracted_functions += len(extracted_documents)
    return tuple(extracted_documents)


def _node_functions(
    node: ast.stmt,
    include_async_functions: bool,
    include_methods: bool,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    match node:
        case ast.FunctionDef():
            return (node,)
        case ast.AsyncFunctionDef():
            if include_async_functions:
                return (node,)
            return ()
        case ast.ClassDef():
            if not include_methods:
                return ()
            return tuple(
                child_node
                for child_node in node.body
                if _is_included_method(
                    node=child_node,
                    include_async_functions=include_async_functions,
                )
            )
        case _:
            return ()


def _is_included_method(node: ast.stmt, include_async_functions: bool) -> bool:
    match node:
        case ast.FunctionDef():
            return True
        case ast.AsyncFunctionDef():
            return include_async_functions
        case _:
            return False


def _text_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _assigned_split(
    document_id: str,
    train_probability: float,
    validation_probability: float,
) -> str:
    hash_value = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16], 16)
    normalized_value = hash_value / float(16**16 - 1)
    validation_threshold = train_probability + validation_probability
    if normalized_value < train_probability:
        return "train"
    if normalized_value < validation_threshold:
        return "validation"
    return "test"


def _effective_worker_count(requested_workers: int, shard_count: int) -> int:
    if shard_count == 0:
        return 1
    return min(requested_workers, shard_count)


def _has_exact_deduplication(
    transforms: tuple[PreprocessingTransformConfiguration, ...],
) -> bool:
    for transform in transforms:
        match transform:
            case ExactDeduplicationTransformConfiguration():
                return True
            case _:
                continue
    return False


def _contiguous_worker_inputs(
    shard_references: tuple[TextShardReference, ...],
    worker_count: int,
    progress_total_documents: int,
) -> tuple[PreprocessingWorkerInput, ...]:
    worker_inputs: list[PreprocessingWorkerInput] = []
    for worker_index in range(worker_count):
        start_index = worker_index * len(shard_references) // worker_count
        end_index = (worker_index + 1) * len(shard_references) // worker_count
        worker_inputs.append(
            PreprocessingWorkerInput(
                worker_index=worker_index,
                shard_references=shard_references[start_index:end_index],
                progress_total_documents=(progress_total_documents if worker_index == 0 else None),
                progress_document_multiplier=worker_count,
            ),
        )
    return tuple(worker_inputs)


def _merge_worker_text_manifests(
    worker_results: tuple[PreprocessingWorkerResult, ...],
    output_artifact_directory: Path,
    output_shard_documents: int,
) -> TextShardCorpusManifest:
    split_counters: dict[str, TextShardSplitCounters] = {}
    for worker_result in sorted(worker_results, key=lambda result: result.worker_index):
        for split_manifest in worker_result.corpus_manifest.splits:
            counters = split_counters.get(split_manifest.split)
            if counters is None:
                counters = TextShardSplitCounters()
                split_counters[split_manifest.split] = counters
            counters.documents += split_manifest.documents
            counters.bytes += split_manifest.bytes
            counters.characters += split_manifest.characters
            counters.shards += split_manifest.shards
    corpus_manifest = TextShardCorpusManifest(
        format_version=1,
        shard_document_limit=output_shard_documents,
        splits=tuple(
            TextShardSplitManifest(
                split=split_name,
                documents=counters.documents,
                bytes=counters.bytes,
                characters=counters.characters,
                shards=counters.shards,
            )
            for split_name, counters in sorted(split_counters.items())
        ),
    )
    (output_artifact_directory / "corpus.json").write_text(
        corpus_manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return corpus_manifest
