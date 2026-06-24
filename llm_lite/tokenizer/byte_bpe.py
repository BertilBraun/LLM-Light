import json
import tarfile
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path

from llm_lite.data.text_shards import TextShardReference, text_shard_references
from llm_lite.pipeline.progress import console_log, progress_bar

ByteToken = tuple[int, ...]
BytePair = tuple[ByteToken, ByteToken]


class ByteBpeWorkerCommand(str, Enum):
    COUNT = "count"
    TOKEN_COUNT = "token_count"
    STOP = "stop"


@dataclass
class ByteBpeTrainingCorpus:
    documents: list[list[ByteToken]]
    bytes: int


@dataclass(frozen=True)
class ByteBpeTrainingResult:
    tokenizer: "ByteBpeTokenizer"
    training_document_count: int
    training_bytes: int
    training_tokens: int
    max_training_documents: int | None
    max_training_bytes: int | None
    worker_count: int
    pair_count_seconds: float
    merge_application_seconds: float

    @property
    def bytes_per_token(self) -> float:
        if self.training_tokens == 0:
            return 0.0
        return self.training_bytes / self.training_tokens


@dataclass(frozen=True)
class ByteBpeDocumentReference:
    shard_reference: TextShardReference
    member_name: str
    byte_count: int


@dataclass(frozen=True)
class ByteBpeTrainingSelection:
    document_references: tuple[ByteBpeDocumentReference, ...]
    bytes: int


@dataclass(frozen=True)
class ByteBpeWorkerInput:
    worker_index: int
    document_references: tuple[ByteBpeDocumentReference, ...]


@dataclass(frozen=True)
class ByteBpeWorkerReady:
    worker_index: int
    document_count: int
    byte_count: int


@dataclass(frozen=True)
class ByteBpeMergeCommand:
    merge_rule: BytePair


class ByteBpeTokenizer:
    def __init__(
        self,
        token_to_id: dict[str, int],
        byte_token_to_id: dict[ByteToken, int],
        merge_rules: tuple[BytePair, ...],
        bos_token: str | None,
        eos_token: str | None,
        pad_token: str | None,
    ) -> None:
        self.token_to_id = token_to_id
        self.byte_token_to_id = byte_token_to_id
        self.id_to_byte_token = {
            token_id: byte_token for byte_token, token_id in byte_token_to_id.items()
        }
        self.merge_rules = merge_rules
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id) + len(self.byte_token_to_id)

    @property
    def merge_count(self) -> int:
        return len(self.merge_rules)

    @property
    def bos_token_id(self) -> int:
        if self.bos_token is None:
            raise ValueError("BOS token is not configured.")
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        if self.eos_token is None:
            raise ValueError("EOS token is not configured.")
        return self.token_to_id[self.eos_token]

    @property
    def pad_token_id(self) -> int | None:
        if self.pad_token is None:
            return None
        return self.token_to_id[self.pad_token]

    def encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]:
        byte_tokens = _byte_tokens(text=text)
        for merge_rule in self.merge_rules:
            byte_tokens = _merge_pair(byte_tokens=byte_tokens, merge_rule=merge_rule)
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        token_ids.extend(self.byte_token_to_id[byte_token] for byte_token in byte_tokens)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        output_bytes = bytearray()
        special_token_ids = {
            self.token_to_id[special_token]
            for special_token in (self.bos_token, self.eos_token, self.pad_token)
            if special_token is not None
        }
        for token_id in token_ids:
            if token_id in special_token_ids:
                continue
            output_bytes.extend(self.id_to_byte_token[token_id])
        return bytes(output_bytes).decode("utf-8", errors="replace")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer_data = {
            "format": "byte_bpe",
            "token_to_id": self.token_to_id,
            "byte_token_to_id": [
                {"bytes": list(byte_token), "token_id": token_id}
                for byte_token, token_id in sorted(
                    self.byte_token_to_id.items(),
                    key=lambda item: item[1],
                )
            ],
            "merge_rules": [
                {"left": list(left_token), "right": list(right_token)}
                for left_token, right_token in self.merge_rules
            ],
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(tokenizer_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "ByteBpeTokenizer":
        tokenizer_data = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        byte_token_to_id = {
            tuple(byte_token_record["bytes"]): byte_token_record["token_id"]
            for byte_token_record in tokenizer_data["byte_token_to_id"]
        }
        merge_rules = tuple(
            (
                tuple(merge_rule["left"]),
                tuple(merge_rule["right"]),
            )
            for merge_rule in tokenizer_data["merge_rules"]
        )
        return cls(
            token_to_id=tokenizer_data["token_to_id"],
            byte_token_to_id=byte_token_to_id,
            merge_rules=merge_rules,
            bos_token=tokenizer_data["bos_token"],
            eos_token=tokenizer_data["eos_token"],
            pad_token=tokenizer_data["pad_token"],
        )


def train_byte_bpe_tokenizer(
    texts: Iterable[str],
    vocabulary_size: int,
    max_training_documents: int | None,
    max_training_bytes: int | None,
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
    workers: int,
) -> ByteBpeTrainingResult:
    if workers < 1:
        raise ValueError("Worker count must be positive.")
    token_to_id = _special_token_to_id(
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        add_pad_token=add_pad_token,
    )
    minimum_vocabulary_size = len(token_to_id) + 256
    if vocabulary_size < minimum_vocabulary_size:
        raise ValueError("Byte BPE vocabulary size must include special tokens and 256 bytes.")
    corpus_sample = _bounded_training_corpus(
        texts=texts,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
    )
    console_log(
        "[tokenizer] byte_bpe sample "
        f"documents={len(corpus_sample.documents)} "
        f"bytes={corpus_sample.bytes} "
        f"target_vocabulary_size={vocabulary_size}",
    )
    training_state = _train_serial_merges(
        documents=corpus_sample.documents,
        vocabulary_size=vocabulary_size,
        token_to_id=token_to_id,
    )
    merge_rules = training_state.merge_rules
    byte_token_to_id = training_state.byte_token_to_id
    corpus_sample.documents = training_state.documents
    training_tokens = sum(len(document) for document in corpus_sample.documents)
    console_log(
        "[tokenizer] byte_bpe complete "
        f"merges={len(merge_rules)} "
        f"training_tokens={training_tokens} "
        f"bytes_per_token={corpus_sample.bytes / max(training_tokens, 1):.4f}",
    )
    tokenizer = ByteBpeTokenizer(
        token_to_id=token_to_id,
        byte_token_to_id=byte_token_to_id,
        merge_rules=tuple(merge_rules),
        bos_token="<bos>" if add_bos_token else None,
        eos_token="<eos>" if add_eos_token else None,
        pad_token="<pad>" if add_pad_token else None,
    )
    return ByteBpeTrainingResult(
        tokenizer=tokenizer,
        training_document_count=len(corpus_sample.documents),
        training_bytes=corpus_sample.bytes,
        training_tokens=training_tokens,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
        worker_count=1,
        pair_count_seconds=training_state.pair_count_seconds,
        merge_application_seconds=training_state.merge_application_seconds,
    )


def train_byte_bpe_tokenizer_from_text_shards(
    artifact_directory: Path,
    split: str | None,
    vocabulary_size: int,
    max_training_documents: int | None,
    max_training_bytes: int | None,
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
    workers: int,
) -> ByteBpeTrainingResult:
    token_to_id = _special_token_to_id(
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        add_pad_token=add_pad_token,
    )
    minimum_vocabulary_size = len(token_to_id) + 256
    if vocabulary_size < minimum_vocabulary_size:
        raise ValueError("Byte BPE vocabulary size must include special tokens and 256 bytes.")
    selection = bounded_training_document_references(
        artifact_directory=artifact_directory,
        split=split,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
    )
    effective_workers = _effective_worker_count(
        requested_workers=workers,
        document_count=len(selection.document_references),
    )
    if effective_workers == 1:
        texts = (
            _read_document_reference_text(document_reference=document_reference)
            for document_reference in selection.document_references
        )
        return train_byte_bpe_tokenizer(
            texts=texts,
            vocabulary_size=vocabulary_size,
            max_training_documents=max_training_documents,
            max_training_bytes=max_training_bytes,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_pad_token=add_pad_token,
            workers=1,
        )
    worker_inputs = _bpe_worker_inputs(
        document_references=selection.document_references,
        worker_count=effective_workers,
    )
    training_state = _train_parallel_merges(
        worker_inputs=worker_inputs,
        vocabulary_size=vocabulary_size,
        token_to_id=token_to_id,
    )
    tokenizer = ByteBpeTokenizer(
        token_to_id=token_to_id,
        byte_token_to_id=training_state.byte_token_to_id,
        merge_rules=tuple(training_state.merge_rules),
        bos_token="<bos>" if add_bos_token else None,
        eos_token="<eos>" if add_eos_token else None,
        pad_token="<pad>" if add_pad_token else None,
    )
    return ByteBpeTrainingResult(
        tokenizer=tokenizer,
        training_document_count=len(selection.document_references),
        training_bytes=selection.bytes,
        training_tokens=training_state.training_tokens,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
        worker_count=effective_workers,
        pair_count_seconds=training_state.pair_count_seconds,
        merge_application_seconds=training_state.merge_application_seconds,
    )


@dataclass(frozen=True)
class ByteBpeMergeTrainingState:
    byte_token_to_id: dict[ByteToken, int]
    merge_rules: tuple[BytePair, ...]
    documents: list[list[ByteToken]]
    training_tokens: int
    pair_count_seconds: float
    merge_application_seconds: float


def _special_token_to_id(
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
) -> dict[str, int]:
    token_to_id: dict[str, int] = {}
    for special_token in (
        "<bos>" if add_bos_token else None,
        "<eos>" if add_eos_token else None,
        "<pad>" if add_pad_token else None,
    ):
        if special_token is not None:
            token_to_id[special_token] = len(token_to_id)
    return token_to_id


def _initial_byte_token_to_id(starting_index: int) -> dict[ByteToken, int]:
    return {(byte_value,): starting_index + byte_value for byte_value in range(256)}


def _byte_tokens(text: str) -> list[ByteToken]:
    return [(byte_value,) for byte_value in text.encode("utf-8")]


def _bounded_training_corpus(
    texts: Iterable[str],
    max_training_documents: int | None,
    max_training_bytes: int | None,
) -> ByteBpeTrainingCorpus:
    if max_training_documents is None and max_training_bytes is None:
        raise ValueError("Byte BPE training sample must be bounded.")
    documents: list[list[ByteToken]] = []
    sampled_bytes = 0
    text_iterator = iter(texts)
    while max_training_documents is None or len(documents) < max_training_documents:
        try:
            text = next(text_iterator)
        except StopIteration:
            break
        text_bytes = text.encode("utf-8")
        if max_training_bytes is not None and sampled_bytes + len(text_bytes) > max_training_bytes:
            if documents:
                break
            raise ValueError("First Byte BPE training document exceeds max_training_bytes.")
        documents.append([(byte_value,) for byte_value in text_bytes])
        sampled_bytes += len(text_bytes)
    if not documents:
        raise ValueError("Byte BPE training sample is empty.")
    return ByteBpeTrainingCorpus(documents=documents, bytes=sampled_bytes)


def bounded_training_document_references(
    artifact_directory: Path,
    split: str | None,
    max_training_documents: int | None,
    max_training_bytes: int | None,
) -> ByteBpeTrainingSelection:
    if max_training_documents is None and max_training_bytes is None:
        raise ValueError("Byte BPE training sample must be bounded.")
    document_references: list[ByteBpeDocumentReference] = []
    sampled_bytes = 0
    shard_references = text_shard_references(
        artifact_directory=artifact_directory,
        split=split,
    )
    with progress_bar(
        description="tokenizer/select_sample",
        total=max_training_documents,
        unit="doc",
    ) as bar:
        for shard_reference in shard_references:
            with tarfile.open(shard_reference.path, mode="r:gz") as shard_file:
                for member in shard_file:
                    if not member.isfile():
                        continue
                    if (
                        max_training_documents is not None
                        and len(document_references) >= max_training_documents
                    ):
                        break
                    if (
                        max_training_bytes is not None
                        and sampled_bytes + member.size > max_training_bytes
                    ):
                        if document_references:
                            return ByteBpeTrainingSelection(
                                document_references=tuple(document_references),
                                bytes=sampled_bytes,
                            )
                        raise ValueError(
                            "First Byte BPE training document exceeds max_training_bytes."
                        )
                    document_references.append(
                        ByteBpeDocumentReference(
                            shard_reference=shard_reference,
                            member_name=member.name,
                            byte_count=member.size,
                        ),
                    )
                    sampled_bytes += member.size
                    bar.update(1)
            if (
                max_training_documents is not None
                and len(document_references) >= max_training_documents
            ):
                break
    if not document_references:
        raise ValueError("Byte BPE training sample is empty.")
    return ByteBpeTrainingSelection(
        document_references=tuple(document_references),
        bytes=sampled_bytes,
    )


def _read_document_reference_text(document_reference: ByteBpeDocumentReference) -> str:
    with tarfile.open(document_reference.shard_reference.path, mode="r:gz") as shard_file:
        extracted_file = shard_file.extractfile(document_reference.member_name)
        if extracted_file is None:
            raise ValueError("Byte BPE training document reference is not readable.")
        return extracted_file.read().decode("utf-8")


def _train_serial_merges(
    documents: list[list[ByteToken]],
    vocabulary_size: int,
    token_to_id: dict[str, int],
) -> ByteBpeMergeTrainingState:
    merge_rules: list[BytePair] = []
    byte_token_to_id = _initial_byte_token_to_id(starting_index=len(token_to_id))
    pair_count_seconds = 0.0
    merge_application_seconds = 0.0
    target_merges = vocabulary_size - len(token_to_id) - len(byte_token_to_id)
    with progress_bar(
        description="tokenizer/byte_bpe_merges",
        total=target_merges,
        unit="merge",
    ) as bar:
        while len(token_to_id) + len(byte_token_to_id) < vocabulary_size:
            pair_count_start = time.perf_counter()
            pair_counts = _pair_counts(corpus=documents)
            pair_count_seconds += time.perf_counter() - pair_count_start
            if not pair_counts:
                break
            best_pair = _best_pair(pair_counts=pair_counts)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token in byte_token_to_id:
                break
            byte_token_to_id[merged_token] = len(token_to_id) + len(byte_token_to_id)
            merge_rules.append(best_pair)
            merge_start = time.perf_counter()
            for document_index, document in enumerate(documents):
                documents[document_index] = _merge_pair(
                    byte_tokens=document,
                    merge_rule=best_pair,
                )
            merge_application_seconds += time.perf_counter() - merge_start
            bar.update(1)
            if len(merge_rules) % 100 == 0:
                console_log(
                    "[tokenizer] byte_bpe "
                    f"merges={len(merge_rules)} "
                    f"vocabulary_size={len(token_to_id) + len(byte_token_to_id)}",
                )
    return ByteBpeMergeTrainingState(
        byte_token_to_id=byte_token_to_id,
        merge_rules=tuple(merge_rules),
        documents=documents,
        training_tokens=sum(len(document) for document in documents),
        pair_count_seconds=pair_count_seconds,
        merge_application_seconds=merge_application_seconds,
    )


def _train_parallel_merges(
    worker_inputs: tuple[ByteBpeWorkerInput, ...],
    vocabulary_size: int,
    token_to_id: dict[str, int],
) -> ByteBpeMergeTrainingState:
    worker_connections: list[Connection] = []
    processes: list[Process] = []
    for worker_input in worker_inputs:
        parent_connection, child_connection = Pipe()
        process = Process(
            target=_byte_bpe_worker_loop,
            args=(child_connection, worker_input),
        )
        process.start()
        child_connection.close()
        worker_connections.append(parent_connection)
        processes.append(process)
    try:
        for connection in worker_connections:
            ready = connection.recv()
            console_log(
                "[tokenizer] byte_bpe worker_ready "
                f"worker={ready.worker_index} "
                f"documents={ready.document_count} "
                f"bytes={ready.byte_count}",
            )
        byte_token_to_id = _initial_byte_token_to_id(starting_index=len(token_to_id))
        merge_rules: list[BytePair] = []
        pair_count_seconds = 0.0
        merge_application_seconds = 0.0
        target_merges = vocabulary_size - len(token_to_id) - len(byte_token_to_id)
        with progress_bar(
            description="tokenizer/byte_bpe_merges",
            total=target_merges,
            unit="merge",
        ) as bar:
            while len(token_to_id) + len(byte_token_to_id) < vocabulary_size:
                pair_count_start = time.perf_counter()
                for connection in worker_connections:
                    connection.send(ByteBpeWorkerCommand.COUNT)
                pair_counts: Counter[BytePair] = Counter()
                for connection in worker_connections:
                    pair_counts.update(connection.recv())
                pair_count_seconds += time.perf_counter() - pair_count_start
                if not pair_counts:
                    break
                best_pair = _best_pair(pair_counts=pair_counts)
                merged_token = best_pair[0] + best_pair[1]
                if merged_token in byte_token_to_id:
                    break
                byte_token_to_id[merged_token] = len(token_to_id) + len(byte_token_to_id)
                merge_rules.append(best_pair)
                merge_start = time.perf_counter()
                for connection in worker_connections:
                    connection.send(ByteBpeMergeCommand(merge_rule=best_pair))
                for connection in worker_connections:
                    connection.recv()
                merge_application_seconds += time.perf_counter() - merge_start
                bar.update(1)
        for connection in worker_connections:
            connection.send(ByteBpeWorkerCommand.TOKEN_COUNT)
        training_tokens = sum(connection.recv() for connection in worker_connections)
    finally:
        for connection in worker_connections:
            connection.send(ByteBpeWorkerCommand.STOP)
            connection.close()
        for process in processes:
            process.join()
    return ByteBpeMergeTrainingState(
        byte_token_to_id=byte_token_to_id,
        merge_rules=tuple(merge_rules),
        documents=[],
        training_tokens=training_tokens,
        pair_count_seconds=pair_count_seconds,
        merge_application_seconds=merge_application_seconds,
    )


def _byte_bpe_worker_loop(
    connection: Connection,
    worker_input: ByteBpeWorkerInput,
) -> None:
    corpus = _load_worker_corpus(document_references=worker_input.document_references)
    connection.send(
        ByteBpeWorkerReady(
            worker_index=worker_input.worker_index,
            document_count=len(corpus.documents),
            byte_count=corpus.bytes,
        ),
    )
    while True:
        command = connection.recv()
        match command:
            case ByteBpeWorkerCommand.COUNT:
                connection.send(_pair_counts(corpus=corpus.documents))
            case ByteBpeMergeCommand(merge_rule=merge_rule):
                for document_index, document in enumerate(corpus.documents):
                    corpus.documents[document_index] = _merge_pair(
                        byte_tokens=document,
                        merge_rule=merge_rule,
                    )
                connection.send(True)
            case ByteBpeWorkerCommand.TOKEN_COUNT:
                connection.send(sum(len(document) for document in corpus.documents))
            case ByteBpeWorkerCommand.STOP:
                return


def _load_worker_corpus(
    document_references: tuple[ByteBpeDocumentReference, ...],
) -> ByteBpeTrainingCorpus:
    documents: list[list[ByteToken]] = []
    byte_count = 0
    for document_reference in document_references:
        text = _read_document_reference_text(document_reference=document_reference)
        text_bytes = text.encode("utf-8")
        documents.append([(byte_value,) for byte_value in text_bytes])
        byte_count += len(text_bytes)
    return ByteBpeTrainingCorpus(documents=documents, bytes=byte_count)


def _bpe_worker_inputs(
    document_references: tuple[ByteBpeDocumentReference, ...],
    worker_count: int,
) -> tuple[ByteBpeWorkerInput, ...]:
    worker_inputs: list[ByteBpeWorkerInput] = []
    for worker_index in range(worker_count):
        start_index = worker_index * len(document_references) // worker_count
        end_index = (worker_index + 1) * len(document_references) // worker_count
        worker_inputs.append(
            ByteBpeWorkerInput(
                worker_index=worker_index,
                document_references=document_references[start_index:end_index],
            ),
        )
    return tuple(worker_inputs)


def _effective_worker_count(requested_workers: int, document_count: int) -> int:
    if requested_workers < 1:
        raise ValueError("Worker count must be positive.")
    if document_count == 0:
        return 1
    return min(requested_workers, document_count)


def _pair_counts(corpus: list[list[ByteToken]]) -> Counter[BytePair]:
    pair_counts: Counter[BytePair] = Counter()
    for document in corpus:
        for token_index in range(len(document) - 1):
            pair_counts[(document[token_index], document[token_index + 1])] += 1
    return pair_counts


def _best_pair(pair_counts: Counter[BytePair]) -> BytePair:
    return min(pair_counts, key=lambda pair: (-pair_counts[pair], pair[0], pair[1]))


def _merge_pair(byte_tokens: list[ByteToken], merge_rule: BytePair) -> list[ByteToken]:
    merged_tokens: list[ByteToken] = []
    token_index = 0
    while token_index < len(byte_tokens):
        next_index = token_index + 1
        if (
            next_index < len(byte_tokens)
            and (
                byte_tokens[token_index],
                byte_tokens[next_index],
            )
            == merge_rule
        ):
            merged_tokens.append(byte_tokens[token_index] + byte_tokens[next_index])
            token_index += 2
        else:
            merged_tokens.append(byte_tokens[token_index])
            token_index += 1
    return merged_tokens
