from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from llm_lite.pipeline.progress import console_log
from llm_lite.tokenizer.byte_bpe import (
    _read_document_reference_texts,
    bounded_training_document_references,
)


@dataclass
class RustByteBpeTrainingSample:
    texts: tuple[str, ...]
    bytes: int


@dataclass(frozen=True)
class RustByteBpeTrainingResult:
    tokenizer: RustByteBpeTokenizer
    training_document_count: int
    training_bytes: int
    training_tokens: int
    max_training_documents: int | None
    max_training_bytes: int | None
    worker_count: int
    training_seconds: float

    @property
    def bytes_per_token(self) -> float:
        if self.training_tokens == 0:
            return 0.0
        return self.training_bytes / self.training_tokens


class RustByteBpeTokenizer:
    def __init__(
        self,
        backend_tokenizer: Tokenizer,
        bos_token: str | None,
        eos_token: str | None,
        pad_token: str | None,
    ) -> None:
        self.backend_tokenizer = backend_tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    @property
    def vocabulary_size(self) -> int:
        return self.backend_tokenizer.get_vocab_size()

    @property
    def merge_count(self) -> int:
        tokenizer_data = json.loads(self.backend_tokenizer.to_str())
        return len(tokenizer_data.get("model", {}).get("merges", []))

    @property
    def bos_token_id(self) -> int:
        if self.bos_token is None:
            raise ValueError("BOS token is not configured.")
        return self._token_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        if self.eos_token is None:
            raise ValueError("EOS token is not configured.")
        return self._token_id(self.eos_token)

    @property
    def pad_token_id(self) -> int | None:
        if self.pad_token is None:
            return None
        return self._token_id(self.pad_token)

    def encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]:
        token_ids = list(
            self.backend_tokenizer.encode(text, add_special_tokens=False).ids,
        )
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.backend_tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer_data = {
            "format": "rust_byte_bpe",
            "tokenizer_json": json.loads(self.backend_tokenizer.to_str()),
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(tokenizer_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> RustByteBpeTokenizer:
        tokenizer_data = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        if tokenizer_data["format"] != "rust_byte_bpe":
            raise ValueError("Tokenizer file is not a Rust Byte BPE tokenizer.")
        backend_tokenizer = Tokenizer.from_str(
            json.dumps(tokenizer_data["tokenizer_json"]),
        )
        return cls(
            backend_tokenizer=backend_tokenizer,
            bos_token=tokenizer_data["bos_token"],
            eos_token=tokenizer_data["eos_token"],
            pad_token=tokenizer_data["pad_token"],
        )

    def _token_id(self, token: str) -> int:
        token_id = self.backend_tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Token is not in vocabulary: {token}")
        return token_id


def train_rust_byte_bpe_tokenizer(
    texts: Iterable[str],
    vocabulary_size: int,
    max_training_documents: int | None,
    max_training_bytes: int | None,
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
    workers: int,
) -> RustByteBpeTrainingResult:
    if workers < 1:
        raise ValueError("Worker count must be positive.")
    special_tokens = _special_tokens(
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        add_pad_token=add_pad_token,
    )
    minimum_vocabulary_size = len(special_tokens) + 256
    if vocabulary_size < minimum_vocabulary_size:
        raise ValueError("Rust Byte BPE vocabulary size must include special tokens and 256 bytes.")
    sample = _bounded_training_text_sample(
        texts=texts,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
    )
    return _train_sample(
        sample=sample,
        vocabulary_size=vocabulary_size,
        special_tokens=special_tokens,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
        worker_count=workers,
    )


def train_rust_byte_bpe_tokenizer_from_text_shards(
    artifact_directory: Path,
    split: str | None,
    vocabulary_size: int,
    max_training_documents: int | None,
    max_training_bytes: int | None,
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
    workers: int,
) -> RustByteBpeTrainingResult:
    if workers < 1:
        raise ValueError("Worker count must be positive.")
    special_tokens = _special_tokens(
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        add_pad_token=add_pad_token,
    )
    minimum_vocabulary_size = len(special_tokens) + 256
    if vocabulary_size < minimum_vocabulary_size:
        raise ValueError("Rust Byte BPE vocabulary size must include special tokens and 256 bytes.")
    console_log(
        "[tokenizer] rust_byte_bpe discover_sample "
        f"split={'all' if split is None else split} "
        f"max_training_documents={max_training_documents} "
        f"max_training_bytes={max_training_bytes}",
    )
    selection = bounded_training_document_references(
        artifact_directory=artifact_directory,
        split=split,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
    )
    console_log(
        "[tokenizer] rust_byte_bpe sample_selected "
        f"documents={len(selection.document_references)} "
        f"bytes={selection.bytes}",
    )
    console_log("[tokenizer] rust_byte_bpe reading_sample")
    texts = _read_document_reference_texts(document_references=selection.document_references)
    sample = RustByteBpeTrainingSample(texts=texts, bytes=selection.bytes)
    return _train_sample(
        sample=sample,
        vocabulary_size=vocabulary_size,
        special_tokens=special_tokens,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
        worker_count=workers,
    )


def _train_sample(
    sample: RustByteBpeTrainingSample,
    vocabulary_size: int,
    special_tokens: tuple[str, ...],
    max_training_documents: int | None,
    max_training_bytes: int | None,
    worker_count: int,
) -> RustByteBpeTrainingResult:
    console_log(
        "[tokenizer] rust_byte_bpe train "
        f"documents={len(sample.texts)} "
        f"bytes={sample.bytes} "
        f"target_vocabulary_size={vocabulary_size} "
        f"workers={worker_count}",
    )
    backend_tokenizer = Tokenizer(BPE(unk_token=None))
    backend_tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    backend_tokenizer.decoder = ByteLevelDecoder()
    bpe_trainer = BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=1,
        special_tokens=list(special_tokens),
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=True,
    )
    training_start = time.perf_counter()
    backend_tokenizer.train_from_iterator(
        sample.texts,
        trainer=bpe_trainer,
        length=len(sample.texts),
    )
    training_seconds = time.perf_counter() - training_start
    tokenizer = RustByteBpeTokenizer(
        backend_tokenizer=backend_tokenizer,
        bos_token="<bos>" if "<bos>" in special_tokens else None,
        eos_token="<eos>" if "<eos>" in special_tokens else None,
        pad_token="<pad>" if "<pad>" in special_tokens else None,
    )
    training_tokens = sum(
        len(backend_tokenizer.encode(text, add_special_tokens=False).ids)
        for text in sample.texts
    )
    console_log(
        "[tokenizer] rust_byte_bpe complete "
        f"vocabulary_size={tokenizer.vocabulary_size} "
        f"merges={tokenizer.merge_count} "
        f"training_tokens={training_tokens} "
        f"bytes_per_token={sample.bytes / max(training_tokens, 1):.4f} "
        f"seconds={training_seconds:.2f}",
    )
    return RustByteBpeTrainingResult(
        tokenizer=tokenizer,
        training_document_count=len(sample.texts),
        training_bytes=sample.bytes,
        training_tokens=training_tokens,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
        worker_count=worker_count,
        training_seconds=training_seconds,
    )


def _bounded_training_text_sample(
    texts: Iterable[str],
    max_training_documents: int | None,
    max_training_bytes: int | None,
) -> RustByteBpeTrainingSample:
    if max_training_documents is None and max_training_bytes is None:
        raise ValueError("Rust Byte BPE training sample must be bounded.")
    sampled_texts: list[str] = []
    sampled_bytes = 0
    text_iterator = iter(texts)
    while max_training_documents is None or len(sampled_texts) < max_training_documents:
        try:
            text = next(text_iterator)
        except StopIteration:
            break
        text_bytes = text.encode("utf-8")
        if max_training_bytes is not None and sampled_bytes + len(text_bytes) > max_training_bytes:
            if sampled_texts:
                break
            raise ValueError("First Rust Byte BPE training document exceeds max_training_bytes.")
        sampled_texts.append(text)
        sampled_bytes += len(text_bytes)
    if not sampled_texts:
        raise ValueError("Rust Byte BPE training sample is empty.")
    return RustByteBpeTrainingSample(texts=tuple(sampled_texts), bytes=sampled_bytes)


def _special_tokens(
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
) -> tuple[str, ...]:
    return tuple(
        token
        for token in (
            "<bos>" if add_bos_token else None,
            "<eos>" if add_eos_token else None,
            "<pad>" if add_pad_token else None,
        )
        if token is not None
    )
