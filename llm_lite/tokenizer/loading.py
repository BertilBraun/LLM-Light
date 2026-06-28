from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm_lite.config.models import (
    ByteBpeTokenizerConfiguration,
    CharacterTokenizerConfiguration,
    RustByteBpeTokenizerConfiguration,
    TokenizerConfiguration,
)
from llm_lite.tokenizer.byte_bpe import ByteBpeTokenizer, train_byte_bpe_tokenizer
from llm_lite.tokenizer.character import CharacterTokenizer, train_character_tokenizer
from llm_lite.tokenizer.rust_byte_bpe import (
    RustByteBpeTokenizer,
    train_rust_byte_bpe_tokenizer,
)


class TextTokenizer(Protocol):
    @property
    def vocabulary_size(self) -> int: ...

    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int: ...

    def encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def save(self, directory: Path) -> None: ...


@dataclass(frozen=True)
class TrainedTokenizer:
    tokenizer: TextTokenizer
    metrics: dict[str, int | float]


def train_tokenizer(
    texts: Iterable[str],
    tokenizer_configuration: TokenizerConfiguration,
) -> TrainedTokenizer:
    match tokenizer_configuration:
        case CharacterTokenizerConfiguration():
            tokenizer = train_character_tokenizer(
                texts=texts,
                add_bos_token=tokenizer_configuration.add_bos_token,
                add_eos_token=tokenizer_configuration.add_eos_token,
                add_pad_token=tokenizer_configuration.add_pad_token,
                additional_special_tokens=tokenizer_configuration.additional_special_tokens,
            )
            return TrainedTokenizer(
                tokenizer=tokenizer,
                metrics={"vocabulary_size": tokenizer.vocabulary_size},
            )
        case ByteBpeTokenizerConfiguration():
            training_result = train_byte_bpe_tokenizer(
                texts=texts,
                vocabulary_size=tokenizer_configuration.vocabulary_size,
                max_training_documents=tokenizer_configuration.max_training_documents,
                max_training_bytes=tokenizer_configuration.max_training_bytes,
                add_bos_token=tokenizer_configuration.add_bos_token,
                add_eos_token=tokenizer_configuration.add_eos_token,
                add_pad_token=tokenizer_configuration.add_pad_token,
                workers=tokenizer_configuration.training_workers,
                additional_special_tokens=tokenizer_configuration.additional_special_tokens,
            )
            return TrainedTokenizer(
                tokenizer=training_result.tokenizer,
                metrics={
                    "vocabulary_size": training_result.tokenizer.vocabulary_size,
                    "merge_count": training_result.tokenizer.merge_count,
                    "training_documents": training_result.training_document_count,
                    "training_bytes": training_result.training_bytes,
                    "training_tokens": training_result.training_tokens,
                    "max_training_documents": _optional_int_metric(
                        value=training_result.max_training_documents,
                    ),
                    "max_training_bytes": _optional_int_metric(
                        value=training_result.max_training_bytes,
                    ),
                    "bytes_per_token": training_result.bytes_per_token,
                    "workers": training_result.worker_count,
                    "pair_count_seconds": training_result.pair_count_seconds,
                    "merge_application_seconds": training_result.merge_application_seconds,
                },
            )
        case RustByteBpeTokenizerConfiguration():
            training_result = train_rust_byte_bpe_tokenizer(
                texts=texts,
                vocabulary_size=tokenizer_configuration.vocabulary_size,
                max_training_documents=tokenizer_configuration.max_training_documents,
                max_training_bytes=tokenizer_configuration.max_training_bytes,
                add_bos_token=tokenizer_configuration.add_bos_token,
                add_eos_token=tokenizer_configuration.add_eos_token,
                add_pad_token=tokenizer_configuration.add_pad_token,
                workers=tokenizer_configuration.training_workers,
                additional_special_tokens=tokenizer_configuration.additional_special_tokens,
            )
            return TrainedTokenizer(
                tokenizer=training_result.tokenizer,
                metrics={
                    "vocabulary_size": training_result.tokenizer.vocabulary_size,
                    "merge_count": training_result.tokenizer.merge_count,
                    "training_documents": training_result.training_document_count,
                    "training_bytes": training_result.training_bytes,
                    "training_tokens": training_result.training_tokens,
                    "max_training_documents": _optional_int_metric(
                        value=training_result.max_training_documents,
                    ),
                    "max_training_bytes": _optional_int_metric(
                        value=training_result.max_training_bytes,
                    ),
                    "bytes_per_token": training_result.bytes_per_token,
                    "workers": training_result.worker_count,
                    "training_seconds": training_result.training_seconds,
                },
            )


def load_tokenizer(
    directory: Path,
    tokenizer_configuration: TokenizerConfiguration,
) -> TextTokenizer:
    match tokenizer_configuration:
        case CharacterTokenizerConfiguration():
            return CharacterTokenizer.load(directory=directory)
        case ByteBpeTokenizerConfiguration():
            return ByteBpeTokenizer.load(directory=directory)
        case RustByteBpeTokenizerConfiguration():
            return RustByteBpeTokenizer.load(directory=directory)


def _optional_int_metric(value: int | None) -> int:
    if value is None:
        return 0
    return value
