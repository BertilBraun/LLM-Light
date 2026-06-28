from pathlib import Path

import pytest

from llm_lite.tokenizer.rust_byte_bpe import (
    RustByteBpeTokenizer,
    train_rust_byte_bpe_tokenizer,
)

pytest.importorskip("tokenizers")


def test_rust_byte_bpe_tokenizer_roundtrips_unicode_and_whitespace() -> None:
    text = "Hello Cafe\n\tindent :)"
    training_result = train_rust_byte_bpe_tokenizer(
        texts=[text],
        vocabulary_size=270,
        max_training_documents=1,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=4,
    )

    token_ids = training_result.tokenizer.encode(text=text, add_bos=True, add_eos=True)

    assert training_result.tokenizer.decode(token_ids) == text
    assert training_result.training_document_count == 1
    assert training_result.training_bytes == len(text.encode("utf-8"))
    assert training_result.tokenizer.vocabulary_size <= 270
    assert training_result.worker_count == 4


def test_rust_byte_bpe_training_is_deterministic() -> None:
    texts = ["abababab", "baba"]

    first_result = train_rust_byte_bpe_tokenizer(
        texts=texts,
        vocabulary_size=263,
        max_training_documents=2,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )
    second_result = train_rust_byte_bpe_tokenizer(
        texts=texts,
        vocabulary_size=263,
        max_training_documents=2,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )

    assert first_result.tokenizer.encode("abab", add_bos=True, add_eos=True) == (
        second_result.tokenizer.encode("abab", add_bos=True, add_eos=True)
    )
    assert first_result.tokenizer.merge_count == second_result.tokenizer.merge_count


def test_rust_byte_bpe_tokenizer_save_load_roundtrip(tmp_path: Path) -> None:
    text = "save load\n"
    training_result = train_rust_byte_bpe_tokenizer(
        texts=[text],
        vocabulary_size=265,
        max_training_documents=1,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )
    training_result.tokenizer.save(directory=tmp_path)

    loaded_tokenizer = RustByteBpeTokenizer.load(directory=tmp_path)

    assert loaded_tokenizer.encode(text=text, add_bos=True, add_eos=True) == (
        training_result.tokenizer.encode(text=text, add_bos=True, add_eos=True)
    )
    assert (
        loaded_tokenizer.decode(
            loaded_tokenizer.encode(text=text, add_bos=True, add_eos=True),
        )
        == text
    )


def test_rust_byte_bpe_tokenizer_encodes_additional_special_token_atomically() -> None:
    training_result = train_rust_byte_bpe_tokenizer(
        texts=["prefixsuffix"],
        vocabulary_size=270,
        max_training_documents=1,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
        additional_special_tokens=("<fim_middle>",),
    )
    tokenizer = training_result.tokenizer

    token_ids = tokenizer.encode(text="prefix<fim_middle>suffix", add_bos=False, add_eos=False)

    assert tokenizer.backend_tokenizer.token_to_id("<fim_middle>") in token_ids
    assert token_ids.count(tokenizer.backend_tokenizer.token_to_id("<fim_middle>")) == 1
    assert tokenizer.decode(token_ids) == "prefixsuffix"


def test_rust_byte_bpe_training_reads_only_bounded_sample() -> None:
    reads = 0

    def iter_texts():
        nonlocal reads
        for text in ("abababab", "bcbcbcbc", "should-not-read"):
            reads += 1
            yield text

    training_result = train_rust_byte_bpe_tokenizer(
        texts=iter_texts(),
        vocabulary_size=263,
        max_training_documents=2,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )

    assert reads == 2
    assert training_result.training_document_count == 2
