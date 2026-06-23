from pathlib import Path

import pytest

from llm_lite.tokenizer.byte_bpe import ByteBpeTokenizer, train_byte_bpe_tokenizer


def test_byte_bpe_tokenizer_roundtrips_unicode_and_whitespace() -> None:
    text = "Hello Café\n\tindent 😀"
    training_result = train_byte_bpe_tokenizer(
        texts=[text],
        vocabulary_size=270,
        max_training_documents=1,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )

    token_ids = training_result.tokenizer.encode(text=text, add_bos=True, add_eos=True)

    assert training_result.tokenizer.decode(token_ids) == text
    assert training_result.training_document_count == 1
    assert training_result.training_bytes == len(text.encode("utf-8"))
    assert training_result.tokenizer.vocabulary_size == 270


def test_byte_bpe_training_is_deterministic() -> None:
    texts = ["abababab", "baba"]

    first_result = train_byte_bpe_tokenizer(
        texts=texts,
        vocabulary_size=263,
        max_training_documents=2,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )
    second_result = train_byte_bpe_tokenizer(
        texts=texts,
        vocabulary_size=263,
        max_training_documents=2,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )

    assert first_result.tokenizer.merge_rules == second_result.tokenizer.merge_rules
    assert first_result.tokenizer.byte_token_to_id == second_result.tokenizer.byte_token_to_id


def test_byte_bpe_tokenizer_save_load_roundtrip(tmp_path: Path) -> None:
    text = "save load\n"
    training_result = train_byte_bpe_tokenizer(
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

    loaded_tokenizer = ByteBpeTokenizer.load(directory=tmp_path)

    assert loaded_tokenizer.encode(text=text, add_bos=True, add_eos=True) == (
        training_result.tokenizer.encode(text=text, add_bos=True, add_eos=True)
    )
    assert (
        loaded_tokenizer.decode(
            loaded_tokenizer.encode(text=text, add_bos=True, add_eos=True),
        )
        == text
    )


def test_byte_bpe_decode_tolerates_invalid_generated_bytes() -> None:
    training_result = train_byte_bpe_tokenizer(
        texts=["valid text"],
        vocabulary_size=260,
        max_training_documents=1,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )
    invalid_byte_token_id = training_result.tokenizer.byte_token_to_id[(250,)]

    decoded_text = training_result.tokenizer.decode([invalid_byte_token_id])

    assert decoded_text == "\ufffd"


def test_byte_bpe_requires_byte_vocabulary_capacity() -> None:
    with pytest.raises(ValueError, match="256 bytes"):
        train_byte_bpe_tokenizer(
            texts=["text"],
            vocabulary_size=258,
            max_training_documents=1,
            max_training_bytes=None,
            add_bos_token=True,
            add_eos_token=True,
            add_pad_token=True,
            workers=1,
        )


def test_byte_bpe_training_reads_only_bounded_sample() -> None:
    reads = 0

    def iter_texts():
        nonlocal reads
        for text in ("abababab", "bcbcbcbc", "should-not-read"):
            reads += 1
            yield text

    training_result = train_byte_bpe_tokenizer(
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


def test_byte_bpe_training_respects_byte_bound() -> None:
    training_result = train_byte_bpe_tokenizer(
        texts=["abcd", "efgh", "ijkl"],
        vocabulary_size=263,
        max_training_documents=None,
        max_training_bytes=8,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )

    assert training_result.training_document_count == 2
    assert training_result.training_bytes == 8
