from pathlib import Path

import pytest

from llm_lite.tokenizer.byte_bpe import ByteBpeTokenizer, train_byte_bpe_tokenizer


def test_byte_bpe_tokenizer_roundtrips_unicode_and_whitespace() -> None:
    text = "Hello Café\n\tindent 😀"
    training_result = train_byte_bpe_tokenizer(
        texts=[text],
        vocabulary_size=270,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
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
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )
    second_result = train_byte_bpe_tokenizer(
        texts=texts,
        vocabulary_size=263,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )

    assert first_result.tokenizer.merge_rules == second_result.tokenizer.merge_rules
    assert first_result.tokenizer.byte_token_to_id == second_result.tokenizer.byte_token_to_id


def test_byte_bpe_tokenizer_save_load_roundtrip(tmp_path: Path) -> None:
    text = "save load\n"
    training_result = train_byte_bpe_tokenizer(
        texts=[text],
        vocabulary_size=265,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
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


def test_byte_bpe_requires_byte_vocabulary_capacity() -> None:
    with pytest.raises(ValueError, match="256 bytes"):
        train_byte_bpe_tokenizer(
            texts=["text"],
            vocabulary_size=258,
            add_bos_token=True,
            add_eos_token=True,
            add_pad_token=True,
        )
