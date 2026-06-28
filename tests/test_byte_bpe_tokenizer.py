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


def test_byte_bpe_tokenizer_encodes_additional_special_token_atomically() -> None:
    training_result = train_byte_bpe_tokenizer(
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

    assert tokenizer.token_to_id["<fim_middle>"] in token_ids
    assert token_ids.count(tokenizer.token_to_id["<fim_middle>"]) == 1
    assert tokenizer.decode(token_ids) == "prefixsuffix"


def test_byte_bpe_encode_applies_ranked_merges() -> None:
    tokenizer = ByteBpeTokenizer(
        token_to_id={"<eos>": 0},
        byte_token_to_id={
            (97,): 1,
            (98,): 2,
            (120,): 3,
            (97, 98): 4,
            (97, 98, 97, 98): 5,
            (120, 120): 6,
        },
        merge_rules=(
            ((97,), (98,)),
            ((120,), (120,)),
            ((97, 98), (97, 98)),
        ),
        bos_token=None,
        eos_token="<eos>",
        pad_token=None,
    )

    token_ids = tokenizer.encode(text="abab", add_bos=False, add_eos=True)

    assert token_ids == [5, 0]


def test_byte_bpe_encode_matches_tokenizers_bpe_for_same_ascii_vocabulary() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    bpe_model = pytest.importorskip("tokenizers.models")
    texts = [
        "abababab xxxxx abba",
        "the quick brown fox jumps over the lazy dog",
        "banana bandana abracadabra",
    ]
    training_result = train_byte_bpe_tokenizer(
        texts=texts,
        vocabulary_size=320,
        max_training_documents=len(texts),
        max_training_bytes=None,
        add_bos_token=False,
        add_eos_token=False,
        add_pad_token=False,
        workers=1,
    )
    tokenizer = training_result.tokenizer
    official_tokenizer = tokenizers.Tokenizer(
        bpe_model.BPE(
            vocab={
                _byte_token_symbol(byte_token): token_id
                for byte_token, token_id in tokenizer.byte_token_to_id.items()
            },
            merges=[
                (_byte_token_symbol(left_token), _byte_token_symbol(right_token))
                for left_token, right_token in tokenizer.merge_rules
            ],
            unk_token=None,
        ),
    )

    for text in texts + ["abab banana dog", "xxxx abracadabra abba"]:
        assert tokenizer.encode(text=text, add_bos=False, add_eos=False) == list(
            official_tokenizer.encode(text, add_special_tokens=False).ids,
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


def _byte_token_symbol(byte_token: tuple[int, ...]) -> str:
    return bytes(byte_token).decode("latin-1")
