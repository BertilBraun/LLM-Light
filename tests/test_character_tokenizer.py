from llm_lite.tokenizer.character import train_character_tokenizer


def test_character_tokenizer_roundtrip() -> None:
    tokenizer = train_character_tokenizer(
        texts=["hello world\n"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
    )

    token_ids = tokenizer.encode(text="hello world\n", add_bos=True, add_eos=True)

    assert tokenizer.decode(token_ids) == "hello world\n"


def test_character_tokenizer_encodes_additional_special_token_atomically() -> None:
    tokenizer = train_character_tokenizer(
        texts=["prefixsuffix"],
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        additional_special_tokens=("<fim_middle>",),
    )

    token_ids = tokenizer.encode(text="prefix<fim_middle>suffix", add_bos=False, add_eos=False)

    assert tokenizer.token_to_id["<fim_middle>"] in token_ids
    assert token_ids.count(tokenizer.token_to_id["<fim_middle>"]) == 1
    assert tokenizer.decode(token_ids) == "prefixsuffix"
