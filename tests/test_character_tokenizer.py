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
