from llm_lite.data.packing import pack_token_sequences


def test_pack_token_sequences_pads_to_context_plus_target() -> None:
    sequences = pack_token_sequences(
        tokenized_documents=[[1, 2, 3]],
        context_length=4,
        pad_token_id=0,
    )

    assert sequences[0].token_ids == (1, 2, 3, 0, 0)
