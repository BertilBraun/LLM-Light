from collections.abc import Iterable, Iterator

from llm_lite.data.datasets import PackedSequence


def pack_token_sequences(
    tokenized_document_stream: Iterable[list[int]],
    context_length: int,
    pad_token_id: int,
) -> Iterator[PackedSequence]:
    for token_ids in tokenized_document_stream:
        if len(token_ids) < 2:
            continue
        start_index = 0
        while start_index < len(token_ids) - 1:
            sequence = token_ids[start_index : start_index + context_length + 1]
            if len(sequence) < context_length + 1:
                sequence = sequence + [pad_token_id] * (context_length + 1 - len(sequence))
            yield PackedSequence(token_ids=tuple(sequence))
            start_index += context_length
