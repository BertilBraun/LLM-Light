import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ByteToken = tuple[int, ...]
BytePair = tuple[ByteToken, ByteToken]


@dataclass
class ByteBpeTrainingCorpus:
    documents: list[list[ByteToken]]
    bytes: int


@dataclass(frozen=True)
class ByteBpeTrainingResult:
    tokenizer: "ByteBpeTokenizer"
    training_document_count: int
    training_bytes: int
    training_tokens: int
    max_training_documents: int | None
    max_training_bytes: int | None

    @property
    def bytes_per_token(self) -> float:
        if self.training_tokens == 0:
            return 0.0
        return self.training_bytes / self.training_tokens


class ByteBpeTokenizer:
    def __init__(
        self,
        token_to_id: dict[str, int],
        byte_token_to_id: dict[ByteToken, int],
        merge_rules: tuple[BytePair, ...],
        bos_token: str | None,
        eos_token: str | None,
        pad_token: str | None,
    ) -> None:
        self.token_to_id = token_to_id
        self.byte_token_to_id = byte_token_to_id
        self.id_to_byte_token = {
            token_id: byte_token for byte_token, token_id in byte_token_to_id.items()
        }
        self.merge_rules = merge_rules
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id) + len(self.byte_token_to_id)

    @property
    def merge_count(self) -> int:
        return len(self.merge_rules)

    @property
    def bos_token_id(self) -> int:
        if self.bos_token is None:
            raise ValueError("BOS token is not configured.")
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        if self.eos_token is None:
            raise ValueError("EOS token is not configured.")
        return self.token_to_id[self.eos_token]

    @property
    def pad_token_id(self) -> int | None:
        if self.pad_token is None:
            return None
        return self.token_to_id[self.pad_token]

    def encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]:
        byte_tokens = _byte_tokens(text=text)
        for merge_rule in self.merge_rules:
            byte_tokens = _merge_pair(byte_tokens=byte_tokens, merge_rule=merge_rule)
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        token_ids.extend(self.byte_token_to_id[byte_token] for byte_token in byte_tokens)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        output_bytes = bytearray()
        special_token_ids = {
            self.token_to_id[special_token]
            for special_token in (self.bos_token, self.eos_token, self.pad_token)
            if special_token is not None
        }
        for token_id in token_ids:
            if token_id in special_token_ids:
                continue
            output_bytes.extend(self.id_to_byte_token[token_id])
        return bytes(output_bytes).decode("utf-8", errors="replace")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer_data = {
            "format": "byte_bpe",
            "token_to_id": self.token_to_id,
            "byte_token_to_id": [
                {"bytes": list(byte_token), "token_id": token_id}
                for byte_token, token_id in sorted(
                    self.byte_token_to_id.items(),
                    key=lambda item: item[1],
                )
            ],
            "merge_rules": [
                {"left": list(left_token), "right": list(right_token)}
                for left_token, right_token in self.merge_rules
            ],
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(tokenizer_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "ByteBpeTokenizer":
        tokenizer_data = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        byte_token_to_id = {
            tuple(byte_token_record["bytes"]): byte_token_record["token_id"]
            for byte_token_record in tokenizer_data["byte_token_to_id"]
        }
        merge_rules = tuple(
            (
                tuple(merge_rule["left"]),
                tuple(merge_rule["right"]),
            )
            for merge_rule in tokenizer_data["merge_rules"]
        )
        return cls(
            token_to_id=tokenizer_data["token_to_id"],
            byte_token_to_id=byte_token_to_id,
            merge_rules=merge_rules,
            bos_token=tokenizer_data["bos_token"],
            eos_token=tokenizer_data["eos_token"],
            pad_token=tokenizer_data["pad_token"],
        )


def train_byte_bpe_tokenizer(
    texts: Iterable[str],
    vocabulary_size: int,
    max_training_documents: int | None,
    max_training_bytes: int | None,
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
) -> ByteBpeTrainingResult:
    token_to_id = _special_token_to_id(
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        add_pad_token=add_pad_token,
    )
    minimum_vocabulary_size = len(token_to_id) + 256
    if vocabulary_size < minimum_vocabulary_size:
        raise ValueError("Byte BPE vocabulary size must include special tokens and 256 bytes.")
    corpus_sample = _bounded_training_corpus(
        texts=texts,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
    )
    print(
        "[tokenizer] byte_bpe sample "
        f"documents={len(corpus_sample.documents)} "
        f"bytes={corpus_sample.bytes} "
        f"target_vocabulary_size={vocabulary_size}",
        flush=True,
    )
    merge_rules: list[BytePair] = []
    byte_token_to_id = _initial_byte_token_to_id(starting_index=len(token_to_id))
    while len(token_to_id) + len(byte_token_to_id) < vocabulary_size:
        pair_counts = _pair_counts(corpus=corpus_sample.documents)
        if not pair_counts:
            break
        best_pair = _best_pair(pair_counts=pair_counts)
        merged_token = best_pair[0] + best_pair[1]
        if merged_token in byte_token_to_id:
            break
        byte_token_to_id[merged_token] = len(token_to_id) + len(byte_token_to_id)
        merge_rules.append(best_pair)
        corpus_sample.documents = [
            _merge_pair(byte_tokens=document, merge_rule=best_pair)
            for document in corpus_sample.documents
        ]
        if len(merge_rules) % 100 == 0:
            print(
                "[tokenizer] byte_bpe "
                f"merges={len(merge_rules)} "
                f"vocabulary_size={len(token_to_id) + len(byte_token_to_id)}",
                flush=True,
            )
    training_tokens = sum(len(document) for document in corpus_sample.documents)
    print(
        "[tokenizer] byte_bpe complete "
        f"merges={len(merge_rules)} "
        f"training_tokens={training_tokens} "
        f"bytes_per_token={corpus_sample.bytes / max(training_tokens, 1):.4f}",
        flush=True,
    )
    tokenizer = ByteBpeTokenizer(
        token_to_id=token_to_id,
        byte_token_to_id=byte_token_to_id,
        merge_rules=tuple(merge_rules),
        bos_token="<bos>" if add_bos_token else None,
        eos_token="<eos>" if add_eos_token else None,
        pad_token="<pad>" if add_pad_token else None,
    )
    return ByteBpeTrainingResult(
        tokenizer=tokenizer,
        training_document_count=len(corpus_sample.documents),
        training_bytes=corpus_sample.bytes,
        training_tokens=training_tokens,
        max_training_documents=max_training_documents,
        max_training_bytes=max_training_bytes,
    )


def _special_token_to_id(
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
) -> dict[str, int]:
    token_to_id: dict[str, int] = {}
    for special_token in (
        "<bos>" if add_bos_token else None,
        "<eos>" if add_eos_token else None,
        "<pad>" if add_pad_token else None,
    ):
        if special_token is not None:
            token_to_id[special_token] = len(token_to_id)
    return token_to_id


def _initial_byte_token_to_id(starting_index: int) -> dict[ByteToken, int]:
    return {(byte_value,): starting_index + byte_value for byte_value in range(256)}


def _byte_tokens(text: str) -> list[ByteToken]:
    return [(byte_value,) for byte_value in text.encode("utf-8")]


def _bounded_training_corpus(
    texts: Iterable[str],
    max_training_documents: int | None,
    max_training_bytes: int | None,
) -> ByteBpeTrainingCorpus:
    if max_training_documents is None and max_training_bytes is None:
        raise ValueError("Byte BPE training sample must be bounded.")
    documents: list[list[ByteToken]] = []
    sampled_bytes = 0
    text_iterator = iter(texts)
    while max_training_documents is None or len(documents) < max_training_documents:
        try:
            text = next(text_iterator)
        except StopIteration:
            break
        text_bytes = text.encode("utf-8")
        if max_training_bytes is not None and sampled_bytes + len(text_bytes) > max_training_bytes:
            if documents:
                break
            raise ValueError("First Byte BPE training document exceeds max_training_bytes.")
        documents.append([(byte_value,) for byte_value in text_bytes])
        sampled_bytes += len(text_bytes)
    if not documents:
        raise ValueError("Byte BPE training sample is empty.")
    return ByteBpeTrainingCorpus(documents=documents, bytes=sampled_bytes)


def _pair_counts(corpus: list[list[ByteToken]]) -> Counter[BytePair]:
    pair_counts: Counter[BytePair] = Counter()
    for document in corpus:
        for token_index in range(len(document) - 1):
            pair_counts[(document[token_index], document[token_index + 1])] += 1
    return pair_counts


def _best_pair(pair_counts: Counter[BytePair]) -> BytePair:
    return min(pair_counts, key=lambda pair: (-pair_counts[pair], pair[0], pair[1]))


def _merge_pair(byte_tokens: list[ByteToken], merge_rule: BytePair) -> list[ByteToken]:
    merged_tokens: list[ByteToken] = []
    token_index = 0
    while token_index < len(byte_tokens):
        next_index = token_index + 1
        if (
            next_index < len(byte_tokens)
            and (
                byte_tokens[token_index],
                byte_tokens[next_index],
            )
            == merge_rule
        ):
            merged_tokens.append(byte_tokens[token_index] + byte_tokens[next_index])
            token_index += 2
        else:
            merged_tokens.append(byte_tokens[token_index])
            token_index += 1
    return merged_tokens
