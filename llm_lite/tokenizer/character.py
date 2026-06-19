import json
from collections.abc import Sequence
from pathlib import Path


class CharacterTokenizer:
    def __init__(
        self,
        token_to_id: dict[str, int],
        bos_token: str | None,
        eos_token: str | None,
        pad_token: str | None,
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)

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
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        token_ids.extend(self.token_to_id[character] for character in text)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        special_tokens = {self.bos_token, self.eos_token, self.pad_token}
        return "".join(
            self.id_to_token[token_id]
            for token_id in token_ids
            if self.id_to_token[token_id] not in special_tokens
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer_data = {
            "token_to_id": self.token_to_id,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(tokenizer_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "CharacterTokenizer":
        tokenizer_data = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        return cls(
            token_to_id=tokenizer_data["token_to_id"],
            bos_token=tokenizer_data["bos_token"],
            eos_token=tokenizer_data["eos_token"],
            pad_token=tokenizer_data["pad_token"],
        )


def train_character_tokenizer(
    texts: list[str],
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
) -> CharacterTokenizer:
    token_to_id: dict[str, int] = {}
    bos_token = "<bos>" if add_bos_token else None
    eos_token = "<eos>" if add_eos_token else None
    pad_token = "<pad>" if add_pad_token else None
    for special_token in (bos_token, eos_token, pad_token):
        if special_token is not None:
            token_to_id[special_token] = len(token_to_id)
    for character in sorted(set("".join(texts))):
        token_to_id[character] = len(token_to_id)
    return CharacterTokenizer(
        token_to_id=token_to_id,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
    )
