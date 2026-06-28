import json
from collections.abc import Iterable, Sequence
from pathlib import Path


class CharacterTokenizer:
    def __init__(
        self,
        token_to_id: dict[str, int],
        bos_token: str | None,
        eos_token: str | None,
        pad_token: str | None,
        additional_special_tokens: tuple[str, ...],
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.additional_special_tokens = additional_special_tokens

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
            if self.bos_token is None:
                raise ValueError("BOS token is not configured.")
            token_ids.append(self.bos_token_id)
        token_ids.extend(
            _encode_text_with_special_tokens(
                text=text,
                token_to_id=self.token_to_id,
                special_tokens=self.additional_special_tokens,
            ),
        )
        if add_eos:
            if self.eos_token is None:
                raise ValueError("EOS token is not configured.")
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        special_tokens = {
            special_token
            for special_token in (
                self.bos_token,
                self.eos_token,
                self.pad_token,
                *self.additional_special_tokens,
            )
            if special_token is not None
        }
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
            "additional_special_tokens": self.additional_special_tokens,
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
            additional_special_tokens=tuple(tokenizer_data.get("additional_special_tokens", ())),
        )


def train_character_tokenizer(
    texts: Iterable[str],
    add_bos_token: bool,
    add_eos_token: bool,
    add_pad_token: bool,
    additional_special_tokens: tuple[str, ...] = (),
) -> CharacterTokenizer:
    token_to_id: dict[str, int] = {}
    bos_token = "<bos>" if add_bos_token else None
    eos_token = "<eos>" if add_eos_token else None
    pad_token = "<pad>" if add_pad_token else None
    for special_token in (bos_token, eos_token, pad_token, *additional_special_tokens):
        if special_token is not None:
            token_to_id[special_token] = len(token_to_id)
    characters: set[str] = set()
    for text in texts:
        characters.update(text)
    for character in sorted(characters):
        token_to_id[character] = len(token_to_id)
    return CharacterTokenizer(
        token_to_id=token_to_id,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        additional_special_tokens=additional_special_tokens,
    )


def _encode_text_with_special_tokens(
    text: str,
    token_to_id: dict[str, int],
    special_tokens: tuple[str, ...],
) -> list[int]:
    token_ids: list[int] = []
    text_index = 0
    ordered_special_tokens = tuple(sorted(special_tokens, key=len, reverse=True))
    while text_index < len(text):
        matched_special_token = _matching_special_token(
            text=text,
            text_index=text_index,
            special_tokens=ordered_special_tokens,
        )
        if matched_special_token is not None:
            token_ids.append(token_to_id[matched_special_token])
            text_index += len(matched_special_token)
            continue
        token_ids.append(token_to_id[text[text_index]])
        text_index += 1
    return token_ids


def _matching_special_token(
    text: str,
    text_index: int,
    special_tokens: tuple[str, ...],
) -> str | None:
    for special_token in special_tokens:
        if text.startswith(special_token, text_index):
            return special_token
    return None
