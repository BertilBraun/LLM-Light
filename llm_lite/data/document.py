from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    split: str | None
