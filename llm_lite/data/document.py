from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


class RawDocumentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


class DocumentMetadataRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    path: str
    metadata: dict[str, str | int | float | bool]
