import io
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from llm_lite.data.document import Document
from llm_lite.pipeline.progress import track_progress

UNSPLIT_DIRECTORY_NAME = "unsplit"


class TextShardSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    split: str
    documents: int = Field(ge=0)
    bytes: int = Field(ge=0)
    characters: int = Field(ge=0)
    shards: int = Field(ge=0)


class TextShardCorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    format_version: int
    shard_document_limit: int = Field(gt=0)
    splits: tuple[TextShardSplitManifest, ...]


@dataclass
class TextShardSplitCounters:
    documents: int = 0
    bytes: int = 0
    characters: int = 0
    shards: int = 0


@dataclass(frozen=True)
class TextShardReference:
    split: str | None
    path: Path


class TextShardWriter:
    def __init__(
        self,
        artifact_directory: Path,
        shard_document_limit: int,
        shard_name_prefix: str,
    ) -> None:
        self.artifact_directory = artifact_directory
        self.shard_document_limit = shard_document_limit
        self.shard_name_prefix = shard_name_prefix
        self.open_shards: dict[str, tarfile.TarFile] = {}
        self.current_shard_documents: dict[str, int] = {}
        self.current_shard_indices: dict[str, int] = {}
        self.split_counters: dict[str, TextShardSplitCounters] = {}
        if self.shard_document_limit <= 0:
            raise ValueError("Text shard document limit must be positive.")

    def append(self, document: Document) -> None:
        split_name = split_directory_name(split=document.split)
        shard_file = self._shard_file(split_name=split_name)
        content_bytes = document.text.encode("utf-8")
        member_name = f"{_safe_document_id(document_id=document.document_id)}.txt"
        tar_info = tarfile.TarInfo(name=member_name)
        tar_info.size = len(content_bytes)
        shard_file.addfile(tarinfo=tar_info, fileobj=io.BytesIO(content_bytes))
        self.current_shard_documents[split_name] += 1
        counters = self._split_counters(split_name=split_name)
        counters.documents += 1
        counters.bytes += len(content_bytes)
        counters.characters += len(document.text)

    def close(self) -> TextShardCorpusManifest:
        for shard_file in self.open_shards.values():
            shard_file.close()
        self.open_shards = {}
        corpus_manifest = TextShardCorpusManifest(
            format_version=1,
            shard_document_limit=self.shard_document_limit,
            splits=tuple(
                TextShardSplitManifest(
                    split=split_name,
                    documents=counters.documents,
                    bytes=counters.bytes,
                    characters=counters.characters,
                    shards=counters.shards,
                )
                for split_name, counters in sorted(self.split_counters.items())
            ),
        )
        (self.artifact_directory / "corpus.json").write_text(
            corpus_manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return corpus_manifest

    def _shard_file(self, split_name: str) -> tarfile.TarFile:
        if self._needs_next_shard(split_name=split_name):
            self._open_next_shard(split_name=split_name)
        shard_file = self.open_shards[split_name]
        return shard_file

    def _needs_next_shard(self, split_name: str) -> bool:
        current_shard_file = self.open_shards.get(split_name)
        if current_shard_file is None:
            return True
        return self.current_shard_documents[split_name] >= self.shard_document_limit

    def _open_next_shard(self, split_name: str) -> None:
        current_shard_file = self.open_shards.get(split_name)
        if current_shard_file is not None:
            current_shard_file.close()
        split_directory = self.artifact_directory / split_name
        split_directory.mkdir(parents=True, exist_ok=True)
        shard_index = self.current_shard_indices.get(split_name, 0)
        shard_path = split_directory / f"{self.shard_name_prefix}shard_{shard_index:06d}.tar.gz"
        self.open_shards[split_name] = tarfile.open(shard_path, mode="w:gz")
        self.current_shard_documents[split_name] = 0
        self.current_shard_indices[split_name] = shard_index + 1
        self._split_counters(split_name=split_name).shards += 1

    def _split_counters(self, split_name: str) -> TextShardSplitCounters:
        counters = self.split_counters.get(split_name)
        if counters is not None:
            return counters
        counters = TextShardSplitCounters()
        self.split_counters[split_name] = counters
        return counters


def write_text_shards(
    documents: Iterator[Document],
    artifact_directory: Path,
    shard_document_limit: int,
    progress_description: str | None = None,
    progress_total: int | None = None,
) -> TextShardCorpusManifest:
    writer = TextShardWriter(
        artifact_directory=artifact_directory,
        shard_document_limit=shard_document_limit,
        shard_name_prefix="",
    )
    document_iterator = documents
    if progress_description is not None:
        document_iterator = track_progress(
            documents,
            description=progress_description,
            total=progress_total,
            unit="doc",
        )
    for document in document_iterator:
        writer.append(document=document)
    return writer.close()


def iter_text_shard_documents(
    artifact_directory: Path,
    split: str | None,
) -> Iterator[Document]:
    split_directories = _split_directories(artifact_directory=artifact_directory, split=split)
    for split_directory in split_directories:
        document_split = (
            None if split_directory.name == UNSPLIT_DIRECTORY_NAME else split_directory.name
        )
        for shard_path in sorted(split_directory.glob("*shard_*.tar.gz")):
            yield from iter_text_shard_reference_documents(
                shard_reference=TextShardReference(
                    split=document_split,
                    path=shard_path,
                ),
            )


def iter_text_shard_texts(artifact_directory: Path, split: str | None) -> Iterator[str]:
    for document in iter_text_shard_documents(artifact_directory=artifact_directory, split=split):
        yield document.text


def text_shard_references(
    artifact_directory: Path,
    split: str | None,
) -> tuple[TextShardReference, ...]:
    references: list[TextShardReference] = []
    split_directories = _split_directories(artifact_directory=artifact_directory, split=split)
    for split_directory in split_directories:
        document_split = (
            None if split_directory.name == UNSPLIT_DIRECTORY_NAME else split_directory.name
        )
        for shard_path in sorted(split_directory.glob("*shard_*.tar.gz")):
            references.append(TextShardReference(split=document_split, path=shard_path))
    return tuple(references)


def iter_text_shard_reference_documents(
    shard_reference: TextShardReference,
) -> Iterator[Document]:
    with tarfile.open(shard_reference.path, mode="r:gz") as shard_file:
        for member in shard_file:
            if not member.isfile():
                continue
            extracted_file = shard_file.extractfile(member)
            if extracted_file is None:
                continue
            yield Document(
                document_id=Path(member.name).stem,
                text=extracted_file.read().decode("utf-8"),
                split=shard_reference.split,
            )


def load_text_shard_corpus_manifest(artifact_directory: Path) -> TextShardCorpusManifest:
    return TextShardCorpusManifest.model_validate_json(
        (artifact_directory / "corpus.json").read_text(encoding="utf-8"),
    )


def split_directory_name(split: str | None) -> str:
    if split is None:
        return UNSPLIT_DIRECTORY_NAME
    return split


def _split_directories(artifact_directory: Path, split: str | None) -> tuple[Path, ...]:
    if split is not None:
        split_directory = artifact_directory / split
        if not split_directory.exists():
            return ()
        return (split_directory,)
    return tuple(
        sorted(
            (child_path for child_path in artifact_directory.iterdir() if child_path.is_dir()),
            key=lambda child_path: child_path.name,
        ),
    )


def _safe_document_id(document_id: str) -> str:
    safe_document_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", document_id)
    if not safe_document_id:
        raise ValueError("Document id must contain at least one path-safe character.")
    return safe_document_id
