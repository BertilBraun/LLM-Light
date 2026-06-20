from collections.abc import Iterator

from llm_lite.data.document import Document, DocumentMetadataRecord, RawDocumentRecord
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName


def iter_raw_documents(registry: ArtifactRegistry) -> Iterator[Document]:
    documents_path = registry.artifact_directory(StageName.RAW_DATASET.value) / "documents.jsonl"
    with documents_path.open("r", encoding="utf-8") as documents_file:
        for line in documents_file:
            document_record = RawDocumentRecord.model_validate_json(line)
            yield Document(
                document_id=document_record.document_id,
                text=document_record.text,
                metadata=document_record.metadata,
            )


def iter_processed_document_texts(registry: ArtifactRegistry) -> Iterator[str]:
    processed_directory = registry.artifact_directory(StageName.PROCESSED_DATASET.value)
    metadata_path = processed_directory / "metadata.jsonl"
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            metadata_record = DocumentMetadataRecord.model_validate_json(line)
            with (processed_directory / metadata_record.path).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as document_file:
                yield document_file.read()
