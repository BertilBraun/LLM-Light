import json
from collections.abc import Iterator

from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName


def iter_document_texts(registry: ArtifactRegistry) -> Iterator[str]:
    documents_path = registry.artifact_directory(StageName.RAW_DATASET.value) / "documents.jsonl"
    with documents_path.open("r", encoding="utf-8") as documents_file:
        for line in documents_file:
            document_data = json.loads(line)
            yield document_data["text"]
