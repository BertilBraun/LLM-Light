import json

from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName


def read_document_texts(registry: ArtifactRegistry) -> list[str]:
    documents_path = registry.artifact_directory(StageName.RAW_DATASET.value) / "documents.jsonl"
    texts: list[str] = []
    with documents_path.open("r", encoding="utf-8") as documents_file:
        for line in documents_file:
            document_data = json.loads(line)
            texts.append(document_data["text"])
    return texts


def read_tokenized_documents(registry: ArtifactRegistry) -> list[list[int]]:
    tokens_path = registry.artifact_directory(StageName.TOKENIZED_DATASET.value) / "tokens.json"
    return json.loads(tokens_path.read_text(encoding="utf-8"))
