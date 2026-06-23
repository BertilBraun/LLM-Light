from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.data.document import RawDocumentRecord
from llm_lite.data.sources import iter_dataset_documents
from llm_lite.pipeline.hashing import hash_model
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import compatible_skip_action


@dataclass(frozen=True)
class RawDatasetStage:
    name: StageName = StageName.RAW_DATASET
    parents: tuple[StageName, ...] = ()

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_model(model=experiment_configuration.dataset)

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        data_path = artifact_directory / "documents.jsonl"
        document_count = 0
        total_bytes = 0
        total_characters = 0
        with data_path.open("w", encoding="utf-8") as data_file:
            for document in iter_dataset_documents(
                experiment_configuration=experiment_configuration,
            ):
                document_count += 1
                document_bytes = len(document.text.encode("utf-8"))
                total_bytes += document_bytes
                total_characters += len(document.text)
                document_record = RawDocumentRecord(
                    document_id=document.document_id,
                    text=document.text,
                    metadata=document.metadata,
                )
                data_file.write(document_record.model_dump_json() + "\n")
        return StageOutput(
            files={"documents": "documents.jsonl"},
            metrics={
                "raw_documents": document_count,
                "processed_documents": document_count,
                "rejected_documents": 0,
                "total_characters": total_characters,
                "total_bytes": total_bytes,
            },
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)
