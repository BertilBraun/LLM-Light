from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.data.document import RawDocumentRecord
from llm_lite.data.sources import load_inline_documents
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
        documents = load_inline_documents(dataset_configuration=experiment_configuration.dataset)
        data_path = artifact_directory / "documents.jsonl"
        with data_path.open("w", encoding="utf-8") as data_file:
            for document in documents:
                document_record = RawDocumentRecord(
                    document_id=document.document_id,
                    text=document.text,
                    metadata=document.metadata,
                )
                data_file.write(document_record.model_dump_json() + "\n")
        return StageOutput(
            files={"documents": "documents.jsonl"},
            metrics={"documents": len(documents)},
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)
