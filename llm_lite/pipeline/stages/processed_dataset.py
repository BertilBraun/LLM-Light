from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.data.document import DocumentMetadataRecord
from llm_lite.data.preprocessing import preprocess_documents
from llm_lite.pipeline.hashing import hash_model
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import compatible_skip_action
from llm_lite.pipeline.stages.io import iter_raw_documents


@dataclass(frozen=True)
class ProcessedDatasetStage:
    name: StageName = StageName.PROCESSED_DATASET
    parents: tuple[StageName, ...] = (StageName.RAW_DATASET,)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_model(model=experiment_configuration.preprocessing)

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        data_directory = artifact_directory / "documents"
        data_directory.mkdir(parents=True, exist_ok=True)
        metadata_path = artifact_directory / "metadata.jsonl"
        preprocessing_result = preprocess_documents(
            documents=iter_raw_documents(registry=registry),
            transforms=experiment_configuration.preprocessing.transforms,
        )
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            for document_index, document in enumerate(preprocessing_result.documents):
                document_path = f"documents/document_{document_index:08d}.txt"
                with (artifact_directory / document_path).open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as document_file:
                    document_file.write(document.text)
                metadata_record = DocumentMetadataRecord(
                    document_id=document.document_id,
                    path=document_path,
                    metadata=document.metadata,
                )
                metadata_file.write(metadata_record.model_dump_json() + "\n")
        counters = preprocessing_result.counters
        return StageOutput(
            files={"metadata": "metadata.jsonl", "documents": "documents"},
            metrics={
                "input_documents": counters.input_documents,
                "output_documents": counters.output_documents,
                "rejected_documents": counters.rejected_documents,
                "input_bytes": counters.input_bytes,
                "output_bytes": counters.output_bytes,
            },
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)
