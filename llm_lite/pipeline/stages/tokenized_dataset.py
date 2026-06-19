import json
from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import compatible_skip_action
from llm_lite.pipeline.stages.io import read_document_texts
from llm_lite.tokenizer.character import CharacterTokenizer


@dataclass(frozen=True)
class TokenizedDatasetStage:
    name: StageName = StageName.TOKENIZED_DATASET
    parents: tuple[StageName, ...] = (StageName.RAW_DATASET, StageName.TOKENIZER)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "packing_add_bos": experiment_configuration.packing.add_bos,
                "packing_add_eos": experiment_configuration.packing.add_eos,
            },
        )

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        tokenizer = CharacterTokenizer.load(
            directory=registry.artifact_directory(StageName.TOKENIZER.value),
        )
        documents = read_document_texts(registry=registry)
        tokenized_documents = [
            tokenizer.encode(
                text=document_text,
                add_bos=experiment_configuration.packing.add_bos,
                add_eos=experiment_configuration.packing.add_eos,
            )
            for document_text in documents
        ]
        tokens_path = artifact_directory / "tokens.json"
        tokens_path.write_text(json.dumps(tokenized_documents, indent=2), encoding="utf-8")
        return StageOutput(
            files={"tokens": "tokens.json"},
            metrics={
                "documents": len(tokenized_documents),
                "tokens": sum(map(len, tokenized_documents)),
            },
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)
