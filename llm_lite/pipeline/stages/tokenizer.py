from pathlib import Path

from llm_lite.config.models import (
    ByteBpeTokenizerConfiguration,
    ExperimentFile,
    RustByteBpeTokenizerConfiguration,
)
from llm_lite.pipeline.hashing import hash_model
from llm_lite.pipeline.progress import console_log
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage
from llm_lite.pipeline.stages.io import iter_processed_document_texts, tokenizer_training_split
from llm_lite.tokenizer.byte_bpe import train_byte_bpe_tokenizer_from_text_shards
from llm_lite.tokenizer.loading import train_tokenizer
from llm_lite.tokenizer.rust_byte_bpe import train_rust_byte_bpe_tokenizer_from_text_shards


class TokenizerStage(BasePipelineStage):
    name: StageName = StageName.TOKENIZER
    parents: tuple[StageName, ...] = (StageName.PROCESSED_DATASET,)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_model(model=experiment_configuration.tokenizer)

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        console_log("[tokenizer] selecting training split")
        split = tokenizer_training_split(registry=registry)
        console_log(f"[tokenizer] training_split={'all' if split is None else split}")
        tokenizer_configuration = experiment_configuration.tokenizer
        match tokenizer_configuration:
            case ByteBpeTokenizerConfiguration():
                console_log(
                    "[tokenizer] byte_bpe start "
                    f"vocabulary_size={tokenizer_configuration.vocabulary_size} "
                    f"max_training_documents={tokenizer_configuration.max_training_documents} "
                    f"max_training_bytes={tokenizer_configuration.max_training_bytes} "
                    f"workers={tokenizer_configuration.training_workers}"
                )
                training_result = train_byte_bpe_tokenizer_from_text_shards(
                    artifact_directory=registry.artifact_directory(
                        StageName.PROCESSED_DATASET.value,
                    ),
                    split=split,
                    vocabulary_size=tokenizer_configuration.vocabulary_size,
                    max_training_documents=tokenizer_configuration.max_training_documents,
                    max_training_bytes=tokenizer_configuration.max_training_bytes,
                    add_bos_token=tokenizer_configuration.add_bos_token,
                    add_eos_token=tokenizer_configuration.add_eos_token,
                    add_pad_token=tokenizer_configuration.add_pad_token,
                    workers=tokenizer_configuration.training_workers,
                )
                tokenizer = training_result.tokenizer
                metrics = {
                    "vocabulary_size": tokenizer.vocabulary_size,
                    "merge_count": tokenizer.merge_count,
                    "training_documents": training_result.training_document_count,
                    "training_bytes": training_result.training_bytes,
                    "training_tokens": training_result.training_tokens,
                    "max_training_documents": training_result.max_training_documents or 0,
                    "max_training_bytes": training_result.max_training_bytes or 0,
                    "bytes_per_token": training_result.bytes_per_token,
                    "workers": training_result.worker_count,
                    "pair_count_seconds": training_result.pair_count_seconds,
                    "merge_application_seconds": training_result.merge_application_seconds,
                    "tokenizer_merges_completed": tokenizer.merge_count,
                }
            case RustByteBpeTokenizerConfiguration():
                console_log(
                    "[tokenizer] rust_byte_bpe start "
                    f"vocabulary_size={tokenizer_configuration.vocabulary_size} "
                    f"max_training_documents={tokenizer_configuration.max_training_documents} "
                    f"max_training_bytes={tokenizer_configuration.max_training_bytes} "
                    f"workers={tokenizer_configuration.training_workers}"
                )
                training_result = train_rust_byte_bpe_tokenizer_from_text_shards(
                    artifact_directory=registry.artifact_directory(
                        StageName.PROCESSED_DATASET.value,
                    ),
                    split=split,
                    vocabulary_size=tokenizer_configuration.vocabulary_size,
                    max_training_documents=tokenizer_configuration.max_training_documents,
                    max_training_bytes=tokenizer_configuration.max_training_bytes,
                    add_bos_token=tokenizer_configuration.add_bos_token,
                    add_eos_token=tokenizer_configuration.add_eos_token,
                    add_pad_token=tokenizer_configuration.add_pad_token,
                    workers=tokenizer_configuration.training_workers,
                )
                tokenizer = training_result.tokenizer
                metrics = {
                    "vocabulary_size": tokenizer.vocabulary_size,
                    "merge_count": tokenizer.merge_count,
                    "training_documents": training_result.training_document_count,
                    "training_bytes": training_result.training_bytes,
                    "training_tokens": training_result.training_tokens,
                    "max_training_documents": training_result.max_training_documents or 0,
                    "max_training_bytes": training_result.max_training_bytes or 0,
                    "bytes_per_token": training_result.bytes_per_token,
                    "workers": training_result.worker_count,
                    "training_seconds": training_result.training_seconds,
                    "tokenizer_merges_completed": tokenizer.merge_count,
                }
            case _:
                trained_tokenizer = train_tokenizer(
                    texts=iter_processed_document_texts(
                        registry=registry,
                        split=split,
                    ),
                    tokenizer_configuration=tokenizer_configuration,
                )
                tokenizer = trained_tokenizer.tokenizer
                metrics = trained_tokenizer.metrics
        tokenizer.save(directory=artifact_directory)
        return StageOutput(
            files={"tokenizer": "tokenizer.json"},
            metrics=metrics,
        )
