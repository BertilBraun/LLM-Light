import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import ExperimentFile
from llm_lite.data.datasets import PackedSequence, PackedSequenceDataset
from llm_lite.data.packing import pack_token_sequences
from llm_lite.data.sources import load_inline_documents
from llm_lite.inference.naive import generate_greedy
from llm_lite.model.gpt import DenseGpt
from llm_lite.pipeline.hashing import hash_json_value, hash_model
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import ORDERED_STAGE_NAMES, StageName
from llm_lite.tokenizer.character import CharacterTokenizer, train_character_tokenizer
from llm_lite.training.checkpoint import latest_checkpoint, load_checkpoint
from llm_lite.training.trainer import train_model
from llm_lite.utilities.random import seed_everything


@dataclass(frozen=True)
class StageReview:
    stage_name: StageName
    action: str


def run_pipeline(
    configuration_path: Path, dry_run: bool, force_stages: tuple[StageName, ...]
) -> int:
    experiment_configuration = load_experiment_configuration(configuration_path=configuration_path)
    seed_everything(seed=experiment_configuration.experiment.seed)
    registry = ArtifactRegistry(run_directory=experiment_configuration.experiment.output_dir)
    force_stage_names = _expanded_force_stages(force_stages=force_stages)
    review = _review_pipeline(
        experiment_configuration=experiment_configuration,
        registry=registry,
        force_stage_names=force_stage_names,
    )
    _print_review(review=review)
    if dry_run:
        return 0
    experiment_configuration.experiment.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_configuration_path = (
        experiment_configuration.experiment.output_dir / "resolved_config.json"
    )
    resolved_configuration_path.write_text(
        experiment_configuration.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _execute_pipeline(
        experiment_configuration=experiment_configuration,
        registry=registry,
        force_stage_names=force_stage_names,
    )
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", required=True, type=Path)
    argument_parser.add_argument("--dry-run", action="store_true")
    argument_parser.add_argument(
        "--force",
        action="append",
        choices=[stage.value for stage in ORDERED_STAGE_NAMES],
        const=StageName.RAW_DATASET.value,
        nargs="?",
    )
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    force_values = tuple(arguments.force) if arguments.force is not None else ()
    force_stages = tuple(StageName(force_value) for force_value in force_values)
    return run_pipeline(
        configuration_path=arguments.config,
        dry_run=arguments.dry_run,
        force_stages=force_stages,
    )


def _review_pipeline(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    force_stage_names: set[StageName],
) -> list[StageReview]:
    review: list[StageReview] = []
    for stage_name in ORDERED_STAGE_NAMES:
        configuration_hash = _stage_configuration_hash(
            experiment_configuration=experiment_configuration,
            stage_name=stage_name,
        )
        parent_hashes = _parent_hashes(registry=registry, stage_name=stage_name)
        if stage_name in force_stage_names:
            review.append(StageReview(stage_name=stage_name, action="force recompute"))
        elif registry.is_compatible(
            artifact_type=stage_name.value,
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
        ):
            action = _compatible_action(registry=registry, stage_name=stage_name)
            review.append(StageReview(stage_name=stage_name, action=action))
        else:
            review.append(StageReview(stage_name=stage_name, action="execute"))
    return review


def _execute_pipeline(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    force_stage_names: set[StageName],
) -> None:
    for stage_name in ORDERED_STAGE_NAMES:
        configuration_hash = _stage_configuration_hash(
            experiment_configuration=experiment_configuration,
            stage_name=stage_name,
        )
        parent_hashes = _parent_hashes(registry=registry, stage_name=stage_name)
        compatible = registry.is_compatible(
            artifact_type=stage_name.value,
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
        )
        if compatible and stage_name not in force_stage_names:
            continue
        artifact_directory = registry.artifact_directory(artifact_type=stage_name.value)
        if stage_name in force_stage_names and artifact_directory.exists():
            shutil.rmtree(artifact_directory)
            artifact_directory.mkdir(parents=True, exist_ok=True)
        registry.write_running_manifest(
            artifact_type=stage_name.value,
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
        )
        _run_stage(
            stage_name=stage_name,
            experiment_configuration=experiment_configuration,
            registry=registry,
            artifact_directory=artifact_directory,
            configuration_hash=configuration_hash,
            parent_hashes=parent_hashes,
        )


def _run_stage(
    stage_name: StageName,
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    configuration_hash: str,
    parent_hashes: dict[str, str],
) -> None:
    match stage_name:
        case StageName.RAW_DATASET:
            _run_raw_dataset_stage(
                experiment_configuration=experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
                configuration_hash=configuration_hash,
                parent_hashes=parent_hashes,
            )
        case StageName.TOKENIZER:
            _run_tokenizer_stage(
                experiment_configuration=experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
                configuration_hash=configuration_hash,
                parent_hashes=parent_hashes,
            )
        case StageName.TOKENIZED_DATASET:
            _run_tokenized_dataset_stage(
                experiment_configuration=experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
                configuration_hash=configuration_hash,
                parent_hashes=parent_hashes,
            )
        case StageName.PACKED_DATASET:
            _run_packed_dataset_stage(
                experiment_configuration=experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
                configuration_hash=configuration_hash,
                parent_hashes=parent_hashes,
            )
        case StageName.PRETRAINING:
            _run_pretraining_stage(
                experiment_configuration=experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
                configuration_hash=configuration_hash,
                parent_hashes=parent_hashes,
            )
        case StageName.EVALUATION:
            _run_evaluation_stage(
                experiment_configuration=experiment_configuration,
                registry=registry,
                artifact_directory=artifact_directory,
                configuration_hash=configuration_hash,
                parent_hashes=parent_hashes,
            )


def _run_raw_dataset_stage(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    configuration_hash: str,
    parent_hashes: dict[str, str],
) -> None:
    documents = load_inline_documents(dataset_configuration=experiment_configuration.dataset)
    data_path = artifact_directory / "documents.jsonl"
    with data_path.open("w", encoding="utf-8") as data_file:
        for document in documents:
            data_file.write(
                json.dumps(
                    {
                        "document_id": document.document_id,
                        "text": document.text,
                        "metadata": document.metadata,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
    registry.write_complete_manifest(
        artifact_type=StageName.RAW_DATASET.value,
        configuration_hash=configuration_hash,
        parent_hashes=parent_hashes,
        files={"documents": "documents.jsonl"},
        metrics={"documents": len(documents)},
    )


def _run_tokenizer_stage(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    configuration_hash: str,
    parent_hashes: dict[str, str],
) -> None:
    documents = _read_document_texts(registry=registry)
    tokenizer = train_character_tokenizer(
        texts=documents,
        add_bos_token=experiment_configuration.tokenizer.add_bos_token,
        add_eos_token=experiment_configuration.tokenizer.add_eos_token,
        add_pad_token=experiment_configuration.tokenizer.add_pad_token,
    )
    tokenizer.save(directory=artifact_directory)
    registry.write_complete_manifest(
        artifact_type=StageName.TOKENIZER.value,
        configuration_hash=configuration_hash,
        parent_hashes=parent_hashes,
        files={"tokenizer": "tokenizer.json"},
        metrics={"vocabulary_size": tokenizer.vocabulary_size},
    )


def _run_tokenized_dataset_stage(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    configuration_hash: str,
    parent_hashes: dict[str, str],
) -> None:
    tokenizer = CharacterTokenizer.load(
        directory=registry.artifact_directory(StageName.TOKENIZER.value)
    )
    documents = _read_document_texts(registry=registry)
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
    registry.write_complete_manifest(
        artifact_type=StageName.TOKENIZED_DATASET.value,
        configuration_hash=configuration_hash,
        parent_hashes=parent_hashes,
        files={"tokens": "tokens.json"},
        metrics={
            "documents": len(tokenized_documents),
            "tokens": sum(map(len, tokenized_documents)),
        },
    )


def _run_packed_dataset_stage(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    configuration_hash: str,
    parent_hashes: dict[str, str],
) -> None:
    tokenizer = CharacterTokenizer.load(
        directory=registry.artifact_directory(StageName.TOKENIZER.value)
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Packing requires a configured pad token.")
    tokenized_documents = _read_tokenized_documents(registry=registry)
    sequences = pack_token_sequences(
        tokenized_documents=tokenized_documents,
        context_length=experiment_configuration.packing.context_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    sequences_path = artifact_directory / "sequences.pt"
    torch.save([sequence.token_ids for sequence in sequences], sequences_path)
    registry.write_complete_manifest(
        artifact_type=StageName.PACKED_DATASET.value,
        configuration_hash=configuration_hash,
        parent_hashes=parent_hashes,
        files={"sequences": "sequences.pt"},
        metrics={"sequences": len(sequences)},
    )


def _run_pretraining_stage(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    configuration_hash: str,
    parent_hashes: dict[str, str],
) -> None:
    tokenizer = CharacterTokenizer.load(
        directory=registry.artifact_directory(StageName.TOKENIZER.value)
    )
    sequences = torch.load(
        registry.artifact_directory(StageName.PACKED_DATASET.value) / "sequences.pt",
        weights_only=False,
    )
    dataset = PackedSequenceDataset(
        sequences=[PackedSequence(token_ids=tuple(token_ids)) for token_ids in sequences],
    )
    model = DenseGpt(
        model_configuration=experiment_configuration.model,
        vocabulary_size=tokenizer.vocabulary_size,
    )
    result = train_model(
        model=model,
        dataset=dataset,
        training_configuration=experiment_configuration.training,
        artifact_directory=artifact_directory,
    )
    registry.write_complete_manifest(
        artifact_type=StageName.PRETRAINING.value,
        configuration_hash=configuration_hash,
        parent_hashes=parent_hashes,
        files={
            "checkpoint": str(result.checkpoint_path.relative_to(artifact_directory)),
            "metrics": "metrics.jsonl",
        },
        metrics={
            "final_step": result.final_step,
            "final_loss": result.final_loss,
            "resumed_from_step": result.resumed_from_step,
        },
    )


def _run_evaluation_stage(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    configuration_hash: str,
    parent_hashes: dict[str, str],
) -> None:
    tokenizer = CharacterTokenizer.load(
        directory=registry.artifact_directory(StageName.TOKENIZER.value)
    )
    model = DenseGpt(
        model_configuration=experiment_configuration.model,
        vocabulary_size=tokenizer.vocabulary_size,
    )
    checkpoint_state = latest_checkpoint(
        checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
        / "checkpoints",
    )
    if checkpoint_state is None:
        raise ValueError("Evaluation requires a completed training checkpoint.")
    load_checkpoint(checkpoint_path=checkpoint_state.checkpoint_path, model=model, optimizer=None)
    generated_text = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        prompt=experiment_configuration.evaluation.prompt,
        maximum_new_tokens=experiment_configuration.inference.maximum_new_tokens,
    )
    expected_text = (
        experiment_configuration.evaluation.prompt
        + experiment_configuration.evaluation.expected_completion
    )
    passed = generated_text == expected_text
    report = {
        "passed": passed,
        "generated_text": generated_text,
        "expected_text": expected_text,
    }
    (artifact_directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    registry.write_complete_manifest(
        artifact_type=StageName.EVALUATION.value,
        configuration_hash=configuration_hash,
        parent_hashes=parent_hashes,
        files={"report": "report.json"},
        metrics={"passed": passed},
    )
    if not passed:
        raise ValueError("Exact reproduction evaluation failed.")


def _read_document_texts(registry: ArtifactRegistry) -> list[str]:
    documents_path = registry.artifact_directory(StageName.RAW_DATASET.value) / "documents.jsonl"
    texts: list[str] = []
    with documents_path.open("r", encoding="utf-8") as documents_file:
        for line in documents_file:
            document_data = json.loads(line)
            texts.append(document_data["text"])
    return texts


def _read_tokenized_documents(registry: ArtifactRegistry) -> list[list[int]]:
    tokens_path = registry.artifact_directory(StageName.TOKENIZED_DATASET.value) / "tokens.json"
    return json.loads(tokens_path.read_text(encoding="utf-8"))


def _stage_configuration_hash(
    experiment_configuration: ExperimentFile,
    stage_name: StageName,
) -> str:
    match stage_name:
        case StageName.RAW_DATASET:
            return hash_model(model=experiment_configuration.dataset)
        case StageName.TOKENIZER:
            return hash_model(model=experiment_configuration.tokenizer)
        case StageName.TOKENIZED_DATASET:
            return hash_json_value(
                value={
                    "packing_add_bos": experiment_configuration.packing.add_bos,
                    "packing_add_eos": experiment_configuration.packing.add_eos,
                },
            )
        case StageName.PACKED_DATASET:
            return hash_model(model=experiment_configuration.packing)
        case StageName.PRETRAINING:
            return hash_json_value(
                value={
                    "model": experiment_configuration.model.model_dump(mode="json"),
                    "training": experiment_configuration.training.model_dump(mode="json"),
                },
            )
        case StageName.EVALUATION:
            return hash_json_value(
                value={
                    "evaluation": experiment_configuration.evaluation.model_dump(mode="json"),
                    "inference": experiment_configuration.inference.model_dump(mode="json"),
                },
            )


def _parent_hashes(registry: ArtifactRegistry, stage_name: StageName) -> dict[str, str]:
    parent_stage_names = _parent_stage_names(stage_name=stage_name)
    parent_hashes: dict[str, str] = {}
    for parent_stage_name in parent_stage_names:
        manifest = registry.read_manifest(artifact_type=parent_stage_name.value)
        if manifest is not None:
            parent_hashes[parent_stage_name.value] = registry.artifact_identifier(
                artifact_type=parent_stage_name.value,
            )
    return parent_hashes


def _parent_stage_names(stage_name: StageName) -> tuple[StageName, ...]:
    match stage_name:
        case StageName.RAW_DATASET:
            return ()
        case StageName.TOKENIZER:
            return (StageName.RAW_DATASET,)
        case StageName.TOKENIZED_DATASET:
            return (StageName.RAW_DATASET, StageName.TOKENIZER)
        case StageName.PACKED_DATASET:
            return (StageName.TOKENIZED_DATASET,)
        case StageName.PRETRAINING:
            return (StageName.PACKED_DATASET, StageName.TOKENIZER)
        case StageName.EVALUATION:
            return (StageName.PRETRAINING, StageName.TOKENIZER)


def _expanded_force_stages(force_stages: tuple[StageName, ...]) -> set[StageName]:
    if not force_stages:
        return set()
    first_forced_index = min(ORDERED_STAGE_NAMES.index(stage_name) for stage_name in force_stages)
    return set(ORDERED_STAGE_NAMES[first_forced_index:])


def _compatible_action(registry: ArtifactRegistry, stage_name: StageName) -> str:
    if stage_name == StageName.PRETRAINING:
        checkpoint_state = latest_checkpoint(
            checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
            / "checkpoints",
        )
        if checkpoint_state is not None:
            return f"complete at step {checkpoint_state.step}, skip"
    return "compatible, skip"


def _print_review(review: list[StageReview]) -> None:
    for review_item in review:
        print(f"{review_item.stage_name.value:18} {review_item.action}")


if __name__ == "__main__":
    raise SystemExit(main())
