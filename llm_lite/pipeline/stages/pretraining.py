import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from llm_lite.config.models import ExperimentFile
from llm_lite.data.datasets import load_packed_sequence_dataset
from llm_lite.evaluation.runner import run_configured_evaluators
from llm_lite.model.gpt import DenseGpt
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.tokenizer.loading import TextTokenizer, load_tokenizer
from llm_lite.training.checkpoint import latest_checkpoint
from llm_lite.training.trainer import train_model


@dataclass(frozen=True)
class PretrainingStage:
    name: StageName = StageName.PRETRAINING
    parents: tuple[StageName, ...] = (StageName.PACKED_DATASET, StageName.TOKENIZER)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "logging_schema_version": 1,
                "model": experiment_configuration.model.model_dump(mode="json"),
                "training": experiment_configuration.training.model_dump(mode="json"),
            },
        )

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        tokenizer = load_tokenizer(
            directory=registry.artifact_directory(StageName.TOKENIZER.value),
            tokenizer_configuration=experiment_configuration.tokenizer,
        )
        dataset = load_packed_sequence_dataset(
            artifact_directory=registry.artifact_directory(StageName.PACKED_DATASET.value),
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
            seed=experiment_configuration.experiment.seed,
            evaluation_callback=_training_evaluation_callback(
                experiment_configuration=experiment_configuration,
                registry=registry,
                tokenizer=tokenizer,
                artifact_directory=artifact_directory,
            ),
        )
        files = {
            "checkpoint": str(result.checkpoint_path.relative_to(artifact_directory)),
            "metrics": "metrics.jsonl",
            "tensorboard": "tensorboard",
        }
        if result.evaluation_path is not None:
            files["training_evaluations"] = str(
                result.evaluation_path.relative_to(artifact_directory),
            )
        return StageOutput(
            files=files,
            metrics={
                "final_step": result.final_step,
                "final_loss": result.final_loss,
                "resumed_from_step": result.resumed_from_step,
            },
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        checkpoint_state = latest_checkpoint(
            checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
            / "checkpoints",
        )
        if checkpoint_state is not None:
            return f"complete at step {checkpoint_state.step}, skip"
        return "compatible, skip"


def _training_evaluation_callback(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    tokenizer: TextTokenizer,
    artifact_directory: Path,
) -> Callable[[int, nn.Module], Path] | None:
    training_evaluation_configuration = experiment_configuration.training.evaluation
    if training_evaluation_configuration is None:
        return None
    evaluation_path = artifact_directory / "training_evaluations.jsonl"

    def run_training_evaluation(step: int, model: nn.Module) -> Path:
        evaluation_result = run_configured_evaluators(
            model=model,
            tokenizer=tokenizer,
            registry=registry,
            evaluation_configuration=training_evaluation_configuration.evaluators,
            inference_configuration=experiment_configuration.inference,
            packing_configuration=experiment_configuration.packing,
        )
        with evaluation_path.open("a", encoding="utf-8") as evaluation_file:
            evaluation_file.write(
                json.dumps(
                    {
                        "step": step,
                        "report": evaluation_result.report,
                        "metrics": evaluation_result.metrics,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        _print_training_evaluation(step=step, metrics=evaluation_result.metrics)
        return evaluation_path

    return run_training_evaluation


def _print_training_evaluation(
    step: int,
    metrics: dict[str, int | float | str | bool],
) -> None:
    if not metrics:
        print(f"[train-eval] step={step} no configured evaluator metrics", flush=True)
        return
    formatted_metrics = " ".join(
        f"{metric_name}={metric_value}" for metric_name, metric_value in sorted(metrics.items())
    )
    print(f"[train-eval] step={step} {formatted_metrics}", flush=True)
