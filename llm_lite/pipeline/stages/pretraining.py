import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import (
    CausalLanguageModelingObjectiveConfiguration,
    DistributedConfiguration,
    ExperimentFile,
    ModelConfiguration,
    TrainingObjective,
)
from llm_lite.data.datasets import load_packed_sequence_dataset
from llm_lite.evaluation.runner import run_configured_evaluators
from llm_lite.evaluation.tensorboard import (
    EVALUATION_TENSORBOARD_DIRECTORY_NAME,
    write_evaluation_metrics_to_tensorboard,
)
from llm_lite.model.factory import build_model
from llm_lite.model.parameters import model_parameter_summary
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.progress import console_log
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage
from llm_lite.tokenizer.loading import TextTokenizer, load_tokenizer
from llm_lite.training.checkpoint import latest_checkpoint
from llm_lite.training.objectives import CausalLanguageModelingObjectiveRunner
from llm_lite.training.trainer import (
    TrainingEvaluationCallback,
    train_model,
    train_model_distributed,
)

PRETRAINING_RECONSTRUCTION_CONTRACT_VERSION = 2


class PretrainingReconstructionContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: int
    model: ModelConfiguration
    objective: TrainingObjective
    causal_language_modeling: CausalLanguageModelingObjectiveConfiguration
    distributed: DistributedConfiguration


class PretrainingStage(BasePipelineStage):
    name: StageName = StageName.PRETRAINING
    parents: tuple[StageName, ...] = (StageName.PACKED_DATASET, StageName.TOKENIZER)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value=_pretraining_reconstruction_contract(
                experiment_configuration=experiment_configuration,
            ).model_dump(mode="json"),
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
        model = build_model(
            model_configuration=experiment_configuration.model,
            vocabulary_size=tokenizer.vocabulary_size,
        )
        parameter_summary = model_parameter_summary(model=model)
        evaluation_callback = _training_evaluation_callback(
            experiment_configuration=experiment_configuration,
            registry=registry,
            tokenizer=tokenizer,
            artifact_directory=artifact_directory,
        )
        if experiment_configuration.distributed.enabled:
            result = train_model_distributed(
                model=model,
                dataset=dataset,
                training_configuration=experiment_configuration.training,
                distributed_configuration=experiment_configuration.distributed,
                artifact_directory=artifact_directory,
                seed=experiment_configuration.experiment.seed,
                evaluation_callback=evaluation_callback,
                model_configuration_hash=hash_json_value(
                    value=experiment_configuration.model.model_dump(mode="json"),
                ),
                objective_runner=CausalLanguageModelingObjectiveRunner(
                    auxiliary_loss_weight=(
                        experiment_configuration.training.causal_language_modeling.auxiliary_loss_weight
                    ),
                    pad_token_id=tokenizer.pad_token_id,
                ),
            )
        else:
            result = train_model(
                model=model,
                dataset=dataset,
                training_configuration=experiment_configuration.training,
                artifact_directory=artifact_directory,
                seed=experiment_configuration.experiment.seed,
                evaluation_callback=evaluation_callback,
                objective_runner=CausalLanguageModelingObjectiveRunner(
                    auxiliary_loss_weight=(
                        experiment_configuration.training.causal_language_modeling.auxiliary_loss_weight
                    ),
                    pad_token_id=tokenizer.pad_token_id,
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
                "model_parameters": parameter_summary.total_parameters,
                "trainable_model_parameters": parameter_summary.trainable_parameters,
                "active_model_parameters": parameter_summary.active_parameters,
                "trainable_active_model_parameters": (
                    parameter_summary.trainable_active_parameters
                ),
                "requested_maximum_steps": experiment_configuration.training.maximum_steps,
                "distributed_world_size": experiment_configuration.distributed.world_size,
                "distributed_strategy": experiment_configuration.distributed.strategy.value,
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

    def continuation_action(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
    ) -> str | None:
        checkpoint_state = latest_checkpoint(
            checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
            / "checkpoints",
        )
        if checkpoint_state is None:
            return None
        maximum_steps = experiment_configuration.training.maximum_steps
        if checkpoint_state.step < maximum_steps:
            return f"resume from step {checkpoint_state.step} to {maximum_steps}"
        return None

    def interrupted_action(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
    ) -> str | None:
        checkpoint_state = latest_checkpoint(
            checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
            / "checkpoints",
        )
        if checkpoint_state is None:
            return None
        maximum_steps = experiment_configuration.training.maximum_steps
        if checkpoint_state.step < maximum_steps:
            return f"resume from step {checkpoint_state.step} to {maximum_steps}"
        return f"recover checkpoint at step {checkpoint_state.step}"


def _pretraining_reconstruction_contract(
    experiment_configuration: ExperimentFile,
) -> PretrainingReconstructionContract:
    return PretrainingReconstructionContract(
        contract_version=PRETRAINING_RECONSTRUCTION_CONTRACT_VERSION,
        model=experiment_configuration.model,
        objective=experiment_configuration.training.objective,
        causal_language_modeling=experiment_configuration.training.causal_language_modeling,
        distributed=experiment_configuration.distributed,
    )


def _training_evaluation_callback(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    tokenizer: TextTokenizer,
    artifact_directory: Path,
) -> TrainingEvaluationCallback | None:
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
        write_evaluation_metrics_to_tensorboard(
            tensorboard_directory=artifact_directory / EVALUATION_TENSORBOARD_DIRECTORY_NAME,
            metrics=evaluation_result.metrics,
            step=step,
        )
        _print_training_evaluation(step=step, metrics=evaluation_result.metrics)
        return evaluation_path

    return run_training_evaluation


def _print_training_evaluation(
    step: int,
    metrics: dict[str, int | float | str | bool],
) -> None:
    if not metrics:
        console_log(f"[train-eval] step={step} no configured evaluator metrics")
        return
    formatted_metrics = " ".join(
        f"{metric_name}={metric_value}" for metric_name, metric_value in sorted(metrics.items())
    )
    console_log(f"[train-eval] step={step} {formatted_metrics}")
