import json
import os
from pathlib import Path

from torch import nn

from llm_lite.config.models import (
    EvaluationConfiguration,
    ExperimentFile,
    NoPostTrainingConfiguration,
)
from llm_lite.evaluation.runner import run_configured_evaluators
from llm_lite.evaluation.tensorboard import (
    EVALUATION_TENSORBOARD_DIRECTORY_NAME,
    write_evaluation_metrics_to_tensorboard,
)
from llm_lite.model.factory import build_model
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage
from llm_lite.tokenizer.loading import load_tokenizer
from llm_lite.training.checkpoint import load_checkpoint, load_latest_checkpoint

EVALUATION_CHECKPOINT_PATH_ENVIRONMENT = "LLM_LITE_EVALUATION_CHECKPOINT_PATH"
EVALUATION_CHECKPOINT_STEP_ENVIRONMENT = "LLM_LITE_EVALUATION_CHECKPOINT_STEP"


class EvaluationStage(BasePipelineStage):
    name: StageName = StageName.EVALUATION
    parents: tuple[StageName, ...] = (StageName.POST_TRAINING, StageName.TOKENIZER)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "evaluation": experiment_configuration.evaluation.model_dump(mode="json"),
                "inference": experiment_configuration.inference.model_dump(mode="json"),
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
        model = build_model(
            model_configuration=experiment_configuration.model,
            vocabulary_size=tokenizer.vocabulary_size,
        )
        checkpoint_step = _load_evaluation_checkpoint(
            experiment_configuration=experiment_configuration,
            registry=registry,
            model=model,
        )
        if checkpoint_step is None:
            raise ValueError("Evaluation requires a completed training checkpoint.")
        evaluation_result = run_configured_evaluators(
            model=model,
            tokenizer=tokenizer,
            registry=registry,
            evaluation_configuration=_evaluation_configuration(
                experiment_configuration=experiment_configuration,
            ),
            inference_configuration=experiment_configuration.inference,
            packing_configuration=experiment_configuration.packing,
        )
        (artifact_directory / "report.json").write_text(
            json.dumps(evaluation_result.report, indent=2),
            encoding="utf-8",
        )
        write_evaluation_metrics_to_tensorboard(
            tensorboard_directory=artifact_directory / EVALUATION_TENSORBOARD_DIRECTORY_NAME,
            metrics=evaluation_result.metrics,
            step=checkpoint_step,
        )
        return StageOutput(
            files={
                "report": "report.json",
                "tensorboard": EVALUATION_TENSORBOARD_DIRECTORY_NAME,
            },
            metrics=evaluation_result.metrics,
        )


def _evaluation_checkpoint_directory(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
) -> Path:
    match experiment_configuration.post_training:
        case NoPostTrainingConfiguration():
            return registry.artifact_directory(StageName.PRETRAINING.value) / "checkpoints"
        case _:
            return registry.artifact_directory(StageName.POST_TRAINING.value) / "checkpoints"


def _load_evaluation_checkpoint(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    model: nn.Module,
) -> int | None:
    checkpoint_path_value = os.environ.get(EVALUATION_CHECKPOINT_PATH_ENVIRONMENT)
    if checkpoint_path_value is not None:
        loaded_step = load_checkpoint(
            checkpoint_path=Path(checkpoint_path_value),
            model=model,
            optimizer=None,
        )
        expected_step_value = os.environ.get(EVALUATION_CHECKPOINT_STEP_ENVIRONMENT)
        if expected_step_value is not None and loaded_step != int(expected_step_value):
            raise ValueError("Loaded checkpoint step does not match evaluation target.")
        return loaded_step
    return load_latest_checkpoint(
        checkpoint_directory=_evaluation_checkpoint_directory(
            experiment_configuration=experiment_configuration,
            registry=registry,
        ),
        model=model,
        optimizer=None,
    )


def _evaluation_configuration(
    experiment_configuration: ExperimentFile,
) -> EvaluationConfiguration:
    if os.environ.get(EVALUATION_CHECKPOINT_PATH_ENVIRONMENT) is None:
        return experiment_configuration.evaluation
    training_evaluation_configuration = experiment_configuration.training.evaluation
    if training_evaluation_configuration is None:
        raise ValueError("Checkpoint evaluation requires training.evaluation.")
    return training_evaluation_configuration.evaluators
