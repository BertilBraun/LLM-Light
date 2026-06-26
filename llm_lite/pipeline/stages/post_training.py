from pathlib import Path

from torch import nn

from llm_lite.config.models import (
    DirectPreferenceOptimizationConfiguration,
    ExperimentFile,
    InferenceConfiguration,
    NoPostTrainingConfiguration,
    PythonGeneratedDirectPreferenceOptimizationConfiguration,
    TrainingConfiguration,
)
from llm_lite.evaluation.python_completion import load_python_completion_tasks
from llm_lite.model.factory import build_model
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import BasePipelineStage
from llm_lite.post_training.generation import (
    generate_python_dpo_data,
    write_python_generated_dpo_data,
)
from llm_lite.post_training.preference import (
    DpoPreferenceDatasetRecord,
    load_dpo_preferences_jsonl,
)
from llm_lite.post_training.training import train_dpo_model
from llm_lite.tokenizer.loading import load_tokenizer
from llm_lite.training.checkpoint import latest_checkpoint, load_latest_checkpoint


class PostTrainingStage(BasePipelineStage):
    name: StageName = StageName.POST_TRAINING
    parents: tuple[StageName, ...] = (StageName.PRETRAINING, StageName.TOKENIZER)

    def configuration_hash(self, experiment_configuration: ExperimentFile) -> str:
        return hash_json_value(
            value={
                "post_training": experiment_configuration.post_training.model_dump(mode="json"),
                "inference": experiment_configuration.inference.model_dump(mode="json"),
                "training_optimizer": experiment_configuration.training.optimizer.model_dump(
                    mode="json",
                ),
                "training_dataloader": experiment_configuration.training.dataloader.model_dump(
                    mode="json",
                ),
                "training_precision": experiment_configuration.training.precision.value,
                "training_gradient_clip_norm": (
                    experiment_configuration.training.gradient_clip_norm
                ),
            },
        )

    def run(
        self,
        experiment_configuration: ExperimentFile,
        registry: ArtifactRegistry,
        artifact_directory: Path,
    ) -> StageOutput:
        match experiment_configuration.post_training:
            case NoPostTrainingConfiguration():
                return StageOutput(
                    files={},
                    metrics={"post_training_enabled": False},
                )
            case DirectPreferenceOptimizationConfiguration() as post_training_configuration:
                preference_dataset = load_dpo_preferences_jsonl(
                    input_path=post_training_configuration.preference_dataset_path,
                )
                return _run_dpo_training(
                    experiment_configuration=experiment_configuration,
                    registry=registry,
                    artifact_directory=artifact_directory,
                    preference_dataset=preference_dataset,
                    maximum_steps=post_training_configuration.maximum_steps,
                    batch_size_pairs=post_training_configuration.batch_size_pairs,
                    beta=post_training_configuration.beta,
                    generated_artifact_files={},
                )
            case PythonGeneratedDirectPreferenceOptimizationConfiguration() as dpo_configuration:
                tokenizer = load_tokenizer(
                    directory=registry.artifact_directory(StageName.TOKENIZER.value),
                    tokenizer_configuration=experiment_configuration.tokenizer,
                )
                policy_model, _reference_model = _load_base_models(
                    experiment_configuration=experiment_configuration,
                    registry=registry,
                    vocabulary_size=tokenizer.vocabulary_size,
                )
                tasks = load_python_completion_tasks(
                    tasks_path=dpo_configuration.tasks_path,
                    maximum_tasks=dpo_configuration.maximum_tasks,
                )
                generation_result = generate_python_dpo_data(
                    model=policy_model,
                    tokenizer=tokenizer,
                    tasks=tasks,
                    samples_per_prompt=dpo_configuration.samples_per_prompt,
                    inference_configuration=_python_generation_inference_configuration(
                        experiment_configuration=experiment_configuration,
                        post_training_configuration=dpo_configuration,
                    ),
                    execution_timeout_seconds=dpo_configuration.execution_timeout_seconds,
                )
                write_python_generated_dpo_data(
                    result=generation_result,
                    artifact_directory=artifact_directory,
                )
                return _run_dpo_training(
                    experiment_configuration=experiment_configuration,
                    registry=registry,
                    artifact_directory=artifact_directory,
                    preference_dataset=generation_result.preferences,
                    maximum_steps=dpo_configuration.maximum_steps,
                    batch_size_pairs=dpo_configuration.batch_size_pairs,
                    beta=dpo_configuration.beta,
                    generated_artifact_files={
                        "candidates": "candidates.jsonl",
                        "scores": "scores.jsonl",
                        "preferences": "preferences.jsonl",
                    },
                )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        checkpoint_state = latest_checkpoint(
            checkpoint_directory=registry.artifact_directory(StageName.POST_TRAINING.value)
            / "checkpoints",
        )
        if checkpoint_state is not None:
            return f"complete at step {checkpoint_state.step}, skip"
        return "disabled or complete, skip"


def _run_dpo_training(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    artifact_directory: Path,
    preference_dataset: DpoPreferenceDatasetRecord,
    maximum_steps: int,
    batch_size_pairs: int,
    beta: float,
    generated_artifact_files: dict[str, str],
) -> StageOutput:
    if len(preference_dataset.preferences) == 0:
        raise ValueError("DPO post-training requires at least one preference pair.")
    tokenizer = load_tokenizer(
        directory=registry.artifact_directory(StageName.TOKENIZER.value),
        tokenizer_configuration=experiment_configuration.tokenizer,
    )
    policy_model, reference_model = _load_base_models(
        experiment_configuration=experiment_configuration,
        registry=registry,
        vocabulary_size=tokenizer.vocabulary_size,
    )
    result = train_dpo_model(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        preference_dataset=preference_dataset,
        training_configuration=_post_training_configuration(
            experiment_configuration=experiment_configuration,
            maximum_steps=maximum_steps,
            batch_size_pairs=batch_size_pairs,
        ),
        artifact_directory=artifact_directory,
        seed=experiment_configuration.experiment.seed,
        beta=beta,
    )
    files = {
        "checkpoint": str(result.checkpoint_path.relative_to(artifact_directory)),
        "metrics": "metrics.jsonl",
        "tensorboard": "tensorboard",
    }
    files.update(generated_artifact_files)
    return StageOutput(
        files=files,
        metrics={
            "post_training_enabled": True,
            "preference_pairs": len(preference_dataset.preferences),
            "final_step": result.final_step,
            "final_loss": result.final_loss,
            "resumed_from_step": result.resumed_from_step,
        },
    )


def _load_base_models(
    experiment_configuration: ExperimentFile,
    registry: ArtifactRegistry,
    vocabulary_size: int,
) -> tuple[nn.Module, nn.Module]:
    policy_model = build_model(
        model_configuration=experiment_configuration.model,
        vocabulary_size=vocabulary_size,
    )
    reference_model = build_model(
        model_configuration=experiment_configuration.model,
        vocabulary_size=vocabulary_size,
    )
    checkpoint_directory = registry.artifact_directory(StageName.PRETRAINING.value) / "checkpoints"
    policy_checkpoint_step = load_latest_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=policy_model,
        optimizer=None,
    )
    reference_checkpoint_step = load_latest_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=reference_model,
        optimizer=None,
    )
    if policy_checkpoint_step is None or reference_checkpoint_step is None:
        raise ValueError("Post-training requires a completed pretraining checkpoint.")
    return policy_model, reference_model


def _post_training_configuration(
    experiment_configuration: ExperimentFile,
    maximum_steps: int,
    batch_size_pairs: int,
) -> TrainingConfiguration:
    return TrainingConfiguration(
        objective=experiment_configuration.training.objective,
        maximum_steps=maximum_steps,
        batch_size_sequences=batch_size_pairs,
        dataloader=experiment_configuration.training.dataloader,
        optimizer=experiment_configuration.training.optimizer,
        precision=experiment_configuration.training.precision,
        gradient_clip_norm=experiment_configuration.training.gradient_clip_norm,
        checkpoint_interval_steps=min(
            experiment_configuration.training.checkpoint_interval_steps,
            maximum_steps,
        ),
        log_interval_steps=min(experiment_configuration.training.log_interval_steps, maximum_steps),
        evaluation=None,
    )


def _python_generation_inference_configuration(
    experiment_configuration: ExperimentFile,
    post_training_configuration: PythonGeneratedDirectPreferenceOptimizationConfiguration,
) -> InferenceConfiguration:
    return experiment_configuration.inference.model_copy(
        update={
            "maximum_new_tokens": experiment_configuration.inference.maximum_new_tokens,
            "stop_sequences": post_training_configuration.stop_sequences,
        },
    )
