from pathlib import Path

from torch import nn

from llm_lite.config.models import TrainingConfiguration
from llm_lite.post_training.dpo import build_dpo_preference_dataset
from llm_lite.post_training.preference import DpoPreferenceDatasetRecord
from llm_lite.tokenizer.loading import TextTokenizer
from llm_lite.training.objectives import DirectPreferenceOptimizationObjectiveRunner
from llm_lite.training.trainer import TrainingResult, train_model


def train_dpo_model(
    policy_model: nn.Module,
    reference_model: nn.Module,
    tokenizer: TextTokenizer,
    preference_dataset: DpoPreferenceDatasetRecord,
    training_configuration: TrainingConfiguration,
    artifact_directory: Path,
    seed: int,
    beta: float,
) -> TrainingResult:
    dataset = build_dpo_preference_dataset(
        preferences=preference_dataset.preferences,
        tokenizer=tokenizer,
    )
    return train_model(
        model=policy_model,
        dataset=dataset,
        training_configuration=training_configuration,
        artifact_directory=artifact_directory,
        seed=seed,
        evaluation_callback=None,
        objective_runner=DirectPreferenceOptimizationObjectiveRunner(
            reference_model=reference_model,
            beta=beta,
        ),
    )
