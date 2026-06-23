import json
from dataclasses import dataclass
from pathlib import Path

from llm_lite.config.models import ExperimentFile
from llm_lite.evaluation.exact_reproduction import evaluate_exact_reproduction
from llm_lite.model.gpt import DenseGpt
from llm_lite.pipeline.hashing import hash_json_value
from llm_lite.pipeline.registry import ArtifactRegistry
from llm_lite.pipeline.stage import StageName, StageOutput
from llm_lite.pipeline.stages.base import compatible_skip_action
from llm_lite.tokenizer.loading import load_tokenizer
from llm_lite.training.checkpoint import load_latest_checkpoint


@dataclass(frozen=True)
class EvaluationStage:
    name: StageName = StageName.EVALUATION
    parents: tuple[StageName, ...] = (StageName.PRETRAINING, StageName.TOKENIZER)

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
        exact_reproduction_configuration = experiment_configuration.evaluation.exact_reproduction
        if exact_reproduction_configuration is None:
            raise ValueError("Exact reproduction evaluation is not configured.")
        tokenizer = load_tokenizer(
            directory=registry.artifact_directory(StageName.TOKENIZER.value),
            tokenizer_configuration=experiment_configuration.tokenizer,
        )
        model = DenseGpt(
            model_configuration=experiment_configuration.model,
            vocabulary_size=tokenizer.vocabulary_size,
        )
        checkpoint_step = load_latest_checkpoint(
            checkpoint_directory=registry.artifact_directory(StageName.PRETRAINING.value)
            / "checkpoints",
            model=model,
            optimizer=None,
        )
        if checkpoint_step is None:
            raise ValueError("Evaluation requires a completed training checkpoint.")
        exact_reproduction_result = evaluate_exact_reproduction(
            model=model,
            tokenizer=tokenizer,
            evaluation_configuration=exact_reproduction_configuration,
            inference_configuration=experiment_configuration.inference,
        )
        report = {
            "exact_reproduction": exact_reproduction_result.model_dump(),
        }
        (artifact_directory / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if not exact_reproduction_result.passed:
            raise ValueError("Exact reproduction evaluation failed.")
        return StageOutput(
            files={"report": "report.json"},
            metrics={"exact_reproduction_passed": exact_reproduction_result.passed},
        )

    def compatible_action(self, registry: ArtifactRegistry) -> str:
        return compatible_skip_action(registry=registry)
