from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import DataLoaderConfiguration, EvaluationConfiguration


def test_load_verify_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )

    assert experiment_configuration.experiment.name == "verify_one_sentence"
    assert experiment_configuration.dataset.documents == ("hello world\n",)
    assert len(experiment_configuration.preprocessing.transforms) == 1
    assert experiment_configuration.training.dataloader.num_workers == 0
    exact_reproduction_configuration = experiment_configuration.evaluation.exact_reproduction

    assert exact_reproduction_configuration is not None
    assert exact_reproduction_configuration.expected_completion == "hello world\n"


def test_load_byte_bpe_verification_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("tests/configs/verify_byte_bpe.yaml"),
    )

    assert experiment_configuration.experiment.name == "verify_byte_bpe"
    assert experiment_configuration.tokenizer.type.value == "byte_bpe"


def test_evaluation_configuration_requires_configured_evaluator() -> None:
    with pytest.raises(ValidationError, match="At least one evaluation block"):
        EvaluationConfiguration.model_validate({})


def test_evaluation_configuration_rejects_unknown_evaluator() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfiguration.model_validate(
            {"other_evaluation_type": {"parameters_for_that": "here"}},
        )


def test_dataloader_configuration_rejects_worker_options_without_workers() -> None:
    with pytest.raises(ValidationError, match="persistent_workers"):
        DataLoaderConfiguration.model_validate(
            {
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": True,
                "prefetch_factor": None,
            },
        )

    with pytest.raises(ValidationError, match="prefetch_factor"):
        DataLoaderConfiguration.model_validate(
            {
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
        )
