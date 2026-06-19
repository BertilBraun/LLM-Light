from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_lite.config.loading import load_experiment_configuration
from llm_lite.config.models import EvaluationConfiguration


def test_load_verify_configuration() -> None:
    experiment_configuration = load_experiment_configuration(
        configuration_path=Path("configs/verify_one_sentence.yaml"),
    )

    assert experiment_configuration.experiment.name == "verify_one_sentence"
    assert experiment_configuration.dataset.documents == ("hello world\n",)
    exact_reproduction_configuration = experiment_configuration.evaluation.exact_reproduction

    assert exact_reproduction_configuration is not None
    assert exact_reproduction_configuration.expected_completion == "hello world\n"


def test_evaluation_configuration_requires_configured_evaluator() -> None:
    with pytest.raises(ValidationError, match="At least one evaluation block"):
        EvaluationConfiguration.model_validate({})


def test_evaluation_configuration_rejects_unknown_evaluator() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfiguration.model_validate(
            {"other_evaluation_type": {"parameters_for_that": "here"}},
        )
