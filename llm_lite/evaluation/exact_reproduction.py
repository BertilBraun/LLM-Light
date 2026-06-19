from dataclasses import dataclass


@dataclass(frozen=True)
class ExactReproductionResult:
    passed: bool
    generated_text: str
    expected_text: str


def evaluate_exact_reproduction(generated_text: str, expected_text: str) -> ExactReproductionResult:
    return ExactReproductionResult(
        passed=generated_text == expected_text,
        generated_text=generated_text,
        expected_text=expected_text,
    )
