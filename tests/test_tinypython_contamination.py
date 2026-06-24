from llm_lite.scripts.report_tinypython_contamination import report_contamination


def test_report_contamination_detects_prompt_code_signature_and_family_overlap() -> None:
    report = report_contamination(
        training_records=[
            {
                "task_description": "Return a cleaned value.",
                "code": "def clean(value: str) -> str:\n    return value.strip()",
                "signature": "def clean(value: str) -> str:",
                "task_family": "string_normalization",
            },
        ],
        eval_records=[
            {
                "task_id": "description_match",
                "task_description": " return   A cleaned VALUE. ",
            },
            {
                "task_id": "signature_match",
                "prompt": "def clean(value: str) -> str:\n",
                "task_family": "string_normalization",
            },
            {
                "task_id": "code_match",
                "code": "def clean(value: str) -> str:\n    return value.strip()",
            },
        ],
    )

    assert report["exact_prompt_or_description_matches"] == []
    assert report["normalized_prompt_or_description_matches"] == ["description_match"]
    assert report["exact_code_matches"] == ["code_match"]
    assert report["signature_overlaps"] == ["code_match", "signature_match"]
    assert report["task_family_overlap_counts"] == {"string_normalization": 1}
