from llm_lite.pipeline.stage import StageName
from llm_lite.pipeline.stages import ORDERED_PIPELINE_STAGES, ORDERED_STAGE_NAMES


def test_ordered_pipeline_stages_expose_names_and_dependencies() -> None:
    assert ORDERED_STAGE_NAMES == (
        StageName.RAW_DATASET,
        StageName.PROCESSED_DATASET,
        StageName.TOKENIZER,
        StageName.PACKED_DATASET,
        StageName.PRETRAINING,
        StageName.POST_TRAINING,
        StageName.EVALUATION,
    )
    assert tuple(stage.name for stage in ORDERED_PIPELINE_STAGES) == ORDERED_STAGE_NAMES
    assert ORDERED_PIPELINE_STAGES[0].parents == ()
    assert ORDERED_PIPELINE_STAGES[1].parents == (StageName.RAW_DATASET,)
    assert ORDERED_PIPELINE_STAGES[2].parents == (StageName.PROCESSED_DATASET,)
    assert ORDERED_PIPELINE_STAGES[3].parents == (
        StageName.PROCESSED_DATASET,
        StageName.TOKENIZER,
    )
    assert ORDERED_PIPELINE_STAGES[4].parents == (StageName.PACKED_DATASET, StageName.TOKENIZER)
    assert ORDERED_PIPELINE_STAGES[5].parents == (StageName.PRETRAINING, StageName.TOKENIZER)
    assert ORDERED_PIPELINE_STAGES[6].parents == (StageName.POST_TRAINING, StageName.TOKENIZER)
