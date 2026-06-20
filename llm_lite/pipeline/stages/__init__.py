from llm_lite.pipeline.stage import PipelineStage, StageName
from llm_lite.pipeline.stages.evaluation import EvaluationStage
from llm_lite.pipeline.stages.packed_dataset import PackedDatasetStage
from llm_lite.pipeline.stages.pretraining import PretrainingStage
from llm_lite.pipeline.stages.processed_dataset import ProcessedDatasetStage
from llm_lite.pipeline.stages.raw_dataset import RawDatasetStage
from llm_lite.pipeline.stages.tokenizer import TokenizerStage

ORDERED_PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    RawDatasetStage(),
    ProcessedDatasetStage(),
    TokenizerStage(),
    PackedDatasetStage(),
    PretrainingStage(),
    EvaluationStage(),
)

ORDERED_STAGE_NAMES: tuple[StageName, ...] = tuple(stage.name for stage in ORDERED_PIPELINE_STAGES)
