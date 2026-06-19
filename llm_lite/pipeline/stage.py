from enum import Enum


class StageName(str, Enum):
    RAW_DATASET = "raw_dataset"
    TOKENIZER = "tokenizer"
    TOKENIZED_DATASET = "tokenized_dataset"
    PACKED_DATASET = "packed_dataset"
    PRETRAINING = "pretraining"
    EVALUATION = "evaluation"


ORDERED_STAGE_NAMES: tuple[StageName, ...] = (
    StageName.RAW_DATASET,
    StageName.TOKENIZER,
    StageName.TOKENIZED_DATASET,
    StageName.PACKED_DATASET,
    StageName.PRETRAINING,
    StageName.EVALUATION,
)
