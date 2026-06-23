from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetType(str, Enum):
    INLINE_TEXT = "inline_text"
    LOCAL_TEXT = "local_text"
    HUGGINGFACE = "huggingface"


class TokenizerType(str, Enum):
    CHARACTER = "character"
    BYTE_BPE = "byte_bpe"


class PreprocessingTransformType(str, Enum):
    NORMALIZE_UNICODE = "normalize_unicode"
    NORMALIZE_LINE_ENDINGS = "normalize_line_endings"
    LOWER_CASE = "lower_case"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    EXACT_DEDUPLICATION = "exact_deduplication"
    ASSIGN_SPLIT = "assign_split"


class ModelType(str, Enum):
    DENSE_GPT = "dense_gpt"


class TrainingObjective(str, Enum):
    CAUSAL_LANGUAGE_MODELING = "causal_language_modeling"


class PostTrainingType(str, Enum):
    NONE = "none"


class InferenceEngine(str, Enum):
    NAIVE = "naive"


class Precision(str, Enum):
    FP32 = "fp32"


class QuantizationType(str, Enum):
    NONE = "none"


class ExperimentConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    seed: int
    output_dir: Path


class InlineTextDatasetConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[DatasetType.INLINE_TEXT]
    documents: tuple[str, ...]


class LocalTextDatasetConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[DatasetType.LOCAL_TEXT]
    paths: tuple[Path, ...]
    glob_patterns: tuple[str, ...]


class HuggingFaceDatasetSplitConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_split: str
    split: str
    max_documents: int | None = Field(default=None, gt=0)


class HuggingFaceDatasetConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[DatasetType.HUGGINGFACE]
    name: str
    text_column: str
    streaming: bool
    splits: tuple[HuggingFaceDatasetSplitConfiguration, ...]


class CharacterTokenizerConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[TokenizerType.CHARACTER]
    add_bos_token: bool
    add_eos_token: bool
    add_pad_token: bool


class ByteBpeTokenizerConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[TokenizerType.BYTE_BPE]
    vocabulary_size: int = Field(ge=256)
    add_bos_token: bool
    add_eos_token: bool
    add_pad_token: bool


TokenizerConfiguration = Annotated[
    CharacterTokenizerConfiguration | ByteBpeTokenizerConfiguration,
    Field(discriminator="type"),
]


class NormalizeLineEndingsTransformConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[PreprocessingTransformType.NORMALIZE_LINE_ENDINGS]


class NormalizeUnicodeTransformConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[PreprocessingTransformType.NORMALIZE_UNICODE]
    form: str

    @model_validator(mode="after")
    def require_supported_unicode_form(self) -> NormalizeUnicodeTransformConfiguration:
        if self.form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError("Unicode normalization form must be NFC, NFD, NFKC, or NFKD.")
        return self


class LowerCaseTransformConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[PreprocessingTransformType.LOWER_CASE]


class MinLengthTransformConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[PreprocessingTransformType.MIN_LENGTH]
    min_characters: int = Field(ge=0)


class MaxLengthTransformConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[PreprocessingTransformType.MAX_LENGTH]
    max_characters: int = Field(gt=0)


class ExactDeduplicationTransformConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[PreprocessingTransformType.EXACT_DEDUPLICATION]


class AssignSplitTransformConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[PreprocessingTransformType.ASSIGN_SPLIT]
    train_probability: float = Field(ge=0.0, le=1.0)
    validation_probability: float = Field(ge=0.0, le=1.0)
    test_probability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_probabilities_sum_to_one(self) -> AssignSplitTransformConfiguration:
        probability_sum = (
            self.train_probability + self.validation_probability + self.test_probability
        )
        if abs(probability_sum - 1.0) > 0.000001:
            raise ValueError("Split assignment probabilities must sum to 1.0.")
        return self


PreprocessingTransformConfiguration = Annotated[
    NormalizeUnicodeTransformConfiguration
    | NormalizeLineEndingsTransformConfiguration
    | LowerCaseTransformConfiguration
    | MinLengthTransformConfiguration
    | MaxLengthTransformConfiguration
    | ExactDeduplicationTransformConfiguration
    | AssignSplitTransformConfiguration,
    Field(discriminator="type"),
]


class PreprocessingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    transforms: tuple[PreprocessingTransformConfiguration, ...]
    output_shard_documents: int = Field(gt=0)


class PackingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    context_length: int = Field(gt=1)
    add_bos: bool
    add_eos: bool
    pack_documents: bool
    maximum_shard_tokens: int = Field(gt=0)


class DenseGptConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: ModelType
    dimension: int = Field(gt=0)
    layers: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    feed_forward_dimension: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    tie_embeddings: bool


class OptimizerConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)


class DataLoaderConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_workers: int = Field(ge=0)
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_workers_for_worker_options(self) -> DataLoaderConfiguration:
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers greater than 0.")
        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers greater than 0.")
        return self


class TrainingEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    interval_steps: int = Field(gt=0)
    evaluators: EvaluationConfiguration


class TrainingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    objective: TrainingObjective
    maximum_steps: int = Field(gt=0)
    batch_size_sequences: int = Field(gt=0)
    dataloader: DataLoaderConfiguration
    optimizer: OptimizerConfiguration
    precision: Precision
    gradient_clip_norm: float = Field(gt=0.0)
    checkpoint_interval_steps: int = Field(gt=0)
    log_interval_steps: int = Field(gt=0)
    evaluation: TrainingEvaluationConfiguration | None


class PostTrainingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: PostTrainingType


class ExactReproductionEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    expected_completion: str


class PerplexityEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    maximum_documents: int | None = Field(default=None, gt=0)


class FixedPromptGenerationEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompts: tuple[str, ...]
    maximum_new_tokens: int = Field(gt=0)


class EvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_reproduction: ExactReproductionEvaluationConfiguration | None = None
    perplexity: PerplexityEvaluationConfiguration | None = None
    fixed_prompt_generation: FixedPromptGenerationEvaluationConfiguration | None = None


class InferenceConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: InferenceEngine
    precision: Precision
    quantization: QuantizationType
    maximum_new_tokens: int = Field(gt=0)


class DistributedConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool


class ExperimentFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment: ExperimentConfiguration
    dataset: Annotated[
        InlineTextDatasetConfiguration
        | LocalTextDatasetConfiguration
        | HuggingFaceDatasetConfiguration,
        Field(discriminator="type"),
    ]
    preprocessing: PreprocessingConfiguration
    tokenizer: TokenizerConfiguration
    packing: PackingConfiguration
    model: DenseGptConfiguration
    training: TrainingConfiguration
    post_training: PostTrainingConfiguration
    evaluation: EvaluationConfiguration
    inference: InferenceConfiguration
    distributed: DistributedConfiguration
