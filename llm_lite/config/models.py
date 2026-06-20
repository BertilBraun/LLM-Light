from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetType(str, Enum):
    INLINE_TEXT = "inline_text"


class TokenizerType(str, Enum):
    CHARACTER = "character"


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

    type: DatasetType
    documents: tuple[str, ...]


class TokenizerConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: TokenizerType
    add_bos_token: bool
    add_eos_token: bool
    add_pad_token: bool


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


class TrainingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    objective: TrainingObjective
    maximum_steps: int = Field(gt=0)
    batch_size_sequences: int = Field(gt=0)
    optimizer: OptimizerConfiguration
    precision: Precision
    gradient_clip_norm: float = Field(gt=0.0)
    checkpoint_interval_steps: int = Field(gt=0)
    log_interval_steps: int = Field(gt=0)


class PostTrainingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: PostTrainingType


class ExactReproductionEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    expected_completion: str


class EvaluationConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_reproduction: ExactReproductionEvaluationConfiguration | None = None

    @model_validator(mode="after")
    def require_at_least_one_evaluator(self) -> "EvaluationConfiguration":
        if self.exact_reproduction is None:
            raise ValueError("At least one evaluation block must be configured.")
        return self


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
    dataset: InlineTextDatasetConfiguration
    tokenizer: TokenizerConfiguration
    packing: PackingConfiguration
    model: DenseGptConfiguration
    training: TrainingConfiguration
    post_training: PostTrainingConfiguration
    evaluation: EvaluationConfiguration
    inference: InferenceConfiguration
    distributed: DistributedConfiguration
