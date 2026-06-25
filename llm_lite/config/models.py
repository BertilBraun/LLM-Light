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
    RUST_BYTE_BPE = "rust_byte_bpe"


class PreprocessingTransformType(str, Enum):
    NORMALIZE_UNICODE = "normalize_unicode"
    NORMALIZE_LINE_ENDINGS = "normalize_line_endings"
    LOWER_CASE = "lower_case"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    EXACT_DEDUPLICATION = "exact_deduplication"
    ASSIGN_SPLIT = "assign_split"
    EXTRACT_PYTHON_FUNCTIONS = "extract_python_functions"


class ModelType(str, Enum):
    DENSE_GPT = "dense_gpt"
    MOE_GPT = "moe_gpt"


class TrainingObjective(str, Enum):
    CAUSAL_LANGUAGE_MODELING = "causal_language_modeling"


class PostTrainingType(str, Enum):
    NONE = "none"
    DIRECT_PREFERENCE_OPTIMIZATION = "direct_preference_optimization"
    PYTHON_GENERATED_DIRECT_PREFERENCE_OPTIMIZATION = (
        "python_generated_direct_preference_optimization"
    )


class InferenceEngine(str, Enum):
    NAIVE = "naive"
    KV_CACHE = "kv_cache"


class DecodingStrategy(str, Enum):
    GREEDY = "greedy"
    SAMPLE = "sample"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class QuantizationType(str, Enum):
    NONE = "none"
    INT8_DYNAMIC = "int8_dynamic"
    INT8_WEIGHT_ONLY = "int8_weight_only"
    INT4_WEIGHT_ONLY = "int4_weight_only"


class GenerationStopReason(str, Enum):
    MAXIMUM_NEW_TOKENS = "maximum_new_tokens"
    EOS_TOKEN = "eos_token"
    STOP_SEQUENCE = "stop_sequence"


class DistributedBackend(str, Enum):
    GLOO = "gloo"
    NCCL = "nccl"


class DistributedStrategy(str, Enum):
    SINGLE_PROCESS = "single_process"
    DATA_PARALLEL = "data_parallel"
    FULLY_SHARDED_DATA_PARALLEL = "fully_sharded_data_parallel"


class DistributedCheckpointType(str, Enum):
    FULL = "full"
    SHARDED = "sharded"


class Configuration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GreedyDecodingConfiguration(Configuration):
    strategy: Literal[DecodingStrategy.GREEDY]


class SamplingDecodingConfiguration(Configuration):
    strategy: Literal[DecodingStrategy.SAMPLE]
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int | None = Field(default=None, gt=0)


DecodingConfiguration = Annotated[
    GreedyDecodingConfiguration | SamplingDecodingConfiguration,
    Field(discriminator="strategy"),
]


class ExperimentConfiguration(Configuration):
    name: str
    seed: int = 0
    output_dir: Path


class InlineTextDatasetConfiguration(Configuration):
    type: Literal[DatasetType.INLINE_TEXT]
    documents: tuple[str, ...]


class LocalTextDatasetConfiguration(Configuration):
    type: Literal[DatasetType.LOCAL_TEXT]
    paths: tuple[Path, ...] = ()
    glob_patterns: tuple[str, ...] = ()


class HuggingFaceDatasetSplitConfiguration(Configuration):
    source_split: str
    split: str
    skip_documents: int = Field(default=0, ge=0)
    max_documents: int | None = Field(default=None, gt=0)


class HuggingFaceDatasetConfiguration(Configuration):
    type: Literal[DatasetType.HUGGINGFACE]
    name: str
    config_name: str | None = None
    text_column: str | None = None
    text_template: str | None = None
    language_column: str | None = None
    languages: tuple[str, ...] = ()
    license_column: str | None = None
    licenses: tuple[str, ...] = ()
    streaming: bool = True
    splits: tuple[HuggingFaceDatasetSplitConfiguration, ...]

    @model_validator(mode="after")
    def require_text_source_and_filter_columns(self) -> HuggingFaceDatasetConfiguration:
        if (self.text_column is None) == (self.text_template is None):
            raise ValueError("Hugging Face dataset requires exactly one text source.")
        if self.languages and self.language_column is None:
            raise ValueError("Hugging Face language filters require language_column.")
        if self.licenses and self.license_column is None:
            raise ValueError("Hugging Face license filters require license_column.")
        return self


class CharacterTokenizerConfiguration(Configuration):
    type: Literal[TokenizerType.CHARACTER]
    add_bos_token: bool = True
    add_eos_token: bool = True
    add_pad_token: bool = True


class ByteBpeTokenizerConfiguration(Configuration):
    type: Literal[TokenizerType.BYTE_BPE]
    vocabulary_size: int = Field(ge=256)
    max_training_documents: int | None = Field(gt=0)
    max_training_bytes: int | None = Field(gt=0)
    training_workers: int = Field(default=1, ge=1)
    add_bos_token: bool = True
    add_eos_token: bool = True
    add_pad_token: bool = True

    @model_validator(mode="after")
    def require_training_sample_bound(self) -> ByteBpeTokenizerConfiguration:
        if self.max_training_documents is None and self.max_training_bytes is None:
            raise ValueError(
                "Byte BPE training requires max_training_documents or max_training_bytes.",
            )
        return self


class RustByteBpeTokenizerConfiguration(Configuration):
    type: Literal[TokenizerType.RUST_BYTE_BPE]
    vocabulary_size: int = Field(ge=256)
    max_training_documents: int | None = Field(gt=0)
    max_training_bytes: int | None = Field(gt=0)
    training_workers: int = Field(default=1, ge=1)
    add_bos_token: bool = True
    add_eos_token: bool = True
    add_pad_token: bool = True

    @model_validator(mode="after")
    def require_training_sample_bound(self) -> RustByteBpeTokenizerConfiguration:
        if self.max_training_documents is None and self.max_training_bytes is None:
            raise ValueError(
                "Rust Byte BPE training requires max_training_documents or max_training_bytes.",
            )
        return self


TokenizerConfiguration = Annotated[
    CharacterTokenizerConfiguration
    | ByteBpeTokenizerConfiguration
    | RustByteBpeTokenizerConfiguration,
    Field(discriminator="type"),
]


class NormalizeLineEndingsTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.NORMALIZE_LINE_ENDINGS]


class NormalizeUnicodeTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.NORMALIZE_UNICODE]
    form: str = "NFC"

    @model_validator(mode="after")
    def require_supported_unicode_form(self) -> NormalizeUnicodeTransformConfiguration:
        if self.form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError("Unicode normalization form must be NFC, NFD, NFKC, or NFKD.")
        return self


class LowerCaseTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.LOWER_CASE]


class MinLengthTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.MIN_LENGTH]
    min_characters: int = Field(ge=0)


class MaxLengthTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.MAX_LENGTH]
    max_characters: int = Field(gt=0)


class ExactDeduplicationTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.EXACT_DEDUPLICATION]


class ExtractPythonFunctionsTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.EXTRACT_PYTHON_FUNCTIONS]
    include_async_functions: bool = False
    include_private_functions: bool = False
    include_methods: bool = False


class AssignSplitTransformConfiguration(Configuration):
    type: Literal[PreprocessingTransformType.ASSIGN_SPLIT]
    train_probability: float = Field(default=0.98, ge=0.0, le=1.0)
    validation_probability: float = Field(default=0.01, ge=0.0, le=1.0)
    test_probability: float = Field(default=0.01, ge=0.0, le=1.0)

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
    | ExtractPythonFunctionsTransformConfiguration
    | AssignSplitTransformConfiguration,
    Field(discriminator="type"),
]


class PreprocessingConfiguration(Configuration):
    transforms: tuple[PreprocessingTransformConfiguration, ...] = ()
    output_shard_documents: int = Field(default=10000, gt=0)
    workers: int = Field(default=1, ge=1)


class PackingConfiguration(Configuration):
    context_length: int = Field(gt=1)
    add_bos: bool = True
    add_eos: bool = True
    pack_documents: bool = True
    maximum_shard_tokens: int = Field(default=1000000, gt=0)
    workers: int = Field(default=1, ge=1)


class DenseGptConfiguration(Configuration):
    type: Literal[ModelType.DENSE_GPT]
    dimension: int = Field(gt=0)
    layers: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    feed_forward_dimension: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    tie_embeddings: bool = True


class MoeGptConfiguration(Configuration):
    type: Literal[ModelType.MOE_GPT]
    dimension: int = Field(gt=0)
    layers: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    expert_feed_forward_dimension: int = Field(gt=0)
    expert_count: int = Field(gt=1)
    router_top_k: int = Field(ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    tie_embeddings: bool = True

    @model_validator(mode="after")
    def require_top_k_not_greater_than_experts(self) -> MoeGptConfiguration:
        if self.router_top_k > self.expert_count:
            raise ValueError("MoE router_top_k must not be greater than expert_count.")
        return self


ModelConfiguration = Annotated[
    DenseGptConfiguration | MoeGptConfiguration,
    Field(discriminator="type"),
]


class CausalLanguageModelingObjectiveConfiguration(Configuration):
    auxiliary_loss_weight: float = Field(default=0.0, ge=0.0)


class OptimizerConfiguration(Configuration):
    learning_rate: float = Field(default=0.0003, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)


class DataLoaderConfiguration(Configuration):
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_workers_for_worker_options(self) -> DataLoaderConfiguration:
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers greater than 0.")
        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers greater than 0.")
        return self


class TrainingEvaluationConfiguration(Configuration):
    interval_steps: int = Field(gt=0)
    evaluators: EvaluationConfiguration


class TrainingConfiguration(Configuration):
    objective: TrainingObjective = TrainingObjective.CAUSAL_LANGUAGE_MODELING
    causal_language_modeling: CausalLanguageModelingObjectiveConfiguration = (
        CausalLanguageModelingObjectiveConfiguration()
    )
    maximum_steps: int = Field(gt=0)
    batch_size_sequences: int = Field(gt=0)
    dataloader: DataLoaderConfiguration = DataLoaderConfiguration()
    optimizer: OptimizerConfiguration = OptimizerConfiguration()
    precision: Precision = Precision.FP32
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    checkpoint_interval_steps: int = Field(default=1000, gt=0)
    log_interval_steps: int = Field(default=10, gt=0)
    evaluation: TrainingEvaluationConfiguration | None = None


class NoPostTrainingConfiguration(Configuration):
    type: Literal[PostTrainingType.NONE] = PostTrainingType.NONE


class DirectPreferenceOptimizationConfiguration(Configuration):
    type: Literal[PostTrainingType.DIRECT_PREFERENCE_OPTIMIZATION]
    preference_dataset_path: Path
    beta: float = Field(default=0.1, gt=0.0)
    maximum_steps: int = Field(gt=0)
    batch_size_pairs: int = Field(gt=0)


class PythonGeneratedDirectPreferenceOptimizationConfiguration(Configuration):
    type: Literal[PostTrainingType.PYTHON_GENERATED_DIRECT_PREFERENCE_OPTIMIZATION]
    tasks_path: Path
    samples_per_prompt: int = Field(gt=1)
    beta: float = Field(default=0.1, gt=0.0)
    maximum_steps: int = Field(gt=0)
    batch_size_pairs: int = Field(gt=0)
    maximum_tasks: int | None = Field(default=None, gt=0)
    execution_timeout_seconds: float = Field(default=2.0, gt=0.0)
    stop_sequences: tuple[str, ...] = ("\n\ndef ", "\nclass ", "\nif __name__")


PostTrainingConfiguration = Annotated[
    NoPostTrainingConfiguration
    | DirectPreferenceOptimizationConfiguration
    | PythonGeneratedDirectPreferenceOptimizationConfiguration,
    Field(discriminator="type"),
]


class ExactReproductionEvaluationConfiguration(Configuration):
    prompt: str
    expected_completion: str


class PerplexityEvaluationConfiguration(Configuration):
    split: str
    maximum_documents: int | None = Field(default=None, gt=0)


class FixedPromptGenerationEvaluationConfiguration(Configuration):
    prompts: tuple[str, ...]
    maximum_new_tokens: int = Field(default=80, gt=0)


class PythonCompletionEvaluationConfiguration(Configuration):
    tasks_path: Path
    maximum_tasks: int | None = Field(default=None, gt=0)
    maximum_new_tokens: int = Field(default=80, gt=0)
    execution_timeout_seconds: float = Field(default=2.0, gt=0.0)
    stop_sequences: tuple[str, ...] = ("\n\ndef ", "\nclass ", "\nif __name__")


class EvaluationConfiguration(Configuration):
    exact_reproduction: ExactReproductionEvaluationConfiguration | None = None
    perplexity: PerplexityEvaluationConfiguration | None = None
    fixed_prompt_generation: FixedPromptGenerationEvaluationConfiguration | None = None
    python_completion: PythonCompletionEvaluationConfiguration | None = None


class InferenceConfiguration(Configuration):
    engine: InferenceEngine = InferenceEngine.KV_CACHE
    precision: Precision = Precision.FP32
    quantization: QuantizationType = QuantizationType.NONE
    decoding: DecodingConfiguration = GreedyDecodingConfiguration(
        strategy=DecodingStrategy.GREEDY,
    )
    maximum_new_tokens: int = Field(default=80, gt=0)
    batch_size: int = Field(default=1, gt=0)
    stop_sequences: tuple[str, ...] = ()


class SimulatedNodesConfiguration(Configuration):
    count: int = Field(default=1, gt=0)
    processes_per_node: int = Field(default=1, gt=0)


class ParallelismConfiguration(Configuration):
    data: int = Field(default=1, gt=0)
    tensor: int = Field(default=1, gt=0)
    pipeline: int = Field(default=1, gt=0)
    context: int = Field(default=1, gt=0)
    expert: int = Field(default=1, gt=0)


class DistributedCheckpointConfiguration(Configuration):
    type: DistributedCheckpointType = DistributedCheckpointType.FULL
    save_rank_local_state: bool = True


class DistributedConfiguration(Configuration):
    enabled: bool = False
    backend: DistributedBackend = DistributedBackend.GLOO
    strategy: DistributedStrategy = DistributedStrategy.SINGLE_PROCESS
    world_size: int = Field(default=1, gt=0)
    simulated_nodes: SimulatedNodesConfiguration = SimulatedNodesConfiguration()
    parallelism: ParallelismConfiguration = ParallelismConfiguration()
    checkpoint: DistributedCheckpointConfiguration = DistributedCheckpointConfiguration()

    @model_validator(mode="after")
    def require_consistent_distributed_configuration(self) -> DistributedConfiguration:
        if not self.enabled and self.strategy is not DistributedStrategy.SINGLE_PROCESS:
            raise ValueError("Disabled distributed configuration must use single_process strategy.")
        if self.enabled and self.strategy is DistributedStrategy.SINGLE_PROCESS:
            raise ValueError("Enabled distributed configuration requires a distributed strategy.")
        if self.simulated_nodes.count * self.simulated_nodes.processes_per_node != self.world_size:
            raise ValueError(
                "simulated_nodes count multiplied by processes_per_node must match world_size."
            )
        parallelism_product = (
            self.parallelism.data
            * self.parallelism.tensor
            * self.parallelism.pipeline
            * self.parallelism.context
            * self.parallelism.expert
        )
        if parallelism_product != self.world_size:
            raise ValueError("Parallelism dimensions must multiply to world_size.")
        if self.parallelism.tensor != 1:
            raise ValueError("Tensor parallelism greater than 1 is not implemented.")
        if self.parallelism.pipeline != 1:
            raise ValueError("Pipeline parallelism greater than 1 is not implemented.")
        if self.parallelism.context != 1:
            raise ValueError("Context parallelism greater than 1 is not implemented.")
        if self.parallelism.expert != 1:
            raise ValueError("Expert parallelism greater than 1 is not implemented.")
        return self


class ExperimentFile(Configuration):
    experiment: ExperimentConfiguration
    dataset: Annotated[
        InlineTextDatasetConfiguration
        | LocalTextDatasetConfiguration
        | HuggingFaceDatasetConfiguration,
        Field(discriminator="type"),
    ]
    preprocessing: PreprocessingConfiguration = PreprocessingConfiguration()
    tokenizer: TokenizerConfiguration
    packing: PackingConfiguration
    model: ModelConfiguration
    training: TrainingConfiguration
    post_training: PostTrainingConfiguration = NoPostTrainingConfiguration()
    evaluation: EvaluationConfiguration = EvaluationConfiguration()
    inference: InferenceConfiguration = InferenceConfiguration()
    distributed: DistributedConfiguration = DistributedConfiguration()
