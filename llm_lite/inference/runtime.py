from dataclasses import dataclass

import torch
from torch import nn

from llm_lite.config.models import (
    GenerationStopReason,
    InferenceConfiguration,
    Precision,
    QuantizationType,
)
from llm_lite.tokenizer.loading import TextTokenizer


@dataclass(frozen=True)
class EncodedPrompt:
    prompt: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class GenerationTiming:
    prefill_seconds: float
    decode_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds


@dataclass(frozen=True)
class GenerationThroughput:
    tokens_per_second: float
    sequences_per_second: float


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    generated_text: str
    full_text: str
    token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stop_reason: GenerationStopReason
    prompt_length: int
    generated_token_count: int
    timing: GenerationTiming
    throughput: GenerationThroughput


@dataclass(frozen=True)
class MutableGenerationState:
    prompt: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: list[int]
    stopped: bool
    stop_reason: GenerationStopReason | None


@dataclass(frozen=True)
class BatchTimingAccumulator:
    prefill_seconds: float
    decode_seconds: float


def prepare_model_for_inference(
    model: nn.Module,
    inference_configuration: InferenceConfiguration,
) -> nn.Module:
    _apply_precision(model=model, precision=inference_configuration.precision)
    _apply_quantization(model=model, quantization=inference_configuration.quantization)
    _move_to_inference_device(model=model)
    model.eval()
    return model


def encode_prompts(
    tokenizer: TextTokenizer,
    prompts: tuple[str, ...],
) -> tuple[EncodedPrompt, ...]:
    encoded_prompts: list[EncodedPrompt] = []
    for prompt in prompts:
        token_ids = tokenizer.encode(text=prompt, add_bos=True, add_eos=False)
        if len(token_ids) == 0:
            raise ValueError("Generation requires at least one prompt token.")
        encoded_prompts.append(EncodedPrompt(prompt=prompt, token_ids=tuple(token_ids)))
    return tuple(encoded_prompts)


def create_generation_states(
    encoded_prompts: tuple[EncodedPrompt, ...],
) -> list[MutableGenerationState]:
    return [
        MutableGenerationState(
            prompt=encoded_prompt.prompt,
            prompt_token_ids=encoded_prompt.token_ids,
            generated_token_ids=[],
            stopped=False,
            stop_reason=None,
        )
        for encoded_prompt in encoded_prompts
    ]


def append_next_token(
    state: MutableGenerationState,
    next_token_id: int,
    tokenizer: TextTokenizer,
    stop_sequences: tuple[str, ...],
) -> MutableGenerationState:
    if next_token_id == tokenizer.eos_token_id:
        return MutableGenerationState(
            prompt=state.prompt,
            prompt_token_ids=state.prompt_token_ids,
            generated_token_ids=state.generated_token_ids,
            stopped=True,
            stop_reason=GenerationStopReason.EOS_TOKEN,
        )
    generated_token_ids = [*state.generated_token_ids, next_token_id]
    if len(stop_sequences) == 0:
        return MutableGenerationState(
            prompt=state.prompt,
            prompt_token_ids=state.prompt_token_ids,
            generated_token_ids=generated_token_ids,
            stopped=False,
            stop_reason=None,
        )
    generated_text = tokenizer.decode(generated_token_ids)
    stop_index = find_stop_sequence_index(text=generated_text, stop_sequences=stop_sequences)
    if stop_index is None:
        return MutableGenerationState(
            prompt=state.prompt,
            prompt_token_ids=state.prompt_token_ids,
            generated_token_ids=generated_token_ids,
            stopped=False,
            stop_reason=None,
        )
    return MutableGenerationState(
        prompt=state.prompt,
        prompt_token_ids=state.prompt_token_ids,
        generated_token_ids=tokenizer.encode(
            text=generated_text[:stop_index],
            add_bos=False,
            add_eos=False,
        ),
        stopped=True,
        stop_reason=GenerationStopReason.STOP_SEQUENCE,
    )


def find_stop_sequence_index(text: str, stop_sequences: tuple[str, ...]) -> int | None:
    stop_indexes = tuple(
        stop_index
        for stop_sequence in stop_sequences
        if (stop_index := text.find(stop_sequence)) >= 0
    )
    if len(stop_indexes) == 0:
        return None
    return min(stop_indexes)


def finalize_generation_results(
    states: tuple[MutableGenerationState, ...],
    tokenizer: TextTokenizer,
    maximum_new_tokens: int,
    timing: BatchTimingAccumulator,
) -> tuple[GenerationResult, ...]:
    total_generated_tokens = sum(len(state.generated_token_ids) for state in states)
    total_seconds = timing.prefill_seconds + timing.decode_seconds
    tokens_per_second = 0.0
    sequences_per_second = 0.0
    if total_seconds > 0.0:
        tokens_per_second = total_generated_tokens / total_seconds
        sequences_per_second = len(states) / total_seconds
    return tuple(
        _finalize_generation_result(
            state=state,
            tokenizer=tokenizer,
            maximum_new_tokens=maximum_new_tokens,
            timing=timing,
            tokens_per_second=tokens_per_second,
            sequences_per_second=sequences_per_second,
        )
        for state in states
    )


def _apply_precision(model: nn.Module, precision: Precision) -> None:
    match precision:
        case Precision.FP32:
            model.float()
        case Precision.FP16:
            model.half()
        case Precision.BF16:
            model.bfloat16()


def _apply_quantization(model: nn.Module, quantization: QuantizationType) -> None:
    match quantization:
        case QuantizationType.NONE:
            return
        case (
            QuantizationType.INT8_DYNAMIC
            | QuantizationType.INT8_WEIGHT_ONLY
            | QuantizationType.INT4_WEIGHT_ONLY
        ):
            raise ValueError(f"Quantization type {quantization.value!r} is not implemented.")


def _move_to_inference_device(model: nn.Module) -> None:
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        return
    current_device = first_parameter.device
    if current_device.type != "cpu" or not torch.cuda.is_available():
        return
    model.to(torch.device("cuda"))


def _finalize_generation_result(
    state: MutableGenerationState,
    tokenizer: TextTokenizer,
    maximum_new_tokens: int,
    timing: BatchTimingAccumulator,
    tokens_per_second: float,
    sequences_per_second: float,
) -> GenerationResult:
    generated_text = tokenizer.decode(state.generated_token_ids)
    full_token_ids = tuple([*state.prompt_token_ids, *state.generated_token_ids])
    stop_reason = state.stop_reason
    if stop_reason is None:
        stop_reason = GenerationStopReason.MAXIMUM_NEW_TOKENS
    return GenerationResult(
        prompt=state.prompt,
        generated_text=generated_text,
        full_text=tokenizer.decode(full_token_ids),
        token_ids=full_token_ids,
        generated_token_ids=tuple(state.generated_token_ids),
        stop_reason=stop_reason,
        prompt_length=len(state.prompt_token_ids),
        generated_token_count=min(len(state.generated_token_ids), maximum_new_tokens),
        timing=GenerationTiming(
            prefill_seconds=timing.prefill_seconds,
            decode_seconds=timing.decode_seconds,
        ),
        throughput=GenerationThroughput(
            tokens_per_second=tokens_per_second,
            sequences_per_second=sequences_per_second,
        ),
    )
