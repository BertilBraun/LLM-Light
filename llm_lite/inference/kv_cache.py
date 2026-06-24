import os
from time import perf_counter

import torch

from llm_lite.config.models import (
    DecodingConfiguration,
    DecodingStrategy,
    GreedyDecodingConfiguration,
)
from llm_lite.inference.decoding import select_next_token_id, select_next_token_ids
from llm_lite.inference.runtime import (
    BatchTimingAccumulator,
    GenerationResult,
    append_next_token,
    create_generation_states,
    encode_prompts,
    finalize_generation_results,
)
from llm_lite.model.protocol import CachedAutoregressiveModel
from llm_lite.pipeline.progress import console_log
from llm_lite.tokenizer.loading import TextTokenizer


def generate_batch(
    model: CachedAutoregressiveModel,
    tokenizer: TextTokenizer,
    prompts: tuple[str, ...],
    maximum_new_tokens: int,
    decoding_configuration: DecodingConfiguration,
    stop_sequences: tuple[str, ...],
) -> tuple[GenerationResult, ...]:
    encoded_prompts = encode_prompts(tokenizer=tokenizer, prompts=prompts)
    prompt_lengths = {len(encoded_prompt.token_ids) for encoded_prompt in encoded_prompts}
    if len(prompt_lengths) != 1:
        return tuple(
            _generate_single_with_cache(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                maximum_new_tokens=maximum_new_tokens,
                decoding_configuration=decoding_configuration,
                stop_sequences=stop_sequences,
            )
            for prompt in prompts
        )
    return _generate_equal_length_batch_with_cache(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        maximum_new_tokens=maximum_new_tokens,
        decoding_configuration=decoding_configuration,
        stop_sequences=stop_sequences,
    )


def generate(
    model: CachedAutoregressiveModel,
    tokenizer: TextTokenizer,
    prompt: str,
    maximum_new_tokens: int,
    decoding_configuration: DecodingConfiguration,
) -> str:
    generation_result = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=(prompt,),
        maximum_new_tokens=maximum_new_tokens,
        decoding_configuration=decoding_configuration,
        stop_sequences=(),
    )[0]
    return generation_result.full_text


def generate_greedy(
    model: CachedAutoregressiveModel,
    tokenizer: TextTokenizer,
    prompt: str,
    maximum_new_tokens: int,
) -> str:
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        maximum_new_tokens=maximum_new_tokens,
        decoding_configuration=GreedyDecodingConfiguration(
            strategy=DecodingStrategy.GREEDY,
        ),
    )


def _generate_single_with_cache(
    model: CachedAutoregressiveModel,
    tokenizer: TextTokenizer,
    prompt: str,
    maximum_new_tokens: int,
    decoding_configuration: DecodingConfiguration,
    stop_sequences: tuple[str, ...],
) -> GenerationResult:
    encoded_prompt = encode_prompts(tokenizer=tokenizer, prompts=(prompt,))[0]
    states = create_generation_states(encoded_prompts=(encoded_prompt,))
    device = _model_device(model=model)
    prefill_start_time = perf_counter()
    input_tensor = torch.tensor(
        [list(encoded_prompt.token_ids)],
        dtype=torch.long,
        device=device,
    )
    debug_generation = _debug_generation_enabled()
    _debug_log(debug_generation, "[generation-debug] kv_cache_single_prefill_forward_start")
    with torch.inference_mode():
        model_output = model.forward_with_cache(
            token_ids=input_tensor,
            inference_cache=model.empty_inference_cache(
                batch_size=input_tensor.shape[0],
                device=device,
            ),
        )
    prefill_seconds = perf_counter() - prefill_start_time
    decode_seconds = 0.0
    with torch.inference_mode():
        for generation_step in range(maximum_new_tokens):
            step_start_time = perf_counter()
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_single_select_start "
                f"step={generation_step + 1}",
            )
            next_token_id = select_next_token_id(
                logits=model_output.logits[0, -1, :],
                decoding_configuration=decoding_configuration,
            )
            select_seconds = perf_counter() - step_start_time
            decode_seconds += select_seconds
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_single_select_done "
                f"step={generation_step + 1} seconds={select_seconds:.4f}",
            )
            states[0] = append_next_token(
                state=states[0],
                next_token_id=next_token_id,
                tokenizer=tokenizer,
                stop_sequences=stop_sequences,
            )
            if states[0].stopped or generation_step == maximum_new_tokens - 1:
                break
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            step_start_time = perf_counter()
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_single_forward_start "
                f"step={generation_step + 1}",
            )
            model_output = model.forward_with_cache(
                token_ids=input_tensor,
                inference_cache=model_output.inference_cache,
            )
            forward_seconds = perf_counter() - step_start_time
            decode_seconds += forward_seconds
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_single_forward_done "
                f"step={generation_step + 1} seconds={forward_seconds:.4f}",
            )
    return finalize_generation_results(
        states=tuple(states),
        tokenizer=tokenizer,
        maximum_new_tokens=maximum_new_tokens,
        timing=BatchTimingAccumulator(
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
        ),
    )[0]


def _generate_equal_length_batch_with_cache(
    model: CachedAutoregressiveModel,
    tokenizer: TextTokenizer,
    prompts: tuple[str, ...],
    maximum_new_tokens: int,
    decoding_configuration: DecodingConfiguration,
    stop_sequences: tuple[str, ...],
) -> tuple[GenerationResult, ...]:
    encoded_prompts = encode_prompts(tokenizer=tokenizer, prompts=prompts)
    states = create_generation_states(encoded_prompts=encoded_prompts)
    device = _model_device(model=model)
    input_tensor = torch.tensor(
        [list(encoded_prompt.token_ids) for encoded_prompt in encoded_prompts],
        dtype=torch.long,
        device=device,
    )
    prefill_start_time = perf_counter()
    debug_generation = _debug_generation_enabled()
    _debug_log(debug_generation, "[generation-debug] kv_cache_batch_prefill_forward_start")
    with torch.inference_mode():
        model_output = model.forward_with_cache(
            token_ids=input_tensor,
            inference_cache=model.empty_inference_cache(
                batch_size=input_tensor.shape[0],
                device=device,
            ),
        )
    prefill_seconds = perf_counter() - prefill_start_time
    decode_seconds = 0.0
    with torch.inference_mode():
        for generation_step in range(maximum_new_tokens):
            active_indexes = tuple(
                sample_index
                for sample_index, state in enumerate(states)
                if not state.stopped
            )
            if len(active_indexes) == 0:
                break
            step_start_time = perf_counter()
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_batch_select_start "
                f"step={generation_step + 1}",
            )
            next_token_ids = select_next_token_ids(
                logits=model_output.logits[:, -1, :],
                decoding_configuration=decoding_configuration,
            )
            select_seconds = perf_counter() - step_start_time
            decode_seconds += select_seconds
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_batch_select_done "
                f"step={generation_step + 1} seconds={select_seconds:.4f}",
            )
            next_input_token_ids: list[int] = []
            for sample_index, state in enumerate(states):
                next_token_id = int(next_token_ids[sample_index].item())
                if state.stopped:
                    next_input_token_ids.append(tokenizer.eos_token_id)
                    continue
                states[sample_index] = append_next_token(
                    state=state,
                    next_token_id=next_token_id,
                    tokenizer=tokenizer,
                    stop_sequences=stop_sequences,
                )
                next_input_token_ids.append(next_token_id)
            if generation_step == maximum_new_tokens - 1:
                break
            if all(state.stopped for state in states):
                break
            input_tensor = torch.tensor(
                [[next_token_id] for next_token_id in next_input_token_ids],
                dtype=torch.long,
                device=device,
            )
            step_start_time = perf_counter()
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_batch_forward_start "
                f"step={generation_step + 1}",
            )
            model_output = model.forward_with_cache(
                token_ids=input_tensor,
                inference_cache=model_output.inference_cache,
            )
            forward_seconds = perf_counter() - step_start_time
            decode_seconds += forward_seconds
            _debug_log(
                debug_generation,
                "[generation-debug] kv_cache_batch_forward_done "
                f"step={generation_step + 1} seconds={forward_seconds:.4f}",
            )
    return finalize_generation_results(
        states=tuple(states),
        tokenizer=tokenizer,
        maximum_new_tokens=maximum_new_tokens,
        timing=BatchTimingAccumulator(
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
        ),
    )


def _model_device(model: CachedAutoregressiveModel) -> torch.device:
    return next(model.parameters()).device


def _debug_generation_enabled() -> bool:
    return os.environ.get("LLM_LITE_GENERATION_DEBUG", "").lower() in {"1", "true", "yes"}


def _debug_log(enabled: bool, message: str) -> None:
    if enabled:
        console_log(message)
