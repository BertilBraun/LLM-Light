from time import perf_counter

import torch
from torch import nn

from llm_lite.config.models import (
    DecodingConfiguration,
    DecodingStrategy,
    GreedyDecodingConfiguration,
)
from llm_lite.inference.decoding import select_next_token_id
from llm_lite.inference.runtime import (
    BatchTimingAccumulator,
    GenerationResult,
    append_next_token,
    create_generation_states,
    encode_prompts,
    finalize_generation_results,
)
from llm_lite.tokenizer.loading import TextTokenizer


def generate_batch(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompts: tuple[str, ...],
    maximum_new_tokens: int,
    decoding_configuration: DecodingConfiguration,
    stop_sequences: tuple[str, ...],
) -> tuple[GenerationResult, ...]:
    model.eval()
    encoded_prompts = encode_prompts(tokenizer=tokenizer, prompts=prompts)
    states = create_generation_states(encoded_prompts=encoded_prompts)
    prefill_seconds = 0.0
    decode_seconds = 0.0
    with torch.no_grad():
        for _generation_step in range(maximum_new_tokens):
            active_indexes = tuple(
                sample_index
                for sample_index, state in enumerate(states)
                if not state.stopped
            )
            if len(active_indexes) == 0:
                break
            for sample_index in active_indexes:
                state = states[sample_index]
                token_ids = [*state.prompt_token_ids, *state.generated_token_ids]
                input_tensor = torch.tensor([token_ids], dtype=torch.long)
                step_start_time = perf_counter()
                model_output = model(input_tensor)
                decode_seconds += perf_counter() - step_start_time
                next_token_id = select_next_token_id(
                    logits=model_output.logits[0, -1, :],
                    decoding_configuration=decoding_configuration,
                )
                states[sample_index] = append_next_token(
                    state=state,
                    next_token_id=next_token_id,
                    tokenizer=tokenizer,
                    stop_sequences=stop_sequences,
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


def generate(
    model: nn.Module,
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
    model: nn.Module,
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
