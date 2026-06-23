from torch import nn

from llm_lite.config.models import InferenceConfiguration, InferenceEngine
from llm_lite.inference import kv_cache, naive
from llm_lite.inference.runtime import GenerationResult, prepare_model_for_inference
from llm_lite.model.gpt import DenseGpt
from llm_lite.tokenizer.loading import TextTokenizer


def generate_text(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    inference_configuration: InferenceConfiguration,
) -> str:
    return generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=(prompt,),
        inference_configuration=inference_configuration,
    )[0].full_text


def generate_batch(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompts: tuple[str, ...],
    inference_configuration: InferenceConfiguration,
) -> tuple[GenerationResult, ...]:
    prepared_model = prepare_model_for_inference(
        model=model,
        inference_configuration=inference_configuration,
    )
    results: list[GenerationResult] = []
    for prompt_batch in _chunk_prompts(
        prompts=prompts,
        batch_size=inference_configuration.batch_size,
    ):
        results.extend(
            _generate_prompt_batch(
                model=prepared_model,
                tokenizer=tokenizer,
                prompts=prompt_batch,
                inference_configuration=inference_configuration,
            ),
        )
    return tuple(results)


def _generate_prompt_batch(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompts: tuple[str, ...],
    inference_configuration: InferenceConfiguration,
) -> tuple[GenerationResult, ...]:
    match inference_configuration.engine:
        case InferenceEngine.NAIVE:
            return naive.generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                maximum_new_tokens=inference_configuration.maximum_new_tokens,
                decoding_configuration=inference_configuration.decoding,
                stop_sequences=inference_configuration.stop_sequences,
            )
        case InferenceEngine.KV_CACHE:
            match model:
                case DenseGpt():
                    return kv_cache.generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        maximum_new_tokens=inference_configuration.maximum_new_tokens,
                        decoding_configuration=inference_configuration.decoding,
                        stop_sequences=inference_configuration.stop_sequences,
                    )
                case _:
                    raise ValueError("KV-cache inference requires a DenseGpt model.")


def _chunk_prompts(
    prompts: tuple[str, ...],
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        prompts[start_index : start_index + batch_size]
        for start_index in range(0, len(prompts), batch_size)
    )
