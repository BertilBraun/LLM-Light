from torch import nn

from llm_lite.config.models import InferenceConfiguration, InferenceEngine
from llm_lite.inference import kv_cache, naive
from llm_lite.inference.runtime import prepare_model_for_inference
from llm_lite.model.gpt import DenseGpt
from llm_lite.tokenizer.loading import TextTokenizer


def generate_text(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    inference_configuration: InferenceConfiguration,
) -> str:
    prepared_model = prepare_model_for_inference(
        model=model,
        inference_configuration=inference_configuration,
    )
    match inference_configuration.engine:
        case InferenceEngine.NAIVE:
            return naive.generate(
                model=prepared_model,
                tokenizer=tokenizer,
                prompt=prompt,
                maximum_new_tokens=inference_configuration.maximum_new_tokens,
                decoding_configuration=inference_configuration.decoding,
            )
        case InferenceEngine.KV_CACHE:
            match prepared_model:
                case DenseGpt():
                    return kv_cache.generate(
                        model=prepared_model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        maximum_new_tokens=inference_configuration.maximum_new_tokens,
                        decoding_configuration=inference_configuration.decoding,
                    )
                case _:
                    raise ValueError("KV-cache inference requires a DenseGpt model.")
