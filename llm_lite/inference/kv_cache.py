import torch

from llm_lite.config.models import (
    DecodingConfiguration,
    DecodingStrategy,
    GreedyDecodingConfiguration,
)
from llm_lite.inference.decoding import select_next_token_id
from llm_lite.model.gpt import DenseGpt
from llm_lite.tokenizer.loading import TextTokenizer


def generate(
    model: DenseGpt,
    tokenizer: TextTokenizer,
    prompt: str,
    maximum_new_tokens: int,
    decoding_configuration: DecodingConfiguration,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(text=prompt, add_bos=True, add_eos=False)
    generated_token_ids = list(token_ids)
    if not generated_token_ids:
        raise ValueError("KV-cache inference requires at least one prompt token.")
    with torch.no_grad():
        input_tensor = torch.tensor([generated_token_ids], dtype=torch.long)
        model_output = model.forward_with_cache(
            token_ids=input_tensor,
            inference_cache=model.empty_inference_cache(
                batch_size=input_tensor.shape[0],
                device=input_tensor.device,
            ),
        )
        for generation_step in range(maximum_new_tokens):
            next_token_id = select_next_token_id(
                logits=model_output.logits[0, -1, :],
                decoding_configuration=decoding_configuration,
            )
            if next_token_id == tokenizer.eos_token_id:
                break
            generated_token_ids.append(next_token_id)
            if generation_step == maximum_new_tokens - 1:
                break
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            model_output = model.forward_with_cache(
                token_ids=input_tensor,
                inference_cache=model_output.inference_cache,
            )
    return tokenizer.decode(generated_token_ids)


def generate_greedy(
    model: DenseGpt,
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
