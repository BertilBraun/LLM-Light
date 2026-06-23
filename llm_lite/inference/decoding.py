import torch

from llm_lite.config.models import (
    DecodingConfiguration,
    GreedyDecodingConfiguration,
    SamplingDecodingConfiguration,
)


def select_next_token_id(
    logits: torch.Tensor,
    decoding_configuration: DecodingConfiguration,
) -> int:
    match decoding_configuration:
        case GreedyDecodingConfiguration():
            return int(torch.argmax(logits).item())
        case SamplingDecodingConfiguration(temperature=temperature, top_k=top_k):
            return _sample_token_id(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
            )


def select_next_token_ids(
    logits: torch.Tensor,
    decoding_configuration: DecodingConfiguration,
) -> torch.Tensor:
    match decoding_configuration:
        case GreedyDecodingConfiguration():
            return torch.argmax(logits, dim=-1)
        case SamplingDecodingConfiguration(temperature=temperature, top_k=top_k):
            return _sample_token_ids(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
            )


def _sample_token_id(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
    sampling_logits = logits / temperature
    if top_k is None:
        probabilities = torch.softmax(sampling_logits, dim=-1)
        return int(torch.multinomial(probabilities, num_samples=1).item())
    selected_logits, selected_token_ids = torch.topk(
        sampling_logits,
        k=min(top_k, sampling_logits.shape[-1]),
    )
    probabilities = torch.softmax(selected_logits, dim=-1)
    sampled_index = int(torch.multinomial(probabilities, num_samples=1).item())
    return int(selected_token_ids[sampled_index].item())


def _sample_token_ids(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    sampling_logits = logits / temperature
    if top_k is None:
        probabilities = torch.softmax(sampling_logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(dim=-1)
    selected_logits, selected_token_ids = torch.topk(
        sampling_logits,
        k=min(top_k, sampling_logits.shape[-1]),
        dim=-1,
    )
    probabilities = torch.softmax(selected_logits, dim=-1)
    sampled_indexes = torch.multinomial(probabilities, num_samples=1)
    return selected_token_ids.gather(dim=-1, index=sampled_indexes).squeeze(dim=-1)
