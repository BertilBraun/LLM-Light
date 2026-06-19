import torch
from torch import nn


def causal_language_modeling_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    input_logits = logits[:, :-1, :].contiguous()
    target_token_ids = token_ids[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        input_logits.view(-1, input_logits.shape[-1]),
        target_token_ids.view(-1),
    )
