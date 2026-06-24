from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ExpertRoutingResult:
    router_logits: torch.Tensor
    top_expert_indices: torch.Tensor
    top_expert_weights: torch.Tensor
    auxiliary_loss: torch.Tensor


class TopKRouter(nn.Module):
    def __init__(self, dimension: int, expert_count: int, top_k: int) -> None:
        super().__init__()
        if top_k > expert_count:
            raise ValueError("Router top-k must not be greater than expert count.")
        self.expert_count = expert_count
        self.top_k = top_k
        self.projection = nn.Linear(dimension, expert_count, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> ExpertRoutingResult:
        router_logits = self.projection(hidden_states)
        routing_probabilities = torch.softmax(router_logits, dim=-1)
        top_expert_weights, top_expert_indices = torch.topk(
            routing_probabilities,
            k=self.top_k,
            dim=-1,
        )
        normalized_top_expert_weights = top_expert_weights / top_expert_weights.sum(
            dim=-1,
            keepdim=True,
        )
        return ExpertRoutingResult(
            router_logits=router_logits,
            top_expert_indices=top_expert_indices,
            top_expert_weights=normalized_top_expert_weights,
            auxiliary_loss=_load_balancing_loss(
                routing_probabilities=routing_probabilities,
                top_expert_indices=top_expert_indices,
                expert_count=self.expert_count,
            ),
        )


def _load_balancing_loss(
    routing_probabilities: torch.Tensor,
    top_expert_indices: torch.Tensor,
    expert_count: int,
) -> torch.Tensor:
    tokens_per_expert = torch.zeros(
        expert_count,
        device=routing_probabilities.device,
        dtype=routing_probabilities.dtype,
    )
    top_one_expert_indices = top_expert_indices[..., 0].reshape(-1)
    tokens_per_expert.scatter_add_(
        dim=0,
        index=top_one_expert_indices,
        src=torch.ones_like(top_one_expert_indices, dtype=routing_probabilities.dtype),
    )
    token_density = tokens_per_expert / top_one_expert_indices.numel()
    probability_density = routing_probabilities.reshape(-1, expert_count).mean(dim=0)
    return expert_count * torch.sum(token_density * probability_density)
