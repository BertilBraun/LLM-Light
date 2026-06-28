from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional
from torch import nn

from llm_lite.config.models import ModernDenseGptConfiguration, ModernMoeGptConfiguration
from llm_lite.model.cache import (
    CachedModelOutput,
    GptInferenceCache,
    TransformerBlockInferenceCache,
)
from llm_lite.model.output import ModelOutput
from llm_lite.model.protocol import CachedAutoregressiveModel
from llm_lite.model.routing import ExpertRoutingResult, RouterUsageSummary, TopKRouter

ModernModelConfiguration = ModernDenseGptConfiguration | ModernMoeGptConfiguration


class ModernDenseGpt(CachedAutoregressiveModel):
    def __init__(
        self,
        model_configuration: ModernDenseGptConfiguration,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        _validate_attention_shape(model_configuration=model_configuration)
        self.model_configuration = model_configuration
        self.token_embedding = nn.Embedding(vocabulary_size, model_configuration.dimension)
        self.blocks = nn.ModuleList(
            [
                ModernDenseTransformerBlock(model_configuration=model_configuration)
                for _ in range(model_configuration.layers)
            ],
        )
        self.final_normalization = RMSNorm(
            dimension=model_configuration.dimension,
            epsilon=model_configuration.normalization_epsilon,
        )
        self.output_projection = nn.Linear(
            model_configuration.dimension,
            vocabulary_size,
            bias=False,
        )
        if model_configuration.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> ModelOutput:
        batch_size, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states=hidden_states, positions=positions)
        hidden_states = self.final_normalization(hidden_states)
        logits = self.output_projection(hidden_states)
        assert logits.shape[0] == batch_size
        return ModelOutput(logits=logits)

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        inference_cache: GptInferenceCache,
    ) -> CachedModelOutput:
        batch_size, sequence_length = token_ids.shape
        cache_sequence_length = _cache_sequence_length(inference_cache=inference_cache)
        positions = torch.arange(
            cache_sequence_length,
            cache_sequence_length + sequence_length,
            device=token_ids.device,
        )
        hidden_states = self.token_embedding(token_ids)
        cache_layers: list[TransformerBlockInferenceCache] = []
        for block, layer_cache in zip(self.blocks, inference_cache.layers, strict=True):
            block_output = block.forward_with_cache(
                hidden_states=hidden_states,
                positions=positions,
                inference_cache=layer_cache,
            )
            hidden_states = block_output.hidden_states
            cache_layers.append(block_output.inference_cache)
        hidden_states = self.final_normalization(hidden_states)
        logits = self.output_projection(hidden_states)
        assert logits.shape[0] == batch_size
        return CachedModelOutput(
            logits=logits,
            inference_cache=GptInferenceCache(layers=tuple(cache_layers)),
        )

    def empty_inference_cache(self, batch_size: int, device: torch.device) -> GptInferenceCache:
        empty_layers = tuple(
            block.empty_inference_cache(batch_size=batch_size, device=device)
            for block in self.blocks
        )
        return GptInferenceCache(layers=empty_layers)


class ModernMoeGpt(CachedAutoregressiveModel):
    def __init__(
        self,
        model_configuration: ModernMoeGptConfiguration,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        _validate_attention_shape(model_configuration=model_configuration)
        self.model_configuration = model_configuration
        self.token_embedding = nn.Embedding(vocabulary_size, model_configuration.dimension)
        self.blocks = nn.ModuleList(
            [
                ModernMoeTransformerBlock(model_configuration=model_configuration)
                for _ in range(model_configuration.layers)
            ],
        )
        self.final_normalization = RMSNorm(
            dimension=model_configuration.dimension,
            epsilon=model_configuration.normalization_epsilon,
        )
        self.output_projection = nn.Linear(
            model_configuration.dimension,
            vocabulary_size,
            bias=False,
        )
        if model_configuration.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> ModelOutput:
        batch_size, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embedding(token_ids)
        auxiliary_losses: list[torch.Tensor] = []
        for block in self.blocks:
            block_output = block(hidden_states=hidden_states, positions=positions)
            hidden_states = block_output.hidden_states
            auxiliary_losses.append(block_output.routing_result.auxiliary_loss)
        hidden_states = self.final_normalization(hidden_states)
        logits = self.output_projection(hidden_states)
        assert logits.shape[0] == batch_size
        return ModelOutput(
            logits=logits,
            auxiliary_loss=torch.stack(auxiliary_losses).mean(),
        )

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        inference_cache: GptInferenceCache,
    ) -> CachedModelOutput:
        batch_size, sequence_length = token_ids.shape
        cache_sequence_length = _cache_sequence_length(inference_cache=inference_cache)
        positions = torch.arange(
            cache_sequence_length,
            cache_sequence_length + sequence_length,
            device=token_ids.device,
        )
        hidden_states = self.token_embedding(token_ids)
        cache_layers: list[TransformerBlockInferenceCache] = []
        for block, layer_cache in zip(self.blocks, inference_cache.layers, strict=True):
            block_output = block.forward_with_cache(
                hidden_states=hidden_states,
                positions=positions,
                inference_cache=layer_cache,
            )
            hidden_states = block_output.hidden_states
            cache_layers.append(block_output.inference_cache)
        hidden_states = self.final_normalization(hidden_states)
        logits = self.output_projection(hidden_states)
        assert logits.shape[0] == batch_size
        return CachedModelOutput(
            logits=logits,
            inference_cache=GptInferenceCache(layers=tuple(cache_layers)),
        )

    def empty_inference_cache(self, batch_size: int, device: torch.device) -> GptInferenceCache:
        empty_layers = tuple(
            block.empty_inference_cache(batch_size=batch_size, device=device)
            for block in self.blocks
        )
        return GptInferenceCache(layers=empty_layers)

    def active_parameter_count(self) -> int:
        return self._shared_parameter_count(trainable_only=False) + self._active_expert_count(
            trainable_only=False,
        )

    def trainable_active_parameter_count(self) -> int:
        return self._shared_parameter_count(trainable_only=True) + self._active_expert_count(
            trainable_only=True,
        )

    def _shared_parameter_count(self, trainable_only: bool) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if _count_parameter(parameter=parameter, trainable_only=trainable_only)
        ) - self._total_expert_parameter_count(trainable_only=trainable_only)

    def _active_expert_count(self, trainable_only: bool) -> int:
        return sum(
            block.active_expert_parameter_count(
                top_k=self.model_configuration.router_top_k,
                trainable_only=trainable_only,
            )
            for block in self.blocks
        )

    def _total_expert_parameter_count(self, trainable_only: bool) -> int:
        return sum(
            block.total_expert_parameter_count(trainable_only=trainable_only)
            for block in self.blocks
        )

    def router_usage_summaries(self) -> tuple[RouterUsageSummary, ...]:
        return tuple(
            block.router_usage_summary(layer_index=layer_index)
            for layer_index, block in enumerate(self.blocks)
        )

    def reset_router_usage(self) -> None:
        for block in self.blocks:
            block.reset_router_usage()


class ModernDenseTransformerBlock(nn.Module):
    def __init__(self, model_configuration: ModernDenseGptConfiguration) -> None:
        super().__init__()
        self.attention_normalization = RMSNorm(
            dimension=model_configuration.dimension,
            epsilon=model_configuration.normalization_epsilon,
        )
        self.attention = RotarySelfAttention(model_configuration=model_configuration)
        self.feed_forward_normalization = RMSNorm(
            dimension=model_configuration.dimension,
            epsilon=model_configuration.normalization_epsilon,
        )
        self.feed_forward = SwiGLUFeedForward(
            dimension=model_configuration.dimension,
            feed_forward_dimension=model_configuration.feed_forward_dimension,
            dropout=model_configuration.dropout,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            hidden_states=self.attention_normalization(hidden_states),
            positions=positions,
        )
        return hidden_states + self.feed_forward(
            self.feed_forward_normalization(hidden_states),
        )

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        inference_cache: TransformerBlockInferenceCache,
    ) -> ModernCachedBlockOutput:
        attention_output, next_cache = self.attention.forward_with_cache(
            hidden_states=self.attention_normalization(hidden_states),
            positions=positions,
            inference_cache=inference_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_normalization(hidden_states),
        )
        return ModernCachedBlockOutput(
            hidden_states=hidden_states,
            inference_cache=next_cache,
        )

    def empty_inference_cache(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TransformerBlockInferenceCache:
        return self.attention.empty_inference_cache(batch_size=batch_size, device=device)


class ModernMoeTransformerBlock(nn.Module):
    def __init__(self, model_configuration: ModernMoeGptConfiguration) -> None:
        super().__init__()
        self.attention_normalization = RMSNorm(
            dimension=model_configuration.dimension,
            epsilon=model_configuration.normalization_epsilon,
        )
        self.attention = RotarySelfAttention(model_configuration=model_configuration)
        self.feed_forward_normalization = RMSNorm(
            dimension=model_configuration.dimension,
            epsilon=model_configuration.normalization_epsilon,
        )
        self.feed_forward = ModernMoeFeedForward(model_configuration=model_configuration)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> ModernMoeBlockOutput:
        hidden_states = hidden_states + self.attention(
            hidden_states=self.attention_normalization(hidden_states),
            positions=positions,
        )
        feed_forward_output = self.feed_forward(
            hidden_states=self.feed_forward_normalization(hidden_states),
        )
        return ModernMoeBlockOutput(
            hidden_states=hidden_states + feed_forward_output.hidden_states,
            routing_result=feed_forward_output.routing_result,
        )

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        inference_cache: TransformerBlockInferenceCache,
    ) -> ModernCachedBlockOutput:
        attention_output, next_cache = self.attention.forward_with_cache(
            hidden_states=self.attention_normalization(hidden_states),
            positions=positions,
            inference_cache=inference_cache,
        )
        hidden_states = hidden_states + attention_output
        feed_forward_output = self.feed_forward(
            hidden_states=self.feed_forward_normalization(hidden_states),
        )
        return ModernCachedBlockOutput(
            hidden_states=hidden_states + feed_forward_output.hidden_states,
            inference_cache=next_cache,
        )

    def empty_inference_cache(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TransformerBlockInferenceCache:
        return self.attention.empty_inference_cache(batch_size=batch_size, device=device)

    def active_expert_parameter_count(self, top_k: int, trainable_only: bool) -> int:
        first_expert = self.feed_forward.experts[0]
        return top_k * sum(
            parameter.numel()
            for parameter in first_expert.parameters()
            if _count_parameter(parameter=parameter, trainable_only=trainable_only)
        )

    def total_expert_parameter_count(self, trainable_only: bool) -> int:
        return sum(
            parameter.numel()
            for expert in self.feed_forward.experts
            for parameter in expert.parameters()
            if _count_parameter(parameter=parameter, trainable_only=trainable_only)
        )

    def router_usage_summary(self, layer_index: int) -> RouterUsageSummary:
        return self.feed_forward.router.usage_summary(layer_index=layer_index)

    def reset_router_usage(self) -> None:
        self.feed_forward.router.reset_usage()


class RotarySelfAttention(nn.Module):
    def __init__(self, model_configuration: ModernModelConfiguration) -> None:
        super().__init__()
        self.dimension = model_configuration.dimension
        self.attention_heads = model_configuration.attention_heads
        self.head_dimension = model_configuration.dimension // model_configuration.attention_heads
        self.dropout = model_configuration.dropout
        self.query_key_value_projection = nn.Linear(
            model_configuration.dimension,
            3 * model_configuration.dimension,
            bias=False,
        )
        self.output_projection = nn.Linear(
            model_configuration.dimension,
            model_configuration.dimension,
            bias=False,
        )
        self.rotary_embedding = RotaryEmbedding(
            head_dimension=self.head_dimension,
            base=model_configuration.rope_base,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query_states, key_states, value_states = self._project_states(hidden_states=hidden_states)
        query_states = self.rotary_embedding(states=query_states, positions=positions)
        key_states = self.rotary_embedding(states=key_states, positions=positions)
        attention_mask = _causal_attention_mask(
            sequence_length=hidden_states.shape[1],
            device=hidden_states.device,
        )
        attention_output = _scaled_dot_product_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            dropout=self.dropout,
            training=self.training,
        )
        return self.output_projection(_combine_heads(states=attention_output))

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        inference_cache: TransformerBlockInferenceCache,
    ) -> tuple[torch.Tensor, TransformerBlockInferenceCache]:
        query_states, key_states, value_states = self._project_states(hidden_states=hidden_states)
        query_states = self.rotary_embedding(states=query_states, positions=positions)
        key_states = self.rotary_embedding(states=key_states, positions=positions)
        cached_key_states = torch.cat((inference_cache.key_states, key_states), dim=2)
        cached_value_states = torch.cat((inference_cache.value_states, value_states), dim=2)
        attention_mask = _causal_cache_mask(
            past_sequence_length=inference_cache.key_states.shape[2],
            sequence_length=hidden_states.shape[1],
            total_sequence_length=cached_key_states.shape[2],
            device=hidden_states.device,
        )
        attention_output = _scaled_dot_product_attention(
            query_states=query_states,
            key_states=cached_key_states,
            value_states=cached_value_states,
            attention_mask=attention_mask,
            dropout=self.dropout,
            training=self.training,
        )
        return self.output_projection(_combine_heads(states=attention_output)), (
            TransformerBlockInferenceCache(
                key_states=cached_key_states,
                value_states=cached_value_states,
            )
        )

    def empty_inference_cache(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TransformerBlockInferenceCache:
        empty_cache_shape = (batch_size, self.attention_heads, 0, self.head_dimension)
        cache_dtype = self.query_key_value_projection.weight.dtype
        return TransformerBlockInferenceCache(
            key_states=torch.empty(empty_cache_shape, device=device, dtype=cache_dtype),
            value_states=torch.empty(empty_cache_shape, device=device, dtype=cache_dtype),
        )

    def _project_states(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_states = self.query_key_value_projection(hidden_states)
        query_states, key_states, value_states = projected_states.chunk(3, dim=-1)
        return (
            _split_heads(states=query_states, attention_heads=self.attention_heads),
            _split_heads(states=key_states, attention_heads=self.attention_heads),
            _split_heads(states=value_states, attention_heads=self.attention_heads),
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dimension: int, base: float) -> None:
        super().__init__()
        if head_dimension % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        inverse_frequencies = 1.0 / (
            base
            ** (torch.arange(0, head_dimension, 2, dtype=torch.float32) / float(head_dimension))
        )
        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )

    def forward(self, states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        frequencies = torch.outer(
            positions.to(dtype=self.inverse_frequencies.dtype),
            self.inverse_frequencies,
        ).to(device=states.device)
        cosine = torch.cos(frequencies).to(dtype=states.dtype)[None, None, :, :]
        sine = torch.sin(frequencies).to(dtype=states.dtype)[None, None, :, :]
        even_states = states[..., 0::2]
        odd_states = states[..., 1::2]
        rotated_even_states = even_states * cosine - odd_states * sine
        rotated_odd_states = even_states * sine + odd_states * cosine
        return torch.stack((rotated_even_states, rotated_odd_states), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized_states = hidden_states * torch.rsqrt(variance + self.epsilon)
        return normalized_states * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dimension: int, feed_forward_dimension: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(dimension, 2 * feed_forward_dimension, bias=False)
        self.output_projection = nn.Linear(feed_forward_dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_states, value_states = self.input_projection(hidden_states).chunk(2, dim=-1)
        hidden_states = torch_functional.silu(gate_states) * value_states
        return self.dropout(self.output_projection(hidden_states))


class ModernMoeFeedForward(nn.Module):
    def __init__(self, model_configuration: ModernMoeGptConfiguration) -> None:
        super().__init__()
        self.expert_count = model_configuration.expert_count
        self.router = TopKRouter(
            dimension=model_configuration.dimension,
            expert_count=model_configuration.expert_count,
            top_k=model_configuration.router_top_k,
        )
        self.experts = nn.ModuleList(
            [
                SwiGLUFeedForward(
                    dimension=model_configuration.dimension,
                    feed_forward_dimension=model_configuration.expert_feed_forward_dimension,
                    dropout=model_configuration.dropout,
                )
                for _ in range(model_configuration.expert_count)
            ],
        )

    def forward(self, hidden_states: torch.Tensor) -> ModernMoeFeedForwardOutput:
        routing_result = self.router(hidden_states=hidden_states)
        combined_states = torch.zeros_like(hidden_states)
        for expert_index, expert in enumerate(self.experts):
            for route_index in range(routing_result.top_expert_indices.shape[-1]):
                expert_token_mask = (
                    routing_result.top_expert_indices[..., route_index] == expert_index
                )
                if not bool(expert_token_mask.any()):
                    continue
                expert_input = hidden_states[expert_token_mask]
                expert_output = expert(expert_input)
                expert_weight = routing_result.top_expert_weights[..., route_index][
                    expert_token_mask
                ].unsqueeze(dim=-1)
                combined_states[expert_token_mask] += expert_output * expert_weight
        return ModernMoeFeedForwardOutput(
            hidden_states=combined_states,
            routing_result=routing_result,
        )


@dataclass(frozen=True)
class ModernMoeFeedForwardOutput:
    hidden_states: torch.Tensor
    routing_result: ExpertRoutingResult


@dataclass(frozen=True)
class ModernMoeBlockOutput:
    hidden_states: torch.Tensor
    routing_result: ExpertRoutingResult


@dataclass(frozen=True)
class ModernCachedBlockOutput:
    hidden_states: torch.Tensor
    inference_cache: TransformerBlockInferenceCache


def _validate_attention_shape(model_configuration: ModernModelConfiguration) -> None:
    if model_configuration.dimension % model_configuration.attention_heads != 0:
        raise ValueError("Model dimension must be divisible by attention heads.")
    head_dimension = model_configuration.dimension // model_configuration.attention_heads
    if head_dimension % 2 != 0:
        raise ValueError("Modern GPT requires an even attention head dimension for RoPE.")


def _cache_sequence_length(inference_cache: GptInferenceCache) -> int:
    if not inference_cache.layers:
        return 0
    first_layer_cache = inference_cache.layers[0]
    return int(first_layer_cache.key_states.shape[2])


def _split_heads(states: torch.Tensor, attention_heads: int) -> torch.Tensor:
    batch_size, sequence_length, embedding_dimension = states.shape
    head_dimension = embedding_dimension // attention_heads
    states = states.view(batch_size, sequence_length, attention_heads, head_dimension)
    return states.transpose(1, 2)


def _combine_heads(states: torch.Tensor) -> torch.Tensor:
    batch_size, attention_heads, sequence_length, head_dimension = states.shape
    states = states.transpose(1, 2).contiguous()
    return states.view(batch_size, sequence_length, attention_heads * head_dimension)


def _scaled_dot_product_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    attention_scores = (query_states @ key_states.transpose(-2, -1)) / math.sqrt(
        query_states.shape[-1],
    )
    attention_scores = attention_scores.masked_fill(
        attention_mask[None, None, :, :],
        float("-inf"),
    )
    attention_probabilities = torch.softmax(attention_scores, dim=-1)
    attention_probabilities = torch_functional.dropout(
        attention_probabilities,
        p=dropout,
        training=training,
    )
    return attention_probabilities @ value_states


def _causal_attention_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool),
        diagonal=1,
    )


def _causal_cache_mask(
    past_sequence_length: int,
    sequence_length: int,
    total_sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(
        past_sequence_length,
        past_sequence_length + sequence_length,
        device=device,
    )
    key_positions = torch.arange(total_sequence_length, device=device)
    return key_positions[None, :] > query_positions[:, None]


def _count_parameter(parameter: nn.Parameter, trainable_only: bool) -> bool:
    return not trainable_only or parameter.requires_grad
