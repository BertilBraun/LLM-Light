import torch
from torch import nn

from llm_lite.model.cache import CachedModelOutput, GptInferenceCache


class CachedAutoregressiveModel(nn.Module):
    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        inference_cache: GptInferenceCache,
    ) -> CachedModelOutput:
        raise NotImplementedError

    def empty_inference_cache(self, batch_size: int, device: torch.device) -> GptInferenceCache:
        raise NotImplementedError
