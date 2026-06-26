from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from llm_lite.post_training.preference import DpoPreferenceRecord
from llm_lite.tokenizer.loading import TextTokenizer
from llm_lite.training.objectives import DpoPreferenceBatch


class DpoPreferenceDataset(Dataset[DpoPreferenceBatch]):
    def __init__(self, samples: tuple[DpoPreferenceBatch, ...]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DpoPreferenceBatch:
        return self.samples[index]


def build_dpo_preference_dataset(
    preferences: tuple[DpoPreferenceRecord, ...],
    tokenizer: TextTokenizer,
) -> DpoPreferenceDataset:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("DPO preference dataset requires a tokenizer pad token.")
    encoded_pairs = tuple(
        _encode_preference_pair(preference=preference, tokenizer=tokenizer)
        for preference in preferences
    )
    if len(encoded_pairs) == 0:
        raise ValueError("DPO preference dataset requires at least one preference.")
    maximum_length = max(
        max(len(pair.chosen_token_ids), len(pair.rejected_token_ids)) for pair in encoded_pairs
    )
    samples = tuple(
        DpoPreferenceBatch(
            chosen_token_ids=_pad_token_ids(
                token_ids=pair.chosen_token_ids,
                pad_token_id=pad_token_id,
                maximum_length=maximum_length,
            ),
            rejected_token_ids=_pad_token_ids(
                token_ids=pair.rejected_token_ids,
                pad_token_id=pad_token_id,
                maximum_length=maximum_length,
            ),
            chosen_completion_mask=_pad_mask(
                mask=pair.chosen_completion_mask,
                maximum_length=maximum_length,
            ),
            rejected_completion_mask=_pad_mask(
                mask=pair.rejected_completion_mask,
                maximum_length=maximum_length,
            ),
        )
        for pair in encoded_pairs
    )
    return DpoPreferenceDataset(samples=samples)


@dataclass(frozen=True)
class EncodedPreferencePair:
    chosen_token_ids: tuple[int, ...]
    rejected_token_ids: tuple[int, ...]
    chosen_completion_mask: tuple[bool, ...]
    rejected_completion_mask: tuple[bool, ...]


def _encode_preference_pair(
    preference: DpoPreferenceRecord,
    tokenizer: TextTokenizer,
) -> EncodedPreferencePair:
    prompt_token_ids = tokenizer.encode(text=preference.prompt, add_bos=True, add_eos=False)
    chosen_completion_token_ids = tokenizer.encode(
        text=preference.chosen_completion,
        add_bos=False,
        add_eos=True,
    )
    rejected_completion_token_ids = tokenizer.encode(
        text=preference.rejected_completion,
        add_bos=False,
        add_eos=True,
    )
    chosen_token_ids = tuple([*prompt_token_ids, *chosen_completion_token_ids])
    rejected_token_ids = tuple([*prompt_token_ids, *rejected_completion_token_ids])
    return EncodedPreferencePair(
        chosen_token_ids=chosen_token_ids,
        rejected_token_ids=rejected_token_ids,
        chosen_completion_mask=_completion_mask(
            prompt_length=len(prompt_token_ids),
            completion_length=len(chosen_completion_token_ids),
        ),
        rejected_completion_mask=_completion_mask(
            prompt_length=len(prompt_token_ids),
            completion_length=len(rejected_completion_token_ids),
        ),
    )


def _completion_mask(prompt_length: int, completion_length: int) -> tuple[bool, ...]:
    return tuple([False] * prompt_length + [True] * completion_length)


def _pad_token_ids(
    token_ids: tuple[int, ...],
    pad_token_id: int,
    maximum_length: int,
) -> torch.Tensor:
    padded_token_ids = [*token_ids, *([pad_token_id] * (maximum_length - len(token_ids)))]
    return torch.tensor(padded_token_ids, dtype=torch.long)


def _pad_mask(mask: tuple[bool, ...], maximum_length: int) -> torch.Tensor:
    padded_mask = [*mask, *([False] * (maximum_length - len(mask)))]
    return torch.tensor(padded_mask, dtype=torch.bool)
