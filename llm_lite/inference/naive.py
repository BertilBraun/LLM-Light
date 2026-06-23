import torch
from torch import nn

from llm_lite.tokenizer.loading import TextTokenizer


def generate_greedy(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    maximum_new_tokens: int,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(text=prompt, add_bos=True, add_eos=False)
    generated_token_ids = list(token_ids)
    with torch.no_grad():
        for _ in range(maximum_new_tokens):
            input_tensor = torch.tensor([generated_token_ids], dtype=torch.long)
            model_output = model(input_tensor)
            next_token_id = int(torch.argmax(model_output.logits[0, -1, :]).item())
            if next_token_id == tokenizer.eos_token_id:
                break
            generated_token_ids.append(next_token_id)
    return tokenizer.decode(generated_token_ids)
