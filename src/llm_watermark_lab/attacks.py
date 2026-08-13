"""Attacks against text watermarks: random token substitution and paraphrase."""

import torch


def substitute_tokens(token_ids: list[int], rate: float, pool: list[int], seed: int = 0) -> list[int]:
    """Replace a fraction `rate` of tokens with random tokens drawn from `pool`."""
    g = torch.Generator().manual_seed(seed)
    n = len(token_ids)
    k = int(n * rate)
    positions = torch.randperm(n, generator=g)[:k].tolist()
    out = list(token_ids)
    for pos in positions:
        idx = int(torch.randint(len(pool), (1,), generator=g))
        out[pos] = pool[idx]
    return out


@torch.no_grad()
def paraphrase(model, tok, text: str, max_new_tokens: int = 300, seed: int = 1) -> list[int]:
    """Paraphrase attack: rewrite the text with the same model, no watermark."""
    from .watermark import generate

    prompt = (
        "Rewrite the following text in your own words. Keep the meaning the same "
        "but change the wording as much as possible. Output only the rewritten "
        f"text.\n\n{text}"
    )
    result = generate(model, tok, prompt, scheme="vanilla", key=0,
                      max_new_tokens=max_new_tokens, seed=seed)
    return result.token_ids
