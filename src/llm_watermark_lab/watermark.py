"""KGW and Gumbel-Max watermarked generation, plus model-free detectors.

KGW (Kirchenbauer et al., ICML 2023):
  At each step, hash the previous token with the secret key to split the
  vocabulary into a green list (fraction gamma) and a red list, then add a
  logit bonus delta to green tokens before sampling. Detection counts the
  fraction of green tokens and runs a one-sided z-test.

Gumbel-Max (Aaronson 2022 / Kuditipudi et al. 2023), distortion-free:
  Replace the sampling randomness with key-derived randoms r and pick
  argmax r_i^(1/p_i). Marginally over the key this samples exactly from p,
  so the output distribution is unchanged. Detection scores
  sum(-log(1 - r[chosen])) which is Exp(1)-distributed per token for
  unrelated text but skews high for watermarked text.
"""

import math
from dataclasses import dataclass

import torch

from .common import green_mask, gumbel_rand


@dataclass
class Generated:
    text: str
    token_ids: list[int]


@torch.no_grad()
def _step_logits(model, input_ids, past):
    out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    return out.logits[0, -1, :].float(), out.past_key_values


def _prompt_ids(tok, prompt: str, device) -> torch.Tensor:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(device)


@torch.no_grad()
def generate(
    model,
    tok,
    prompt: str,
    scheme: str,
    key: int,
    max_new_tokens: int = 200,
    gamma: float = 0.25,
    delta: float = 2.0,
    seed: int = 0,
) -> Generated:
    """Generate with scheme in {"vanilla", "kgw", "gumbel"} at temperature 1.0."""
    device = next(model.parameters()).device
    input_ids = _prompt_ids(tok, prompt, device)
    sampler = torch.Generator().manual_seed(seed)

    generated: list[int] = []
    prev_token = int(input_ids[0, -1])
    logits, past = _step_logits(model, input_ids, None)

    for _ in range(max_new_tokens):
        vocab = logits.shape[-1]
        if scheme == "kgw":
            mask = green_mask(key, prev_token, vocab, gamma).to(device)
            logits = logits + mask * delta
            probs = torch.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs.cpu(), 1, generator=sampler))
        elif scheme == "gumbel":
            probs = torch.softmax(logits, dim=-1).cpu().double()
            r = gumbel_rand(key, prev_token, vocab)
            # argmax r^(1/p) == argmax log(r)/p  (log r < 0, so higher p wins)
            next_token = int(torch.argmax(torch.log(r) / (probs + 1e-300)))
        elif scheme == "vanilla":
            probs = torch.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs.cpu(), 1, generator=sampler))
        else:
            raise ValueError(f"unknown scheme: {scheme}")

        if next_token == tok.eos_token_id:
            break
        generated.append(next_token)
        prev_token = next_token
        step_input = torch.tensor([[next_token]], device=device)
        logits, past = _step_logits(model, step_input, past)

    return Generated(text=tok.decode(generated), token_ids=generated)


@dataclass
class Detection:
    z: float
    stat: float
    n_tokens: int


def detect_kgw(token_ids: list[int], key: int, vocab_size: int, gamma: float = 0.25) -> Detection:
    """One-sided z-test on the green-token fraction. z > 4 is a confident hit."""
    hits = 0
    pairs = list(zip(token_ids, token_ids[1:]))
    for prev, cur in pairs:
        if cur < vocab_size and green_mask(key, prev, vocab_size, gamma)[cur]:
            hits += 1
    n = len(pairs)
    if n == 0:
        return Detection(z=0.0, stat=0.0, n_tokens=0)
    z = (hits - gamma * n) / math.sqrt(n * gamma * (1 - gamma))
    return Detection(z=z, stat=hits / n, n_tokens=n)


def detect_gumbel(token_ids: list[int], key: int, vocab_size: int) -> Detection:
    """Score sum(-log(1 - r[chosen])); Exp(1) per token under the null.

    z is the normal approximation of the Gamma(n, 1) null: (S - n) / sqrt(n).
    """
    score = 0.0
    pairs = list(zip(token_ids, token_ids[1:]))
    for prev, cur in pairs:
        if cur >= vocab_size:
            continue
        r = float(gumbel_rand(key, prev, vocab_size)[cur])
        score += -math.log(1.0 - r)
    n = len(pairs)
    if n == 0:
        return Detection(z=0.0, stat=0.0, n_tokens=0)
    z = (score - n) / math.sqrt(n)
    return Detection(z=z, stat=score / n, n_tokens=n)


@torch.no_grad()
def mean_nll(model, tok, prompt: str, token_ids: list[int]) -> float:
    """Mean negative log-likelihood (nats/token) of a continuation under the model."""
    device = next(model.parameters()).device
    prompt_ids = _prompt_ids(tok, prompt, device)
    cont = torch.tensor([token_ids], device=device)
    full = torch.cat([prompt_ids, cont], dim=1)
    logits = model(input_ids=full).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    start = prompt_ids.shape[1]
    nll = 0.0
    for i, t in enumerate(token_ids):
        nll -= float(logprobs[start + i - 1, t])
    return nll / max(len(token_ids), 1)
