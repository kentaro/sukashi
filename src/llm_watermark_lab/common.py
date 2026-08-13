"""Shared primitives: per-step seeding and key-derived pseudo-randomness.

Both schemes derive their randomness from (secret key, recent context tokens).
The generator and the detector recompute the exact same random values,
which is what makes model-free detection possible.

The context is a sliding window of the last CONTEXT_WIDTH generated tokens
(left-padded with a constant at the start). A 1-token context makes
generation deterministic given the previous token, which causes repetition
loops; widening the window is the standard mitigation.
"""

import torch

_PRIME_A = 1_000_003
_PRIME_B = 2_654_435_761
_MOD = 2**31 - 1

CONTEXT_WIDTH = 4
_PAD = 0


def window(tokens: list[int], index: int) -> tuple[int, ...]:
    """Context window for position `index`: the CONTEXT_WIDTH tokens before it."""
    ctx = tokens[max(0, index - CONTEXT_WIDTH):index]
    return tuple([_PAD] * (CONTEXT_WIDTH - len(ctx)) + list(ctx))


def step_seed(key: int, context: tuple[int, ...]) -> int:
    """Deterministic seed for one generation step from key and context tokens."""
    s = key % _MOD
    for t in context:
        s = (s * _PRIME_A + t * _PRIME_B) % _MOD
    return s


def green_mask(key: int, context: tuple[int, ...], vocab_size: int, gamma: float) -> torch.Tensor:
    """KGW green-list membership mask over the vocabulary (CPU bool tensor)."""
    g = torch.Generator().manual_seed(step_seed(key, context))
    return torch.rand(vocab_size, generator=g) < gamma


def gumbel_rand(key: int, context: tuple[int, ...], vocab_size: int) -> torch.Tensor:
    """Key-derived uniform randoms r in (0, 1) for the Gumbel-Max scheme (CPU float64)."""
    g = torch.Generator().manual_seed(step_seed(key, context))
    r = torch.rand(vocab_size, generator=g, dtype=torch.float64)
    return r.clamp(1e-12, 1.0 - 1e-12)
