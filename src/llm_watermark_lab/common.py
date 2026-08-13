"""Shared primitives: per-step seeding and key-derived pseudo-randomness.

Both schemes derive their randomness from (secret key, previous token id).
The generator and the detector recompute the exact same random values,
which is what makes model-free detection possible.
"""

import torch

_PRIME_A = 1_000_003
_PRIME_B = 2_654_435_761
_MOD = 2**31 - 1


def step_seed(key: int, prev_token: int) -> int:
    """Deterministic seed for one generation step from key and previous token."""
    return (key * _PRIME_A + prev_token * _PRIME_B) % _MOD


def green_mask(key: int, prev_token: int, vocab_size: int, gamma: float) -> torch.Tensor:
    """KGW green-list membership mask over the vocabulary (CPU bool tensor)."""
    g = torch.Generator().manual_seed(step_seed(key, prev_token))
    return torch.rand(vocab_size, generator=g) < gamma


def gumbel_rand(key: int, prev_token: int, vocab_size: int) -> torch.Tensor:
    """Key-derived uniform randoms r in (0, 1) for the Gumbel-Max scheme (CPU float64)."""
    g = torch.Generator().manual_seed(step_seed(key, prev_token))
    r = torch.rand(vocab_size, generator=g, dtype=torch.float64)
    return r.clamp(1e-12, 1.0 - 1e-12)
