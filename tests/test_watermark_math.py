"""Model-free tests: run both schemes against a synthetic token distribution
and check that the detectors separate watermarked from unwatermarked sequences.
"""

import math

import torch

from sukashi.common import green_mask, gumbel_rand, step_seed, window
from sukashi.watermark import detect_gumbel, detect_kgw

VOCAB = 1000
KEY = 12345
GAMMA = 0.25
DELTA = 2.0
N_TOKENS = 400


def synthetic_logits(prev_token: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 7919 + prev_token)
    return torch.randn(VOCAB, generator=g) * 2.0


def gen_kgw(n: int, watermark: bool) -> list[int]:
    sampler = torch.Generator().manual_seed(0)
    prev, out = 1, []
    for i in range(n):
        logits = synthetic_logits(prev, i)
        if watermark:
            logits = logits + green_mask(KEY, window(out, i), VOCAB, GAMMA) * DELTA
        probs = torch.softmax(logits, dim=-1)
        prev = int(torch.multinomial(probs, 1, generator=sampler))
        out.append(prev)
    return out


def gen_gumbel(n: int, watermark: bool) -> list[int]:
    sampler = torch.Generator().manual_seed(0)
    prev, out = 1, []
    for i in range(n):
        probs = torch.softmax(synthetic_logits(prev, i), dim=-1).double()
        if watermark:
            r = gumbel_rand(KEY, window(out, i), VOCAB)
            prev = int(torch.argmax(torch.log(r) / (probs + 1e-300)))
        else:
            prev = int(torch.multinomial(probs, 1, generator=sampler))
        out.append(prev)
    return out


def test_seed_is_deterministic():
    ctx = (1, 2, 3, 4)
    assert step_seed(KEY, ctx) == step_seed(KEY, ctx)
    assert step_seed(KEY, ctx) != step_seed(KEY, (1, 2, 3, 5))
    assert torch.equal(green_mask(KEY, ctx, VOCAB, GAMMA), green_mask(KEY, ctx, VOCAB, GAMMA))


def test_window_pads_and_slides():
    assert window([], 0) == (0, 0, 0, 0)
    assert window([7, 8], 2) == (0, 0, 7, 8)
    assert window([5, 6, 7, 8, 9], 5) == (6, 7, 8, 9)


def test_green_mask_fraction():
    frac = green_mask(KEY, (1, 2, 3, 4), 100_000, GAMMA).float().mean().item()
    assert abs(frac - GAMMA) < 0.01


def test_kgw_detects_watermarked_only():
    z_wm = detect_kgw(gen_kgw(N_TOKENS, True), KEY, VOCAB, GAMMA).z
    z_plain = detect_kgw(gen_kgw(N_TOKENS, False), KEY, VOCAB, GAMMA).z
    assert z_wm > 4.0, f"watermarked z too low: {z_wm}"
    assert abs(z_plain) < 3.0, f"plain z too high: {z_plain}"


def test_gumbel_detects_watermarked_only():
    z_wm = detect_gumbel(gen_gumbel(N_TOKENS, True), KEY, VOCAB).z
    z_plain = detect_gumbel(gen_gumbel(N_TOKENS, False), KEY, VOCAB).z
    assert z_wm > 4.0, f"watermarked z too low: {z_wm}"
    assert abs(z_plain) < 3.0, f"plain z too high: {z_plain}"


def test_wrong_key_does_not_detect():
    ids = gen_kgw(N_TOKENS, True)
    assert abs(detect_kgw(ids, KEY + 1, VOCAB, GAMMA).z) < 3.0
    ids = gen_gumbel(N_TOKENS, True)
    assert abs(detect_gumbel(ids, KEY + 1, VOCAB).z) < 3.0


def test_gumbel_is_distortion_free_marginally():
    """Over many keys, Gumbel-Max sampling should reproduce the base distribution."""
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
    counts = torch.zeros(4)
    n = 20_000
    for k in range(n):
        r = gumbel_rand(k, (0, 0, 0, 0), 4)
        counts[int(torch.argmax(torch.log(r) / probs))] += 1
    freq = counts / n
    for i in range(4):
        se = math.sqrt(float(probs[i]) * (1 - float(probs[i])) / n)
        assert abs(float(freq[i]) - float(probs[i])) < 5 * se, (
            f"token {i}: freq {float(freq[i]):.4f} vs p {float(probs[i]):.4f}"
        )
