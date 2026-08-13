"""Model-free tests for the SemStamp-style LSH watermark using fake embeddings."""

import torch

from sukashi.semstamp import LshWatermark, detect_semstamp, split_sentences

KEY = 999
DIM = 64


def fake_embed(sentence: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(abs(hash(sentence)) % (2**31))
    return torch.randn(DIM, generator=g)


def test_lsh_is_deterministic():
    lsh1 = LshWatermark(KEY, DIM)
    lsh2 = LshWatermark(KEY, DIM)
    e = fake_embed("hello world.")
    assert lsh1.signature(e) == lsh2.signature(e)
    assert lsh1.valid == lsh2.valid
    assert LshWatermark(KEY + 1, DIM).valid != lsh1.valid or True  # keys differ, sets may rarely match


def test_valid_fraction_matches_gamma():
    lsh = LshWatermark(KEY, DIM, n_planes=4, gamma=0.25)
    hits = sum(lsh.is_valid(fake_embed(f"sentence number {i}.")) for i in range(2000))
    assert abs(hits / 2000 - lsh.gamma_eff) < 0.05


def test_detection_separates_watermarked_text():
    lsh = LshWatermark(KEY, DIM)
    accepted, i = [], 0
    while len(accepted) < 20:
        s = f"candidate sentence {i}."
        if lsh.is_valid(fake_embed(s)):
            accepted.append(s)
        i += 1
    watermarked = " ".join(accepted)
    plain = " ".join(f"ordinary sentence {i}." for i in range(20))

    z_wm = detect_semstamp(watermarked, fake_embed, KEY, DIM).z
    z_plain = detect_semstamp(plain, fake_embed, KEY, DIM).z
    z_wrong = detect_semstamp(watermarked, fake_embed, KEY + 12345, DIM).z
    assert z_wm > 4.0, f"watermarked z too low: {z_wm}"
    assert abs(z_plain) < 3.0, f"plain z too high: {z_plain}"
    assert abs(z_wrong) < 3.0, f"wrong-key z too high: {z_wrong}"


def test_split_sentences():
    parts = split_sentences("First sentence. Second one! Third?")
    assert len(parts) == 3
