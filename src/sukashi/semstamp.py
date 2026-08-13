"""SemStamp-style semantic watermarking (Hou et al., NAACL 2024), simplified.

Instead of biasing token selection, the watermark lives in sentence-embedding
space. A secret key derives random LSH hyperplanes that partition the
embedding space into 2^n_planes regions, of which a key-selected fraction
gamma are "valid". Generation proceeds sentence by sentence: candidate
sentences are sampled freely and rejected until one lands in a valid region
(rejection sampling), optionally with a margin from region boundaries for
robustness. Detection embeds each sentence of a text and z-tests the
fraction that falls in valid regions.

Because paraphrasing preserves sentence meaning, the embedding stays near
its original point and usually within the same region, so the watermark
survives paraphrase attacks that destroy token-level schemes.
"""

import math
import re
from dataclasses import dataclass

import torch

from .watermark import Detection, _prompt_ids

_TERMINATORS = "。！？.!?\n"
_SPLIT_RE = re.compile(f"[^{re.escape(_TERMINATORS)}]+[{re.escape(_TERMINATORS)}]*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT_RE.findall(text) if len(s.strip()) >= 4]


class LshWatermark:
    """Key-derived LSH partition of embedding space with a valid-region set."""

    def __init__(self, key: int, dim: int, n_planes: int = 4, gamma: float = 0.25):
        self.n_planes = n_planes
        self.gamma = gamma
        g = torch.Generator().manual_seed(key % (2**31 - 1))
        planes = torch.randn(n_planes, dim, generator=g, dtype=torch.float64)
        self.planes = planes / planes.norm(dim=1, keepdim=True)
        n_regions = 2**n_planes
        n_valid = max(1, round(gamma * n_regions))
        perm = torch.randperm(n_regions, generator=g)
        self.valid = set(perm[:n_valid].tolist())
        self.gamma_eff = n_valid / n_regions

    def _projections(self, emb: torch.Tensor) -> torch.Tensor:
        e = emb.to(torch.float64)
        e = e / e.norm()
        return self.planes @ e

    def signature(self, emb: torch.Tensor) -> int:
        bits = (self._projections(emb) > 0).tolist()
        return sum(1 << i for i, b in enumerate(bits) if b)

    def is_valid(self, emb: torch.Tensor, margin: float = 0.0) -> bool:
        proj = self._projections(emb)
        if float(proj.abs().min()) < margin:
            return False
        return self.signature(emb) in self.valid


@dataclass
class SemStampResult:
    text: str
    sentences: list[str]
    tries: list[int]


@torch.no_grad()
def _sample_sentence(model, tok, context_ids, seed: int, max_tokens: int = 80) -> tuple[str, torch.Tensor]:
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=context_ids,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=max_tokens,
        pad_token_id=tok.eos_token_id,
    )
    new_ids = out[0, context_ids.shape[1]:]
    # Truncate at the first sentence terminator (decode incrementally).
    for j in range(1, len(new_ids) + 1):
        piece = tok.decode(new_ids[:j])
        if piece and piece[-1] in _TERMINATORS:
            new_ids = new_ids[:j]
            break
    return tok.decode(new_ids), new_ids.unsqueeze(0)


@torch.no_grad()
def generate_semstamp(
    model,
    tok,
    embed_fn,
    prompt: str,
    key: int,
    n_planes: int = 4,
    gamma: float = 0.25,
    margin: float = 0.01,
    max_sentences: int = 12,
    max_tries: int = 16,
    watermark: bool = True,
) -> SemStampResult:
    """Sentence-level rejection sampling against a key-derived LSH partition.

    embed_fn maps a sentence string to a 1-D embedding tensor.
    """
    device = next(model.parameters()).device
    context = _prompt_ids(tok, prompt, device)
    lsh: LshWatermark | None = None
    sentences: list[str] = []
    tries_log: list[int] = []

    for si in range(max_sentences):
        accepted = None
        for attempt in range(max_tries if watermark else 1):
            seed = key * 7919 + si * 104729 + attempt
            text, ids = _sample_sentence(model, tok, context, seed)
            if not text.strip() or ids.shape[1] == 0:
                break
            if not watermark:
                accepted = (text, ids)
                break
            emb = embed_fn(text.strip())
            if lsh is None:
                lsh = LshWatermark(key, emb.shape[-1], n_planes, gamma)
            if lsh.is_valid(emb, margin):
                accepted = (text, ids)
                break
        else:
            # No candidate passed; keep the last one so generation continues.
            accepted = (text, ids)
            attempt = max_tries
        if accepted is None:
            break
        sentences.append(accepted[0].strip())
        tries_log.append(attempt + 1)
        context = torch.cat([context, accepted[1].to(device)], dim=1)
        if int(accepted[1][0, -1]) == tok.eos_token_id:
            break

    return SemStampResult(text="".join(sentences), sentences=sentences, tries=tries_log)


def detect_semstamp(
    text: str,
    embed_fn,
    key: int,
    dim: int,
    n_planes: int = 4,
    gamma: float = 0.25,
) -> Detection:
    """z-test on the fraction of sentences whose embedding lands in a valid region."""
    lsh = LshWatermark(key, dim, n_planes, gamma)
    sentences = split_sentences(text)
    if not sentences:
        return Detection(z=0.0, stat=0.0, n_tokens=0)
    hits = sum(1 for s in sentences if lsh.is_valid(embed_fn(s)))
    n = len(sentences)
    g = lsh.gamma_eff
    z = (hits - g * n) / math.sqrt(n * g * (1 - g))
    return Detection(z=z, stat=hits / n, n_tokens=n)
