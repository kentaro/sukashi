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


@dataclass
class Detection:
    z: float
    stat: float
    n_tokens: int


def _prompt_ids(tok, prompt: str, device) -> torch.Tensor:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(device)

_TERMINATORS = "。！？.!?\n"
_SPLIT_RE = re.compile(f"[^{re.escape(_TERMINATORS)}]+[{re.escape(_TERMINATORS)}]*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT_RE.findall(text) if len(s.strip()) >= 4]


class LshWatermark:
    """Key-derived LSH partition of embedding space with chained valid regions.

    As in the SemStamp paper, the set of valid regions for each sentence is
    derived from the LSH signature of the PREVIOUS sentence (seeded with the
    secret key), so the watermark is a chain rather than a fixed global set.
    """

    def __init__(self, key: int, dim: int, n_planes: int = 4, gamma: float = 0.25,
                 center: torch.Tensor | None = None):
        self.key = key
        self.n_planes = n_planes
        self.gamma = gamma
        self.center = None if center is None else center.to(torch.float64)
        g = torch.Generator().manual_seed(key % (2**31 - 1))
        planes = torch.randn(n_planes, dim, generator=g, dtype=torch.float64)
        self.planes = planes / planes.norm(dim=1, keepdim=True)
        self.n_regions = 2**n_planes
        self.n_valid = max(1, round(gamma * self.n_regions))
        self.gamma_eff = self.n_valid / self.n_regions

    def _projections(self, emb: torch.Tensor) -> torch.Tensor:
        e = emb.to(torch.float64)
        if self.center is not None:
            e = e - self.center
        e = e / e.norm()
        return self.planes @ e

    def signature(self, emb: torch.Tensor) -> int:
        bits = (self._projections(emb) > 0).tolist()
        return sum(1 << i for i, b in enumerate(bits) if b)

    def valid_set(self, prev_sig: int) -> set[int]:
        seed = (self.key * 1_000_003 + prev_sig * 2_654_435_761) % (2**31 - 1)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(self.n_regions, generator=g)
        return set(perm[: self.n_valid].tolist())

    def is_valid(self, emb: torch.Tensor, prev_sig: int, margin: float = 0.0) -> bool:
        proj = self._projections(emb)
        if float(proj.abs().min()) < margin:
            return False
        return self.signature(emb) in self.valid_set(prev_sig)


@dataclass
class SemStampResult:
    text: str
    sentences: list[str]
    tries: list[int]


@torch.no_grad()
def _sample_sentence(model, tok, context_ids, seed: int, max_tokens: int = 80,
                     temperature: float = 1.0, top_p: float = 0.95) -> tuple[str, torch.Tensor]:
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=context_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_tokens,
        pad_token_id=tok.eos_token_id,
    )
    new_ids = out[0, context_ids.shape[1]:]
    # Truncate at the first sentence terminator that follows actual content,
    # so leading newlines or paragraph breaks do not yield empty candidates.
    for j in range(1, len(new_ids) + 1):
        piece = tok.decode(new_ids[:j], skip_special_tokens=True)
        if piece.strip() and piece[-1] in _TERMINATORS:
            new_ids = new_ids[:j]
            break
    return tok.decode(new_ids, skip_special_tokens=True), new_ids.unsqueeze(0)


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
    center: torch.Tensor | None = None,
    candidate_filter=None,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> SemStampResult:
    """Sentence-level rejection sampling against a key-derived LSH partition.

    embed_fn maps a sentence string to a 1-D embedding tensor. candidate_filter,
    if given, is a str -> bool quality gate; rejected candidates are resampled
    (this applies to unwatermarked generation as well, so both sides of a
    comparison get the same quality constraint).
    """
    device = next(model.parameters()).device
    context = _prompt_ids(tok, prompt, device)
    lsh: LshWatermark | None = None
    prev_sig = 0
    sentences: list[str] = []
    tries_log: list[int] = []

    for si in range(max_sentences):
        accepted = None
        for attempt in range(max_tries if (watermark or candidate_filter) else 1):
            seed = key * 7919 + si * 104729 + attempt
            text, ids = _sample_sentence(model, tok, context, seed,
                                         temperature=temperature, top_p=top_p)
            if not text.strip() or ids.shape[1] == 0:
                break
            if candidate_filter is not None and not candidate_filter(text.strip()):
                continue
            if not watermark:
                accepted = (text, ids)
                break
            emb = embed_fn(text.strip())
            if lsh is None:
                lsh = LshWatermark(key, emb.shape[-1], n_planes, gamma, center)
            if lsh.is_valid(emb, prev_sig, margin):
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
        if watermark and lsh is not None:
            prev_sig = lsh.signature(embed_fn(accepted[0].strip()))
        context = torch.cat([context, accepted[1].to(device)], dim=1)
        if tok.eos_token_id in accepted[1][0].tolist():
            break

    return SemStampResult(text="".join(sentences), sentences=sentences, tries=tries_log)


def detect_semstamp(
    text: str,
    embed_fn,
    key: int,
    dim: int,
    n_planes: int = 4,
    gamma: float = 0.25,
    center: torch.Tensor | None = None,
) -> Detection:
    """z-test on the fraction of sentences whose embedding lands in a valid region."""
    lsh = LshWatermark(key, dim, n_planes, gamma, center)
    sentences = split_sentences(text)
    if not sentences:
        return Detection(z=0.0, stat=0.0, n_tokens=0)
    hits = 0
    prev_sig = 0
    for s in sentences:
        emb = embed_fn(s)
        if lsh.signature(emb) in lsh.valid_set(prev_sig):
            hits += 1
        prev_sig = lsh.signature(emb)
    n = len(sentences)
    g = lsh.gamma_eff
    z = (hits - g * n) / math.sqrt(n * g * (1 - g))
    return Detection(z=z, stat=hits / n, n_tokens=n)
