# sukashi

A from-scratch implementation of **semantic (embedding-space) watermarking**
for LLM-generated text, inspired by SemStamp (Hou et al., NAACL 2024) and
[this Zenn article](https://zenn.dev/hellorusk/articles/3328866ca9e922) on
how systems like Claude's text watermarking work.

Token-level watermarks (KGW, Gumbel-Max) live in surface token choices, so a
single paraphrase pass destroys them. sukashi embeds the watermark in
**sentence-embedding space** instead:

1. A secret key derives random LSH hyperplanes that partition the embedding
   space into `2^n_planes` regions; a key-selected fraction `gamma` of the
   regions are "valid".
2. Generation runs sentence by sentence: candidates are sampled freely and
   rejected until one embeds into a valid region (rejection sampling, with a
   margin from region boundaries for robustness).
3. Detection embeds each sentence of a text and z-tests the fraction that
   lands in valid regions. `z > 4` is a confident hit; human text and
   wrong-key detection stay around `|z| < 2`.

Paraphrasing preserves sentence meaning, so the embedding stays near its
original point — usually within the same region — and the watermark
survives attacks that wipe out token-level schemes.

An earlier revision of this repository implemented the token-level schemes
(KGW and Gumbel-Max) with their own detectors and attack experiments; that
code is preserved in the git history.

## Usage

```sh
uv sync

# Unit tests (synthetic embeddings, no model download)
uv run pytest
```

Generation needs a local causal LM (tested with
`sbintuitions/sarashina2.2-3b-instruct-v0.1` on Apple Silicon MPS) and a
sentence-embedding model (tested with `cl-nagoya/ruri-v3-30m`):

```python
from sukashi.semstamp import generate_semstamp, detect_semstamp

result = generate_semstamp(model, tok, embed_fn, prompt, key=SECRET_KEY)
detection = detect_semstamp(result.text, embed_fn, key=SECRET_KEY, dim=DIM)
```
