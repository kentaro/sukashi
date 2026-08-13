# llm-watermark-lab

From-scratch implementation of LLM text watermarking, built to understand how
schemes like the one Anthropic announced for Claude (and Google's SynthID-Text)
actually work. Inspired by
[this Zenn article](https://zenn.dev/hellorusk/articles/3328866ca9e922).

Two schemes are implemented against a local model
(Qwen2.5-0.5B-Instruct on Apple Silicon MPS), with model-free detectors:

- **KGW** (Kirchenbauer et al., ICML 2023): hash the previous token with a
  secret key to split the vocabulary into green/red lists, add a logit bonus
  to green tokens, detect via a z-test on the green-token fraction.
- **Gumbel-Max** (Aaronson 2022 / Kuditipudi et al. 2023): distortion-free.
  Replace sampling randomness with key-derived randoms and pick
  `argmax r_i^(1/p_i)`; the output distribution is provably unchanged.
  Detect via `sum(-log(1 - r[chosen]))`, which is Exp(1) per token under
  the null.

The experiment also reproduces the attacks discussed in the article:
random token substitution and a paraphrase attack (rewriting the watermarked
text with the same model), plus a human-text false-positive check and a
quality (NLL) comparison.

## Usage

```sh
uv sync

# Unit tests (synthetic distribution, no model download)
uv run pytest

# Full experiment (downloads Qwen2.5-0.5B-Instruct on first run)
uv run python -m llm_watermark_lab.experiment
```

Results are written to `results/results.json`.

## Interpreting z-scores

`z > 4` means the watermark is detected with overwhelming confidence
(one-sided p < 3e-5). Human text and wrong-key detection should stay
around `|z| < 2`.

## Results (Qwen2.5-0.5B-Instruct, MPS, 200 tokens/sample, 39s total)

Both schemes hash a sliding window of the last 4 generated tokens
(`CONTEXT_WIDTH` in `common.py`). Detection z-scores, averaged over 4 prompts:

| text under test        | KGW detector | Gumbel detector |
|------------------------|-------------:|----------------:|
| vanilla (no watermark) |        -0.76 |            0.73 |
| KGW-watermarked        |     **9.78** |           -0.07 |
| Gumbel-watermarked     |        -0.90 |       **34.99** |
| wrong key              |        -0.42 |            0.16 |
| human text (Austen)    |        -1.51 |            0.18 |

Quality (mean NLL in nats/token under the same model, averaged):
vanilla 2.25, KGW 3.02, Gumbel 2.77. KGW measurably distorts the
distribution (worst case 1.22 -> 4.39); the Gumbel gap to vanilla is
sampling noise, consistent with its distortion-free construction.

Attacks (on prompt 1 outputs, re-tokenized from text):

| attack            | KGW z | Gumbel z |
|-------------------|------:|---------:|
| none              |  5.55 |    38.79 |
| substitution 10%  |  2.45 |    19.61 |
| substitution 30%  | -0.82 |     7.27 |
| substitution 50%  |  1.31 |     0.40 |
| paraphrase (LLM)  | -1.45 |    -0.66 |

A single paraphrase pass wipes out both watermarks, reproducing the key
weakness discussed in the article.

Two trade-offs observed while building this:

- With a 1-token hash context (an earlier revision), generation is
  deterministic given the previous token, which causes visible repetition
  loops — especially in Japanese. Widening the window to 4 tokens fixes
  the loops.
- The wider window costs edit robustness: each substituted token corrupts
  its own position plus the next 4 windows, so token substitution degrades
  z much faster than with a 1-token context. This is the same
  quality/robustness tension the literature resolves with schemes like
  SemStamp (semantic-space watermarks).
