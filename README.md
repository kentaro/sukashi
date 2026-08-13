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
