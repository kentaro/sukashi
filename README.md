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

## Results (Qwen2.5-0.5B-Instruct, MPS, 200 tokens/sample, 42s total)

Detection z-scores, averaged over 4 prompts:

| text under test        | KGW detector | Gumbel detector |
|------------------------|-------------:|----------------:|
| vanilla (no watermark) |        -0.70 |            0.54 |
| KGW-watermarked        |    **12.07** |           -0.60 |
| Gumbel-watermarked     |        -2.91 |       **39.96** |
| wrong key              |        -0.57 |           -1.30 |
| human text (Austen)    |        -2.86 |            1.03 |

Quality (mean NLL in nats/token under the same model): KGW distorts the
distribution as predicted (e.g. 2.73 -> 4.76 on prompt 1), while Gumbel-Max
does not degrade likelihood.

Attacks (on prompt 1 outputs):

| attack            | KGW z | Gumbel z |
|-------------------|------:|---------:|
| none              | 13.79 |    29.74 |
| substitution 10%  | 11.34 |    23.64 |
| substitution 30%  |  8.55 |    13.04 |
| substitution 50%  |  3.97 |     6.52 |
| paraphrase (LLM)  |  0.05 |     7.17 |

The paraphrase attack wipes out the KGW watermark in a single pass,
reproducing the key weakness discussed in the article. The residual Gumbel
signal after paraphrase is explained by the weak 0.5B paraphraser copying
several phrases verbatim, not by semantic robustness.

Caveat observed in practice: with a fixed key and a 1-token hash context,
Gumbel-Max generation is deterministic given the previous token, which
noticeably increases phrase repetition (and lowers NLL below the vanilla
baseline). Production schemes mitigate this with longer hash windows and
key rotation.
