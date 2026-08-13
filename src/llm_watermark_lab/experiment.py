"""End-to-end experiment: generate, detect, attack, and measure quality.

Run with: uv run python -m llm_watermark_lab.experiment
Writes results/results.json and prints a summary table.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attacks import paraphrase, substitute_tokens
from .watermark import detect_gumbel, detect_kgw, generate, mean_nll

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SECRET_KEY = 42_424_243
WRONG_KEY = 777
GAMMA = 0.25
DELTA = 2.0
MAX_NEW_TOKENS = 200

PROMPTS = [
    "Write a short essay about why people enjoy walking in the rain.",
    "Describe an imaginary city built entirely on the backs of giant turtles.",
    "Explain to a curious child why the sky changes color at sunset.",
    "Tell a short story about a lighthouse keeper who collects bottled messages.",
]

# Human-written baseline: opening of "Pride and Prejudice" (public domain).
HUMAN_TEXT = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife. However little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered as the rightful property of "
    "some one or other of their daughters. My dear Mr. Bennet, said his lady "
    "to him one day, have you heard that Netherfield Park is let at last? "
    "Mr. Bennet replied that he had not. But it is, returned she; for Mrs. "
    "Long has just been here, and she told me all about it. Mr. Bennet made "
    "no answer. Do you not want to know who has taken it? cried his wife "
    "impatiently. You want to tell me, and I have no objection to hearing it."
)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.to(device).eval()
    vocab_size = model.config.vocab_size

    results: dict = {"model": MODEL_NAME, "device": device, "key": SECRET_KEY,
                     "gamma": GAMMA, "delta": DELTA, "samples": []}

    all_tokens: list[int] = []
    t0 = time.time()

    for i, prompt in enumerate(PROMPTS):
        print(f"\n=== Prompt {i + 1}/{len(PROMPTS)}: {prompt[:60]}...")
        sample: dict = {"prompt": prompt, "texts": {}, "detection": {}}

        gens = {}
        for scheme in ("vanilla", "kgw", "gumbel"):
            g = generate(model, tok, prompt, scheme=scheme, key=SECRET_KEY,
                         max_new_tokens=MAX_NEW_TOKENS, gamma=GAMMA, delta=DELTA, seed=i)
            gens[scheme] = g
            all_tokens.extend(g.token_ids)
            sample["texts"][scheme] = g.text
            print(f"  [{scheme}] {len(g.token_ids)} tokens")

        # Detection matrix: each detector against each text
        for scheme, g in gens.items():
            dk = detect_kgw(g.token_ids, SECRET_KEY, vocab_size, GAMMA)
            dg = detect_gumbel(g.token_ids, SECRET_KEY, vocab_size)
            sample["detection"][scheme] = {"kgw_z": dk.z, "gumbel_z": dg.z,
                                           "n_tokens": dk.n_tokens}
            print(f"  detect on {scheme:>7}: kgw z={dk.z:6.2f}  gumbel z={dg.z:6.2f}")

        # Wrong-key detection (should be near 0)
        wk = detect_kgw(gens["kgw"].token_ids, WRONG_KEY, vocab_size, GAMMA)
        wg = detect_gumbel(gens["gumbel"].token_ids, WRONG_KEY, vocab_size)
        sample["detection"]["wrong_key"] = {"kgw_z": wk.z, "gumbel_z": wg.z}
        print(f"  wrong key       : kgw z={wk.z:6.2f}  gumbel z={wg.z:6.2f}")

        # Quality: mean NLL (nats/token) under the same model
        sample["mean_nll"] = {s: mean_nll(model, tok, prompt, g.token_ids)
                              for s, g in gens.items()}
        print("  mean NLL: " + "  ".join(f"{s}={v:.3f}" for s, v in sample["mean_nll"].items()))

        results["samples"].append(sample)

    # Human-text false positive check
    human_ids = tok(HUMAN_TEXT, add_special_tokens=False)["input_ids"]
    hk = detect_kgw(human_ids, SECRET_KEY, vocab_size, GAMMA)
    hg = detect_gumbel(human_ids, SECRET_KEY, vocab_size)
    results["human_baseline"] = {"kgw_z": hk.z, "gumbel_z": hg.z, "n_tokens": hk.n_tokens}
    print(f"\nHuman text ({hk.n_tokens} tokens): kgw z={hk.z:.2f}  gumbel z={hg.z:.2f}")

    # Attacks on sample 0
    print("\n=== Attacks (on prompt 1 outputs) ===")
    pool = sorted(set(all_tokens))
    attack: dict = {"substitution": {}, "paraphrase": {}}
    kgw_tokens = tok(results["samples"][0]["texts"]["kgw"], add_special_tokens=False)["input_ids"]
    gumbel_tokens = tok(results["samples"][0]["texts"]["gumbel"], add_special_tokens=False)["input_ids"]

    for rate in (0.1, 0.3, 0.5):
        sk = detect_kgw(substitute_tokens(kgw_tokens, rate, pool), SECRET_KEY, vocab_size, GAMMA)
        sg = detect_gumbel(substitute_tokens(gumbel_tokens, rate, pool), SECRET_KEY, vocab_size)
        attack["substitution"][str(rate)] = {"kgw_z": sk.z, "gumbel_z": sg.z}
        print(f"  substitution {int(rate * 100):>2}%: kgw z={sk.z:6.2f}  gumbel z={sg.z:6.2f}")

    print("  paraphrasing (this takes a while) ...")
    para_kgw = paraphrase(model, tok, results["samples"][0]["texts"]["kgw"])
    para_gum = paraphrase(model, tok, results["samples"][0]["texts"]["gumbel"])
    pk = detect_kgw(para_kgw, SECRET_KEY, vocab_size, GAMMA)
    pg = detect_gumbel(para_gum, SECRET_KEY, vocab_size)
    attack["paraphrase"] = {"kgw_z": pk.z, "gumbel_z": pg.z,
                            "kgw_text": tok.decode(para_kgw),
                            "gumbel_text": tok.decode(para_gum)}
    print(f"  paraphrase      : kgw z={pk.z:6.2f}  gumbel z={pg.z:6.2f}")

    results["attacks"] = attack
    results["elapsed_sec"] = time.time() - t0

    out = Path(__file__).resolve().parents[2] / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {out} ({results['elapsed_sec']:.0f}s total)")


if __name__ == "__main__":
    main()
