#!/usr/bin/env python3
"""
Maatuska audit — Neon's two holes, checked on the saved hidden states (no GPU).

  python audit.py results/EleutherAI_pythia-410m_states.npz --tokenizer EleutherAI/pythia-410m
  python audit.py results/EleutherAI_pythia-1b_states.npz   --tokenizer EleutherAI/pythia-1b

1. Single-token words only: does K1 survive when multi-token test words (averaged over
   sub-word pieces) are dropped?
2. Lid-lifted prefixes: in upper layers, is the hidden Matryoshka still there just
   under the lid? Read PCA components k0 .. k0+15 for k0 in {0, 1, 2, 4, 8}, against
   16 random directions.

Exploratory, added after the first results; not preregistered. Uses the same purity,
random bases and vocabulary PCA as maatuska.py.
"""
import argparse
import math
import os

import numpy as np

import maatuska as M

K0S = (0, 1, 2, 4, 8)
WIDTH = 16


def band_purity(X_test, X_fit, labels, k0, d):
    X_test = np.asarray(X_test, dtype=np.float64)
    X_fit = np.asarray(X_fit, dtype=np.float64)
    mu = X_fit.mean(0)
    _, _, vt = np.linalg.svd(X_fit - mu, full_matrices=False)
    V = vt.T
    Xt = X_test - mu
    pca = M.knn_purity_per_word(Xt @ V[:, k0:k0 + d], labels).mean()
    rand = [M.knn_purity_per_word(Xt @ M.random_orthonormal(Xt.shape[1], 1000 + s)[:, :d], labels).mean()
            for s in range(M.N_RANDOM_SEEDS)]
    return float(pca), float(np.mean(rand)), float(np.max(rand))


def single_token_mask(words, tokenizer_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return np.array([len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1 for w in words])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.states, allow_pickle=False)
    words = [str(w) for w in z["test_words"]]
    family = np.array([str(f) for f in z["family"]])
    kingdom = np.array([str(k) for k in z["kingdom"]])
    freq = z["freq"]
    layers = [int(l) for l in z["layers"]]
    primary = layers[len(layers) // 2] if len(layers) % 2 else None
    mask = single_token_mask(words, a.tokenizer)
    lines = [f"# Maatuska audit — {os.path.basename(a.states)}", "",
             "Exploratory, added after the first results; not preregistered.", "",
             f"Single-token test words: {int(mask.sum())} of {len(words)}.", ""]
    fams, counts = np.unique(family[mask], return_counts=True)
    lines += ["Single-token words per family: " +
              ", ".join(f"{f} {c}" for f, c in zip(fams, counts)), ""]

    # 1. single-token-only K1-style table at every layer
    lines += ["## 1. Single-token words only — family log2(PCA / random) at d = 8, 16, 32", "",
              "| layer | all words d16 | single-token d8 | d16 | d32 | single-token full purity | chance |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for l in layers:
        Xt, Xf = z[f"test_L{l}"].astype(np.float64), z[f"vocab_L{l}"].astype(np.float64)
        full = M.analyse_layer(Xt, Xf, kingdom, family, freq)
        sub = M.analyse_layer(Xt[mask], Xf, kingdom[mask], family[mask], freq[mask])
        g = full["grid"]
        def adv(res, d):
            i = res["grid"].index(d)
            p, r = res["family"]["pca"][i], res["family"]["random_mean"][i]
            mx = max(s[i] for s in res["family"]["random_seeds"])
            return f"{math.log2(p / r):+.2f}{'*' if p > mx else ''}"
        lines.append(f"| {l} | {adv(full, 16)} | {adv(sub, 8)} | {adv(sub, 16)} | {adv(sub, 32)} |"
                     f" {sub['family']['pca'][-1]:.3f} | {sub['family']['chance']:.3f} |")
    lines += ["", "`*` = PCA beats all 5 random seeds.", ""]

    # 2. lid-lifted prefixes
    lines += [f"## 2. Lid-lifted prefixes — family log2(PCA[k0 : k0+{WIDTH}] / random {WIDTH}), all words", "",
              "| layer | " + " | ".join(f"k0={k}" for k in K0S) + " |",
              "|---:|" + "---:|" * len(K0S)]
    for l in layers:
        Xt, Xf = z[f"test_L{l}"], z[f"vocab_L{l}"]
        cells = []
        for k0 in K0S:
            p, r, mx = band_purity(Xt, Xf, family, k0, WIDTH)
            cells.append(f"{math.log2(p / r):+.2f}{'*' if p > mx else ''}")
        lines.append(f"| {l} | " + " | ".join(cells) + " |")
    lines += ["", "`*` = PCA band beats all 5 random seeds.", ""]
    out = a.out or os.path.join(os.path.dirname(a.states) or ".",
                                "AUDIT_" + os.path.basename(a.states).replace("_states.npz", ".md"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
