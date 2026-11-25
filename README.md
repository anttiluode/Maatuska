# Maatuska

**Is an ordinary LLM's residual vector already a hidden Matryoshka?**

PerceptionLab / Antti Luode, with Claude (Opus 5.5). Helsinki, 25 November 2026.

> Do not hype. Do not lie. Just show.

---

## The question in one picture

A Matryoshka embedding is a vector whose first few numbers already say the
coarse thing (*animal*), the next ones the finer thing (*bird*), and the rest
the details (*sparrow*). You can cut it short and still read it, just at a blurrier grain.

PEEL (Kim & Joe, arXiv 2610.41785, 30 Oct 2026) showed that embedding models
**trained** to be nested this way also line up with real taxonomies:
shorter prefix means coarser categories, against shuffled and reversed controls.

Ordinary LLMs are not trained that way. Their raw coordinates have no order.
But there is one label-free order every vector space has: **principal
components**, sorted by variance. So:

> Fit PCA on thousands of ordinary words, with no labels. Read held-out words
> through the top-d components only. Does that prefix carry *more* taxonomy than
> d random directions do? Does the coarse level arrive before the fine one?

Classical word-embedding work predicts **no**. The top principal components of
word vectors mostly encode frequency, and removing them improves semantics
(Mu & Viswanath, *All-but-the-Top*, 2018). PEEL's result predicts something like
**yes**, if the structure is not a product of MRL training. The outcome is open.

## Why this repo exists

Every "reader / regulator" direction from the 25 Nov 2026 morning (the March 2025
iris, the grain lens of RakeisuusClaude/Theory, Neon's closed-loop valve) needs a knob
that changes semantic resolution **without labels**. Neon named the check
*kill condition 1*. If PCA order is not coarse-to-fine in a frozen model, that
knob is not where we hoped, and the iris needs a different basis (context-relative
mean/residue, or a learned one).

This is also the oldest question in the corpus in its plainest form. The 2024
chocolate-wafer *superweights* (DaFuc) and the 2026 NumberHell "one long vector
read at zoom levels" both assumed a vector can be read in blocks from coarse to fine.
Here we ask whether a real trained one already is.

## What it measures

- 192 test words: 4 kingdoms × 4 families × 12 words (animal/plant/artifact/person).
- Each word's residual state at every 2nd layer: mean over its tokens, averaged over 3 neutral templates.
- PCA basis fitted on up to 4000 **other** vocabulary words from the model's own tokenizer, disjoint and unlabeled.
- kNN label purity (k = 10) at kingdom and family level, for width d = 2, 4, 8, … , full, under:
  - **PCA** — the label-free order under test;
  - **random** — 5 random orthonormal bases (PEEL's shuffle control);
  - **reverse** — the lowest-variance components (PEEL's tail control);
  - **raw** — the literal first d coordinates.
- Frequency check: correlation of top PCs with the model's own log P(word | "The"), and purity after removing the top k PCs.
- Neighbourhood overlap between each prefix and full width (PEEL's non-nesting).

The gates (K1 hidden Matryoshka, K2 frequency on top, K3 tail dilution) are frozen in
[`PREREGISTRATION.md`](PREREGISTRATION.md). `REPORT.md` is written mechanically from them.

## Instrument check (already run)

Before touching a real model, the instrument has to tell outcomes apart in
synthetic worlds built to make each gate pass and fail:

| world | built so that | K1 | K2 | K3 |
|---|---|:-:|:-:|:-:|
| planted | hierarchy is the top variance | PASS | – | – |
| diluted | hierarchy on top + harmful lowest-variance tail | PASS | – | PASS |
| buried | frequency + junk on top, hierarchy spread thin underneath | FAIL | TRUE | – |

All three came out as expected: [`results_synthetic/REPORT.md`](results_synthetic/REPORT.md).
The full model pipeline was also run end to end on a tiny *untrained* GPT-NeoX.
It sat at chance (family purity ≈ 0.06 against chance 0.058), which is the right null.

One lesson from building the instrument, now noted in the preregistration. When a
vocabulary has only a few categories, their variance is low-rank and
lands in the top PCs on its own. A real vocabulary has thousands of categories,
so the synthetic worlds were given many too.

## Run it (Windows, RTX 3060)

```
pip install torch transformers numpy matplotlib
python maatuska.py --synthetic
python maatuska.py
```

The second command downloads `EleutherAI/pythia-410m` and `EleutherAI/pythia-1b`
(about 3 GB together) and should take a few minutes on the 3060. Everything lands in
`results/`:

- `REPORT.md` — the verdicts;
- `*.png` — purity vs width, the PEEL-Figure-1 analogue across layers, all-but-the-top;
- `*.json` — every number;
- `*_states.npz` — the hidden states, so the next experiment (the iris) can reuse
  them without the GPU. These files are large and git-ignored.

An extra model is descriptive only:

```
python maatuska.py --models Qwen/Qwen3-1.7B --out results_qwen
```

## Results so far (25 Nov 2026)

- **Main claim PASS on three models**: Pythia-410m, Pythia-1b and Qwen3-1.7B. A label-free
  PCA prefix carries more family structure than random directions of the same width.
  Reports: [`results/REPORT.md`](results/REPORT.md), [`results_qwen/REPORT.md`](results_qwen/REPORT.md).
- Layer profiles, the "lid" of top components, and what it means for the iris:
  [`ANALYSIS.md`](ANALYSIS.md).
- Follow-ups: [`audit.py`](audit.py) checks single-token words and lid-lifted prefixes on the saved states
  (no GPU). The repair test now lives in Theory as
  [Bridge](https://github.com/anttiluode/Theory/blob/main/BRIDGE.md).

## Lineage

DaFuc (superweights, Oct 2024) → NumberHell (one long vector at zoom levels) →
RakeisuusClaude / Theory (semantic grain in Qwen and Pythia) → PEEL (Oct 2026, external) →
Neon's kill condition 1 → **Maatuska** → the iris (DendriticAttentionSystem, Mar 2025)
on a frozen model, if K1 passes.
