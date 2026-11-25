# Maatuska — analysis of the first real run

25 November 2026. Pythia-410m and Pythia-1b, run by Antti on an RTX 3060; read by Claude (Opus 5.5)
from `results/*.json`. The verdicts in `results/REPORT.md` follow the preregistered rules. Everything
below the verdicts is descriptive and was **not** preregistered.

## Verdicts (preregistered)

| gate | Pythia-410m (layer 12/24) | Pythia-1b (layer 8/16) |
|---|---|---|
| **K1 hidden Matryoshka** | **PASS** at d = 16: PCA 0.335 vs random 0.260 (best seed 0.315), log2 +0.37 | **PASS** at d = 8, 16, 32: best d = 16, PCA 0.321 vs random 0.238 (best seed 0.260), log2 +0.43 |
| K2 frequency on top | FALSE: corr(PC1, log-freq) = +0.44 | FALSE: +0.17 |
| K3 tail dilution | FAIL: +0.005, CI [−0.002, +0.012] | FAIL: +0.001, CI [−0.005, +0.006] |

**Main claim: PASS.** In both frozen models, at the preregistered middle layer, the top principal
components of an unlabeled 4000-word vocabulary carry more family-level taxonomy than random
directions of the same width. The 410m pass is narrow (one width, PCA 0.335 against a best random
seed of 0.315).

## What the layers show (descriptive)

Family-level log2(PCA / random) at d = 16, with * where PCA beats all 5 random seeds:

| layer | 410m | layer | 1b |
|---|---|---|---|
| 0 (embeddings) | +1.20* | 0 | +1.10* |
| 4 | +0.78* | 4 | +0.80* |
| 8 | +0.59* | 8 (primary) | +0.43* |
| 12 (primary) | +0.37* | 12 | +0.24* |
| 16 | +0.26* | 14 | +0.08 |
| 20 | −0.09 | 16 (last) | −0.18 |
| 24 (last) | −0.09 | | |

1. **The hidden Matryoshka is strongest at the bottom of the network and gone at the top.**
   Early and lower-middle layers reach log2 advantages of +0.6 to +1.2. That is the same range PEEL
   reports for embeddings *trained* to be nested. Above roughly two thirds of the depth, the PCA
   order carries no extra taxonomy, and at the top it is slightly worse than random.
2. **At the top, the nuisance sits on the lid, not in the tail** (and the order is still underneath it; see the audit below). Removing the top few principal
   components helps more and more with depth. In 410m, family purity goes from 0.552 to 0.661 in
   layer 22 when the top 1 PC is removed, and from 0.562 to 0.591 at the primary layer when the top 3
   are removed. The top PC's share of variance grows from about 5% in middle layers to 17–22% in the
   last ones. This is the opposite of PEEL. In a nested (MRL) embedding, the harmful residue is in the
   tail, so cutting the end helps. In an ordinary LLM's late layers, it is on top, so cutting the
   start helps. It matches a known effect: "rogue dimensions" dominating similarity in late
   transformer layers (Timkey & van Schijndel, 2021). Whether these top PCs are exactly those rogue
   dimensions was not checked here.
3. **The top PCs are only partly frequency.** |corr| with the model's own log-frequency proxy stays
   between 0.4 and 0.48 in most late layers, under the 0.5 bar. Something else (position, template,
   or the rogue dimensions) is mixed in.
4. **Coarse before fine, weakly.** Counting how wide a prefix must be to cover 90% of the way from
   chance to full-width purity, kingdom needs the same or a narrower prefix than family. For example,
   410m layer 4 needs 64 dims for kingdom and 128 for family; 1b layer 8 needs 128 and 256. PCA gets
   there with half the width random needs. This is a descriptive measure, not a gate.
5. **The advantage peaks at very small widths.** Across layers the family advantage peaks around
   d = 8–16: about 1–1.5% of the vector. A 16-number PCA prefix reaches about 60% of full-width family
   purity, where random 16 directions reach about 46%.
6. **Partly nested, like PEEL.** 10-NN neighbourhoods at d = 16 share only about 0.34 Jaccard with
   full width. At d = 512 the overlap is 0.83–0.89. Short prefixes are a different neighbourhood
   structure, not just a blurred copy.
7. **Raw coordinates carry no order.** The literal first-d-coordinate prefix tracks random, as it
   should for an unordered residual stream.

## What this does and does not establish

- It does establish that a **label-free** coarse-to-fine reading axis exists in two frozen,
  non-Matryoshka models. It is strongest in early and lower-middle layers. Neon's kill condition 1 is
  passed. The iris has a knob.
- It does not establish that a reader can *use* that knob to do anything better. That is the next
  experiment.
- One model family (Pythia, same training data), one 192-word taxonomy, three templates. Before any
  external claim it needs a non-Pythia model (`--models Qwen/Qwen3-1.7B`) and a second word set.
- Prior art to cite: PEEL (Kim & Joe, 2026) for the prefix/taxonomy method; Mu & Viswanath (2018) and
  Timkey & van Schijndel (2021) for top-component and rogue-dimension effects; Park et al. (2024) for
  hierarchical concept geometry in LLMs.

## Qwen3-1.7B (added the same morning)

Third model, different family and training. Preregistered verdicts at layer 14 of 28:

- **K1 PASS** at d = 8, 16 and 32. Best d = 16: PCA 0.280 vs random 0.207 (best seed 0.228), log2 +0.44.
- **K2 TRUE.** corr(PC1, log-freq) = +0.54. Removing the top 1–3 PCs raises family purity by +0.088,
  three times the Pythia effect.
- **K3 FAIL.** No tail dilution: −0.001, CI [−0.005, +0.003].

So the main claim now holds on three models from two families.

The layer profile is **not** Pythia's. Qwen keeps a moderate advantage through almost the whole
depth: d = 16 log2 between +0.1 and +0.5 at nearly every layer, with PCA beating every random seed at
most layers. It only fades in the very last layer. Its lid is frequency, and it is heavy in the middle
layers (8–16, |corr(PC1, freq)| 0.53–0.62) and in the last layer (layer 28: purity 0.495 to 0.659 when
the top 10 PCs are removed). The Pythia picture, a Matryoshka at the bottom and a lid at the top, is a
Pythia picture. The shared part is a label-free coarse knob plus a lid, at depths that depend on the
model.

## Audit (exploratory, after the first results)

`audit.py` on the saved Pythia states. Full tables: [`results/AUDIT_EleutherAI_pythia-410m.md`](results/AUDIT_EleutherAI_pythia-410m.md),
[`results/AUDIT_EleutherAI_pythia-1b.md`](results/AUDIT_EleutherAI_pythia-1b.md).

1. **Not a tokenization artifact.** With only the 100 single-token test words, the advantage is
   nearly unchanged. Layer 0 is +1.13 (410m) and +1.03 (1b) at d = 16, and middle layers keep PCA above
   every random seed.
2. **The Matryoshka does not vanish at the top; it is under the lid.** Reading components
   k0 … k0+15 instead of 1 … 16 restores the advantage in every upper layer of both models:

   | model, layer | top 16 (k0 = 0) | skip 1 | skip 4 | skip 8 |
   |---|---:|---:|---:|---:|
   | 410m, layer 22 | −0.06 | +0.34* | +0.41* | +0.65* |
   | 410m, layer 24 (last) | −0.09 | +0.20* | +0.38* | +0.51* |
   | 1b, layer 16 (last) | −0.18 | +0.16 | +0.30* | +0.44* |

   So the earlier sentence "above about two thirds of the depth the order disappears" was wrong.
   The order is there at every depth. In upper layers a few top components sit on top of it, and the
   iris has to lift them before it reads. Lifting more (k0 = 8) helps even in the middle layers.

## Bridge (in Theory, 25 Nov 2026)

Writing the PCA reading back into Pythia-410m does **not** repair its answers: 0.306 vs native 0.339,
the same as an equally long random move. Iris's neighbour blend does repair (0.452), and random neighbours
harm (0.234). The knob is a good way to **read** neighbourhoods, not a state the model can use as its own.
Details: [Theory/BRIDGE.md](https://github.com/anttiluode/Theory/blob/main/BRIDGE.md).

## Consequence for the iris

A reader on a frozen model has **two** label-free knobs, and which one matters depends on depth:

- **prefix width** (how much of the PCA order to keep) works best in early and lower-middle layers;
- **lid removal** (drop the top few PCs) matters most in late layers.

The saved `*_states.npz` files hold every word state, so the next experiment can run on them
without the GPU.
