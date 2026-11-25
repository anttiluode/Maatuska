# Maatuska — preregistration

Written before any real model was run. The verdict lines in `REPORT.md` are
computed mechanically from these rules by `maatuska.py`; the thresholds below are
not to be changed after seeing results. If a rule turns out to be badly chosen,
the fix goes into a new, dated amendment section and the original verdict stays
in the record.

## Question

Is the residual stream of an ordinary pretrained LLM, a model never trained with a
Matryoshka objective, already ordered coarse-to-fine along a **label-free** axis?

PEEL (Kim & Joe, arXiv 2610.41785, Oct 2026) found coarse-to-fine taxonomic order
in the prefix dimensions of retrieval embeddings that were *trained* to be nested.
Its §3 depends on that MRL training and leaves open whether standard models carry
the same order. The opposite prediction comes from classical word-embedding
work: the top principal components of word vectors mostly encode frequency and
removing them helps semantics (Mu & Viswanath, "All-but-the-Top", 2018).

The answer matters for this research line: the reader/regulator experiments
(the "iris", grain lens, Neon's kill condition 1) all need a knob that changes
semantic resolution without labels. If no label-free axis orders the frozen
model's vector coarse-to-fine, that knob does not exist in the obvious place.

## Substrate

- Frozen models: `EleutherAI/pythia-410m` and `EleutherAI/pythia-1b` (primary
  replication pair). Others optional and descriptive only.
- Representation: mean over the word's own tokens of the residual stream at each
  layer, averaged over three neutral templates
  (`Here is a {w}`, `I saw the {w}`, `The word is {w}`).
- Test set: 192 words, 4 kingdoms (animal, plant, artifact, person) × 4 families
  × 12 words. Labels are used **only for scoring**.
- Label-free basis: PCA fitted on up to 4000 ordinary vocabulary words taken from
  the model's own tokenizer, disjoint from the test words, centred on that
  vocabulary's mean. The test words are never used to fit any basis.

## Measurements

kNN label purity (k = 10, cosine, leave-one-out) at depth 1 (kingdom) and
depth 2 (family), for prefixes of width d ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512,
… , full} under four orderings:

- **PCA** — top-d principal components of the disjoint vocabulary;
- **reverse** — bottom-d principal components (PEEL's tail control);
- **random** — d directions of a random orthonormal basis, 5 seeds (PEEL's
  shuffled control);
- **raw** — the first d raw coordinates (literal prefix; descriptive only).

Also: correlation of the top 10 PCs with a frequency proxy (the model's own
log-probability of the word after "The"); purity after removing the top k PCs
(k = 0…10); 10-NN set overlap between each PCA prefix and full width.

## Gates

The primary layer is the middle layer, `L // 2`, of each model. All other
layers are descriptive.

**K1 — hidden Matryoshka (the main question).** PASS for a model if, at the
primary layer, for at least one d ∈ {8, 16, 32, 64}: the depth-2 PCA purity
exceeds the purity of **every** one of the 5 random seeds, and
log2(PCA / mean random) ≥ 0.20, and PCA > reverse. The claim "an ordinary LLM
carries a label-free coarse-to-fine order" requires K1 to pass on **both**
primary models. Fail on either → KILL for "PCA order is the iris knob".

**K2 — frequency on top.** Reported as TRUE for a model if |corr(PC1, frequency
proxy)| ≥ 0.5 at the primary layer. Separately reported: whether removing the
top 1–3 PCs raises full-width depth-2 purity by more than 0.02.

**K3 — tail dilution (PEEL's d = 512 < 768 bump).** At the primary layer, compare
depth-2 purity of the PCA prefix at d = round(2/3 · D) against full width D,
with a paired bootstrap over words (2000 resamples). PASS if the lower 95% bound
of (prefix − full) is above 0. The peak d is reported descriptively but is not
the test, because choosing the best d would bias the comparison.

**K4 — non-nesting.** Descriptive only: 10-NN Jaccard between PCA-prefix
neighbourhoods and full width, as a counterpart to PEEL's < 0.18 tree-edge
Jaccard.

## Instrument check

Before real models, `python maatuska.py --synthetic` must show that the
instrument can tell the two outcomes apart. In the *planted* world, hierarchy
sits in high-variance directions, and K1 must PASS. In the *buried* world,
high-variance nuisance (a frequency axis plus junk) sits on top and the
hierarchy is in low-variance directions; there K1 must FAIL. If the
instrument cannot separate these, the real-model result means nothing.

## What a result would and would not mean

- K1 PASS on both: label-free PCA order in a frozen ordinary LLM carries extra
  coarse-to-fine taxonomic structure beyond what random subspaces carry at
  equal width. It does not show that a reader can *use* it, and it does not
  show the order is nested.
- K1 FAIL: the obvious label-free grain knob is not there. The grain lens would
  need labels, context (the within-context mean / residue split), or a learned
  basis.
- Random subspaces also show coarse-before-fine, because coarse distinctions have
  more variance. That is why the comparison is always PCA *versus random at
  equal width*, never PCA alone.

## Amendment 1 — instrument worlds (25 Nov 2026, before any real model run)

While building the instrument check, the synthetic worlds were revised twice. No gate,
threshold, width set or primary layer was changed.

1. The first "buried" world was invisible even at full width (purity at chance).
   That tests nothing, so it was rebuilt so the hierarchy is present but placed late
   in variance order.
2. With only 4 kingdoms and 16 families in the unlabeled vocabulary, categorical
   variance is low-rank and lands in the top PCs on its own, so "buried" could not
   be built. Real vocabularies contain thousands of categories, so the synthetic
   vocabularies now draw from 40 kingdoms and 160 families, and the test words use
   4 and 16 of them.
3. A third world, "diluted", was added so K3 is seen passing as well as failing.

Final instrument check: planted (K1 PASS), diluted (K1 PASS, K3 PASS),
buried (K1 FAIL, K2 TRUE), as expected. An untrained tiny GPT-NeoX run end to end
gave chance-level purity, as it should.
