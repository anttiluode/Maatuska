# Maatuska audit — EleutherAI_pythia-1b_states.npz

Exploratory, added after the first results; not preregistered.

Single-token test words: 100 of 192.

Single-token words per family: arts 9, bird 3, family 12, fish 5, flower 2, furniture 11, grain 5, herb 4, insect 6, instrument 4, mammal 12, medicine 6, tool 5, trade 1, tree 4, vehicle 11

## 1. Single-token words only — family log2(PCA / random) at d = 8, 16, 32

| layer | all words d16 | single-token d8 | d16 | d32 | single-token full purity | chance |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +1.10* | +0.93* | +1.03* | +0.83* | 0.587 | 0.073 |
| 2 | +0.80* | +0.63* | +0.71* | +0.40* | 0.574 | 0.073 |
| 4 | +0.80* | +0.53* | +0.73* | +0.45* | 0.583 | 0.073 |
| 6 | +0.76* | +0.50* | +0.77* | +0.39* | 0.583 | 0.073 |
| 8 | +0.43* | +0.29* | +0.45* | +0.20* | 0.609 | 0.073 |
| 10 | +0.40* | +0.09 | +0.39* | +0.18* | 0.601 | 0.073 |
| 12 | +0.24* | +0.18 | +0.31* | +0.10* | 0.618 | 0.073 |
| 14 | +0.08 | +0.06 | +0.11 | +0.02 | 0.613 | 0.073 |
| 16 | -0.18 | -0.31 | -0.19 | -0.11 | 0.592 | 0.073 |

`*` = PCA beats all 5 random seeds.

## 2. Lid-lifted prefixes — family log2(PCA[k0 : k0+16] / random 16), all words

| layer | k0=0 | k0=1 | k0=2 | k0=4 | k0=8 |
|---:|---:|---:|---:|---:|---:|
| 0 | +1.10* | +1.20* | +1.22* | +1.33* | +1.24* |
| 2 | +0.80* | +0.92* | +0.90* | +0.92* | +0.74* |
| 4 | +0.80* | +0.97* | +1.00* | +1.04* | +1.02* |
| 6 | +0.76* | +0.78* | +0.88* | +0.84* | +0.88* |
| 8 | +0.43* | +0.47* | +0.58* | +0.68* | +0.76* |
| 10 | +0.40* | +0.44* | +0.47* | +0.58* | +0.71* |
| 12 | +0.24* | +0.38* | +0.40* | +0.44* | +0.61* |
| 14 | +0.08 | +0.27* | +0.33* | +0.36* | +0.51* |
| 16 | -0.18 | +0.16 | +0.16 | +0.30* | +0.44* |

`*` = PCA band beats all 5 random seeds.
