#!/usr/bin/env python3
"""
Maatuska — is an ordinary LLM's residual vector already a hidden Matryoshka?

Rules are frozen in PREREGISTRATION.md. This script measures, then applies those
rules mechanically and writes REPORT.md, results JSON and a figure.

  python maatuska.py --synthetic                 # instrument check, numpy only
  python maatuska.py                             # pythia-410m + pythia-1b
  python maatuska.py --models Qwen/Qwen3-1.7B    # extra model, descriptive

Do not hype. Do not lie. Just show.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Test words: 4 kingdoms x 4 families x 12 words. Labels are used only to score.
# --------------------------------------------------------------------------
TAXONOMY = {
    "animal": {
        "mammal": "dog cat horse cow lion tiger elephant rabbit wolf bear monkey deer",
        "bird": "eagle sparrow owl parrot crow pigeon swan duck penguin hawk robin flamingo",
        "fish": "salmon trout tuna cod shark carp herring sardine catfish mackerel perch pike",
        "insect": "ant bee beetle butterfly mosquito wasp moth cricket grasshopper dragonfly ladybug termite",
    },
    "plant": {
        "tree": "oak pine birch maple willow spruce cedar elm beech aspen fir cypress",
        "flower": "rose tulip daisy lily orchid sunflower daffodil violet poppy carnation peony marigold",
        "herb": "basil mint parsley thyme rosemary oregano dill sage cilantro chive lavender tarragon",
        "grain": "wheat rice barley oats corn rye millet sorghum quinoa buckwheat spelt maize",
    },
    "artifact": {
        "tool": "hammer saw screwdriver wrench drill chisel pliers axe shovel rake crowbar trowel",
        "furniture": "chair table sofa bed desk wardrobe shelf bench stool dresser cabinet couch",
        "vehicle": "car truck bus bicycle motorcycle train tractor van taxi tram scooter ambulance",
        "instrument": "piano guitar violin drum flute trumpet cello harp clarinet saxophone trombone accordion",
    },
    "person": {
        "medicine": "doctor nurse surgeon dentist pharmacist paramedic therapist midwife physician radiologist optometrist veterinarian",
        "trade": "carpenter plumber electrician mechanic welder mason blacksmith locksmith roofer bricklayer glazier tiler",
        "arts": "singer dancer actor poet novelist sculptor composer painter photographer playwright illustrator choreographer",
        "family": "mother father sister brother uncle aunt cousin grandmother grandfather nephew niece daughter",
    },
}

TEMPLATES = ["Here is a {w}", "I saw the {w}", "The word is {w}"]
K_NN = 10
N_RANDOM_SEEDS = 5
K1_DS = (8, 16, 32, 64)
K1_MIN_LOG2 = 0.20
K2_MIN_CORR = 0.5
N_BOOT = 2000


def build_words():
    words, kingdom, family = [], [], []
    for k, fams in TAXONOMY.items():
        for f, ws in fams.items():
            for w in ws.split():
                words.append(w)
                kingdom.append(k)
                family.append(f)
    assert len(words) == len(set(words)) == 192, "test words must be 192 and unique"
    return words, np.array(kingdom), np.array(family)


# --------------------------------------------------------------------------
# Measurement primitives
# --------------------------------------------------------------------------
def knn_purity_per_word(Z, labels, k=K_NN):
    """Leave-one-out cosine kNN: fraction of each word's k neighbours sharing its label."""
    Z = np.asarray(Z, dtype=np.float64)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(n, 1e-12)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return (labels[nn] == labels[:, None]).mean(axis=1)


def knn_sets(Z, k=K_NN):
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def chance_purity(labels):
    _, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    return float(sum(c * (c - 1) for c in counts) / (n * (n - 1)))


def width_grid(D):
    g, d = [], 2
    while d < D:
        g.append(d)
        d *= 2
    two_thirds = int(round(2 * D / 3))
    g += [two_thirds, D]
    return sorted(set(g))


_ROT_CACHE = {}


def random_orthonormal(D, seed):
    key = (D, seed)
    if key in _ROT_CACHE:
        return _ROT_CACHE[key]
    _ROT_CACHE[key] = _random_orthonormal(D, seed)
    return _ROT_CACHE[key]


def _random_orthonormal(D, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((D, D)))
    return q * np.sign(np.diag(r))


def analyse_layer(X_test, X_fit, kingdom, family, freq):
    """All measurements for one layer. X_* are (n, D) float arrays."""
    X_test = np.asarray(X_test, dtype=np.float64)
    X_fit = np.asarray(X_fit, dtype=np.float64)
    D = X_test.shape[1]
    mu = X_fit.mean(axis=0)
    Xt, Xf = X_test - mu, X_fit - mu
    # PCA of the disjoint, unlabeled vocabulary
    _, svals, vt = np.linalg.svd(Xf, full_matrices=Xf.shape[0] < D)
    V = vt.T  # columns ordered by variance; columns past rank(Xf) are arbitrary null directions
    rank = int(np.sum(svals > svals[0] * 1e-10))
    var = svals[:rank] ** 2
    grid = width_grid(D)
    bases = {"pca": V, "reverse": V[:, ::-1][:, : D], "raw": np.eye(D)}
    # reverse must start from the smallest *real* component, not from null directions
    bases["reverse"] = np.concatenate([V[:, :rank][:, ::-1], V[:, rank:]], axis=1)
    rand = [random_orthonormal(D, 1000 + s) for s in range(N_RANDOM_SEEDS)]

    out = {"D": D, "rank_fit": rank, "grid": grid,
           "explained_top10": (var[:10] / var.sum()).tolist()}
    for depth, lab in (("kingdom", kingdom), ("family", family)):
        curves = {name: [] for name in ("pca", "reverse", "raw")}
        rcurves = [[] for _ in rand]
        for d in grid:
            for name in curves:
                curves[name].append(float(knn_purity_per_word(Xt @ bases[name][:, :d], lab).mean()))
            for i, R in enumerate(rand):
                rcurves[i].append(float(knn_purity_per_word(Xt @ R[:, :d], lab).mean()))
        out[depth] = {**curves, "random_seeds": rcurves,
                      "random_mean": np.mean(rcurves, axis=0).tolist(),
                      "chance": chance_purity(lab)}

    # K3: tail dilution at a preregistered width 2/3 D vs full, paired bootstrap over words
    d23 = int(round(2 * D / 3))
    p23 = knn_purity_per_word(Xt @ V[:, :d23], family)
    pfull = knn_purity_per_word(Xt, family)
    diff = p23 - pfull
    rng = np.random.default_rng(7)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)])
    out["dilution"] = {"d": d23, "prefix": float(p23.mean()), "full": float(pfull.mean()),
                       "diff": float(diff.mean()),
                       "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}

    # K2: top PCs vs frequency proxy; all-but-the-top
    proj = Xt @ V[:, :10]
    out["freq_corr_top10"] = [float(np.corrcoef(proj[:, i], freq)[0, 1]) if np.std(proj[:, i]) > 0 else 0.0
                              for i in range(10)]
    abtt = []
    for k in range(0, 11):
        P = Xt - (Xt @ V[:, :k]) @ V[:, :k].T if k else Xt
        abtt.append(float(knn_purity_per_word(P, family).mean()))
    out["abtt_family"] = abtt

    # K4: neighbourhood overlap between PCA prefixes and full width
    full_nn = knn_sets(Xt)
    jac = []
    for d in grid:
        nn = knn_sets(Xt @ V[:, :d])
        inter = np.array([len(set(a) & set(b)) for a, b in zip(nn, full_nn)])
        jac.append(float(np.mean(inter / (2 * K_NN - inter))))
    out["knn_jaccard_vs_full"] = jac
    return out


def gates(primary):
    """Apply PREREGISTRATION.md rules to the primary-layer analysis."""
    fam = primary["family"]
    grid = primary["grid"]
    k1_rows = []
    for d in K1_DS:
        if d not in grid:
            continue
        i = grid.index(d)
        pca, rev = fam["pca"][i], fam["reverse"][i]
        seeds = [s[i] for s in fam["random_seeds"]]
        rmean = float(np.mean(seeds))
        log2 = math.log2(pca / rmean) if pca > 0 and rmean > 0 else float("-inf")
        ok = pca > max(seeds) and log2 >= K1_MIN_LOG2 and pca > rev
        k1_rows.append({"d": d, "pca": pca, "random_mean": rmean, "random_max": max(seeds),
                        "reverse": rev, "log2_pca_over_random": log2, "pass": bool(ok)})
    k1 = any(r["pass"] for r in k1_rows)
    pc1 = primary["freq_corr_top10"][0]
    k2 = abs(pc1) >= K2_MIN_CORR
    abtt = primary["abtt_family"]
    abtt_gain = max(abtt[1:4]) - abtt[0]
    dil = primary["dilution"]
    k3 = dil["ci95"][0] > 0
    return {"K1_hidden_matryoshka": {"pass": k1, "rows": k1_rows},
            "K2_frequency_on_top": {"true": k2, "corr_pc1": pc1,
                                    "abtt_gain_top1to3": abtt_gain,
                                    "abtt_helps": abtt_gain > 0.02},
            "K3_tail_dilution": {"pass": k3, **dil},
            }


# --------------------------------------------------------------------------
# Synthetic worlds for the instrument check (numpy only)
# --------------------------------------------------------------------------
def synthetic_world(kind, kingdom, family, D=256, n_vocab=1500, seed=0):
    """
    Three worlds, each hidden inside a random rotation so no raw coordinate is special.
    Components are blocks of directions [start:stop] with a per-direction scale.

    planted: kingdom and family codes are the highest-variance directions.
    diluted: same order, plus a block of *lowest*-variance nuisance in the last
             third of the spectrum that still corrupts cosine neighbourhoods
             (the PEEL tail). Truncating at 2/3 D removes it.
    buried:  a frequency axis and junk carry the highest per-direction variance;
             the hierarchy is fully present but spread thinly over many
             low-variance directions, so PCA order puts it late.
    The unlabeled vocabulary shares the same generative structure with many
    random categories, so a PCA fitted on it can find these blocks without our
    labels. (With only a few categories, categorical variance is low-rank and
    concentrates into top PCs by itself, which would make "buried" impossible.)
    """
    rng = np.random.default_rng(seed)
    basis = random_orthonormal(D, seed + 99)
    worlds = {
        "planted": dict(k=(0, 8, 6.0), f=(8, 24, 3.0), freq=(24, 1.0),
                        noise=[(25, D, 0.5)]),
        "diluted": dict(k=(0, 8, 1.0), f=(8, 24, 0.8), freq=(24, 0.3),
                        noise=[(25, 171, 0.40), (171, D, 0.28)]),
        "buried": dict(k=(70, 150, 0.5), f=(150, 230, 0.45), freq=(0, 4.0),
                       noise=[(1, 70, 1.0), (230, D, 0.05)]),
    }
    w = worlds[kind]
    # the vocabulary has many categories (as real vocabularies do); the test words use 4 and 16 of them
    n_k, n_f = 40, 160
    (k0, k1, sk), (f0, f1, sf), (q, sq) = w["k"], w["f"], w["freq"]
    k_codes = rng.standard_normal((n_k, k1 - k0))
    f_codes = rng.standard_normal((n_f, f1 - f0))

    def make(kidx, fidx, n):
        freq = rng.standard_normal(n)
        x = (sk * k_codes[kidx] @ basis[:, k0:k1].T + sf * f_codes[fidx] @ basis[:, f0:f1].T
             + sq * freq[:, None] * basis[:, q][None, :])
        for a0, a1, s in w["noise"]:
            x = x + s * rng.standard_normal((n, a1 - a0)) @ basis[:, a0:a1].T
        return x, freq

    kmap = {k: i for i, k in enumerate(sorted(set(kingdom)))}
    fmap = {f: i for i, f in enumerate(sorted(set(family)))}
    ki = np.array([kmap[k] for k in kingdom])
    fi = np.array([fmap[f] for f in family])
    X_test, freq_test = make(ki, fi, len(ki))
    X_fit, _ = make(rng.integers(0, n_k, n_vocab), rng.integers(0, n_f, n_vocab), n_vocab)
    return X_test, X_fit, freq_test


# --------------------------------------------------------------------------
# Real models
# --------------------------------------------------------------------------
def vocab_words(tok, exclude, n_max):
    out, seen = [], set(exclude)
    items = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    for t, _ in items:
        if not (t.startswith("Ġ") or t.startswith("▁")):
            continue
        w = t[1:]
        if len(w) >= 4 and w.isalpha() and w.islower() and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= n_max:
            break
    return out


def embed_words(model, tok, words, layers, device, batch=64):
    """Mean residual over the word's tokens, averaged over TEMPLATES. Returns {layer: (n, D)}."""
    import torch
    acc = {l: None for l in layers}
    for tmpl in TEMPLATES:
        prefix = tmpl.split("{w}")[0].rstrip()
        plen = len(tok(prefix, add_special_tokens=True)["input_ids"])
        reps = {l: [] for l in layers}
        for s in range(0, len(words), batch):
            texts = [tmpl.format(w=w) for w in words[s:s + batch]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            lens = enc["attention_mask"].sum(1).tolist()
            for l in layers:
                h = hs[l].float()
                rows = [h[i, plen:lens[i]].mean(0) if lens[i] > plen else h[i, lens[i] - 1]
                        for i in range(len(texts))]
                reps[l].append(torch.stack(rows).cpu().numpy())
        for l in layers:
            arr = np.concatenate(reps[l])
            acc[l] = arr if acc[l] is None else acc[l] + arr
    return {l: acc[l] / len(TEMPLATES) for l in layers}


def freq_proxy(model, tok, words, device, batch=64):
    """log P(word | 'The') under the model itself: a label-free frequency proxy."""
    import torch
    plen = len(tok("The", add_special_tokens=True)["input_ids"])
    out = []
    for s in range(0, len(words), batch):
        texts = [f"The {w}" for w in words[s:s + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        with torch.no_grad():
            logp = torch.log_softmax(model(**enc).logits.float(), -1)
        ids = enc["input_ids"]
        lens = enc["attention_mask"].sum(1).tolist()
        for i in range(len(texts)):
            lp = sum(float(logp[i, p - 1, ids[i, p]]) for p in range(plen, lens[i]))
            out.append(lp)
    return np.array(out)


def run_model(name, n_vocab, layer_step, device_pref, save_dir=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if (device_pref != "cpu" and torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[{name}] loading on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import transformers
    major = int(transformers.__version__.split(".")[0])
    kw = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
    n_layers = model.config.num_hidden_layers
    words, kingdom, family = build_words()
    mid = n_layers // 2
    layers = sorted(set(list(range(0, n_layers + 1, layer_step)) + [mid, n_layers]))
    vocab = vocab_words(tok, words, n_vocab)
    print(f"[{name}] {n_layers} layers, {len(vocab)} unlabeled vocabulary words, layers {layers}", flush=True)
    t0 = time.time()
    test = embed_words(model, tok, words, layers, device)
    fit = embed_words(model, tok, vocab, layers, device)
    freq = freq_proxy(model, tok, words, device)
    print(f"[{name}] embedded in {time.time() - t0:.0f}s; analysing ...", flush=True)
    if save_dir:
        tag = name.replace("/", "_")
        np.savez_compressed(os.path.join(save_dir, f"{tag}_states.npz"),
                            test_words=np.array(words), vocab_words=np.array(vocab), freq=freq,
                            kingdom=kingdom, family=family, layers=np.array(layers),
                            **{f"test_L{l}": test[l].astype(np.float16) for l in layers},
                            **{f"vocab_L{l}": fit[l].astype(np.float16) for l in layers})
    per_layer = {}
    for l in layers:
        per_layer[l] = analyse_layer(test[l], fit[l], kingdom, family, freq)
        fam = per_layer[l]["family"]
        print(f"  layer {l:3d}: family purity full={fam['pca'][-1]:.3f}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"model": name, "n_layers": n_layers, "primary_layer": mid, "n_vocab": len(vocab),
            "layers": per_layer, "gates": gates(per_layer[mid])}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def plot(result, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    prim = result["layers"][result["primary_layer"]]
    grid = prim["grid"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for depth, ls in (("kingdom", "--"), ("family", "-")):
        p = prim[depth]
        ax[0].plot(grid, p["pca"], ls, color="#1f6feb", label=f"PCA ({depth})")
        ax[0].plot(grid, p["random_mean"], ls, color="#888888", label=f"random ({depth})")
        ax[0].plot(grid, p["reverse"], ls, color="#d1242f", label=f"reverse ({depth})")
        ax[0].axhline(p["chance"], ls=":", color="#bbbbbb", lw=1)
    ax[0].set_xscale("log", base=2)
    ax[0].set_title(f"layer {result['primary_layer']}: kNN purity vs width")
    ax[0].set_xlabel("prefix width d")
    ax[0].legend(fontsize=7)
    cmap = plt.get_cmap("viridis")
    ls_ = sorted(result["layers"])
    for j, l in enumerate(ls_):
        f = result["layers"][l]["family"]
        adv = [math.log2(a / b) if a > 0 and b > 0 else 0 for a, b in zip(f["pca"], f["random_mean"])]
        ax[1].plot(result["layers"][l]["grid"], adv, color=cmap(j / max(1, len(ls_) - 1)),
                   lw=2.5 if l == result["primary_layer"] else 1, label=f"L{l}")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].axhline(K1_MIN_LOG2, color="#1f6feb", ls=":", lw=1)
    ax[1].set_xscale("log", base=2)
    ax[1].set_title("family: log2(PCA / random)  (PEEL Fig.1 analogue)")
    ax[1].set_xlabel("prefix width d")
    ax[1].legend(fontsize=6, ncol=2)
    ax[2].plot(range(11), prim["abtt_family"], "o-", color="#8250df")
    ax[2].set_title("family purity after removing top-k PCs")
    ax[2].set_xlabel("k removed")
    fig.suptitle(result["model"])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def verdict_lines(result):
    g = result["gates"]
    k1, k2, k3 = g["K1_hidden_matryoshka"], g["K2_frequency_on_top"], g["K3_tail_dilution"]
    best = max(k1["rows"], key=lambda r: r["log2_pca_over_random"]) if k1["rows"] else None
    L = [f"### {result['model']}  (primary layer {result['primary_layer']} of {result['n_layers']})",
         "",
         f"- **K1 hidden Matryoshka: {'PASS' if k1['pass'] else 'FAIL'}**"
         + (f" — best d={best['d']}: PCA {best['pca']:.3f} vs random {best['random_mean']:.3f}"
            f" (max seed {best['random_max']:.3f}), reverse {best['reverse']:.3f},"
            f" log2 = {best['log2_pca_over_random']:+.2f}" if best else ""),
         f"- **K2 frequency on top: {'TRUE' if k2['true'] else 'FALSE'}** — corr(PC1, log-freq) = {k2['corr_pc1']:+.2f};"
         f" removing top 1–3 PCs changes family purity by {k2['abtt_gain_top1to3']:+.3f}"
         f" ({'helps' if k2['abtt_helps'] else 'does not help'} by the 0.02 rule)",
         f"- **K3 tail dilution: {'PASS' if k3['pass'] else 'FAIL'}** — d={k3['d']}: {k3['prefix']:.3f} vs full {k3['full']:.3f},"
         f" diff {k3['diff']:+.3f}, 95% CI [{k3['ci95'][0]:+.3f}, {k3['ci95'][1]:+.3f}]",
         ""]
    rows = ["| d | PCA | random mean | random max | reverse | log2 PCA/random | pass |",
            "|---:|---:|---:|---:|---:|---:|:---:|"]
    for r in k1["rows"]:
        rows.append(f"| {r['d']} | {r['pca']:.3f} | {r['random_mean']:.3f} | {r['random_max']:.3f} |"
                    f" {r['reverse']:.3f} | {r['log2_pca_over_random']:+.2f} | {'✓' if r['pass'] else '✗'} |")
    return L + rows + [""]


def write_report(results, path, synthetic):
    lines = ["# Maatuska — report", "",
             "Generated by `maatuska.py`; rules from `PREREGISTRATION.md`.", ""]
    if synthetic:
        # expected (K1, K2, K3) per world; each gate must be seen both passing and failing
        want = {"planted": (True, False, False), "diluted": (True, False, True), "buried": (False, True, False)}
        ok = True
        for r in results:
            g = r["gates"]
            got = (g["K1_hidden_matryoshka"]["pass"], g["K2_frequency_on_top"]["true"], g["K3_tail_dilution"]["pass"])
            exp = want[r["model"].split(":")[1]]
            ok &= got == exp
            lines += [f"- {r['model']}: (K1, K2, K3) = {got}, expected {exp}"]
        lines = lines[:4] + [f"## Instrument check: {'PASSED' if ok else 'FAILED'}", "",
                             "Each gate must be seen both passing and failing in worlds built to make it so.", ""] + lines[4:] + [""]
    else:
        primary = [r for r in results if "pythia-410m" in r["model"] or "pythia-1b" in r["model"]]
        if len(primary) == 2:
            both = all(r["gates"]["K1_hidden_matryoshka"]["pass"] for r in primary)
            lines += [f"## Main claim (K1 on both primary models): {'PASS' if both else 'KILL'}", ""]
    for r in results:
        lines += verdict_lines(r)
        if r.get("figure"):
            lines += [f"![{r['model']}]({os.path.basename(r['figure'])})", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="instrument check on synthetic worlds (numpy only)")
    ap.add_argument("--models", nargs="+", default=["EleutherAI/pythia-410m", "EleutherAI/pythia-1b"])
    ap.add_argument("--n-vocab", type=int, default=4000)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-save-states", action="store_true",
                    help="do not save hidden states (.npz) for reuse by later experiments")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    words, kingdom, family = build_words()
    results = []
    if a.synthetic:
        for kind in ("planted", "diluted", "buried"):
            Xt, Xf, fq = synthetic_world(kind, kingdom, family)
            lay = analyse_layer(Xt, Xf, kingdom, family, fq)
            res = {"model": f"synthetic:{kind}", "n_layers": 0, "primary_layer": 0, "n_vocab": len(Xf),
                   "layers": {0: lay}, "gates": gates(lay)}
            results.append(res)
    else:
        for m in a.models:
            results.append(run_model(m, a.n_vocab, a.layer_step, a.device,
                                     None if a.no_save_states else a.out))
    for r in results:
        tag = r["model"].replace("/", "_").replace(":", "_")
        r["figure"] = plot(r, os.path.join(a.out, f"{tag}.png"))
        with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
            json.dump({**r, "layers": {str(k): v for k, v in r["layers"].items()}}, f, indent=1)
        print("\n".join(verdict_lines(r)))
    write_report(results, os.path.join(a.out, "REPORT.md"), a.synthetic)
    print(f"\nwrote {os.path.join(a.out, 'REPORT.md')}")


if __name__ == "__main__":
    main()
