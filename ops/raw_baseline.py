#!/usr/bin/env python
"""Raw-spectra baseline, standalone: nested-CV parameter_0 R² and RMSE of a linear probe on the raw measured
spectra, exactly as the SpectralFM-LeJEPA evaluation scores its `raw` arm. Needs only numpy, scipy,
scikit-learn (1.6.1 used for the reference numbers), soundfile and joblib; no GPU, no other repo file.

    python raw_baseline.py --data-root /path/to/nova_data [--components 0 | 0,1 | 0,1,2]
                           [--fraction 1.0 | 0.25 | 0.0625] [--sets dataset0055 ...] [--jobs 8] [--output r.json]

Data layout (as on our servers): <data-root>/labeled_regression/dataset<D>/ and <data-root>/labeled_data/, each
with labels.tsv (`filename<TAB>parameter_0`) and wavs named dataset<D>_comp<C>_spec_<S>.wav under wav/, wavs/
or the set directory itself. Each wav is truncated / zero-padded to 245 points; the probe sees raw values.

Protocol (fixed seeds, so every run is reproducible):
- outer CV: 2 repeats x 5 folds, KFold(shuffle, random_state=42+r); inner CV: 5 folds on each outer training fold;
- 12 recipes = normalizer {none, standardize, whiten, whiten8, whiten32, whiten128} x probe {RidgeCV(alphas =
  logspace(-3, 3, 20), leave-one-out), LinearRegression}; the recipe with the best inner R² is refit on the whole
  outer training fold and predicts the held-out fold. Recipe seed per fold = 42 + 1000 + 10r + f.
- R² and RMSE are means over the 2 repeats of the out-of-fold predictions; ± is the bootstrap SD (500 resamples).
- `--components 0`: component 0 only, all eight sets (labeled_data + seven labeled_regression sets); inner folds
  are row-level KFold(shuffle, random_state=fold seed). Reproduces the report's raw R² (e.g. labeled_data .830).
- `--components 0,1[,2]`: the joined-component study (report Appendix K), labeled_regression sets only. Samples
  with every listed component; scores comp<c> (one component) and concat ([x_c0 | x_c1 | ...], one vector per
  sample); inner folds are seeded group folds (as in the component study). Reproduces Appendix K's raw columns.
- `--fraction F`: keep a seeded share F of each set's samples (default_rng(0) permutation of the sorted samples,
  first ceil(F x n) kept, so smaller fractions are subsets of larger ones), before the CV. 1.0 = all samples.
  Sets left with fewer than 20 samples are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import warnings

import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore", category=LinAlgWarning)

SEED = 42                # evaluation seed: outer folds, subsample, recipe seeds
FRACTION_SEED = 0        # sample-selection seed for --fraction
MAX_SAMPLES = 5000       # larger sets are subsampled (default_rng(SEED)); none of ours is
MIN_N = 20
LENGTH = 245
N_REPEATS, N_FOLDS, N_INNER = 2, 5, 5
NORMALIZERS = ("none", "standardize", "whiten", "whiten8", "whiten32", "whiten128")
PROBES = ("ridgecv", "ols")
RECIPES = tuple((n, p) for n in NORMALIZERS for p in PROBES)
WHITEN_SAMPLES_PER_DIRECTION = 2
SETS = ("labeled_data", "dataset0055", "dataset0106", "dataset0109", "dataset0112", "dataset0113", "dataset0114",
        "dataset0120")
PATTERN = re.compile(r"dataset(\d+)_comp(\d+)_spec_(\d+)\.wav")


# ---------------------------------------------------------------- data

def set_dir(root, name):
    return os.path.join(root, "labeled_data" if name == "labeled_data" else os.path.join("labeled_regression", name))


def _wav_root(d):
    for sub in ("wav", "wavs"):
        if glob.glob(os.path.join(d, sub, "*.wav")):
            return os.path.join(d, sub)
    return d


def load(d, components):
    """(X {component: [n, 245] float32}, y [n], n_all): samples (dataset, spec) that have every listed component
    in labels.tsv and on disk, sorted; n_all = number of such samples before any subsample."""
    files, labels = {}, {}
    root = _wav_root(d)
    with open(os.path.join(d, "labels.tsv")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2 or (m := PATTERN.match(parts[0])) is None:
                continue
            key = (int(m[1]), int(m[3]), int(m[2]))
            if key[2] not in components:
                continue
            path = os.path.join(root, parts[0])
            files[key] = path if os.path.isfile(path) else os.path.join(d, parts[0])
            labels[key] = float(parts[1])
    keys = [k for k in sorted({k[:2] for k in files})
            if all((*k, c) in files and os.path.isfile(files[(*k, c)]) for c in components)]
    n_all = len(keys)
    if len(keys) > MAX_SAMPLES:
        idx = np.sort(np.random.default_rng(SEED).choice(len(keys), MAX_SAMPLES, replace=False))
        keys = [keys[i] for i in idx]
    for k in keys:
        if len({labels[(*k, c)] for c in components}) != 1:
            raise ValueError(f"dataset {k[0]}, spec {k[1]}: components disagree on the label")
    X = {}
    for c in components:
        rows = []
        for k in keys:
            data = np.asarray(sf.read(files[(*k, c)], dtype="float32")[0]).flatten()
            row = np.zeros(LENGTH, np.float32)
            row[:min(len(data), LENGTH)] = data[:LENGTH]
            rows.append(row)
        X[c] = np.stack(rows)
    y = np.array([labels[(*k, components[0])] for k in keys], np.float64)
    return X, y, n_all


def take_fraction(n, fraction):
    """Sorted indices of the kept samples: the first ceil(fraction x n) of one seeded permutation."""
    if fraction >= 1:
        return np.arange(n)
    perm = np.random.default_rng(FRACTION_SEED).permutation(n)
    return np.sort(perm[:max(1, math.ceil(fraction * n))])


# ---------------------------------------------------------------- probes

def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _standardize(X_tr, X_te):
    X_tr32 = np.asarray(X_tr).astype(np.float32, copy=False)
    mu = X_tr32.mean(axis=0).astype(np.float32)
    sd = X_tr32.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return [((np.asarray(X, dtype=np.float32) - mu) / sd).astype(np.float32) for X in (X_tr, X_te)]


def _whiten(X_tr, X_te, seed):
    """Exact PCA whitening fit on X_tr, at most one direction per 2 training rows, dead directions dropped."""
    X64 = np.asarray(X_tr, dtype=np.float64)
    n, d = X64.shape
    k = min(d, max(1, n // WHITEN_SAMPLES_PER_DIRECTION))
    pca = PCA(n_components=k, whiten=True, svd_solver="full", random_state=seed).fit(X64)
    ev = pca.explained_variance_
    tol = (np.finfo(np.float64).eps * max(n, d)) ** 2
    k_keep = max(1, int(np.sum(ev > tol * ev[0])) if ev.size else 0)
    return [pca.transform(np.asarray(X, dtype=np.float64))[:, :k_keep].astype(np.float32) for X in (X_tr, X_te)]


def fold_transforms(X_tr, X_te, seed):
    out = {"none": (np.asarray(X_tr, np.float32), np.asarray(X_te, np.float32)),
           "standardize": tuple(_standardize(X_tr, X_te))}
    Ztr, Zte = _whiten(X_tr, X_te, seed)
    for name in NORMALIZERS[2:]:
        k = Ztr.shape[1] if name == "whiten" else min(Ztr.shape[1], int(name[len("whiten"):]))
        out[name] = (Ztr[:, :k], Zte[:, :k])
    return out


def recipe_predictions(X_tr, y_tr, X_te, recipes, seed):
    T = fold_transforms(X_tr, X_te, seed)
    out = {}
    for norm, probe in recipes:
        m = RidgeCV(alphas=np.logspace(-3, 3, 20)) if probe == "ridgecv" else LinearRegression()
        m.fit(T[norm][0], y_tr)
        out[(norm, probe)] = np.asarray(m.predict(T[norm][1])).ravel()
    return out


def inner_folds(n, seed, grouped):
    """Row-level KFold (comp0 protocol) or the component study's seeded group folds (one group per sample)."""
    if not grouped:
        yield from KFold(N_INNER, shuffle=True, random_state=seed).split(np.zeros(n))
        return
    for chunk in np.array_split(np.random.default_rng(seed).permutation(n), N_INNER):
        test = np.isin(np.arange(n), chunk)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def outer_fold(arms, y, tr, te, seed, grouped):
    """{arm: (held-out prediction, chosen recipe, inner R²)}: recipe picked by inner CV on tr, refit on tr."""
    out = {}
    with threadpool_limits(limits=1):
        for a, X in arms.items():
            Xtr, ytr = X[tr], y[tr]
            preds = {rc: np.zeros(len(tr)) for rc in RECIPES}
            for itr, ite in inner_folds(len(tr), seed, grouped):
                P = recipe_predictions(Xtr[itr], ytr[itr], Xtr[ite], RECIPES, seed)
                for rc in RECIPES:
                    preds[rc][ite] = P[rc]
            inner = {rc: r2(ytr, p) for rc, p in preds.items()}
            best = max(RECIPES, key=lambda rc: inner[rc])
            out[a] = (recipe_predictions(Xtr, ytr, X[te], [best], seed)[best], best, inner[best])
    return out


def nested_cv(arms, y, grouped, jobs):
    n = len(y)
    folds = [(r, f, tr, te) for r in range(N_REPEATS)
             for f, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(np.zeros(n)))]
    outs = Parallel(n_jobs=jobs)(delayed(outer_fold)(arms, y, tr, te, SEED + 1000 + 10 * r + f, grouped)
                                 for r, f, tr, te in folds)
    oof = {a: np.zeros((N_REPEATS, n)) for a in arms}
    chosen = {a: [] for a in arms}
    for (r, f, tr, te), out in zip(folds, outs):
        for a, (pred, (norm, probe), ir2) in out.items():
            oof[a][r, te] = pred
            chosen[a].append({"repeat": r, "fold": f, "norm": norm, "probe": probe, "inner_r2": ir2})
    return oof, chosen


def bootstrap_sd(y, P, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.std([np.mean([r2(y[i], p[i]) for p in P])
                         for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]))


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, help="directory holding labeled_regression/ and labeled_data/")
    ap.add_argument("--components", default="0", help="0 (single component) or e.g. 0,1 / 0,1,2 (joined)")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of each set's samples to keep (0, 1]")
    ap.add_argument("--sets", nargs="+", help="default: all eight sets (components 0) or the seven "
                                              "labeled_regression sets (joined components)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="outer folds run in parallel")
    ap.add_argument("--output", help="write per-set results (scores, chosen recipes, OOF predictions) as JSON")
    a = ap.parse_args(argv)
    comps = tuple(int(c) for c in a.components.split(","))
    if not 0 < a.fraction <= 1:
        ap.error("--fraction must be in (0, 1]")
    joined = len(comps) > 1
    sets = a.sets or [s for s in SETS if not (joined and s == "labeled_data")]
    scores = ([f"comp{c}" for c in comps] + ["concat"]) if joined else ["comp0"]
    print(f"raw baseline: components {comps}, fraction {a.fraction}, seed {SEED}, "
          f"{N_REPEATS}x{N_FOLDS} outer / {N_INNER} inner folds, {len(RECIPES)} recipes")
    print(f"{'set':<13}{'n_all':>7}{'n':>6}  " + "  ".join(f"{s + ' R²':>16}{'RMSE':>9}" for s in scores))
    results = {}
    for name in sets:
        X, y, n_all = load(set_dir(a.data_root, name), comps)
        keep = take_fraction(len(y), a.fraction)
        X, y = {c: v[keep] for c, v in X.items()}, y[keep]
        if len(y) < MIN_N:
            print(f"{name:<13}{n_all:>7}{len(y):>6}  skipped: fewer than {MIN_N} samples")
            continue
        arms = {f"comp{c}": X[c] for c in comps}
        if joined:
            arms["concat"] = np.concatenate([X[c] for c in comps], axis=1)
        oof, chosen = nested_cv(arms, y, grouped=joined, jobs=a.jobs)
        res = {s: {"r2": float(np.mean([r2(y, p) for p in oof[s]])), "r2_sd": bootstrap_sd(y, oof[s]),
                   "rmse": float(np.mean([rmse(y, p) for p in oof[s]]))} for s in scores}
        print(f"{name:<13}{n_all:>7}{len(y):>6}  " + "  ".join(
            f"{res[s]['r2']:>9.3f} ±{res[s]['r2_sd']:.3f}{res[s]['rmse']:>9.4g}" for s in scores), flush=True)
        results[name] = {"n_all": n_all, "n": int(len(y)), "kept": keep.tolist(), "scores": res,
                         "chosen": chosen, "oof": {s: oof[s].tolist() for s in scores}, "y": y.tolist()}
    if results:
        print(f"{'mean':<26}" + "  ".join(f"{np.mean([r['scores'][s]['r2'] for r in results.values()]):>9.3f}"
                                          f"{'':>7}{np.mean([r['scores'][s]['rmse'] for r in results.values()]):>9.4g}"
                                          for s in scores))
    if a.output:
        with open(a.output, "w") as f:
            json.dump({"protocol": {"components": comps, "fraction": a.fraction, "seed": SEED,
                                    "fraction_seed": FRACTION_SEED, "recipes": [f"{n}+{p}" for n, p in RECIPES]},
                       "sets": results}, f)


if __name__ == "__main__":
    main()
