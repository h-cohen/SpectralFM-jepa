"""Nested fixed-block head comparison for the frozen feature study.

Each arm uses the same repeated outer and inner folds. Standardization is refit on
the training rows at each level, and candidate heads are selected from pooled inner
out-of-fold predictions only.
"""
from __future__ import annotations

import warnings
import math

import numpy as np
from scipy.linalg import eigh
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from spectral_lejepa.evaluation.nested import bootstrap_sd, paired_delta
from spectral_lejepa.evaluation.probe import r2


def _scale(X_train, X_valid):
    scaler = StandardScaler()
    return (scaler.fit_transform(X_train).astype(np.float32),
            scaler.transform(X_valid).astype(np.float32))


def _menu(n_rows, n_features):
    # PLSRegression requires components <= min(n_samples, n_features). Leave one
    # sample of slack to avoid degenerate full-rank fits in small inner folds.
    max_components = min(n_rows - 1, n_features)
    recipes = [("ridgecv", "ridgecv")]
    recipes += [(f"pls_{min(c, max_components)}", ("pls", min(c, max_components)))
                for c in (2, 8, 16) if max_components >= 1]
    recipes += [(f"rbf_a{alpha:g}_g{gamma:g}", ("rbf", alpha, gamma))
                for gamma in (0.1, 1.0, 10.0) for alpha in (0.01, 0.1, 1.0)]
    # Small feature spaces can map several nominated PLS sizes to the same cap.
    return list(dict(recipes).items())


def _predict_menu(X_train, y_train, X_valid, recipes):
    """Fit candidate heads; reuse each RBF kernel eigendecomposition across alphas."""
    out = {}
    rbf_eigens = {}
    for name, spec in recipes:
        if spec == "ridgecv":
            model = RidgeCV(alphas=np.logspace(-3, 3, 20))
            model.fit(X_train, y_train)
            out[name] = np.asarray(model.predict(X_valid)).ravel()
        elif spec[0] == "pls":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = PLSRegression(n_components=spec[1], scale=False, max_iter=500)
                model.fit(X_train, y_train)
            out[name] = np.asarray(model.predict(X_valid)).ravel()
        else:
            _, alpha, gamma_multiplier = spec
            gamma = gamma_multiplier / X_train.shape[1]
            key = gamma_multiplier
            if key not in rbf_eigens:
                K = rbf_kernel(X_train, gamma=gamma)
                eigval, eigvec = eigh(K, check_finite=False, driver="evd")
                k_valid = rbf_kernel(X_valid, X_train, gamma=gamma)
                rbf_eigens[key] = (eigval, eigvec, k_valid)
            eigval, eigvec, k_valid = rbf_eigens[key]
            weights = eigvec.T @ y_train
            coef = eigvec @ (weights / (eigval + alpha))
            out[name] = np.asarray(k_valid @ coef).ravel()
    return out


def _arm_outer(X, y, tr, te, n_inner, seed):
    # Size the shared menu to the smallest inner-training split, so every
    # nominated PLS model is valid in every inner fold.
    smallest_inner_train = len(tr) - math.ceil(len(tr) / n_inner)
    recipes = _menu(smallest_inner_train, X.shape[1])
    inner_folds = list(KFold(n_inner, shuffle=True, random_state=seed).split(tr))
    inner_preds = {name: np.zeros(len(tr), dtype=np.float64) for name, _ in recipes}
    # Cache scaler outputs once per inner fold; each candidate shares these arrays.
    for itr, iva in inner_folds:
        inner_tr, inner_va = tr[itr], tr[iva]
        Xtr, Xva = _scale(X[inner_tr], X[inner_va])
        preds = _predict_menu(Xtr, y[inner_tr], Xva, recipes)
        for name, pred in preds.items():
            inner_preds[name][iva] = pred
    scores = {name: r2(y[tr], pred) for name, pred in inner_preds.items()}
    chosen = max(scores, key=scores.get)
    Xtr, Xte = _scale(X[tr], X[te])
    pred = _predict_menu(Xtr, y[tr], Xte, [(chosen, dict(recipes)[chosen])])[chosen]
    return pred, {"recipe": chosen, "inner_r2": float(scores[chosen])}


def compare_heads(model_flat: np.ndarray, random_flat: np.ndarray, raw: np.ndarray, y, *,
                  n_repeats: int = 2, n_folds: int = 5, n_inner: int = 5,
                  seed: int = 42) -> dict:
    """Compare predeclared standardized heads on raw, trained, and random features."""
    arrays = {"raw": np.asarray(raw), "trained": np.asarray(model_flat),
              "random": np.asarray(random_flat)}
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if any(a.ndim != 2 for a in arrays.values()):
        raise ValueError("all feature arms must be two-dimensional matrices")
    row_counts = {len(y), *(len(a) for a in arrays.values())}
    if len(row_counts) != 1:
        raise ValueError("all feature arms and labels must have the same number of rows")
    if n_repeats < 1 or min(n_folds, n_inner) < 2 or len(y) < n_folds:
        raise ValueError("n_repeats must be >= 1; n_folds and n_inner >= 2; n_folds <= n")
    if any(a.shape[1] == 0 for a in arrays.values()):
        raise ValueError("feature arms must have at least one column")
    if arrays["raw"].shape[1] != 245:
        raise ValueError("raw features must have 245 columns")
    smallest_outer_train = len(y) - math.ceil(len(y) / n_folds)
    if smallest_outer_train < n_inner:
        raise ValueError("n_inner must not exceed the smallest outer training fold")

    n = len(y)
    predictions = {name: np.zeros((n_repeats, n), dtype=np.float64) for name in arrays}
    chosen = {name: [] for name in arrays}
    test_folds = []
    for repeat in range(n_repeats):
        folds = list(KFold(n_folds, shuffle=True, random_state=seed + repeat).split(y))
        test_folds.append([te.tolist() for _, te in folds])
        for fold, (tr, te) in enumerate(folds):
            fold_seed = seed + 1000 + 10 * repeat + fold
            for name, X in arrays.items():
                pred, pick = _arm_outer(X, y, tr, te, n_inner, fold_seed)
                predictions[name][repeat, te] = pred
                chosen[name].append({"repeat": repeat, "fold": fold, **pick})

    arms = {}
    for name, P in predictions.items():
        per_repeat = [r2(y, p) for p in P]
        arms[name] = {"r2_mean": float(np.mean(per_repeat)),
                      "bootstrap_sd": bootstrap_sd(y, P, seed=seed),
                      "oof": P, "chosen": chosen[name]}
    return {
        "protocol": {"n": n, "n_repeats": n_repeats, "n_folds": n_folds,
                     "n_inner": n_inner, "seed": seed, "readout": "fixedblock2_flat",
                     "standardization": "StandardScaler fitted on each training split",
                     "heads": [name for name, _ in _menu(
                         max(1, n - math.ceil(n / n_folds) - math.ceil((n - math.ceil(n / n_folds)) / n_inner)),
                         max(a.shape[1] for a in arrays.values()))]},
        "arms": arms,
        "vs_raw": paired_delta(y, predictions["trained"], predictions["raw"], seed=seed),
        "vs_random": paired_delta(y, predictions["trained"], predictions["random"], seed=seed),
        "test_folds": test_folds,
    }
