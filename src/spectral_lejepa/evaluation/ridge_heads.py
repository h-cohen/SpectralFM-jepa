"""Fixed linear heads -- no recipe selection at all -- scored on the outer folds of the nested CV.

  ridge_ens   standardize (fit on the training fold), Ridge at alpha 1e-3 ... 1e3 (one per decade),
              the 7 predictions averaged.
  std_ols     standardize + ordinary least squares.
  client_raw  raw spectra only: ridge at the same 7 alphas on each of 4 preprocessings (none,
              standardize, SNV per spectrum, first derivative along wavelength then standardize),
              the 4 x 7 = 28 predictions averaged. Approximates the client's ensemble of ridges.

The outer folds are nested.group_kfold(groups, n_folds, seed + r), the very folds nested_cv draws, so
every head is paired with the selected-recipe results and with raw. Everything is CPU float64.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from .nested import group_kfold
from .probe import _standardize, r2

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
HEADS = ("ridge_ens", "std_ols", "client_raw")
RAW_ONLY_HEADS = ("client_raw",)


def standardize(X_tr, X_te):
    """(Z_tr, Z_te) float32: mean and SD from X_tr only (SD < 1e-8 counts as 1), as probe.fold_transforms."""
    return tuple(_standardize(X_tr, X_te))


def snv(X):
    """Standard normal variate: every spectrum (row) centred and scaled by its own mean and SD."""
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(axis=1, keepdims=True)
    return ((X - X.mean(axis=1, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)).astype(np.float32)


def first_derivative(X):
    """Finite difference along wavelength: [n, d] -> [n, d - 1]."""
    return np.diff(np.asarray(X, dtype=np.float32), axis=1)


def ridge_multi_alpha(X_tr, y_tr, X_te, alphas=ALPHAS):
    """[len(alphas), n_te] Ridge predictions (fit_intercept, features centred on X_tr), one
    eigendecomposition for all alphas: the Gram matrix when n <= p, else the covariance. float64."""
    X = np.asarray(X_tr, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    y = np.asarray(y_tr, dtype=np.float64)
    yc = y - y.mean()
    Tc = np.asarray(X_te, dtype=np.float64) - mu
    out = []
    if X.shape[0] <= X.shape[1]:
        s, U = np.linalg.eigh(Xc @ Xc.T)                  # [n], [n, n]
        s = np.clip(s, 0, None)
        K_te = Tc @ Xc.T @ U                              # [n_te, n]
        b = U.T @ yc
        for a in alphas:
            out.append(K_te @ (b / (s + a)) + y.mean())
    else:
        s, V = np.linalg.eigh(Xc.T @ Xc)                  # [p], [p, p]
        s = np.clip(s, 0, None)
        b = V.T @ (Xc.T @ yc)
        for a in alphas:
            out.append(Tc @ (V @ (b / (s + a))) + y.mean())
    return np.stack(out)


CLIENT_PREPROCESSINGS = {
    "none": lambda A, B: (np.asarray(A, np.float64), np.asarray(B, np.float64)),
    "standardize": standardize,
    "snv": lambda A, B: (snv(A), snv(B)),
    "derivative": lambda A, B: standardize(first_derivative(A), first_derivative(B)),
}


def head_predict(head, X_tr, y_tr, X_te):
    """[n_te] prediction of a fixed head, every statistic fit on (X_tr, y_tr) only."""
    y_tr = np.asarray(y_tr, dtype=np.float64)
    if head == "ridge_ens":
        Z_tr, Z_te = standardize(X_tr, X_te)
        return ridge_multi_alpha(Z_tr, y_tr, Z_te).mean(axis=0)      # [7, n_te] -> [n_te]
    if head == "std_ols":
        Z_tr, Z_te = standardize(X_tr, X_te)
        return np.asarray(LinearRegression().fit(Z_tr, y_tr).predict(Z_te)).ravel()
    if head == "client_raw":
        P = [ridge_multi_alpha(A, y_tr, B) for A, B in (f(X_tr, X_te) for f in CLIENT_PREPROCESSINGS.values())]
        return np.concatenate(P).mean(axis=0)               # [4 * 7, n_te] -> [n_te]
    raise ValueError(f"unknown head {head!r}; choose from {HEADS}")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def run_fixed_heads(arms, y, heads=HEADS, n_repeats=2, n_folds=5, seed=42, groups=None):
    """Every head on every arm, on nested_cv's outer folds. arms {name: X [n, d]}; the raw-only heads
    (client_raw) run on the arm named 'raw' alone. Returns {'oof': {head: {arm: [R, n]}},
    'scores': {head: {arm: r2/rmse mean and per repeat}}, 'test_folds': [[fold indices] per repeat]}."""
    y = np.asarray(y, dtype=np.float64)
    if bad := [h for h in heads if h not in HEADS]:
        raise ValueError(f"unknown head {bad[0]!r}; choose from {HEADS}")
    folds = [list(group_kfold(groups, n_folds, seed + r, n=len(y))) for r in range(n_repeats)]
    oof, scores = {}, {}
    for head in heads:
        oof[head], scores[head] = {}, {}
        for arm, X in arms.items():
            if head in RAW_ONLY_HEADS and arm != "raw":
                continue
            P = np.zeros((n_repeats, len(y)))
            for r, rf in enumerate(folds):
                for tr, te in rf:
                    P[r, te] = head_predict(head, X[tr], y[tr], X[te])
            oof[head][arm] = P
            scores[head][arm] = {"r2_per_repeat": [r2(y, p) for p in P], "r2_mean": float(np.mean([r2(y, p) for p in P])),
                                 "rmse_per_repeat": [rmse(y, p) for p in P],
                                 "rmse_mean": float(np.mean([rmse(y, p) for p in P]))}
    return {"oof": oof, "scores": scores,
            "test_folds": [[te.tolist() for _, te in rf] for rf in folds]}
