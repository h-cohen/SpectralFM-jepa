"""Feature normalizers, linear probes and R² -- copied from clean-eval (6237feb) with behaviour
preserved: normalize.py (none/standardize/whiten), regressors.make_regressor (ridgecv/ols),
protocol.r2, nested.fold_transforms. Every normalizer is fit on training rows only.

Whitening is exact float64 PCA, keeping at most one direction per 2 training samples and
dropping only numerically dead directions: on this task the label signal lives in
low-variance directions, so "explains ~all the variance" is not a reason to drop one.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, RidgeCV

# RidgeCV's small-alpha end is ill-conditioned by construction at high p/n; CV picks a
# well-conditioned alpha. Silence the warnings, not the numerics (as the parent does).
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

NORMALIZERS = ("none", "standardize", "whiten", "whiten8", "whiten32", "whiten128")
PROBES = ("ridgecv", "ols")
RECIPES = tuple((n, p) for n in NORMALIZERS for p in PROBES)
WHITEN_SAMPLES_PER_DIRECTION = 2
EXACT_SVD_MAX_DIM = 6000


def r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _standardize(X_tr, X_te):
    X_tr32 = np.asarray(X_tr).astype(np.float32, copy=False)
    mu = X_tr32.mean(axis=0).astype(np.float32)
    sd = X_tr32.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return [((np.asarray(X, dtype=np.float32) - mu) / sd).astype(np.float32) for X in (X_tr, X_te)]


def _whiten(X_tr, X_te, seed):
    """Full-rank whitening fit on X_tr: returns (Z_tr, Z_te) with the k_keep alive directions."""
    X64 = np.asarray(X_tr, dtype=np.float64)
    n, d = X64.shape
    k = min(d, max(1, n // WHITEN_SAMPLES_PER_DIRECTION))
    solver = "full" if min(n, d) <= EXACT_SVD_MAX_DIM else "randomized"
    pca = PCA(n_components=k, whiten=True, svd_solver=solver, random_state=seed).fit(X64)
    ev = pca.explained_variance_
    tol = (np.finfo(np.float64).eps * max(n, d)) ** 2
    k_keep = max(1, int(np.sum(ev > tol * ev[0])) if ev.size else 0)
    return [pca.transform(np.asarray(X, dtype=np.float64))[:, :k_keep].astype(np.float32)
            for X in (X_tr, X_te)]


def fold_transforms(X_tr, X_te, seed=42) -> dict:
    """{normalizer: (X_tr', X_te')}, every normalizer fit on X_tr only. whitenK = the first K
    columns of one full-rank whitening (identical to fitting each separately with exact SVD)."""
    out = {"none": (np.asarray(X_tr, dtype=np.float32), np.asarray(X_te, dtype=np.float32)),
           "standardize": tuple(_standardize(X_tr, X_te))}
    Ztr, Zte = _whiten(X_tr, X_te, seed)
    for name in NORMALIZERS:
        if name.startswith("whiten"):
            k = Ztr.shape[1] if name == "whiten" else min(Ztr.shape[1], int(name[len("whiten"):]))
            out[name] = (Ztr[:, :k], Zte[:, :k])
    return out


def make_regressor(name):
    if name == "ridgecv":
        # cv=None: efficient leave-one-out over the alpha grid via one SVD
        return RidgeCV(alphas=np.logspace(-3, 3, 20))
    if name == "ols":
        return LinearRegression()
    raise ValueError(f"unknown probe {name!r}")


def fit_predict(X_tr, y_tr, X_te, probe, seed):
    m = make_regressor(probe)
    m.fit(X_tr, y_tr)
    return np.asarray(m.predict(X_te)).ravel()
