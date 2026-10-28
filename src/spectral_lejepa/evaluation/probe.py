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
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor

# RidgeCV's small-alpha end is ill-conditioned by construction at high p/n; CV picks a
# well-conditioned alpha. Silence the warnings, not the numerics (as the parent does).
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

NORMALIZERS = ("none", "standardize", "whiten", "whiten8", "whiten32", "whiten128")
PROBES = ("ridgecv", "ols")
RECIPES = tuple((n, p) for n in NORMALIZERS for p in PROBES)
NONLINEAR_PROBES = ("pls", "krr", "mlp")
NONLINEAR_NORMALIZERS = ("standardize", "whiten128")
EXTENDED_RECIPES = RECIPES + tuple((n, p) for n in NONLINEAR_NORMALIZERS for p in NONLINEAR_PROBES)
PLS_COMPONENTS = (2, 4, 8, 16, 32)
KRR_PCA_DIM = 256
KRR_BANDWIDTH_SCALES = (0.5, 1.0, 2.0)
KRR_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
MLP_WEIGHT_DECAYS = (1e-3, 1e-2)
MLP_SEEDS = 3


def recipe_set(name):
    if name == "linear":
        return RECIPES
    if name == "extended":
        return EXTENDED_RECIPES
    raise ValueError(f"unknown recipe set {name!r}; choose 'linear' or 'extended'")


def _inner_cv(n, seed):
    return KFold(min(5, n), shuffle=True, random_state=seed)


def _fit_nonlinear(X_tr, y_tr, X_te, probe, seed):
    """Nonlinear probes; every hyperparameter chosen by internal CV on (X_tr, y_tr) only."""
    X_tr = np.asarray(X_tr, np.float64); X_te = np.asarray(X_te, np.float64); y_tr = np.asarray(y_tr, np.float64)
    n = len(y_tr)
    if np.ptp(y_tr) == 0 or n < 4:
        return np.full(len(X_te), y_tr.mean())
    cv = _inner_cv(n, seed)
    if probe == "pls":
        cap = min(X_tr.shape[1], n - n // cv.get_n_splits() - 1)
        grid = [k for k in PLS_COMPONENTS if k <= cap] or [1]
        m = GridSearchCV(PLSRegression(scale=False), {"n_components": grid}, cv=cv, scoring="r2")
        return np.asarray(m.fit(X_tr, y_tr).predict(X_te)).ravel()
    if probe == "krr":
        k = min(KRR_PCA_DIM, n - 1, X_tr.shape[1])
        pca = PCA(n_components=k, random_state=seed).fit(X_tr)
        Z_tr, Z_te = pca.transform(X_tr), pca.transform(X_te)
        d2 = euclidean_distances(Z_tr, squared=True)      # (n, n), not (n, n, k)
        med = np.median(d2[np.triu_indices(n, 1)]) or 1.0
        gammas = [1.0 / (2 * s * s * med) for s in KRR_BANDWIDTH_SCALES]
        m = GridSearchCV(KernelRidge(kernel="rbf"), {"gamma": gammas, "alpha": list(KRR_ALPHAS)}, cv=cv, scoring="r2")
        return np.asarray(m.fit(Z_tr, y_tr).predict(Z_te)).ravel()
    if probe == "mlp":
        mu, sd = y_tr.mean(), y_tr.std()
        early = n >= 20                       # MLPRegressor's 10% validation split needs >= 2 rows
        preds = []
        for s in range(MLP_SEEDS):
            base = MLPRegressor(hidden_layer_sizes=(256,), early_stopping=early, validation_fraction=0.1,
                                max_iter=500, random_state=seed + s)
            m = GridSearchCV(base, {"alpha": list(MLP_WEIGHT_DECAYS)}, cv=cv, scoring="r2")
            preds.append(m.fit(X_tr, (y_tr - mu) / sd).predict(X_te) * sd + mu)
        return np.mean(preds, axis=0).ravel()
    raise ValueError(f"unknown probe {probe!r}")
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
    if probe in NONLINEAR_PROBES:
        return _fit_nonlinear(X_tr, y_tr, X_te, probe, seed)
    m = make_regressor(probe)
    m.fit(X_tr, y_tr)
    return np.asarray(m.predict(X_te)).ravel()
