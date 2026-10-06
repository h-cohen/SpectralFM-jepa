"""Parity of the torch probe backend with the sklearn reference (probe.py), recipe by recipe."""
import numpy as np
import pytest
import torch

from spectral_lejepa.evaluation import probe, probe_torch
from spectral_lejepa.evaluation.probe import RECIPES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def problem(n, d, seed=0, dup=False, offset=0.0):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n + 20, d)) * np.logspace(0, -2, d) + offset).astype(np.float32)
    if dup:   # near-duplicate columns: a numerically rank-deficient design
        X[:, d // 2:d // 2 + 5] = X[:, :5] + 1e-6 * rng.normal(size=(n + 20, 5)).astype(np.float32)
    y = 3.0 * X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=n + 20) + 10.0
    return X[:n], y[:n], X[n:]


def sklearn_alpha(X_tr, y_tr):
    m = probe.make_regressor("ridgecv").fit(X_tr, y_tr)
    return float(m.alpha_)


CASES = {"n>p": (80, 30, False), "n<p": (40, 120, False), "n>p dup": (90, 40, True),
         "n<p dup": (40, 100, True), "n=p": (50, 50, False)}


@pytest.mark.parametrize("case", list(CASES))
def test_recipe_predictions_match_sklearn(case):
    n, d, dup = CASES[case]
    X_tr, y_tr, X_te = problem(n, d, seed=n + d, dup=dup)
    T = probe.fold_transforms(X_tr, X_te, seed=7)
    got = probe_torch.recipe_predictions(X_tr, y_tr, X_te, RECIPES, seed=7, device=DEVICE)
    assert list(got) == list(RECIPES)
    scale = np.std(y_tr)
    for norm, pr in RECIPES:
        a, b = T[norm]
        ref = probe.fit_predict(a, y_tr, b, pr, 7)
        diff = np.max(np.abs(got[(norm, pr)] - ref)) / scale
        # ridgecv runs in float64 in sklearn too; LinearRegression keeps float32 inputs in
        # float32 (sgelsd), so its reference carries float32 rounding of its own
        tol = 1e-6 if pr == "ridgecv" else 1e-4
        assert diff < tol, (case, norm, pr, diff)


@pytest.mark.parametrize("case", list(CASES))
def test_ridge_alpha_matches_sklearn(case):
    n, d, dup = CASES[case]
    X_tr, y_tr, X_te = problem(n, d, seed=n + d, dup=dup)
    T = probe.fold_transforms(X_tr, X_te, seed=7)
    for norm in probe.NORMALIZERS:
        a, b = T[norm]
        fit = probe_torch.ridge_gcv(torch.as_tensor(a, device=DEVICE), torch.as_tensor(y_tr, device=DEVICE))
        assert fit["alpha"] == sklearn_alpha(a, y_tr), (case, norm)


def test_fold_transforms_match_sklearn():
    X_tr, _, X_te = problem(60, 200, seed=3, offset=5.0)
    ref = probe.fold_transforms(X_tr, X_te, seed=7)
    got = probe_torch.fold_transforms(X_tr, X_te, seed=7, device=DEVICE)
    assert list(got) == list(ref)
    for norm in ref:
        for g, r in zip(got[norm], ref[norm]):
            g = g.cpu().numpy()
            assert g.dtype == np.float32 and g.shape == r.shape
            if norm in ("none", "standardize"):
                assert np.array_equal(g, r), norm
            else:   # PCA sign is free; whitened columns agree up to sign
                sign = np.sign(np.sum(g * r, axis=0))
                np.testing.assert_allclose(g * sign, r, rtol=0, atol=1e-5)


def test_fit_predict_contract():
    X_tr, y_tr, X_te = problem(50, 20, seed=1)
    for pr in probe.PROBES:
        got = probe_torch.fit_predict(X_tr, y_tr, X_te, pr, seed=7, device=DEVICE)
        ref = probe.fit_predict(X_tr, y_tr, X_te, pr, 7)
        assert isinstance(got, np.ndarray) and got.shape == ref.shape
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)
    with pytest.raises(ValueError):
        probe_torch.fit_predict(X_tr, y_tr, X_te, "lasso", seed=7, device=DEVICE)


@pytest.mark.parametrize("n,d", [(60, 20), (30, 80)])
def test_ols_own_decomposition_matches_sklearn(n, d):
    """ols() without the ridge decomposition (its fallback when the intercept component mixes)."""
    X_tr, y_tr, X_te = problem(n, d, seed=5)
    m = probe_torch.ols(torch.as_tensor(X_tr, device=DEVICE), y_tr)
    got = (torch.as_tensor(X_te, device=DEVICE).double() @ m["coef"] + m["intercept"]).cpu().numpy()
    ref = probe.fit_predict(X_tr, y_tr, X_te, "ols", 7)
    assert np.max(np.abs(got - ref)) / np.std(y_tr) < 1e-4
