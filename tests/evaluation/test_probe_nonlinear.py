import numpy as np
import pytest
from spectral_lejepa.evaluation.probe import (EXTENDED_RECIPES, RECIPES, fit_predict, r2, recipe_set)


def _toy(n=200, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2
    return X[:150], y[:150], X[150:], y[150:]


@pytest.mark.parametrize("probe", ["krr", "mlp"])
def test_nonlinear_probe_beats_ridge_on_nonlinear_target(probe):
    Xtr, ytr, Xte, yte = _toy()
    lin = r2(yte, fit_predict(Xtr, ytr, Xte, "ridgecv", 0))
    nl = r2(yte, fit_predict(Xtr, ytr, Xte, probe, 0))
    assert nl > lin + 0.2


def test_pls_matches_linear_target():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 30)); y = X[:, :3].sum(1)
    assert r2(y[100:], fit_predict(X[:100], y[:100], X[100:], "pls", 0)) > 0.95


def test_pls_caps_components_on_tiny_training_set():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(16, 50)); y = rng.normal(size=16)
    p = fit_predict(X[:12], y[:12], X[12:], "pls", 0)
    assert p.shape == (4,) and np.isfinite(p).all()


@pytest.mark.parametrize("probe", ["pls", "krr", "mlp"])
def test_constant_target_gives_finite_predictions(probe):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 8))
    p = fit_predict(X[:15], np.ones(15), X[15:], probe, 0)
    assert np.isfinite(p).all()


def test_mlp_is_deterministic_for_a_seed():
    Xtr, ytr, Xte, _ = _toy(n=180)
    assert np.array_equal(fit_predict(Xtr, ytr, Xte, "mlp", 7), fit_predict(Xtr, ytr, Xte, "mlp", 7))


def test_recipe_sets():
    assert recipe_set("linear") == RECIPES and len(RECIPES) == 12
    assert len(EXTENDED_RECIPES) == 18 and EXTENDED_RECIPES[:12] == RECIPES
    with pytest.raises(ValueError):
        recipe_set("nope")
