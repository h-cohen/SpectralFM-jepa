import json

import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from scripts.frozen_heads import compare_heads


def test_compare_heads_scores_nonlinear_signal_and_returns_paired_results():
    rng = np.random.default_rng(4)
    n = 40
    raw = rng.normal(size=(n, 245))
    trained = rng.normal(size=(n, 6))
    random = rng.normal(size=(n, 6))
    y = trained[:, 0] * trained[:, 1] + 0.03 * rng.normal(size=n)

    result = compare_heads(trained, random, raw, y, n_repeats=1, n_folds=2, n_inner=2, seed=7)

    assert set(result["arms"]) == {"raw", "trained", "random"}
    for arm in result["arms"].values():
        assert arm["oof"].shape == (1, n)
        assert np.isfinite(arm["oof"]).all()
        assert np.isfinite(arm["r2_mean"])
        assert np.isfinite(arm["bootstrap_sd"])
        assert len(arm["chosen"]) == 2
    assert result["arms"]["trained"]["r2_mean"] > -0.2
    assert set(result) >= {"vs_raw", "vs_random"}
    assert np.isfinite(result["vs_raw"]["delta"])
    assert np.isfinite(result["vs_random"]["delta"])
    json_ready = {**result, "arms": {
        name: {key: value for key, value in arm.items() if key != "oof"}
        for name, arm in result["arms"].items()
    }}
    json.dumps(json_ready)


def test_outer_test_labels_do_not_select_the_first_fold_recipe():
    rng = np.random.default_rng(5)
    n = 30
    trained = rng.normal(size=(n, 5))
    random = rng.normal(size=(n, 5))
    raw = rng.normal(size=(n, 245))
    y = trained[:, 0] ** 2 + 0.1 * rng.normal(size=n)
    test_idx = next(KFold(2, shuffle=True, random_state=13).split(y))[1]

    original = compare_heads(trained, random, raw, y, n_repeats=1, n_folds=2, n_inner=2, seed=13)
    changed_y = y.copy()
    changed_y[test_idx] += 1000 * rng.normal(size=len(test_idx))
    changed = compare_heads(trained, random, raw, changed_y, n_repeats=1, n_folds=2, n_inner=2, seed=13)

    for arm in ("raw", "trained", "random"):
        assert original["arms"][arm]["chosen"][0] == changed["arms"][arm]["chosen"][0]


def test_scaler_is_fit_without_outer_heldout_rows(monkeypatch):
    rng = np.random.default_rng(6)
    n = 24
    X = rng.normal(size=(n, 3))
    trained, random, raw = X, X + 0.01, rng.normal(size=(n, 245))
    trained[:, 0] = np.arange(n)
    random[:, 0] = np.arange(n) + 0.01
    raw[:, 0] = np.arange(n)
    y = X[:, 0] ** 2
    heldout = set(next(KFold(2, shuffle=True, random_state=19).split(y))[1].tolist())
    original_fit = StandardScaler.fit
    fit_rows = []

    def recording_fit(self, values, *args, **kwargs):
        fit_rows.append(np.asarray(values).copy())
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)
    compare_heads(trained, random, raw, y, n_repeats=1, n_folds=2, n_inner=2, seed=19)
    # First 2 inner fits plus outer refit for each arm belong to outer fold 0.
    for values in fit_rows[:9]:
        observed = set(np.rint(values[:, 0]).astype(int).tolist())
        assert observed.isdisjoint(heldout)


def test_mismatched_rows_raise_clear_value_error():
    with pytest.raises(ValueError, match="same number of rows"):
        compare_heads(np.zeros((10, 3)), np.zeros((9, 3)), np.zeros((10, 245)), np.zeros(10))
