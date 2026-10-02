import numpy as np
import pytest

from spectral_lejepa.evaluation import nested as nc
from spectral_lejepa.evaluation.probe import RECIPES, fold_transforms, r2


def synthetic(n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = X[:, 0] * 2 + 0.1 * rng.normal(size=n)
    return X, y


def test_recipes_and_r2():
    assert len(RECIPES) == 12 and RECIPES[0] == ("none", "ridgecv")
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, y) == pytest.approx(1.0)


def test_fold_transforms_fit_on_train_only():
    X, _ = synthetic()
    T = fold_transforms(X[:40], X[40:])
    assert set(T) == {"none", "standardize", "whiten", "whiten8", "whiten32", "whiten128"}
    a, _ = T["standardize"]
    np.testing.assert_allclose(a.mean(0), 0, atol=1e-5)
    w, _ = T["whiten"]
    np.testing.assert_allclose(np.cov(w.T, bias=False), np.eye(w.shape[1]), atol=1e-3)
    assert np.array_equal(T["whiten8"][0], w[:, :8])


def test_run_nested_schema_and_signal():
    X, y = synthetic()
    bank = {"layer0": X, "layer1": np.random.default_rng(1).normal(size=X.shape).astype(np.float32)}
    raw = np.random.default_rng(2).normal(size=(60, 245)).astype(np.float32)
    res, oof = nc.run_nested(bank, raw, y, seed=42)
    assert list(res["families"]) == ["raw", "embedding", "layer0", "layer1", "embedding_top3"]
    assert len(res["raw_fixed_recipes"]) == 12
    assert res["families"]["layer0"]["r2_mean"] > 0.9
    assert res["families"]["embedding"]["r2_mean"] > 0.9
    assert oof["embedding"].shape == (2, 60)
    assert nc.best_block(res) == "layer0"
    again, _ = nc.run_nested(bank, raw, y, seed=42)
    assert again["families"]["embedding"]["r2_mean"] == res["families"]["embedding"]["r2_mean"]


def test_paired_delta_identical_is_zero():
    X, y = synthetic()
    P = np.stack([y + 0.1, y - 0.1])
    d = nc.paired_delta(y, P, P)
    assert d["delta"] == 0 and d["sd"] == 0


def test_canary_passes_on_real_signal_and_noise():
    X, y = synthetic(n=80)
    c = nc.run_canary(X, X, y)
    assert c["block"]["real_r2"] > 0.9 and c["block"]["passed"] and c["raw"]["passed"]


def test_ladder_schema():
    X, y = synthetic()
    out = nc.nested_ladder({"raw": X[:, 1:], "block": X}, y, max_draws=2)
    assert out["rungs"] == [10, 20, 48]
    assert set(out["arms"]) == {"raw", "block"} and set(out["gaps"]) == {"block"}
    assert out["arms"]["block"]["48"]["median"] > 0.9


def test_compare_with_reference_detects_differences():
    X, y = synthetic()
    bank = {"layer0": X}
    raw = X[:, ::-1].copy()
    res, _ = nc.run_nested(bank, raw, y)
    assert nc.compare_with_reference(res, res) == {"max_abs_diff": 0.0, "choice_mismatches": 0}
    other = {**res, "embedding_minus_raw": {**res["embedding_minus_raw"], "delta": res["embedding_minus_raw"]["delta"] + 1}}
    assert nc.compare_with_reference(res, other)["max_abs_diff"] == pytest.approx(1.0)
