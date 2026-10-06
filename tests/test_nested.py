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


def torch_device():
    torch = pytest.importorskip("torch")
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_run_nested_torch_backend_matches_sklearn():
    X, y = synthetic()
    bank = {"layer0": X, "layer1": np.random.default_rng(1).normal(size=X.shape).astype(np.float32)}
    raw = np.random.default_rng(2).normal(size=(60, 245)).astype(np.float32)
    ref, ref_oof = nc.run_nested(bank, raw, y, seed=42)
    got, got_oof = nc.run_nested(bank, raw, y, seed=42, backend="torch", device=torch_device())
    # max_abs_diff covers every chosen recipe's inner R² too, so a flipped choice must be a near-tie
    # (OLS is invariant to the normalizer when n > p; sklearn's float32 lstsq rounding breaks that tie)
    cmp = nc.compare_with_reference(got, ref)
    assert cmp["max_abs_diff"] < 1e-5, cmp
    assert got["test_folds"] == ref["test_folds"] and set(got_oof) == set(ref_oof)


def test_ladder_and_canary_torch_backend_match_sklearn():
    X, y = synthetic()
    arms = {"raw": X[:, 1:], "block": X}
    ref = nc.nested_ladder(arms, y, max_draws=2)
    got = nc.nested_ladder(arms, y, max_draws=2, backend="torch", device=torch_device())
    assert ref["rungs"] == got["rungs"]
    for arm in arms:
        for n, o in got["arms"][arm].items():
            # recipe counts may differ by OLS ties across normalizers (n > p), never the scores
            assert sum(o["recipes"].values()) == sum(ref["arms"][arm][n]["recipes"].values())
            for k in ("median", "p25", "p75"):
                assert abs(o[k] - ref["arms"][arm][n][k]) < 1e-5
    c_ref = nc.run_canary(X, X[:, ::-1].copy(), y)
    c_got = nc.run_canary(X, X[:, ::-1].copy(), y, backend="torch", device=torch_device())
    for k in ("block", "raw"):
        assert abs(c_ref[k]["real_r2"] - c_got[k]["real_r2"]) < 1e-9
        assert abs(c_ref[k]["shuffled_r2"] - c_got[k]["shuffled_r2"]) < 1e-9


def test_torch_backend_threads_do_not_change_results():
    X, y = synthetic()
    bank = {"layer0": X, "layer1": X[:, ::-1] * 2}
    one, oof1 = nc.run_nested(bank, X[:, :5], y, seed=3, backend="torch", device=torch_device())
    many, oof3 = nc.run_nested(bank, X[:, :5], y, seed=3, backend="torch", device=torch_device(), n_jobs=3)
    assert one == many and all(np.array_equal(oof1[k], oof3[k]) for k in oof1)


def test_unknown_backend_rejected():
    X, y = synthetic()
    with pytest.raises(ValueError, match="backend"):
        nc.run_nested({"layer0": X}, X, y, backend="jax")


def test_map_on_streams_matches_sequential_on_every_device_layout():
    torch = pytest.importorskip("torch")
    from spectral_lejepa.evaluation.probe_torch import map_on_streams
    X, y = synthetic(n=80, d=30)
    folds = list(nc.KFold(5, shuffle=True, random_state=0).split(X))
    args = [(X[tr], y[tr], X[te], RECIPES, 7, "torch") for tr, te in folds]
    dev = torch_device()
    seq = [nc._recipe_predictions(*a, device=dev) for a in args]
    layouts = [(dev, 2)] + ([(["cuda:0", "cuda:1"], 2)] if torch.cuda.device_count() > 1 else [])
    for devices, workers in layouts:
        got = map_on_streams(lambda *a, device: nc._recipe_predictions(*a, device=device), args, workers, devices)
        for g, s in zip(got, seq):
            assert all(np.array_equal(g[rc], s[rc]) for rc in RECIPES), devices
