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


def test_nnls_weights_convex_and_picks_signal():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    P = np.stack([y + 0.1 * rng.normal(size=200), rng.normal(size=200)], 1)
    w = nc.nnls_weights(P, y)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    assert w[0] > 0.8


def test_nnls_weights_zero_solution_falls_back_to_equal():
    y = np.arange(10.)
    P = np.stack([-y, -2 * y], 1)          # only negative weights would help
    assert np.allclose(nc.nnls_weights(P, y), [0.5, 0.5])


def test_nnls_weights_collinear_heads_finite():
    y = np.random.default_rng(1).normal(size=50)
    P = np.stack([y, y, y], 1)
    w = nc.nnls_weights(P, y)
    assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)


def test_stack_of_one_head_equals_that_head():
    X, y = synthetic()
    res = nc.nested_cv({"a": X}, {"a": ["a"]}, y, stacks={"s": ["a"]})
    np.testing.assert_allclose(res["s"]["oof"], res["a"]["oof"])
    assert res["s"]["chosen"][0]["members"][0]["weight"] == pytest.approx(1.0)


def test_stack_weights_favour_informative_head():
    X, y = synthetic()
    noise = np.random.default_rng(3).normal(size=X.shape).astype(np.float32)
    res = nc.nested_cv({"good": X, "bad": noise}, {}, y, stacks={"s": ["good", "bad"]})
    w = {m["arm"]: m["weight"] for m in res["s"]["chosen"][0]["members"]}
    assert w["good"] > w["bad"]
    assert res["s"]["r2_mean"] > 0.9


def test_nested_cv_without_stacks_unchanged():
    X, y = synthetic()
    a = nc.nested_cv({"a": X}, {"a": ["a"]}, y)
    b = nc.nested_cv({"a": X}, {"a": ["a"]}, y, stacks={})
    np.testing.assert_array_equal(a["a"]["oof"], b["a"]["oof"])
    assert set(a) == set(b)


def test_mix_prefers_informative_family():
    X, y = synthetic()
    rng = np.random.default_rng(3)
    arms = {"a0": X, "a1": X + 0.01 * rng.normal(size=X.shape).astype(np.float32),
            "b0": rng.normal(size=X.shape).astype(np.float32), "b1": rng.normal(size=X.shape).astype(np.float32)}
    res = nc.nested_cv(arms, {}, y, mixes={"m": (["a0", "a1"], ["b0", "b1"])})
    ws = [c["weight_a"] for c in res["m"]["chosen"]]
    assert len(ws) == 10 and set(ws) <= set(nc.MIX_GRID)
    assert sum(w == 1 for w in ws) >= 6
    assert res["m"]["r2_mean"] > 0.9
    c = res["m"]["chosen"][0]
    assert c["members_a"] and c["members_b"] and "inner_r2" in c


def test_mix_of_identical_families_equals_their_average():
    X, y = synthetic()
    res = nc.nested_cv({"a": X, "b": X}, {}, y, ensembles={"avg": (["a", "b"], 2)},
                       mixes={"m": (["a"], ["b"])})
    np.testing.assert_allclose(res["m"]["oof"], res["avg"]["oof"], atol=1e-8)


def test_nested_cv_without_mixes_unchanged():
    X, y = synthetic()
    a = nc.nested_cv({"a": X}, {"a": ["a"]}, y)
    b = nc.nested_cv({"a": X}, {"a": ["a"]}, y, mixes={})
    np.testing.assert_array_equal(a["a"]["oof"], b["a"]["oof"])
    assert set(a) == set(b)


def test_mix_grid_sizes():
    assert len(nc.mix_grid(2)) == 5 and len(nc.mix_grid(3)) == 15
    assert all(abs(sum(w) - 1) < 1e-12 for w in nc.mix_grid(3))
    assert nc.mix_grid(2) == [(w, 1 - w) for w in nc.MIX_GRID]


def test_mix_legacy_two_tuple_matches_stored_expectation():
    # values from the pre-generalisation code on this fixed case
    X, y = synthetic()
    rng = np.random.default_rng(3)
    arms = {"a0": X, "a1": X + 0.01 * rng.normal(size=X.shape).astype(np.float32),
            "b0": rng.normal(size=X.shape).astype(np.float32), "b1": rng.normal(size=X.shape).astype(np.float32)}
    r = nc.nested_cv(arms, {}, y, mixes={"m": (["a0", "a1"], ["b0", "b1"])})["m"]
    assert r["r2_mean"] == pytest.approx(0.9955940517513133, abs=1e-12)
    assert float(r["oof"].sum()) == pytest.approx(40.71476669805168, abs=1e-9)
    np.testing.assert_allclose(r["oof"][0, :3], [0.29651474952697754, -4.682342913720108, 1.7927823121605546], atol=1e-10)

    n, d = X.shape
    rng = np.random.default_rng(5)
    A = np.stack([y + rng.normal(size=n) * 2] + [rng.normal(size=n) for _ in range(d - 1)], 1).astype(np.float32)
    B = np.stack([y + rng.normal(size=n) * 2] + [rng.normal(size=n) for _ in range(d - 1)], 1).astype(np.float32)
    r = nc.nested_cv({"a": A, "b": B}, {}, y, mixes={"m": (["a"], ["b"])})["m"]
    assert [c["weight_a"] for c in r["chosen"]] == [.5, .5, .5, .5, .5, .5, .5, .75, .5, .25]
    assert r["r2_mean"] == pytest.approx(0.5058091638444773, abs=1e-12)
    assert float(r["oof"].sum()) == pytest.approx(45.00528952709628, abs=1e-9)
    assert [c["weights"] for c in r["chosen"]][7] == [.75, .25]
    # list form and dict form give the same result as the tuple
    r2_ = nc.nested_cv({"a": A, "b": B}, {}, y, mixes={"m": {"families": [["a"], ["b"]], "shrink": False}})["m"]
    np.testing.assert_array_equal(r["oof"], r2_["oof"])


def _three_family(n=200, informative=0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    y = X[:, 0] * 2 + 0.1 * rng.normal(size=n)
    arms = {f"f{k}": (X if k == informative else rng.normal(size=X.shape).astype(np.float32)) for k in range(3)}
    return arms, y


def test_mix_k3_picks_informative_family():
    arms, y = _three_family(n=100, informative=1)
    res = nc.nested_cv(arms, {}, y, mixes={"m": [["f0"], ["f1"], ["f2"]]})["m"]
    assert len(res["chosen"]) == 10 and all(len(c["weights"]) == 3 for c in res["chosen"])
    assert sum(c["weights"][1] == 1 for c in res["chosen"]) >= 8
    assert res["r2_mean"] > 0.9 and "weight_a" not in res["chosen"][0]


def test_mix_shrink_fires_on_noise_and_not_on_signal():
    rng = np.random.default_rng(2)
    n = 80
    y = rng.normal(size=n)
    arms = {f"f{k}": rng.normal(size=(n, 12)).astype(np.float32) for k in range(3)}
    res = nc.nested_cv(arms, {}, y, mixes={"s": {"families": [["f0"], ["f1"], ["f2"]], "shrink": True},
                                            "u": [["f0"], ["f1"], ["f2"]]})
    ch = res["s"]["chosen"]
    assert sum(c["shrunk"] for c in ch) >= 7
    assert all(np.allclose(c["weights"], 1 / 3) for c in ch if c["shrunk"])
    assert not any(c["shrunk"] for c in res["u"]["chosen"])
    arms, y = _three_family(n=200, informative=0)
    ch = nc.nested_cv(arms, {}, y, mixes={"s": {"families": [["f0"], ["f1"], ["f2"]], "shrink": True}})["s"]["chosen"]
    assert sum(c["shrunk"] for c in ch) == 0 and all(c["weights"][0] == 1 for c in ch)
