import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from spectral_lejepa.evaluation import nested as nc
from spectral_lejepa.evaluation import ridge_heads as rh


def synthetic(n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n, d)) * rng.uniform(0.5, 5, d) + rng.normal(size=d) * 10).astype(np.float32)
    y = X[:, 0] * 0.4 + 0.1 * rng.normal(size=n)
    return X, y


def test_alphas_are_the_seven_decades():
    assert rh.ALPHAS == (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


@pytest.mark.parametrize("shape", [(40, 12), (30, 80)])     # n > p and n < p
def test_ridge_multi_alpha_matches_sklearn(shape):
    X, y = synthetic(*shape)
    P = rh.ridge_multi_alpha(X[:25], y[:25], X[25:], rh.ALPHAS)
    assert P.shape == (7, len(X) - 25)
    for a, p in zip(rh.ALPHAS, P):
        ref = Ridge(alpha=a).fit(X[:25].astype(np.float64), y[:25]).predict(X[25:].astype(np.float64))
        np.testing.assert_allclose(p, ref, rtol=1e-5, atol=1e-5)


def test_ridge_ens_is_mean_of_seven_ridges():
    X, y = synthetic()
    tr, te = slice(0, 45), slice(45, None)
    Ztr, Zte = rh.standardize(X[tr], X[te])
    each = [Ridge(alpha=a).fit(Ztr.astype(np.float64), y[tr]).predict(Zte.astype(np.float64)) for a in rh.ALPHAS]
    got = rh.head_predict("ridge_ens", X[tr], y[tr], X[te])
    np.testing.assert_allclose(got, np.mean(each, axis=0), rtol=1e-5, atol=1e-5)


def test_standardization_is_fit_on_train_only():
    X, y = synthetic()
    Ztr, Zte = rh.standardize(X[:40], X[40:])
    np.testing.assert_allclose(Ztr.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(Ztr.std(0), 1, atol=1e-5)
    assert np.abs(Zte.mean(0)).max() > 1e-3          # test rows do not get their own mean removed
    # changing the test rows cannot change the train transform nor the fit
    Xb = X.copy(); Xb[40:] += 100
    np.testing.assert_array_equal(rh.standardize(Xb[:40], Xb[40:])[0], Ztr)
    p1 = rh.head_predict("std_ols", X[:40], y[:40], X[40:])
    Xc = X.copy(); Xc[40:] = X[40:]
    np.testing.assert_allclose(p1, rh.head_predict("std_ols", Xc[:40], y[:40], Xc[40:]))


def test_std_ols_matches_sklearn_on_standardized():
    X, y = synthetic(n=50, d=8)
    Ztr, Zte = rh.standardize(X[:35], X[35:])
    ref = LinearRegression().fit(Ztr, y[:35]).predict(Zte)
    np.testing.assert_allclose(rh.head_predict("std_ols", X[:35], y[:35], X[35:]), ref, rtol=1e-5, atol=1e-5)


def test_snv_rows_have_zero_mean_unit_sd_and_ignore_scale_offset():
    X, _ = synthetic()
    S = rh.snv(X)
    np.testing.assert_allclose(S.mean(1), 0, atol=1e-5)
    np.testing.assert_allclose(S.std(1), 1, atol=1e-5)
    np.testing.assert_allclose(rh.snv(3 * X + 7), S, atol=1e-4)


def test_first_derivative_is_difference_along_wavelength():
    X = np.array([[1., 4., 9., 16.], [0., 0., 1., 3.]], np.float32)
    np.testing.assert_array_equal(rh.first_derivative(X), [[3, 5, 7], [0, 1, 2]])


def test_client_raw_averages_28_models():
    X, y = synthetic(n=50, d=14)
    tr, te = slice(0, 35), slice(35, None)
    pieces = []
    for name in rh.CLIENT_PREPROCESSINGS:
        A, B = rh.CLIENT_PREPROCESSINGS[name](X[tr], X[te])
        pieces.append(rh.ridge_multi_alpha(A, y[tr], B, rh.ALPHAS))
    assert len(rh.CLIENT_PREPROCESSINGS) == 4 and np.concatenate(pieces).shape == (28, 15)
    got = rh.head_predict("client_raw", X[tr], y[tr], X[te])
    assert got.shape == (15,)
    np.testing.assert_allclose(got, np.concatenate(pieces).mean(0), rtol=1e-6, atol=1e-6)


def test_unknown_head_refused():
    X, y = synthetic()
    with pytest.raises(ValueError, match="unknown head"):
        rh.head_predict("nope", X[:30], y[:30], X[30:])


def test_fixed_heads_use_the_nested_outer_test_folds():
    X, y = synthetic(n=60)
    res = nc.nested_cv({"a": X}, {"a": ["a"]}, y, seed=42)
    out = rh.run_fixed_heads({"a": X}, y, ("ridge_ens", "std_ols"), seed=42)
    assert out["test_folds"] == res["test_folds"]
    oof = out["oof"]["ridge_ens"]["a"]
    assert oof.shape == (2, 60)
    # every row is predicted exactly once per repeat, by a model that did not see it
    r, f = 0, 0
    te = np.array(res["test_folds"][r][f]); tr = np.setdiff1d(np.arange(60), te)
    np.testing.assert_allclose(oof[r, te], rh.head_predict("ridge_ens", X[tr], y[tr], X[te]), rtol=1e-6, atol=1e-6)


def test_run_fixed_heads_schema_and_client_raw_only_on_raw():
    X, y = synthetic(n=60)
    out = rh.run_fixed_heads({"raw": X, "layer2/flat": X[:, :6]}, y, ("ridge_ens", "client_raw"), seed=42)
    assert set(out["oof"]["ridge_ens"]) == {"raw", "layer2/flat"}
    assert set(out["oof"]["client_raw"]) == {"raw"}
    s = out["scores"]["ridge_ens"]["raw"]
    assert set(s) >= {"r2_mean", "r2_per_repeat", "rmse_mean", "rmse_per_repeat"}
    assert s["rmse_mean"] == pytest.approx(np.mean([np.sqrt(np.mean((y - p) ** 2)) for p in out["oof"]["ridge_ens"]["raw"]]))


def test_default_nested_unchanged_and_alpha_logged_only_when_enabled():
    X, y = synthetic(n=60)
    kw = dict(seed=42, recipes=(("standardize", "ridgecv"), ("none", "ols")))
    base = nc.nested_cv({"a": X}, {"a": ["a"]}, y, **kw)
    assert all("alpha" not in c for c in base["a"]["chosen"])
    logged = nc.nested_cv({"a": X}, {"a": ["a"]}, y, log_alpha=True, **kw)
    np.testing.assert_array_equal(base["a"]["oof"], logged["a"]["oof"])
    for c in logged["a"]["chosen"]:
        if c["probe"] == "ridgecv":
            assert c["alpha"] in set(np.logspace(-3, 3, 20))
        else:
            assert c["alpha"] is None


def test_chosen_alpha_matches_ridgecv_on_the_training_fold():
    from sklearn.linear_model import RidgeCV
    X, y = synthetic(n=60)
    tr = np.arange(0, 48)
    Ztr, _ = rh.standardize(X[tr], X[:2])
    ref = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(Ztr, y[tr]).alpha_
    assert nc.ridge_alpha(X[tr], y[tr], "standardize", seed=0) == pytest.approx(ref)
    assert nc.ridge_alpha(X[tr], y[tr], "standardize", seed=0, backend="torch", device="cpu") == pytest.approx(ref)


def test_write_fixed_heads_files(tmp_path):
    import json
    X, y = synthetic(n=60)
    rh.write_fixed_heads(tmp_path, {"raw": X, "layer2/flat": X[:, :6]}, {"layer2/flat": X[:, 6:]}, y,
                         ("ridge_ens", "client_raw"), seed=42)
    z = np.load(tmp_path / "fixed_heads_oof.npz")
    assert set(z.files) == {"y", "ridge_ens__raw", "ridge_ens__layer2_flat", "ridge_ens__random",
                            "client_raw__raw"}
    meta = json.loads((tmp_path / "fixed_heads.json").read_text())
    assert meta["scores"]["ridge_ens"]["random"]["r2_mean"] == pytest.approx(
        rh.run_fixed_heads({"x": X[:, 6:]}, y, ("ridge_ens",), seed=42)["scores"]["ridge_ens"]["x"]["r2_mean"])
    assert meta["test_folds"] and meta["heads"] == ["ridge_ens", "client_raw"]
