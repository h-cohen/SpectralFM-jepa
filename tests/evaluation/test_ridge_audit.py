import json

import numpy as np
import pytest

from ops import ridge_audit as ra
from spectral_lejepa.evaluation import ridge_heads as rh
from spectral_lejepa.evaluation.nested import nested_cv, write_json


def chosen(*recipes, alpha=None):
    return [{"repeat": 0, "fold": i, "arm": "a", "norm": n, "probe": p, "inner_r2": .5,
             **({} if alpha is None else {"alpha": alpha[i]})} for i, (n, p) in enumerate(recipes)]


def test_chosen_frequencies_modal_and_stability():
    c = chosen(("none", "ols"), ("none", "ols"), ("whiten", "ols"), ("standardize", "ridgecv"))
    f = ra.chosen_frequencies(c)
    assert f["n_folds"] == 4 and f["counts"]["none+ols"] == 2
    assert f["modal"] == "none+ols" and f["stability"] == pytest.approx(0.5)
    assert f["n_distinct"] == 3


def make_run(root, n=60, seed=0):
    """A fake eval run dir with one set: nested_oof, nested_results (with alphas), random_control, heads."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 10)).astype(np.float32)
    y = raw[:, 0] + .3 * rng.normal(size=n)
    flat = rng.normal(size=(n, 8)).astype(np.float32); flat[:, 0] = y + .1 * rng.normal(size=n)
    rnd = rng.normal(size=(n, 8)).astype(np.float32)
    rc = {"recipes": (("standardize", "ridgecv"),), "log_alpha": True, "seed": 42}
    res = nested_cv({"raw": raw, "layer2/flat": flat}, {"raw": ["raw"], "layer2/flat": ["layer2/flat"]}, y, **rc)
    rres = nested_cv({"layer2/flat": rnd}, {"layer2/flat": ["layer2/flat"]}, y, **rc)
    d = root / "set1"; (d / "random_control").mkdir(parents=True)
    for dd, r in ((d, res), (d / "random_control", rres)):
        write_json(dd / "nested_results.json", {"families": {k: {"chosen": v["chosen"]} for k, v in r.items()
                                                              if k in ("raw", "layer2/flat")}})
        np.savez(dd / "nested_oof.npz", y=y, **{k: r[k]["oof"] for k in ("raw", "layer2/flat") if k in r})
    rh.write_fixed_heads(d, {"raw": raw, "layer2/flat": flat}, {"layer2/flat": rnd}, y, rh.HEADS, seed=42)
    return root


def test_recipe_audit_reads_every_arm(tmp_path):
    make_run(tmp_path)
    out = ra.recipe_audit(tmp_path)
    assert set(out["set1"]) == {"raw", "layer2/flat", "random"}
    assert out["set1"]["raw"]["n_folds"] == 10
    assert out["set1"]["raw"]["modal"] == "standardize+ridgecv"


def test_heads_audit_columns_and_pairings(tmp_path):
    make_run(tmp_path)
    out = ra.heads_audit(tmp_path)
    s = out["per_set"]["set1"]
    assert set(s["ridge_ens"]) == {"raw", "layer2/flat", "random"} and set(s["client_raw"]) == {"raw"}
    row = s["ridge_ens"]["layer2/flat"]
    assert {"r2", "rmse", "delta_r2", "delta_r2_sd", "delta_rmse"} <= set(row)
    assert s["ridge_ens"]["raw"]["delta_r2"] == 0
    assert row["delta_r2"] > 0 and row["delta_rmse"] < 0
    assert out["summary"]["ridge_ens"]["layer2/flat"]["wins"] == 1
    assert "jepa_h1_vs_client_raw" in s and "selected_vs_h1" in s
    assert set(s["selected_vs_h1"]) == {"raw", "layer2/flat", "random"}
    assert out["alphas"]["layer2/flat"]            # chosen-alpha distribution
