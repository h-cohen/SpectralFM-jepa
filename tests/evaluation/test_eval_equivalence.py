"""Our copied nested CV must reproduce the parent's saved ref_feb25 numbers exactly."""
import json
import os

import numpy as np
import pytest

from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.nested import compare_ladders, compare_with_reference, ladder_for_set, run_nested

PARENT_OUT = "/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged/code/eval_outputs"
SETS = {
    "dataset0055": "label_probe_regression_ref_feb25/label_probe/dataset0055",
    "dataset0120": "label_probe_regression_ref_feb25/label_probe/dataset0120",
    "labeled_regression_all": "label_probe_regression_merged_ref_feb25/label_probe/labeled_regression_all",
}
TOL = 1e-10


def parent(rel):
    d = os.path.join(PARENT_OUT, rel)
    if not os.path.exists(os.path.join(d, "nested_results.json")):
        pytest.skip("parent outputs not available")
    return d


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SETS))
def test_nested_matches_parent(name):
    d = parent(SETS[name])
    bank, raw, y, _ = load_bank(os.path.join(d, "bank.npz"))
    ours, oof = run_nested(bank, raw, y, seed=42, n_jobs=8)
    check = compare_with_reference(ours, json.load(open(os.path.join(d, "nested_results.json"))))
    assert check["choice_mismatches"] == 0, check
    assert check["max_abs_diff"] <= TOL, check
    ref_oof = np.load(os.path.join(d, "nested_oof.npz"))
    for fam, P in oof.items():
        np.testing.assert_allclose(P, ref_oof[fam], rtol=0, atol=TOL, err_msg=fam)


@pytest.mark.slow
def test_ladder_matches_parent_on_merged_set():
    d = parent(SETS["labeled_regression_all"])
    ref = json.load(open(os.path.join(d, "nested_ladder.json")))
    bank, raw, y, _ = load_bank(os.path.join(d, "bank.npz"))
    ours = ladder_for_set(bank, raw, y, ref["block"], seed=42, n_jobs=8)
    check = compare_ladders(ours, ref)
    assert check["recipe_mismatches"] == 0 and check["max_abs_diff"] <= TOL, check
