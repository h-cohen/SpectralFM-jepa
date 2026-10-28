import json
from spectral_lejepa.evaluation.adapt_select import composite, dev_score, select_steps


def _res(path, scores):
    path.mkdir(parents=True)
    (path / "nested_results.json").write_text(json.dumps(
        {"families": {"layer2/flat": {"chosen": [{"inner_r2": s} for s in scores]}}}))
    (path / "bank.npz").write_bytes(b"x")


def test_dev_score_mean(tmp_path):
    _res(tmp_path / "s", [0.1, 0.3])
    assert abs(dev_score(tmp_path / "s" / "nested_results.json") - 0.2) < 1e-12


def test_select_by_member_mean_ties_to_smaller_step(tmp_path):
    runs = {}
    for m, by_step in {"m1": {250: .5, 500: .6}, "m2": {250: .7, 500: .6}}.items():
        runs[m] = {}
        for st, s in by_step.items():
            _res(tmp_path / m / str(st) / "dataset0112", [s]); runs[m][st] = tmp_path / m / str(st)
    assert select_steps(runs, ["dataset0112"]) == {"dataset0112": 250}


def test_composite_links_chosen_step(tmp_path):
    _res(tmp_path / "m1" / "500" / "dataset0112", [.5])
    composite({"m1": {500: tmp_path / "m1" / "500"}}, {"dataset0112": 500}, tmp_path / "out")
    assert (tmp_path / "out" / "m1" / "dataset0112" / "bank.npz").exists()
