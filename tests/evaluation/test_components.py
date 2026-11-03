import numpy as np

from spectral_lejepa.evaluation import components as cp


def toy(n=60, d=10, noise=1.0, seed=0):
    """comp1 is a noisy copy of comp0 (independent noise per component, same sample label)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = X[:, 0] * 2 + 0.05 * rng.normal(size=n)
    comps = {c: (X + noise * rng.normal(size=X.shape)).astype(np.float32) for c in (0, 1)}
    return comps, y


def test_arms_share_folds_and_average_beats_each_component():
    comps, y = toy()
    arms = {"raw": comps}
    res = cp.run_components(arms, y, n_repeats=2, n_folds=5, n_inner=3, seed=1, pooled=True)
    r2s = {k: v["r2_mean"] for k, v in res["scores"]["raw"].items()}
    assert set(r2s) == {"comp0", "comp1", "pooled_comp0", "pooled_comp1", "pooled_avg"}
    assert r2s["pooled_avg"] > r2s["pooled_comp0"] and r2s["pooled_avg"] > r2s["pooled_comp1"]
    oof = res["oof"]["raw"]
    assert oof["comp0"].shape == (2, 60) and oof["pooled_avg"].shape == (2, 60)
    assert res["deltas_vs_comp0"]["raw"]["pooled_avg"]["r2"]["delta"] > 0
    assert "rmse_mean" in res["scores"]["raw"]["comp0"]


def test_no_sample_in_both_sides_of_any_fold():
    comps, y = toy(n=40)
    res = cp.run_components({"raw": comps}, y, n_repeats=1, n_folds=4, n_inner=3, seed=2, pooled=True)
    n = 40
    folds = res["test_folds"][0]
    assert sorted(i for f in folds for i in f) == list(range(n))     # each sample tested exactly once per repeat
    assert all(set(a).isdisjoint(b) for i, a in enumerate(folds) for b in folds[i + 1:])


def test_run_study_writes_json_and_npz(tmp_path):
    import json

    import soundfile as sf
    import torch
    from spectral_lejepa.evaluation.data import normalize_like_fairseq
    from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
    from tests.models.test_model import SMALL
    rng = np.random.default_rng(0)
    d = tmp_path / "set"
    d.mkdir()
    lines = []
    for spec in range(30):
        for comp in (0, 1):
            name = f"dataset0007_comp{comp}_spec_{spec}.wav"
            sf.write(d / name, rng.normal(size=245).astype(np.float32), 16000, subtype="FLOAT")
            lines.append(f"{name}\t{spec * 0.5}")
    (d / "labels.tsv").write_text("\n".join(lines) + "\n")
    torch.manual_seed(0)
    mk = lambda: EvalBackbone(build_model(SMALL, 245))
    vcfg = {"max_samples": 100, "device": "cpu", "batch_size": 8}
    res = cp.run_study(mk(), mk(), {}, "set", [str(d)], (0, 1), True, vcfg, 0, tmp_path / "out",
                       {"n_jobs": 1, "backend": "sklearn", "device": None}, lambda raw, cfg: normalize_like_fairseq(raw))
    out = json.loads((tmp_path / "out" / "results_components.json").read_text())
    assert set(out["scores"]) == {"raw", "jepa", "random"}
    assert out["scores"]["jepa"]["pooled_avg"]["rmse_mean"] >= 0
    assert "r2" in out["deltas_vs_comp0"]["jepa"]["comp1"]
    z = np.load(tmp_path / "out" / "components_oof.npz")
    assert z["jepa__pooled_avg"].shape == (2, 30) and z["y"].shape == (30,)
