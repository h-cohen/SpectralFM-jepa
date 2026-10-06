import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from scripts import evaluate
from spectral_lejepa.config import load_config
from spectral_lejepa.evaluation.data import load_labeled_data, normalize_like_fairseq
from spectral_lejepa.evaluation.nested import run_nested, write_json
from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.training.trainer import train
from tests.test_train_smoke import TINY_OVERRIDES, REPO, data_dir, tiny_cfg  # noqa: F401  (fixture reuse)

SET = "toyset"


def make_label_set(root):
    rng = np.random.default_rng(1)
    t = np.arange(245) / 245
    d = root / SET
    d.mkdir(parents=True)
    lines = []
    for i in range(30):
        f = rng.uniform(1, 5)
        x = np.sin(2 * np.pi * f * t) + 0.05 * rng.normal(size=245)
        name = f"dataset0001_comp0_spec_{i}.wav"
        sf.write(d / name, x.astype(np.float32), 16000, subtype="FLOAT")
        lines.append(f"{name}\t{f + 0.1 * rng.normal():.5f}\n")
    (d / "labels.tsv").write_text("".join(lines))
    return d


def test_evaluate_main_end_to_end(data_dir, tmp_path, monkeypatch):  # noqa: F811
    cfg = tiny_cfg(data_dir, tmp_path / "pre")
    cfg["training"]["max_steps"] = 4
    cfg["training"]["ckpt_every"] = 4
    cfg["training"]["val_every"] = 4
    cfg["training"]["diag_every"] = 4
    cfg["training"]["log_every"] = 2
    with pytest.warns(UserWarning, match="batch_size=8"):
        result = train(cfg)
    ckpt = Path(result["output_dir"]) / "checkpoint_last.pt"

    set_dir = make_label_set(tmp_path)
    raw, y = load_labeled_data([str(set_dir)])
    # fake parent baseline: any bank works, since only the raw arm and folds must match
    bank = {"layer0": np.random.default_rng(0).normal(size=(len(y), 8)).astype(np.float32)}
    nested, oof = run_nested(bank, raw, y, seed=42, n_jobs=1)
    parent = tmp_path / "parent"
    base_dir = parent / "base"
    base_dir.mkdir(parents=True)
    write_json(base_dir / "nested_results.json", nested)
    np.savez(base_dir / "nested_oof.npz", y=y, **oof)

    ecfg = {"experiment": {"name": "toy_eval", "seed": 42, "output_dir": str(tmp_path / "eval_out")},
            "evaluation": {"device": "cpu", "batch_size": 16, "max_samples": 5000, "n_jobs": 1, "min_n": 20,
                           "ladder_sets": [], "ladder_block_from": SET,
                           "readouts": ["mean", "seg4", "flat"], "flat_blocks": [0], "random_control": True},
            "label_sets": {SET: [str(set_dir)]},
            "baseline": {"tag": "fake", "checkpoint": "none", "parent_repo": str(parent),
                         "run_dirs": {SET: "base"}},
            "wandb": {"enabled": False, "entity": None, "project": "x", "group": None, "job_type": "eval",
                      "tags": []}}
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(ecfg))

    evaluate.main(["--checkpoint", str(ckpt), "--config", str(cfg_path)])

    out_root = Path(ecfg["experiment"]["output_dir"]) / f"{ckpt.parent.name}_step4"
    top = json.loads((out_root / "summary.json").read_text())
    lin = top["lineage"]
    assert len(lin["pretraining_checkpoint_sha256"]) == 64
    assert lin["pretraining_run_id"] is None
    for key in ("pretraining_git_commit", "evaluation_git_commit", "evaluation_git_dirty"):
        assert key in lin
    assert lin["evaluation_backend"] == "sklearn"
    s = json.loads((out_root / SET / "summary.json").read_text())
    assert top["sets"][SET]["n"] == 30
    for key in ("embedding_r2", "raw_r2", "canary", "vs_ref_mean_only", "vs_random_control"):
        assert key in s
    assert s["vs_ref_mean_only"]["raw"]["delta"] == 0.0
    assert "vs_baseline" not in s
    assert set(s["vs_random_control"]) >= {"delta", "sd", "p_a_better"}
    assert "pretraining_artifact" not in lin and "pretraining_artifact_digest" not in lin
    assert lin["pretraining_checkpoint"] == str(ckpt)
    assert [k for k in s["blocks"] if k.endswith("/flat")] == ["layer0/flat"] and any(k.endswith("/seg4") for k in s["blocks"])
    assert s["verdict"] in ("strong-win", "win", "no-win") and s["delta_vs_raw"] == s["embedding_r2"] - s["raw_r2"]
    rc = s["random_control"]
    assert set(rc) >= {"embedding_r2", "best_block", "delta_vs_raw", "verdict"}
    assert (out_root / SET / "random_control" / "nested_results.json").exists()
    card = top["scorecard"]
    for row in ("model", "random_control"):
        assert set(card[row]) == {"wins", "strong_wins", "sets", "mean_delta"}
        assert card[row]["sets"] == 1

    cfg_path.write_text(yaml.safe_dump(ecfg))
    with pytest.raises(ValueError, match="bogus"):
        evaluate.main(["--checkpoint", str(ckpt), "--config", str(cfg_path), "evaluation.readouts=[mean,bogus]"])
    with pytest.raises(ValueError, match="flat_blocks"):
        evaluate.main(["--checkpoint", str(ckpt), "--config", str(cfg_path), "evaluation.flat_blocks=[-1]"])
    ecfg["evaluation"]["backend"] = "jax"
    cfg_path.write_text(yaml.safe_dump(ecfg))
    with pytest.raises(ValueError, match="backend"):
        evaluate.main(["--checkpoint", str(ckpt), "--config", str(cfg_path)])


def test_verdict_and_scorecard():
    # user rule (2026-10-06): any gain over raw wins; +0.05 is the aspiration ("strong-win")
    assert evaluate.verdict(0.80, 0.86) == (pytest.approx(0.06), "strong-win")
    assert evaluate.verdict(0.80, 0.84)[1] == "win"
    assert evaluate.verdict(0.983, 0.985)[1] == "win"
    assert evaluate.verdict(0.80, 0.80)[1] == "no-win"
    sets = {"a": {"delta_vs_raw": 0.06, "verdict": "strong-win"}, "b": {"delta_vs_raw": -0.1, "verdict": "no-win"},
            "c": {"delta_vs_raw": 0.002, "verdict": "win"}}
    assert evaluate.scorecard(sets, None) == {"wins": 2, "strong_wins": 1, "sets": 3,
                                              "mean_delta": pytest.approx(-0.038 / 3)}


def test_model_input_methods():
    raw = np.random.default_rng(0).normal(5.0, 2.0, size=(4, 245)).astype(np.float32)
    z = normalize_like_fairseq(raw)
    np.testing.assert_array_equal(evaluate.model_input(raw, {"data": {"normalization": "sample_zscore"}}), z)
    np.testing.assert_array_equal(evaluate.model_input(raw, {"data": {}}), z)      # an old checkpoint
    g = {"data": {"normalization": "global"}, "derived": {"global_stats": {"mean": 5.0, "std": 2.0}}}
    out = evaluate.model_input(raw, g)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, (raw - 5.0) / 2.0, rtol=1e-6)


def test_evaluate_global_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    packed = tmp_path / "packed"
    packed.mkdir()
    np.save(packed / "spectra.npy", rng.normal(5.0, 2.0, size=(100, 245)).astype(np.float32))
    (packed / "stats.json").write_text(json.dumps({"mean": 5.0, "std": 2.0, "n_rows": 100, "n_dropped": 0}))
    np.save(packed / "drop_rows.npy", np.array([], dtype=np.int64))
    np.save(packed / "valid_rows.npy", np.arange(10, 30))
    cfg = load_config(REPO / "configs" / "pretrain.yaml", [
        "experiment.name=packed", f"experiment.output_dir={tmp_path / 'pre'}",
        "data.source=packed", f"data.packed_dir={packed}", "data.normalization=global",
        *TINY_OVERRIDES, "training.max_steps=2", "training.ckpt_every=2", "training.val_every=2",
        "training.diag_every=2", "training.log_every=1"])
    with pytest.warns(UserWarning, match="batch_size=8"):
        result = train(cfg)
    ckpt = Path(result["output_dir"]) / "checkpoint_last.pt"
    set_dir = make_label_set(tmp_path)
    ecfg = {"experiment": {"name": "toy_eval", "seed": 42, "output_dir": str(tmp_path / "eval_out")},
            "evaluation": {"device": "cpu", "batch_size": 16, "max_samples": 5000, "n_jobs": 1, "min_n": 20,
                           "ladder_sets": [], "ladder_block_from": SET,
                           "readouts": ["mean"], "flat_blocks": None, "random_control": True},
            "label_sets": {SET: [str(set_dir)]},
            "baseline": {"tag": "fake", "checkpoint": "none", "parent_repo": str(tmp_path), "run_dirs": {}},
            "wandb": {"enabled": False, "entity": None, "project": "x", "group": None, "job_type": "eval",
                      "tags": []}}
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(ecfg))
    evaluate.main(["--checkpoint", str(ckpt), "--config", str(cfg_path)])
    out_root = Path(ecfg["experiment"]["output_dir"]) / f"{ckpt.parent.name}_step2"
    assert load_bank(out_root / SET / "bank.npz")[3]["model_input"] == "global"
    assert json.loads((out_root / "summary.json").read_text())["lineage"]["model_input"] == "global"
