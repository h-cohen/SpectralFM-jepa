import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import screen
from tests.test_evaluate_script import make_label_set
from tests.test_train_smoke import TINY_OVERRIDES, data_dir  # noqa: F401  (fixture reuse)

REPO = Path(__file__).resolve().parents[1]


def fake_row(tmp_path, arm, r2, oof, y, status="ok"):
    d = tmp_path / arm
    d.mkdir()
    np.savez(d / "nested_oof.npz", y=y, embedding=oof)
    return {"arm": arm, "status": status, "embedding_r2": r2, "oof": str(d / "nested_oof.npz")}


def test_decide_rule(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    noise = lambda s: np.stack([y + s * rng.normal(size=200) for _ in range(2)])
    control = noise(0.7)
    rows = [fake_row(tmp_path, "control_s0", 0.50, control, y),
            fake_row(tmp_path, "control_s1", 0.49, noise(0.7), y),
            fake_row(tmp_path, "good", 0.90, noise(0.3), y),
            fake_row(tmp_path, "bad", 0.10, noise(1.5), y),
            fake_row(tmp_path, "same", 0.50, control.copy(), y),   # identical predictions -> delta 0
            {"arm": "broken", "status": "failed", "log": "x.log"}]
    out = {r["arm"]: r for r in screen.decide(rows)}
    assert out["control_s0"]["verdict"] == "control" and out["control_s1"]["verdict"] == "control"
    assert out["good"]["verdict"] == "win" and out["good"]["delta_vs_control"] > 0
    assert out["bad"]["verdict"] == "loss"
    assert out["same"]["verdict"] == "neutral"
    assert out["broken"]["verdict"] == "failed"
    assert out["good"]["floor"] >= abs(0.50 - 0.49)
    assert "| good |" in screen.to_markdown(list(out.values()))


def test_decide_without_control(tmp_path):
    y = np.arange(50.0)
    rows = [{"arm": "control_s0", "status": "failed", "log": "x"},
            fake_row(tmp_path, "control_s1", 0.5, np.stack([y, y]), y),
            fake_row(tmp_path, "arm", 0.6, np.stack([y, y]), y)]
    out = {r["arm"]: r for r in screen.decide(rows)}
    assert out["arm"]["verdict"] == "no-control"


def test_decide_refuses_different_rows(tmp_path):
    y = np.arange(50.0)
    rows = [fake_row(tmp_path, "control_s0", 0.5, np.stack([y, y]), y),
            fake_row(tmp_path, "control_s1", 0.5, np.stack([y, y]), y),
            fake_row(tmp_path, "arm", 0.6, np.stack([y[::-1]] * 2), y[::-1])]
    with pytest.raises(ValueError, match="different rows"):
        screen.decide(rows)


def test_two_checkpoint_dirs_is_a_failure(tmp_path):
    for d in ("arm_20260101-000000", "arm_20260101-000001"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "checkpoint_last.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="2 checkpoint dirs"):
        screen.find_checkpoint(tmp_path, "arm")


def test_screen_end_to_end_tiny(data_dir, tmp_path):  # noqa: F811
    set_dir = make_label_set(tmp_path)
    out = tmp_path / "screen"
    eval_cfg = {"experiment": {"name": "e", "seed": 42, "output_dir": str(out / "eval")},
                "evaluation": {"device": "cpu", "batch_size": 16, "max_samples": 5000, "n_jobs": 1,
                               "min_n": 20, "ladder_sets": [], "ladder_block_from": "labeled_data",
                               "readouts": ["mean"], "random_control": False},
                "label_sets": {"labeled_data": [str(set_dir)]},
                "baseline": {"tag": "none", "checkpoint": "none", "parent_repo": str(tmp_path), "run_dirs": {}},
                "wandb": {"enabled": False, "entity": None, "project": "x", "group": None,
                          "job_type": "eval", "tags": []}}
    (tmp_path / "eval.yaml").write_text(yaml.safe_dump(eval_cfg))
    scfg = {"name": "tiny", "output_dir": str(out), "eval_config": str(tmp_path / "eval.yaml"),
            "shared_overrides": [f"data.manifest_dir={data_dir}", f"experiment.output_dir={out}",
                                 *TINY_OVERRIDES, "training.max_steps=4", "training.ckpt_every=4",
                                 "training.val_every=4", "training.diag_every=4", "training.log_every=2"],
            "arms": {"control_s0": ["experiment.seed=0"], "control_s1": ["experiment.seed=1"],
                     "broken": ["model.no_such_key=1"]},
            "wandb": {"enabled": False, "entity": None, "project": "x", "group": "tiny", "tags": []}}
    (tmp_path / "screen.yaml").write_text(yaml.safe_dump(scfg))

    screen.main(["--config", str(tmp_path / "screen.yaml"), "--gpus", "none", "--max_evals", "1"])

    rows = {r["arm"]: r for r in json.loads((out / "results.json").read_text())}
    assert rows["broken"]["status"] == "failed" and rows["broken"]["verdict"] == "failed"
    assert Path(rows["broken"]["log"]).exists()
    for arm in ("control_s0", "control_s1"):
        assert rows[arm]["status"] == "ok" and rows[arm]["verdict"] == "control"
        assert rows[arm]["n"] == 30 and "valid_mse_loss" in rows[arm]
    assert (out / "results.md").read_text().count("\n") >= 4
