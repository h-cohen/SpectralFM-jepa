import math
from pathlib import Path

import pytest

from spectral_lejepa.training.checkpoint import load_checkpoint
from spectral_lejepa.training.early_stop import EarlyStopping
from spectral_lejepa.training.trainer import train
from tests.training.test_train_smoke import _stop_cfg, data_dir  # noqa: F401  (fixture)


def _feed(es, values, start=1):
    return [es.update(start + i, {"valid/loss": v}) for i, v in enumerate(values)]


def test_improves_then_stops_after_patience():
    es = EarlyStopping("valid/loss", 2, 0.0, 0)
    assert _feed(es, [1.0, 0.9, 0.95, 0.91]) == [True, True, False, False]
    assert es.should_stop and es.best_step == 2 and es.best_value == 0.9


def test_min_delta():
    es = EarlyStopping("valid/loss", 5, 0.05, 0)
    assert _feed(es, [1.0, 0.97]) == [True, False]
    assert es.bad_checks == 1


def test_ignores_checks_before_start_step():
    es = EarlyStopping("valid/loss", 1, 0.0, 100)
    assert es.update(50, {"valid/loss": 0.1}) is False
    assert es.best_value is None and es.bad_checks == 0 and not es.should_stop


def test_nan_is_not_improvement():
    es = EarlyStopping("valid/loss", 5, 0.0, 0)
    assert _feed(es, [1.0, math.nan]) == [True, False]
    assert es.bad_checks == 1 and es.best_value == 1.0


def test_missing_metric_fails():
    with pytest.raises(ValueError, match="valid/loss"):
        EarlyStopping("valid/loss", 1, 0.0, 0).update(0, {})


def test_state_roundtrip():
    es = EarlyStopping("valid/loss", 3, 0.0, 0)
    _feed(es, [1.0, 0.8, 0.9])
    other = EarlyStopping("valid/loss", 3, 0.0, 0)
    other.load_state_dict(es.state_dict())
    assert (other.best_step, other.best_value, other.bad_checks) == (2, 0.8, 1)


ALWAYS_WORSE = {"metric": "valid/loss", "min_delta": 1e9, "start_step": 0}


def _es_cfg(data_dir, tmp_path, name, **training):  # noqa: F811
    cfg = _stop_cfg(data_dir, tmp_path, name)
    cfg["training"].update(val_every=2, **training)
    return cfg


def test_trainer_stops_early_and_writes_best(data_dir, tmp_path):  # noqa: F811
    cfg = _es_cfg(data_dir, tmp_path, "es",
                    early_stopping={**ALWAYS_WORSE, "patience": 1})
    with pytest.warns(UserWarning):
        result = train(cfg)
    out = Path(result["output_dir"])
    assert result["stopped_at"] == 4 and result["best_step"] == 2
    best = load_checkpoint(out / "checkpoint_best.pt")
    assert best["step"] == 2 and best["early_stopping"]["best_step"] == 2
    assert result["best_checkpoint"] == str(out / "checkpoint_best.pt")
    assert load_checkpoint(out / "checkpoint_last.pt")["step"] == 4


def test_early_stop_state_survives_resume(data_dir, tmp_path):  # noqa: F811
    es_cfg = {**ALWAYS_WORSE, "patience": 3}
    with pytest.warns(UserWarning):
        first = train(_es_cfg(data_dir, tmp_path, "a", stop_step=6, early_stopping=es_cfg))
    assert first["stopped_at"] is None
    assert load_checkpoint(first["final_checkpoint"])["early_stopping"]["bad_checks"] == 2
    with pytest.warns(UserWarning):
        second = train(_es_cfg(data_dir, tmp_path, "b", early_stopping=es_cfg,
                                 resume=first["final_checkpoint"]))
    assert second["stopped_at"] == 8 and second["best_step"] == 2


def test_no_early_stopping_key_unchanged(data_dir, tmp_path):  # noqa: F811
    with pytest.warns(UserWarning):
        result = train(_stop_cfg(data_dir, tmp_path, "plain", stop_step=4))
    assert result["stopped_at"] is None
    assert not (Path(result["output_dir"]) / "checkpoint_best.pt").exists()
