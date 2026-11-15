import json
from pathlib import Path

import numpy as np
import pytest

from spectral_lejepa.config import load_config
from spectral_lejepa.training.checkpoint import load_checkpoint
from spectral_lejepa.training.trainer import train
from tests.training.test_train_smoke import REPO, TINY_OVERRIDES

ROOT = REPO / "configs"


def _packed(tmp_path):
    """60 synthetic rows, 30 labeled_data rows, 2 measured sets x 20 groups x 3 components (120 rows)."""
    rng = np.random.default_rng(0)
    packed = tmp_path / "packed"
    packed.mkdir()
    entries = [("single_channel_all", f"s_{i}.wav") for i in range(60)]
    entries += [("labeled_data", f"l_{i}.wav") for i in range(30)]
    for ds in ("dataset0055", "dataset0106"):
        entries += [(f"labeled_regression/{ds}", f"{ds}_comp{c}_spec_{g}.wav") for g in range(20) for c in (0, 1, 2)]
    (packed / "index.tsv").write_text("".join(f"{i}\t{s}\t/x/{n}\n" for i, (s, n) in enumerate(entries)))
    n = len(entries)
    np.save(packed / "spectra.npy", rng.normal(5.0, 2.0, size=(n, 245)).astype(np.float32))
    counts = {"single_channel_all": 60, "labeled_data": 30, "labeled_regression/dataset0055": 60,
              "labeled_regression/dataset0106": 60}
    (packed / "stats.json").write_text(json.dumps({"mean": 5.0, "std": 2.0, "n_rows": n, "n_dropped": 0,
                                                   "counts": counts}))
    np.save(packed / "drop_rows.npy", np.zeros(n, dtype=bool))
    np.save(packed / "valid_rows.npy", np.array([0, 1, 60]))
    return packed


def _cfg(name, tmp_path, packed, **data):
    cfg = load_config(ROOT / name, [f"experiment.output_dir={tmp_path / 'out'}", f"data.packed_dir={packed}",
                                    *TINY_OVERRIDES, "training.max_steps=3", "training.warmup_steps=1"])
    cfg["data"].update(data)
    return cfg


def _derived(result):
    return load_checkpoint(Path(result["output_dir"]) / "checkpoint_last.pt")["config"]["derived"]


def test_existing_configs_have_no_budget_keys():
    for name in ("final_model.yaml", "measured_only.yaml", "pretrain.yaml"):
        d = load_config(ROOT / name)["data"]
        assert not {"measured_fraction", "exclude_sources", "blend"} & set(d), name


def test_measured_only_run_has_no_budget_derived(tmp_path):
    packed = _packed(tmp_path)
    cfg = _cfg("budget_m.yaml", tmp_path, packed, measured_fraction=None)
    with pytest.warns(UserWarning):
        d = _derived(train(cfg))
    assert not {"measured_fraction", "n_measured_groups", "exclude_sources", "blend"} & set(d)
    assert d["n_train_rows"] == 120


def test_budget_m_fraction_rows(tmp_path):
    packed = _packed(tmp_path)
    with pytest.warns(UserWarning):
        r = train(_cfg("budget_m.yaml", tmp_path, packed, measured_fraction=0.25))
    d = _derived(r)
    assert d["measured_fraction"] == 0.25 and d["n_measured_groups"] == 10 and d["n_train_rows"] == 30
    groups = np.load(Path(r["output_dir"]) / "measured_groups.npy")
    assert len(groups) == 10 and sum(g.startswith("dataset0055/") for g in groups) == 5


def test_budget_ms_excludes_labeled_data(tmp_path):
    packed = _packed(tmp_path)
    with pytest.warns(UserWarning):
        d = _derived(train(_cfg("budget_ms.yaml", tmp_path, packed, measured_fraction=0.25)))
    assert d["exclude_sources"] == ["labeled_data"]
    assert d["n_train_rows"] == 58 + 30          # synthetic minus 2 validation rows, plus 10 groups x 3
    assert d["source_mass"] == pytest.approx({"labeled_regression": 0.5, "default": 0.5})


def test_blend_requires_source_weights(tmp_path):
    packed = _packed(tmp_path)
    cfg = _cfg("budget_m.yaml", tmp_path, packed, blend={"prob": 0.5, "lam_low": 0.5})
    with pytest.raises(ValueError, match="data.blend needs data.source_weights"):
        train(cfg)


def test_budget_mx_trains_and_records_blend(tmp_path):
    packed = _packed(tmp_path)
    with pytest.warns(UserWarning):
        d = _derived(train(_cfg("budget_mx.yaml", tmp_path, packed, measured_fraction=0.0625)))
    assert d["blend"] == {"prob": 0.5, "lam_low": 0.5} and d["n_measured_groups"] == 4


def test_resume_refuses_changed_fraction(tmp_path):
    packed = _packed(tmp_path)
    cfg = _cfg("budget_m.yaml", tmp_path, packed, measured_fraction=0.25)
    cfg["training"]["ckpt_every"] = 1
    with pytest.warns(UserWarning):
        r = train(cfg)
    cfg = _cfg("budget_m.yaml", tmp_path, packed, measured_fraction=0.5)
    cfg["training"]["resume"] = str(Path(r["output_dir"]) / "checkpoint_step1.pt")
    with pytest.raises(ValueError, match="data.measured_fraction"):
        train(cfg)
