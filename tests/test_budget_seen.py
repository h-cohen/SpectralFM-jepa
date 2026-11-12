import numpy as np
import pytest

from ops.budget_seen import bank_group_keys, seen_mask, seen_r2
from spectral_lejepa.evaluation.bank import save_bank


def _packed(tmp_path, rng):
    names = [f"dataset0106_comp{c}_spec_{g}.wav" for g in range(6) for c in (0, 1)]
    names += ["spectra0000_batch1_spec_0.wav"]
    spectra = rng.standard_normal((len(names), 245)).astype(np.float32)
    np.save(tmp_path / "spectra.npy", spectra)
    src = ["labeled_regression/dataset0106"] * 12 + ["single_channel_all"]
    (tmp_path / "index.tsv").write_text("".join(f"{i}\t{s}\t/x/{n}\n" for i, (s, n) in enumerate(zip(src, names))))
    return tmp_path, spectra


def _bank(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_bank(path, {"layer2/flat": np.zeros((len(raw), 4), np.float32)}, raw, np.arange(len(raw), dtype=float),
              {"set": "dataset0106"})
    return path


def test_bank_group_keys_match_packed_rows(tmp_path):
    (tmp_path / "p").mkdir()
    packed, spectra = _packed(tmp_path / "p", np.random.default_rng(0))
    bank = _bank(tmp_path / "b" / "dataset0106" / "bank.npz", spectra[[0, 4, 10]])   # comp0 of groups 0, 2, 5
    assert bank_group_keys(bank, packed).tolist() == ["dataset0106/0", "dataset0106/2", "dataset0106/5"]


def test_unmatched_row_raises(tmp_path):
    (tmp_path / "p").mkdir()
    packed, spectra = _packed(tmp_path / "p", np.random.default_rng(0))
    bank = _bank(tmp_path / "b" / "dataset0106" / "bank.npz", np.vstack([spectra[[0]], spectra[[0]] + 1]))
    with pytest.raises(ValueError, match="row 1"):
        bank_group_keys(bank, packed)


def test_seen_mask_matches_groups():
    keys = np.array(["dataset0106/0", "dataset0106/2", "dataset0106/5"])
    assert seen_mask(keys, np.array(["dataset0106/2", "dataset0055/0"])).tolist() == [False, True, False]


def test_seen_r2_splits():
    y = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0])
    out = seen_r2(y, pred, np.arange(8) < 4)
    assert out["seen"] == pytest.approx(1.0) and out["unseen"] == pytest.approx(-3.0)
    assert out["n_seen"] == 4 and out["n_unseen"] == 4
    assert np.isnan(seen_r2(y, pred, np.ones(8, bool))["unseen"])
