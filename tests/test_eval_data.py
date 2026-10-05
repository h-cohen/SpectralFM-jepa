import os

import numpy as np
import pytest
import soundfile as sf
import torch

from spectral_lejepa.evaluation.bank import READOUTS, extract_bank, load_bank, readout_arms, save_bank
from spectral_lejepa.evaluation.data import load_labeled_data, normalize_like_fairseq
from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
from tests.test_model import SMALL

NOVA = "/mnt5/noy/SpectralFM/fairseq/data/nova_data"
PARENT_OUT = "/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged/code/eval_outputs"
SMALL_SETS = ["dataset0055", "dataset0106", "dataset0109", "dataset0112", "dataset0113",
              "dataset0114", "dataset0115", "dataset0117", "dataset0118", "dataset0120"]


def make_set(root, rows, wav_subdir, lengths=None):
    root.mkdir(parents=True)
    wav_dir = root / wav_subdir if wav_subdir else root
    wav_dir.mkdir(exist_ok=True)
    lines = []
    for name, label in rows:
        n = (lengths or {}).get(name, 245)
        sf.write(wav_dir / name, np.full(n, float(len(lines) + 1), np.float32), 16000, subtype="FLOAT")
        lines.append(f"{name}\t{label}")
    (root / "labels.tsv").write_text("\n".join(lines) + "\n")


def test_order_component_filter_and_merge(tmp_path):
    make_set(tmp_path / "a", [("dataset0007_comp0_spec_2.wav", 0.5), ("dataset0007_comp1_spec_2.wav", 0.5),
                              ("dataset0007_comp0_spec_1.wav", -1.0), ("dataset0007_comp1_spec_9.wav", 2.0)],
             "wavs", lengths={"dataset0007_comp0_spec_1.wav": 200})
    make_set(tmp_path / "b", [("dataset0003_comp0_spec_4.wav", 3.0)], None)
    raw, y = load_labeled_data([str(tmp_path / "a"), str(tmp_path / "b")])
    assert raw.shape == (3, 245) and raw.dtype == np.float32 and y.dtype == np.float64
    assert y.tolist() == [3.0, -1.0, 0.5]           # sorted by (dataset, spec); comp1-only spec dropped
    assert (raw[1, 200:] == 0).all() and (raw[1, :200] != 0).all()   # short wav zero-padded


def test_duplicate_spectrum_across_sets_rejected(tmp_path):
    make_set(tmp_path / "a", [("dataset0007_comp0_spec_2.wav", 0.5)], None)
    make_set(tmp_path / "b", [("dataset0007_comp0_spec_2.wav", 0.5)], None)
    with pytest.raises(ValueError, match="dataset 7, spec 2"):
        load_labeled_data([str(tmp_path / "a"), str(tmp_path / "b")])


def test_max_samples_subsample_is_seeded(tmp_path):
    make_set(tmp_path / "a", [(f"dataset0001_comp0_spec_{i}.wav", float(i)) for i in range(30)], None)
    a = load_labeled_data([str(tmp_path / "a")], max_samples=10, seed=42)[1]
    b = load_labeled_data([str(tmp_path / "a")], max_samples=10, seed=42)[1]
    assert len(a) == 10 and np.array_equal(a, b)


def test_normalize_like_fairseq():
    x = np.random.default_rng(0).normal(3, 2, size=(5, 245))
    z = normalize_like_fairseq(x)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z, (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8), rtol=1e-6)


def test_bank_roundtrip_and_extraction(tmp_path):
    torch.manual_seed(0)
    backbone = EvalBackbone(build_model(SMALL, 245))
    z = normalize_like_fairseq(np.random.default_rng(0).normal(size=(10, 245)))
    bank = extract_bank(backbone, z, device="cpu", batch_size=4, readouts=("mean",))
    assert list(bank) == ["layer0", "layer1", "layer2"]
    assert all(v.shape == (10, 32) and v.dtype == np.float32 for v in bank.values())
    with torch.no_grad():
        hs = backbone(input_values=torch.from_numpy(z)).hidden_states
    np.testing.assert_allclose(bank["layer2"], hs[2].mean(1).numpy(), rtol=1e-5, atol=1e-6)

    raw, y = np.random.default_rng(1).normal(size=(10, 245)).astype(np.float32), np.arange(10.0)
    save_bank(tmp_path / "bank.npz", bank, raw, y, {"checkpoint": "x", "backbone": "EvalBackbone"})
    bank2, raw2, y2, meta = load_bank(tmp_path / "bank.npz")
    assert list(bank2) == list(bank) and all(np.array_equal(bank[k], bank2[k]) for k in bank)
    assert np.array_equal(raw2, raw) and np.array_equal(y2, y) and meta["backbone"] == "EvalBackbone"


def test_load_parent_format_bank(tmp_path):
    """Parent banks are [N, K comps, 10 stats, D]; we take component 0, statistic 'mean'."""
    stats = ("mean", "std", "max", "min", "first", "last", "seg0", "seg1", "seg2", "seg3")
    arr = np.random.default_rng(0).normal(size=(6, 1, 10, 4)).astype(np.float32)
    np.savez(tmp_path / "bank.npz", bank__fe=arr, bank__layer0=arr + 1, input_raw=np.ones((6, 1, 245), np.float32),
             input_z=np.ones((6, 1, 245), np.float32), y=np.arange(6.0),
             _meta=np.array([repr({"comps": (0,), "pool_stats": stats})]))
    bank, raw, y, _ = load_bank(tmp_path / "bank.npz")
    assert list(bank) == ["fe", "layer0"]
    assert np.array_equal(bank["fe"], arr[:, 0, 0, :]) and raw.shape == (6, 245)


def test_readout_arms_shapes_and_values():
    h = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)   # B=2, T=6, D=3
    arms = readout_arms([h, h + 1], ("mean", "seg4", "flat"))
    assert list(arms) == ["layer0", "layer0/seg4", "layer0/flat", "layer1", "layer1/seg4", "layer1/flat"]
    np.testing.assert_allclose(arms["layer0"], h.mean(1).numpy())
    e = np.linspace(0, 6, 5).astype(int)          # [0, 1, 3, 4, 6]
    seg = np.concatenate([h[:, e[s]:max(e[s] + 1, e[s + 1])].mean(1).numpy() for s in range(4)], 1)
    np.testing.assert_allclose(arms["layer0/seg4"], seg)
    assert arms["layer0/seg4"].shape == (2, 12) and arms["layer0/flat"].shape == (2, 18)
    np.testing.assert_allclose(arms["layer1/flat"], (h + 1).reshape(2, -1).numpy())


def test_bank_roundtrip_with_readouts(tmp_path):
    torch.manual_seed(0)
    backbone = EvalBackbone(build_model(SMALL, 245))
    z = normalize_like_fairseq(np.random.default_rng(0).normal(size=(10, 245)))
    bank = extract_bank(backbone, z, device="cpu", batch_size=4, readouts=READOUTS)
    assert len(bank) == 3 * (SMALL["depth"] + 1) and bank["layer2/flat"].shape == (10, 24 * 32)
    raw, y = np.zeros((10, 245), np.float32), np.arange(10.0)
    save_bank(tmp_path / "bank.npz", bank, raw, y, {"readouts": READOUTS})
    assert "bank__layer1__seg4" in np.load(tmp_path / "bank.npz").files
    back, *_ = load_bank(tmp_path / "bank.npz", readouts=READOUTS)
    assert list(back) == list(bank) and all(np.array_equal(back[k], bank[k]) for k in bank)
    only_mean, *_ = load_bank(tmp_path / "bank.npz")
    assert list(only_mean) == ["layer0", "layer1", "layer2"]


def test_parent_bank_gives_mean_and_seg4_never_flat(tmp_path):
    stats = ("mean", "std", "max", "min", "first", "last", "seg0", "seg1", "seg2", "seg3")
    arr = np.random.default_rng(0).normal(size=(6, 1, 10, 4)).astype(np.float32)
    np.savez(tmp_path / "bank.npz", bank__layer0=arr, input_raw=np.ones((6, 1, 245), np.float32),
             input_z=np.ones((6, 1, 245), np.float32), y=np.arange(6.0),
             _meta=np.array([repr({"comps": (0,), "pool_stats": stats})]))
    bank, *_ = load_bank(tmp_path / "bank.npz", readouts=READOUTS)
    assert list(bank) == ["layer0", "layer0/seg4"]
    np.testing.assert_array_equal(bank["layer0/seg4"], arr[:, 0, 6:10, :].reshape(6, -1))


def _parent_bank(rel):
    path = os.path.join(PARENT_OUT, rel, "bank.npz")
    if not os.path.exists(path):
        pytest.skip("parent outputs not available")
    data = np.load(path)
    return data["input_raw"][:, 0, :], data["y"]


@pytest.mark.slow
@pytest.mark.parametrize("set_dirs,rel", [
    ([f"{NOVA}/labeled_regression/dataset0120"], "label_probe_regression_ref_feb25/label_probe/dataset0120"),
    ([f"{NOVA}/labeled_regression/{s}" for s in SMALL_SETS],
     "label_probe_regression_merged_ref_feb25/label_probe/labeled_regression_all"),
    ([f"{NOVA}/labeled_data"], "label_probe_regression_labeled_data_ref_feb25/label_probe/labeled_data"),
])
def test_same_rows_as_parent(set_dirs, rel):
    parent_raw, parent_y = _parent_bank(rel)
    raw, y = load_labeled_data(set_dirs)
    assert np.array_equal(y, parent_y)
    assert np.array_equal(raw, parent_raw)
