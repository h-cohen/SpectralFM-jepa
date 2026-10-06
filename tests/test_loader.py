import json
import os

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from spectral_lejepa.data.loader import (
    SEQUENCE_LENGTH, PackedSpectra, SourceWeightedSampler, SpectraDataset, make_loader, manifest_fingerprint,
    normalize, normalize_signal, packed_train_rows, read_manifest, remap_root, source_mass, source_row_probs,
    subsample, valid_subset,
)

REAL_VALID = "/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_one/valid.tsv"


def write_set(tmp_path, n=10, length=245, bad=None):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    rng = np.random.default_rng(0)
    names = []
    for i in range(n):
        x = rng.uniform(0.0, 0.8, length).astype(np.float32)
        if bad == "nan" and i == 3:
            x[10] = np.nan
        if bad == "short" and i == 3:
            x = x[:200]
        name = f"spec_{i}.wav"
        sf.write(wav_dir / name, x, 16000, subtype="FLOAT")
        names.append(name)
    manifest = tmp_path / "train.tsv"
    manifest.write_text(str(wav_dir) + "\n" + "".join(f"{n}\t{length}\n" for n in names))
    return manifest


def test_remap_storage_root():
    assert remap_root("/storage/noy/SpectralFM/x/wav") == "/mnt5/noy/SpectralFM/x/wav"
    assert remap_root("/mnt5/noy/SpectralFM/x/wav") == "/mnt5/noy/SpectralFM/x/wav"


def test_read_manifest(tmp_path):
    paths = read_manifest(write_set(tmp_path, n=4))
    assert len(paths) == 4
    assert paths[0] == os.path.join(str(tmp_path / "wav"), "spec_0.wav")


def test_batch_shape_dtype_and_normalization(tmp_path):
    loader = make_loader(SpectraDataset(read_manifest(write_set(tmp_path))), batch_size=4, shuffle=False,
                         seed=0, num_workers=0, drop_last=True)
    batch = next(iter(loader))
    assert batch.shape == (4, SEQUENCE_LENGTH)
    assert batch.dtype == torch.float32
    assert torch.isfinite(batch).all()
    assert batch.mean(dim=1).abs().max() < 1e-5
    assert torch.allclose(batch.var(dim=1, unbiased=False), torch.ones(4), atol=1e-3)


def test_normalization_matches_fairseq(tmp_path):
    path = read_manifest(write_set(tmp_path, n=1))[0]
    raw = torch.from_numpy(sf.read(path, dtype="float32")[0])
    assert torch.equal(SpectraDataset([path])[0], F.layer_norm(raw, raw.shape))
    assert torch.equal(normalize_signal(raw[None])[0], F.layer_norm(raw, raw.shape))


def test_wrong_length_names_the_file(tmp_path):
    ds = SpectraDataset(read_manifest(write_set(tmp_path, bad="short")))
    with pytest.raises(ValueError, match="spec_3.wav"):
        ds[3]


def test_non_finite_names_the_file(tmp_path):
    ds = SpectraDataset(read_manifest(write_set(tmp_path, bad="nan")))
    with pytest.raises(ValueError, match="spec_3.wav"):
        ds[3]


def test_subsample_is_deterministic_and_sorted():
    paths = [f"p{i}" for i in range(100)]
    a, b = subsample(paths, 10, seed=1), subsample(paths, 10, seed=1)
    assert a == b and len(a) == 10
    assert [paths.index(p) for p in a] == sorted(paths.index(p) for p in a)
    assert subsample(paths, None, seed=1) == paths
    assert subsample(paths, 1000, seed=1) == paths


def test_fingerprint(tmp_path):
    fp = manifest_fingerprint(write_set(tmp_path, n=3))
    assert fp["rows"] == 3 and len(fp["sha256"]) == 64


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(REAL_VALID), reason="real data not mounted")
def test_real_valid_split():
    paths = read_manifest(REAL_VALID)
    assert len(paths) == 1000 and paths[0].startswith("/mnt5/noy/")
    x = SpectraDataset(paths)[0]
    assert x.shape == (SEQUENCE_LENGTH,) and torch.isfinite(x).all()


def write_packed(tmp_path, n=20):
    raw = np.random.default_rng(1).normal(2.0, 3.0, size=(n, SEQUENCE_LENGTH)).astype(np.float32)
    np.save(tmp_path / "spectra.npy", raw)
    np.save(tmp_path / "drop_rows.npy", np.array([0, 5]))
    np.save(tmp_path / "valid_rows.npy", np.array([3, 7, 9]))
    return raw


def test_normalize_methods():
    x = torch.randn(3, SEQUENCE_LENGTH) * 4 + 1
    assert torch.equal(normalize(x, "sample_zscore", None), F.layer_norm(x, x.shape[-1:]))
    assert torch.allclose(normalize(x, "global", {"mean": 1.0, "std": 4.0}), (x - 1.0) / 4.0)
    with pytest.raises(ValueError):
        normalize(x, "bogus", None)


def test_packed_spectra_normalization(tmp_path):
    raw = write_packed(tmp_path)
    stats = {"mean": 2.0, "std": 3.0}
    g = PackedSpectra(tmp_path, [4, 2], "global", stats)
    assert len(g) == 2
    assert torch.allclose(g[0], (torch.from_numpy(raw[4]) - 2.0) / 3.0)
    z = PackedSpectra(tmp_path, [2], "sample_zscore", None)[0]
    assert torch.allclose(z, F.layer_norm(torch.from_numpy(raw[2]), (SEQUENCE_LENGTH,)))


def test_packed_train_rows_exclude_dropped_and_valid(tmp_path):
    write_packed(tmp_path)
    rows = packed_train_rows(tmp_path)
    assert len(rows) == 20 - 2 - 3
    assert not {0, 5, 3, 7, 9} & set(rows.tolist())


def test_packed_train_rows_bool_drop_mask(tmp_path):
    np.save(tmp_path / "spectra.npy", np.zeros((8, SEQUENCE_LENGTH), dtype=np.float32))
    mask = np.zeros(8, dtype=bool)
    mask[3] = True
    np.save(tmp_path / "drop_rows.npy", mask)
    np.save(tmp_path / "valid_rows.npy", np.array([1]))
    assert packed_train_rows(tmp_path).tolist() == [0, 2, 4, 5, 6, 7]


def test_valid_subset_is_seeded_sorted_and_spread():
    valid = np.arange(0, 10000, 5)
    a, b = valid_subset(valid, k=100, seed=0), valid_subset(valid, k=100, seed=0)
    assert np.array_equal(a, b) and len(a) == 100 and np.all(np.diff(a) > 0)
    assert set(a.tolist()) <= set(valid.tolist()) and a.max() > valid[200]   # not a prefix
    assert np.array_equal(valid_subset(valid[:10], k=100), valid[:10])


def write_sources(tmp_path, counts=None):
    counts = counts or {"single": 60, "labeled_regression/a": 10, "labeled_regression/b": 20, "other": 10}
    json.dump({"counts": counts, "n_rows": sum(counts.values())}, open(tmp_path / "stats.json", "w"))
    return np.arange(sum(counts.values()))


def test_source_row_probs_group_masses(tmp_path):
    rows = write_sources(tmp_path)
    p = source_row_probs(tmp_path, rows, {"labeled_regression": 0.5, "default": 0.5})
    assert p.sum() == pytest.approx(1.0)
    assert p[:60].sum() + p[90:].sum() == pytest.approx(0.5)
    assert p[60:90].sum() == pytest.approx(0.5)
    assert p[60] == pytest.approx(0.5 / 30) and p[0] == pytest.approx(0.5 / 70)   # uniform within a group
    assert source_mass(tmp_path, rows, {"labeled_regression": 0.5, "default": 0.5}) == pytest.approx(
        {"labeled_regression": 0.5, "default": 0.5})


def test_source_row_probs_longest_prefix_and_renormalization(tmp_path):
    rows = write_sources(tmp_path)
    w = {"labeled_regression": 1.0, "labeled_regression/a": 3.0, "default": 1.0}
    mass = source_mass(tmp_path, rows, w)
    assert mass == pytest.approx({"labeled_regression": 0.2, "labeled_regression/a": 0.6, "default": 0.2})
    held_out = rows[(rows < 60) | (rows >= 90)]            # no training rows in the labeled group
    p = source_row_probs(tmp_path, held_out, {"labeled_regression": 0.5, "default": 0.5})
    assert p.sum() == pytest.approx(1.0) and np.allclose(p, 1 / len(held_out))
    assert source_mass(tmp_path, held_out, {"labeled_regression": 0.5, "default": 0.5}) == {"default": 1.0}


def test_source_row_probs_errors(tmp_path):
    rows = write_sources(tmp_path)
    with pytest.raises(ValueError, match="no 'default'"):
        source_row_probs(tmp_path, rows, {"labeled_regression": 1.0})
    with pytest.raises(ValueError, match="matches no source"):
        source_row_probs(tmp_path, rows, {"nope": 1.0, "default": 1.0})
    with pytest.raises(ValueError, match="positive"):
        source_row_probs(tmp_path, rows, {"labeled_regression": 0.0, "default": 1.0})
    json.dump({"counts": {"x": 5}, "n_rows": 9}, open(tmp_path / "stats.json", "w"))
    with pytest.raises(ValueError, match="n_rows"):
        source_row_probs(tmp_path, rows, {"default": 1.0})


def test_source_weighted_sampler(tmp_path):
    rows = write_sources(tmp_path, {"big": 990, "small": 10})
    p = source_row_probs(tmp_path, rows, {"small": 0.5, "default": 0.5})
    sampler = SourceWeightedSampler(p, seed=3)
    assert len(sampler) == 1000
    first = list(sampler)
    assert first == list(SourceWeightedSampler(p, seed=3)) and len(first) == 1000
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first
    drawn = []
    for e in range(20):
        sampler.set_epoch(e)
        drawn.append(np.array(list(sampler)))
    assert (np.concatenate(drawn) >= 990).mean() == pytest.approx(0.5, abs=0.02)
