import os

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from spectral_lejepa.data.loader import (
    SEQUENCE_LENGTH, SpectraDataset, make_loader, manifest_fingerprint, normalize_signal,
    read_manifest, remap_root, subsample,
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
    loader = make_loader(read_manifest(write_set(tmp_path)), batch_size=4, shuffle=False,
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
