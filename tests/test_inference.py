import ast
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from spectral_lejepa import inference as inf
from spectral_lejepa.models.export import export_ensemble
from spectral_lejepa.models.vit_1d import EvalBackbone
from spectral_lejepa.training.checkpoint import load_model


@pytest.fixture
def bundle(make_checkpoint, tmp_path):
    members = {'a': make_checkpoint('a', 0), 'b': make_checkpoint('b', 1)}
    export_ensemble(members, tmp_path / 'bundle')
    return tmp_path / 'bundle', members


def spectra(n=5, seed=0):
    return np.random.default_rng(seed).normal(.4, .5, (n, 245)).astype(np.float32)


def test_module_is_self_contained():
    tree = ast.parse(open(inf.__file__).read())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert not [m for m in names if m.startswith('spectral_lejepa')]


def test_embedding_is_each_members_block2_flat_in_manifest_order(bundle):
    path, members = bundle
    x = spectra()
    e = inf.SpectralEmbedder(path, device='cpu', batch_size=2).embed(x)   # batches keep row order
    assert e.shape == (5, 2, 12288) and e.dtype == np.float32
    for j, name in enumerate(['a', 'b']):
        model, _ = load_model(members[name])
        with torch.no_grad():
            ref = EvalBackbone(model)((torch.from_numpy(x) - .3784) / .5126).hidden_states[2].flatten(1).numpy()
        assert np.allclose(e[:, j], ref, atol=1e-5)


def test_single_spectrum_and_float64_are_accepted(bundle):
    emb = inf.SpectralEmbedder(bundle[0], device='cpu')
    one = spectra(1)[0].astype(np.float64)
    assert emb.embed(one).shape == (1, 2, 12288)


def test_tampered_member_is_rejected(bundle):
    path, _ = bundle
    with open(path / 'member_a.pt', 'ab') as f:
        f.write(b'x')
    with pytest.raises(ValueError, match='checksum'):
        inf.SpectralEmbedder(path, device='cpu')


def test_bundle_resolution_order(bundle, monkeypatch, tmp_path):
    path, _ = bundle
    monkeypatch.setenv('SPECTRALFM_BUNDLE', str(path))
    assert inf.resolve_bundle() == path
    assert inf.resolve_bundle(str(path)) == path
    monkeypatch.delenv('SPECTRALFM_BUNDLE')
    monkeypatch.setattr(inf, 'DEFAULT_BUNDLE', tmp_path / 'missing')
    with pytest.raises(FileNotFoundError, match='export-ensemble'):
        inf.resolve_bundle()


def test_load_spectra_formats_and_ids(tmp_path):
    x = spectra(3)
    np.save(tmp_path / 'x.npy', x)
    assert inf.load_spectra(x)[1] == ['0', '1', '2']
    assert np.array_equal(inf.load_spectra(tmp_path / 'x.npy')[0], x)
    d = tmp_path / 'wavs'
    d.mkdir()
    for name, row in zip(['b.wav', 'a.wav', 'c.wav'], x):
        sf.write(d / name, row, 16000, subtype='FLOAT')
    (d / 'notes.txt').write_text('ignored')
    arr, ids = inf.load_spectra(d)
    assert ids == ['a.wav', 'b.wav', 'c.wav'] and np.allclose(arr[1], x[0])
    arr2, ids2 = inf.load_spectra([d / 'c.wav', d / 'a.wav'])
    assert ids2 == ['c.wav', 'a.wav'] and np.allclose(arr2[0], x[2])


@pytest.mark.parametrize('bad, message', [
    (np.zeros((2, 240), np.float32), '245'),
    (np.full((1, 245), np.nan, np.float32), 'finite'),
    (np.zeros((0, 245), np.float32), 'empty'),
])
def test_bad_arrays_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        inf.load_spectra(bad)


def test_bad_wavs_are_rejected(tmp_path):
    sf.write(tmp_path / 'short.wav', np.zeros(240, np.float32), 16000)
    with pytest.raises(ValueError, match='short.wav'):
        inf.load_spectra(tmp_path / 'short.wav')
    sf.write(tmp_path / 'stereo.wav', np.zeros((245, 2), np.float32), 16000)
    with pytest.raises(ValueError, match='mono'):
        inf.load_spectra(tmp_path / 'stereo.wav')
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError, match='no .wav'):
        inf.load_spectra(tmp_path / 'empty')


def test_ensemble_regressor_averages_member_predictions():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(60, 2, 8))
    y = F[:, 0, 0] + F[:, 1, 1]
    reg = inf.EnsembleRegressor().fit(F, y)
    each = [m.predict(F[:, j]) for j, m in enumerate(reg.models_)]
    assert np.allclose(reg.predict(F), np.mean(each, 0))
