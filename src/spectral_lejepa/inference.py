"""SpectralFM inference: raw 245-point spectra -> robust-ensemble embedding (N, M, 12288).

Self-contained (torch, numpy, soundfile, scikit-learn only): copy this file and a bundle anywhere.
    from spectral_lejepa.inference import SpectralEmbedder, EnsembleRegressor
    F = SpectralEmbedder().embed('spectra/')            # (N, M, 12288): one block-2 flat vector per member
    reg = EnsembleRegressor().fit(F_train, y_train)     # one ridge per member, predictions averaged
    python -m spectral_lejepa embed INPUT --output OUT.npz [--bundle DIR] [--device auto|cpu|cuda]
Build the bundle once with: python -m spectral_lejepa export-ensemble
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEFAULT_BUNDLE = Path('outputs/inference/robust-ensemble')
LENGTH = 245


def resolve_bundle(bundle=None):
    path = Path(bundle or os.environ.get('SPECTRALFM_BUNDLE') or DEFAULT_BUNDLE)
    if not (path / 'manifest.json').is_file():
        raise FileNotFoundError(f'no inference bundle at {path}; build it with '
                                '`python -m spectral_lejepa export-ensemble` or set SPECTRALFM_BUNDLE')
    return path


def _check(arr, where):
    if arr.ndim != 2 or arr.shape[1] != LENGTH:
        raise ValueError(f'{where}: expected spectra of {LENGTH} points, got shape {arr.shape}')
    if len(arr) == 0:
        raise ValueError(f'{where}: empty input')
    if not np.isfinite(arr).all():
        raise ValueError(f'{where}: values must be finite (row {int(np.where(~np.isfinite(arr).all(1))[0][0])})')
    return arr.astype(np.float32, copy=False)


def _read_wav(path):
    data, _ = sf.read(str(path), dtype='float32', always_2d=True)
    if data.shape[1] != 1:
        raise ValueError(f'{path.name}: expected a mono wav, got {data.shape[1]} channels')
    if data.shape[0] != LENGTH:
        raise ValueError(f'{path.name}: expected {LENGTH} samples, got {data.shape[0]}')
    return data[:, 0]


def load_spectra(x):
    """(spectra (N, 245) float32, ids). x: array, .npy, .wav, folder of .wav, or list of .wav paths."""
    if isinstance(x, np.ndarray):
        arr = x[None] if x.ndim == 1 else x
        return _check(np.asarray(arr), 'array'), [str(i) for i in range(len(arr))]
    if isinstance(x, (list, tuple)):
        paths = [Path(p) for p in x]
    else:
        p = Path(x)
        if p.suffix == '.npy':
            arr = np.load(p)
            arr = arr[None] if arr.ndim == 1 else arr
            return _check(arr, p.name), [str(i) for i in range(len(arr))]
        paths = sorted(p.glob('*.wav')) if p.is_dir() else [p]
        if p.is_dir() and not paths:
            raise ValueError(f'{p}: no .wav files')
    if not paths:
        raise ValueError('empty input')
    return _check(np.stack([_read_wav(q) for q in paths]), 'wav input'), [q.name for q in paths]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class SpectralEmbedder:
    """Loads the bundle's TorchScript members; embed() returns (N, M, 12288) float32 in manifest order."""

    def __init__(self, bundle=None, device='auto', batch_size=256):
        self.bundle = resolve_bundle(bundle)
        self.manifest = json.loads((self.bundle / 'manifest.json').read_text())
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        self.batch_size = batch_size
        self.encoders = []
        for m in self.manifest['members']:
            f = self.bundle / m['file']
            if _sha256(f) != m['sha256']:
                raise ValueError(f'{f.name}: checksum does not match manifest.json')
            self.encoders.append(torch.jit.load(str(f), map_location=self.device).eval())
        self.members = [m['name'] for m in self.manifest['members']]

    @torch.no_grad()
    def embed_with_ids(self, x):
        arr, ids = load_spectra(x)
        out = np.empty((len(arr), len(self.encoders), 12288), np.float32)
        for i in range(0, len(arr), self.batch_size):
            batch = torch.from_numpy(arr[i:i + self.batch_size]).to(self.device)
            for j, enc in enumerate(self.encoders):
                out[i:i + len(batch), j] = enc(batch).cpu().numpy()
        return out, ids

    def embed(self, x):
        return self.embed_with_ids(x)[0]


class EnsembleRegressor:
    """One StandardScaler + RidgeCV per member on (N, M, D) features; predict() averages the members.
    A convenience baseline; the benchmark chose each probe recipe by inner cross-validation."""

    def __init__(self, alphas=np.logspace(-3, 3, 20)):
        self.alphas = alphas

    def fit(self, features, y):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.models_ = [make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas)).fit(features[:, j], y)
                        for j in range(features.shape[1])]
        return self

    def predict(self, features):
        return np.mean([m.predict(features[:, j]) for j, m in enumerate(self.models_)], 0)
