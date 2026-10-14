"""SpectralFM inference: raw 245-point spectra -> robust-ensemble embedding (N, M, 12288).

Self-contained (torch, numpy, soundfile, scikit-learn only). Inside the repo, import it or use the CLI below;
elsewhere, copy this file and a bundle and run `python inference.py INPUT --output OUT.npz --bundle DIR`.
A bundle is TorchScript, which executes code when loaded: load only bundles you built or trust.
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

_HERE = Path(__file__).resolve()
DEFAULT_BUNDLE = (_HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent) / 'outputs/inference/robust-ensemble'
LENGTH = 245


def resolve_bundle(bundle=None):
    path = Path(bundle or os.environ.get('SPECTRALFM_BUNDLE') or DEFAULT_BUNDLE)
    if not (path / 'manifest.json').is_file():
        raise FileNotFoundError(f'no inference bundle at {path}; pass --bundle DIR (bundle=...) or set '
                                'SPECTRALFM_BUNDLE; inside the repo, build it with '
                                '`python -m spectral_lejepa export-ensemble`')
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
    if not path.is_file():
        raise FileNotFoundError(f'{path}: no such file')
    subtype = sf.info(str(path)).subtype
    if subtype not in ('FLOAT', 'DOUBLE'):   # integer PCM would be rescaled to [-1, 1]
        raise ValueError(f'{path.name}: spectra must be stored as float wavs, got {subtype}')
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
        if p.suffix.lower() == '.npy':
            arr = np.load(p)
            arr = arr[None] if arr.ndim == 1 else arr
            return _check(arr, p.name), [str(i) for i in range(len(arr))]
        paths = sorted(q for q in p.iterdir() if q.suffix.lower() == '.wav') if p.is_dir() else [p]
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
    def embed_with_ids(self, x, out=None):
        """out: optional preallocated (N, M, 12288) float32 array, e.g. a memmap for inputs larger than RAM."""
        arr, ids = load_spectra(x)
        shape = (len(arr), len(self.encoders), 12288)
        if out is None:
            out = np.empty(shape, np.float32)
        elif out.shape != shape:
            raise ValueError(f'out has shape {out.shape}, expected {shape}')
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


def main(argv=None):
    p = argparse.ArgumentParser(description='Embed raw spectra with the robust ensemble -> .npz '
                                            '(embeddings (N, M, 12288), ids, members, bundle_manifest).')
    p.add_argument('input', help='.npy (N, 245), a .wav, or a folder of .wav files')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--bundle')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=256)
    a = p.parse_args(argv)
    embedder = SpectralEmbedder(a.bundle, a.device, a.batch_size)
    spectra, ids = load_spectra(a.input)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name + '.embeddings.tmp.npy')   # embeddings stream to disk, not RAM
    try:
        emb = np.lib.format.open_memmap(tmp, 'w+', np.float32, (len(spectra), len(embedder.members), 12288))
        embedder.embed_with_ids(spectra, out=emb)
        np.savez(a.output, embeddings=emb, ids=np.array(ids), members=np.array(embedder.members),
                 bundle_manifest=json.dumps(embedder.manifest))
        shape = emb.shape
        del emb
    finally:
        tmp.unlink(missing_ok=True)
    print(f'{a.output}: embeddings {shape} from members {embedder.members}')


if __name__ == '__main__':
    main()
