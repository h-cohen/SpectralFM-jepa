"""Unlabeled SpectralFM spectra for pretraining.

Data definition (inherited from the parent project, unchanged):
- one spectrum = one mono float32 WAV of exactly 245 samples (the 16 kHz rate is nominal);
- fairseq manifests: line 1 is the wav root, then `filename<TAB>num_samples`;
- preprocessing = per-sample z-score via F.layer_norm (eps 1e-5), exactly what fairseq's
  FileAudioDataset(normalize=True) applied for data2vec. No global statistics, no augmentation.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEQUENCE_LENGTH = 245
STORAGE_PREFIX = "/storage/noy/"  # how RunAI jobs see the volume
LOCAL_PREFIX = "/mnt5/noy/"       # how this server sees it


def remap_root(root: str) -> str:
    return LOCAL_PREFIX + root[len(STORAGE_PREFIX):] if root.startswith(STORAGE_PREFIX) else root


def read_manifest(path) -> list[str]:
    with open(path) as f:
        root = remap_root(f.readline().strip())
        names = [line.split("\t")[0] for line in f if line.strip()]
    return [os.path.join(root, name) for name in names]


def manifest_fingerprint(path) -> dict:
    """Identifies the exact input list a run used: content hash + row count."""
    data = open(path, "rb").read()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "rows": data.count(b"\n") - 1}


def normalize_signal(x: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score over the last axis (population variance, eps 1e-5 inside the sqrt)."""
    return F.layer_norm(x, x.shape[-1:])


class SpectraDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        x, _ = sf.read(path, dtype="float32")
        if x.ndim != 1 or x.shape[0] != SEQUENCE_LENGTH:
            raise ValueError(f"{path}: expected shape ({SEQUENCE_LENGTH},), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: contains NaN or inf")
        return normalize_signal(torch.from_numpy(x))


def subsample(paths, max_samples, seed):
    """A seeded subset (kept in manifest order) for dev runs; all paths when max_samples is None."""
    if max_samples is None or max_samples >= len(paths):
        return list(paths)
    idx = np.sort(np.random.default_rng(seed).choice(len(paths), max_samples, replace=False))
    return [paths[i] for i in idx]


def make_loader(paths, batch_size, shuffle, seed, num_workers, drop_last) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)  # makes the shuffle order reproducible
    return DataLoader(SpectraDataset(paths), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers, generator=generator,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
