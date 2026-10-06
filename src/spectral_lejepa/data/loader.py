"""Unlabeled SpectralFM spectra for pretraining.

Data definition (inherited from the parent project, unchanged):
- one spectrum = one mono float32 WAV of exactly 245 samples (the 16 kHz rate is nominal);
- fairseq manifests: line 1 is the wav root, then `filename<TAB>num_samples`;
- preprocessing = per-sample z-score via F.layer_norm (eps 1e-5), exactly what fairseq's
  FileAudioDataset(normalize=True) applied for data2vec. No global statistics, no augmentation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

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


def normalize(x: torch.Tensor, method: str, stats: dict | None) -> torch.Tensor:
    """`sample_zscore` (per spectrum) or `global` ((x - mean) / std from the packed stats.json)."""
    if method == "sample_zscore":
        return normalize_signal(x)
    if method == "global":
        return (x - stats["mean"]) / stats["std"]
    raise ValueError(f"unknown normalization {method!r}: use 'sample_zscore' or 'global'")


class SpectraDataset(Dataset):
    def __init__(self, paths, normalization="sample_zscore", stats=None):
        self.paths = list(paths)
        self.normalization, self.stats = normalization, stats

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        x, _ = sf.read(path, dtype="float32")
        if x.ndim != 1 or x.shape[0] != SEQUENCE_LENGTH:
            raise ValueError(f"{path}: expected shape ({SEQUENCE_LENGTH},), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: contains NaN or inf")
        return normalize(torch.from_numpy(x), self.normalization, self.stats)


class PackedSpectra(Dataset):
    """Selected rows of a packed `spectra.npy`; the memmap opens lazily so DataLoader workers are safe."""

    def __init__(self, packed_dir, rows, normalization="sample_zscore", stats=None):
        self.path = os.path.join(packed_dir, "spectra.npy")
        self.rows = np.asarray(rows)
        self.normalization, self.stats = normalization, stats
        self._array = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        x = torch.from_numpy(np.array(self._array[self.rows[i]], dtype=np.float32))
        return normalize(x, self.normalization, self.stats)


def _row_indices(a: np.ndarray) -> np.ndarray:
    """Row indices from either an index array or a boolean mask over all rows."""
    return np.flatnonzero(a) if a.dtype == bool else a.astype(np.int64)


def packed_train_rows(packed_dir) -> np.ndarray:
    """All rows of the packed array except dropped (constant/non-finite) and validation rows."""
    n = len(np.load(os.path.join(packed_dir, "spectra.npy"), mmap_mode="r"))
    excluded = np.concatenate([_row_indices(np.load(os.path.join(packed_dir, f"{name}_rows.npy")))
                               for name in ("drop", "valid")])
    return np.setdiff1d(np.arange(n), excluded)


def valid_subset(valid_rows, k=2048, seed=0) -> np.ndarray:
    """A seeded random subset (sorted) of the validation rows, so every source is represented."""
    return np.sort(np.random.default_rng(seed).choice(valid_rows, min(k, len(valid_rows)), replace=False))


def _source_groups(packed_dir, train_rows, source_weights):
    """(group key per training row, normalized mass per group that has training rows)."""
    stats = json.loads((Path(packed_dir) / "stats.json").read_text())
    counts = stats["counts"]
    if sum(counts.values()) != stats["n_rows"]:
        raise ValueError(f"stats.json counts sum to {sum(counts.values())} but n_rows={stats['n_rows']}")
    bad = {k: v for k, v in source_weights.items() if not v > 0}
    if bad:
        raise ValueError(f"data.source_weights masses must be positive, got {bad}")
    keys = [k for k in source_weights if k != "default"]
    group_of = {}
    for source in counts:
        matches = [k for k in keys if source.startswith(k)]
        if matches:
            group_of[source] = max(matches, key=len)
        elif "default" in source_weights:
            group_of[source] = "default"
        else:
            raise ValueError(f"source {source!r} matches no data.source_weights key and there is no 'default'")
    unused = [k for k in keys if k not in group_of.values()]
    if unused:
        raise ValueError(f"data.source_weights keys {unused} matches no source in stats.json counts")
    ends = np.cumsum(list(counts.values()))
    source_idx = np.searchsorted(ends, np.asarray(train_rows), side="right")
    names = np.array([group_of[s] for s in counts])
    groups = names[source_idx]
    present = [g for g in dict.fromkeys(group_of.values()) if (groups == g).any()]
    total = sum(source_weights[g] for g in present)
    return groups, {g: source_weights[g] / total for g in present}


def source_row_probs(packed_dir, train_rows, source_weights) -> np.ndarray:
    """Per-training-row sampling probability: each source group gets its mass, uniform within the group."""
    groups, mass = _source_groups(packed_dir, train_rows, source_weights)
    p = np.zeros(len(groups))
    for g, m in mass.items():
        sel = groups == g
        p[sel] = m / sel.sum()
    return p


def source_mass(packed_dir, train_rows, source_weights) -> dict:
    """Realized probability mass per source group (masses are renormalized over groups with rows)."""
    return _source_groups(packed_dir, train_rows, source_weights)[1]


class SourceWeightedSampler(Sampler):
    """Each epoch draws len(probs) dataset positions with replacement from `probs`."""

    def __init__(self, probs, seed=0):
        self.probs = np.asarray(probs, dtype=np.float64)
        self.probs = self.probs / self.probs.sum()
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.probs)

    def __iter__(self):
        n = len(self.probs)
        return iter(np.random.default_rng(self.seed + self.epoch).choice(n, size=n, p=self.probs).tolist())


def subsample(paths, max_samples, seed):
    """A seeded subset (kept in manifest order) for dev runs; all paths when max_samples is None."""
    if max_samples is None or max_samples >= len(paths):
        return list(paths)
    idx = np.sort(np.random.default_rng(seed).choice(len(paths), max_samples, replace=False))
    return [paths[i] for i in idx]


def make_loader(dataset, batch_size, shuffle, seed, num_workers, drop_last, sampler=None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)  # makes the shuffle order reproducible
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      drop_last=drop_last, num_workers=num_workers, generator=generator,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
