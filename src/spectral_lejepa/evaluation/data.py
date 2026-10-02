"""Labeled spectra for the downstream parameter_0 probe.

Copied semantics of clean-eval `data_loader.load_labeled_data(comps=(0,))` (+ the merged
set built by `merge_label_sets.py`), commit 6237feb:
- each set dir has labels.tsv (`filename<TAB>parameter_0`) and wavs under wav/, wavs/ or itself;
- filenames are `dataset<D>_comp<C>_spec_<S>.wav`; component 0 is the probe input;
- spectra are ordered by (dataset, spec) across ALL given dirs (merging = union of sets);
- more than max_samples spectra -> seeded subsample (default_rng(seed).choice, sorted);
- each wav is zero-padded / truncated to 245 points; spectra whose wav is missing are skipped.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import soundfile as sf

LABELED_PATTERN = re.compile(r"dataset(\d+)_comp(\d+)_spec_(\d+)\.wav")
COMPONENT = 0


def _wav_root(set_dir):
    for sub in ("wav", "wavs"):
        cand = os.path.join(set_dir, sub)
        if glob.glob(os.path.join(cand, "*.wav")):
            return cand
    return set_dir


def normalize_like_fairseq(arr: np.ndarray) -> np.ndarray:
    """The eval's per-row z-score (eps 1e-8 on the std), exactly as clean-eval feeds backbones."""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32)


def load_labeled_data(set_dirs, max_samples=5000, seed=42, target_length=245):
    files, labels, owner = {}, {}, {}   # (dataset, spec) -> wav path / label / set dir
    for set_dir in set_dirs:
        wav_root = _wav_root(set_dir)
        with open(os.path.join(set_dir, "labels.tsv")) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2 or (m := LABELED_PATTERN.match(parts[0])) is None:
                    continue
                ds, comp, spec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if comp != COMPONENT:
                    continue
                key = (ds, spec)
                if key in owner and owner[key] != set_dir:
                    raise ValueError(f"dataset {ds}, spec {spec} appears in both {owner[key]} and {set_dir}")
                owner[key] = set_dir
                path = os.path.join(wav_root, parts[0])
                files[key] = path if os.path.isfile(path) else os.path.join(set_dir, parts[0])
                labels[key] = float(parts[1])

    keys = sorted(files)
    if not keys:
        raise RuntimeError(f"no component-{COMPONENT} spectra under {set_dirs}")
    if len(keys) > max_samples:
        idx = np.random.default_rng(seed).choice(len(keys), max_samples, replace=False)
        idx.sort()
        keys = [keys[i] for i in idx]

    rows, ys, missing = [], [], 0
    for key in keys:
        if not os.path.isfile(files[key]):
            missing += 1
            continue
        data = np.asarray(sf.read(files[key], dtype="float32")[0]).flatten()
        row = np.zeros(target_length, dtype=np.float32)
        row[: min(len(data), target_length)] = data[:target_length]
        rows.append(row)
        ys.append(labels[key])
    if not rows:
        raise RuntimeError(f"no labeled wavs found under {set_dirs}")
    print(f"[eval data] {len(rows)} spectra from {len(set_dirs)} set(s); {missing} labels without a wav skipped")
    return np.stack(rows), np.array(ys, dtype=np.float64)
