"""Whole-stack subsets of a fairseq manifest, selected exactly as the previous project did; raw spectra."""
import os

import numpy as np
import soundfile as sf

REMAPS = [('/storage/noy/', '/mnt5/noy/'), ('/storage/', '/mnt5/'), ('/mnt5/noy/', '/storage/noy/'), ('/mnt5/', '/storage/')]


def _root(root):
    if os.path.isdir(root):
        return root
    for src, dst in REMAPS:
        if root.startswith(src) and os.path.isdir(dst + root[len(src):]):
            return dst + root[len(src):]
    return root


def load_stack_subset(dataset_dir, split, n, seed=42, stack_size=10, length=245):
    """{'raw': [N, length] float32, 'stack_idx': [N], 'filename': [N]}; short wavs are zero-padded.
    Unlike the previous loader, an unreadable wav raises instead of being skipped."""
    with open(os.path.join(dataset_dir, f'{split}.tsv')) as f:
        root = _root(f.readline().strip())
        rows = [ln.strip().split('\t')[0] for ln in f if ln.strip()]
    n_total = len(rows) // stack_size
    chosen = np.sort(np.random.default_rng(seed).choice(n_total, max(1, min(n // stack_size, n_total)), replace=False))
    raw, stack_idx, names = [], [], []
    for s, b in enumerate(chosen):
        for name in rows[b * stack_size:(b + 1) * stack_size]:
            data = np.asarray(sf.read(os.path.join(root, name), dtype='float32')[0]).reshape(-1)
            row = np.zeros(length, np.float32)
            row[:min(len(data), length)] = data[:length]
            raw.append(row)
            stack_idx.append(s)
            names.append(name)
    return {'raw': np.stack(raw), 'stack_idx': np.array(stack_idx), 'filename': names}
