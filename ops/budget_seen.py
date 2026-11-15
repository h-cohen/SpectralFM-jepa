"""Which evaluation spectra a measured-budget run saw in pretraining (spec
docs/superpowers/specs/2026-11-12-measured-budget-design.md); descriptive only.

    PYTHONPATH=src:. python ops/budget_seen.py BANK_ROOT PACKED_DIR OUT_JSON

Writes {set: [group key per bank row]} for every <set>/bank.npz under BANK_ROOT. Each bank row is matched
to the packed row with the identical float32 spectrum of the same set; the group key is
`datasetXXXX/<spectrum index>` as in `measured_groups.npy`.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.data.loader import _GROUP_NAME
from spectral_lejepa.evaluation.bank import load_bank


def bank_group_keys(bank_path, packed_dir) -> np.ndarray:
    bank_path = Path(bank_path)
    _, raw, _, meta = load_bank(bank_path, readouts=('flat',))
    ds = meta.get('set', bank_path.parent.name)
    rows, names = [], []
    with open(Path(packed_dir) / 'index.tsv') as f:   # `row<TAB>source<TAB>path`
        for line in f:
            if f'{ds}_' in line:
                row, _, path = line.rstrip('\n').split('\t', 2)
                rows.append(int(row))
                names.append(path.rsplit('/', 1)[-1])
    spectra = np.load(Path(packed_dir) / 'spectra.npy', mmap_mode='r')[np.array(rows)]
    by_bytes = {spectra[i].tobytes(): names[i] for i in range(len(rows))}
    keys = []
    for i, x in enumerate(np.asarray(raw, np.float32)):
        name = by_bytes.get(x.tobytes())
        if name is None:
            raise ValueError(f'{bank_path}: row {i} matches no packed {ds} spectrum')
        m = _GROUP_NAME.fullmatch(name)
        keys.append(f'{m[1]}/{m[2]}')
    return np.array(keys)


def seen_mask(keys, measured_groups) -> np.ndarray:
    return np.isin(np.asarray(keys), np.asarray(measured_groups))


def _r2(y, p):
    if len(y) < 2:
        return float('nan')
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def seen_r2(y, pred, mask) -> dict:
    """R² on seen and unseen rows; `pred` (n,) or (repeats, n), R² averaged over repeats."""
    y, mask = np.asarray(y, float), np.asarray(mask, bool)
    pred = np.atleast_2d(np.asarray(pred, float))
    part = lambda m: float(np.mean([_r2(y[m], p[m]) for p in pred]))
    return {'seen': part(mask), 'unseen': part(~mask), 'n_seen': int(mask.sum()), 'n_unseen': int((~mask).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bank_root', type=Path)
    ap.add_argument('packed_dir', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args(argv)
    out = {f.parent.name: bank_group_keys(f, a.packed_dir).tolist()
           for f in sorted(a.bank_root.glob('*/bank.npz')) if f.parent.name.startswith('dataset')}
    a.out.write_text(json.dumps(out))
    print({k: len(v) for k, v in out.items()})


if __name__ == '__main__':
    main()
