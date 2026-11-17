"""Build the multichannel measured pretraining set: one [8, 245] input per measured sample (row of a
labeled_regression source pickle), channels [comp0, comp1, N1, C1, S1, N2, C2, S2] = components
[0, 1, 26, 27, 28, 29, 30, 31]. (N, C, S) = (cos 2ψ, sin 2ψ cos Δ, sin 2ψ sin Δ), so z = C + iS.

An absent channel is NaN: missing component, non-finite or constant row, or (for a triplet) any |value| > 1.05
(ellipsometric N, C, S lie in [-1, 1]; larger values are corrupt, e.g. N up to 23,045 on 0120). A triplet is
kept or dropped whole. Rows with no channel present are dropped. Labels are never read.

    PYTHONPATH=src:. python ops/build_multichannel.py [--root .../labeled_regression] [--out .../multichannel_measured]

Writes signals.npy [N, 8, 245] float32, rows.tsv (`index<TAB>dataset<TAB>row`), stats.json (per-channel mean /
std over present values, presence counts).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression')
OUT = Path('/mnt5/home/hadar/nova/data/multichannel_measured')
COMPONENTS = (0, 1, 26, 27, 28, 29, 30, 31)
TRIPLETS = ((2, 3, 4), (5, 6, 7))     # channel positions of (N, C, S)
LENGTH = 245
BOUND = 1.05


def load_set(path):
    df = pd.read_pickle(path)
    cols = [c for c in df.columns if ':' in str(c)]
    n = len(df)
    X = np.full((n, len(COMPONENTS), LENGTH), np.nan, np.float32)
    if not cols:
        return X
    pre = cols[0].split(':')[0].rsplit('_', 1)[0]
    have = {int(c.split(':')[0].rsplit('_', 1)[1]) for c in cols}
    for j, c in enumerate(COMPONENTS):
        if c in have:
            a = df[[f'{pre}_{c}:{i}' for i in range(LENGTH)]].to_numpy(np.float64)
            bad = ~np.isfinite(a).all(1) | (np.nan_to_num(a).std(1) == 0)
            a[bad] = np.nan
            X[:, j] = a
    for t in TRIPLETS:
        bad = np.isnan(X[:, t]).any((1, 2)) | (np.nan_to_num(np.abs(X[:, t])) > BOUND).any((1, 2))
        X[np.ix_(bad, t)] = np.nan
    return X


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args(argv)
    sigs, rows = [], []
    for d in sorted(p for p in a.root.iterdir() if (p / p.name).is_file()):
        X = load_set(d / d.name)
        keep = np.flatnonzero(~np.isnan(X).all((1, 2)))
        sigs.append(X[keep])
        rows += [(d.name, int(r)) for r in keep]
        print(d.name, 'rows', len(X), 'kept', len(keep), 'present per channel', (~np.isnan(X[keep]).all(2)).sum(0).tolist())
    S = np.concatenate(sigs)
    present = ~np.isnan(S).all(2)
    mean = [float(np.nanmean(S[:, j])) for j in range(S.shape[1])]
    std = [float(np.nanstd(S[:, j])) for j in range(S.shape[1])]
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / 'signals.npy', S)
    (a.out / 'rows.tsv').write_text(''.join(f'{i}\t{ds}\t{r}\n' for i, (ds, r) in enumerate(rows)))
    (a.out / 'stats.json').write_text(json.dumps({'components': COMPONENTS, 'mean': mean, 'std': std, 'n': len(S),
                                                  'present': present.sum(0).tolist()}, indent=1))
    print('total', len(S), 'present per channel', present.sum(0).tolist())


if __name__ == '__main__':
    main()
