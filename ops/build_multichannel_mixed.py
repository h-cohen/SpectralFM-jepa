"""Build the mixed multichannel measured set: the whole-sample [8, 245] inputs of ops/build_multichannel.py plus
every single-component row of the same source pickles as its own input with one channel present, so the
encoder sees at least bm100's measured spectra (the whole-sample-only set has ~12x fewer inputs and its
encoders lost accuracy as training went on).

Single rows go to their own slot: components 0, 1, 26..31 to slots [0, 1, 2..7]; every other component to slot 0
(the generic spectrum channel). A single row is kept if finite and non-constant; a 26..31 row also needs
|value| <= 1.05. Labels are never read.

    PYTHONPATH=src:. python ops/build_multichannel_mixed.py [--root ...] [--out .../multichannel_mixed]

Writes signals.npy [N, 8, 245] float32 (whole-sample rows first), rows.tsv (`index<TAB>dataset<TAB>row<TAB>kind`,
kind = `all` or the component number), stats.json (per-channel mean / std over present values).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ops.build_multichannel import BOUND, COMPONENTS, LENGTH, ROOT, load_set

OUT = Path('/mnt5/home/hadar/nova/data/multichannel_mixed')


def singles(path):
    """[(component, rows [m], values [m, 245])] for every component of the pickle."""
    df = pd.read_pickle(path)
    cols = [c for c in df.columns if ':' in str(c)]
    if not cols:
        return []
    pre = cols[0].split(':')[0].rsplit('_', 1)[0]
    out = []
    for c in sorted({int(c.split(':')[0].rsplit('_', 1)[1]) for c in cols}):
        a = df[[f'{pre}_{c}:{i}' for i in range(LENGTH)]].to_numpy(np.float64)
        ok = np.isfinite(a).all(1) & (np.nan_to_num(a).std(1) > 0)
        if c >= 26:
            ok &= (np.abs(np.nan_to_num(a)) <= BOUND).all(1)
        out.append((c, np.flatnonzero(ok), a[ok].astype(np.float32)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args(argv)
    whole, single, rows_w, rows_s = [], [], [], []
    for d in sorted(p for p in a.root.iterdir() if (p / p.name).is_file()):
        X = load_set(d / d.name)
        keep = np.flatnonzero(~np.isnan(X).all((1, 2)))
        whole.append(X[keep])
        rows_w += [(d.name, int(r), 'all') for r in keep]
        n_s = 0
        for c, r, v in singles(d / d.name):
            S = np.full((len(r), len(COMPONENTS), LENGTH), np.nan, np.float32)
            S[:, COMPONENTS.index(c) if c in COMPONENTS else 0] = v
            single.append(S)
            rows_s += [(d.name, int(i), str(c)) for i in r]
            n_s += len(r)
        print(d.name, 'whole', len(keep), 'single', n_s, flush=True)
    S = np.concatenate(whole + single)
    rows = rows_w + rows_s
    present = ~np.isnan(S).all(2)
    mean = [float(np.nanmean(S[:, j])) for j in range(S.shape[1])]
    std = [float(np.nanstd(S[:, j])) for j in range(S.shape[1])]
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / 'signals.npy', S)
    (a.out / 'rows.tsv').write_text(''.join(f'{i}\t{ds}\t{r}\t{k}\n' for i, (ds, r, k) in enumerate(rows)))
    (a.out / 'stats.json').write_text(json.dumps({'components': COMPONENTS, 'mean': mean, 'std': std, 'n': len(S),
                                                  'n_whole': len(rows_w), 'present': present.sum(0).tolist()},
                                                 indent=1))
    print('total', len(S), 'whole', len(rows_w), 'present per channel', present.sum(0).tolist())


if __name__ == '__main__':
    main()
