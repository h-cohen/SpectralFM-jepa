"""Do the complex (ellipsometric) channels help the parameter_0 probe? (exploratory study, extends report Appendix K)

EDA (2026-11-17): components 26-31 form two triplets (26, 27, 28) and (29, 30, 31) with N² + C² + S² ≈ 1 (median
.98-1.0 on 0055/0112/0113/0120/0115), i.e. N = cos 2ψ, C = sin 2ψ cos Δ, S = sin 2ψ sin Δ: the complex signal is
z = C + iS = sin 2ψ e^{iΔ}. Labeled samples with finite comp0 and 26-31 (read from the source pickles, row = spec
index): labeled_data 4716, 0055 25, 0106 64, 0109 96, 0112 50 (18 all-NaN), 0120 125; 0113 (all NaN) and 0114
(no such channels) are out.

Feature sets per sample (raw values; z_k = C_k + i S_k, triplet k = 1, 2):
- c0, c01 = [comp0 | comp1] (the Appendix K baselines);
- cx = [26 | 27 | ... | 31] (1470 values); pol = [N_k, |z_k|, unwrapped arg z_k] (1470);
- fft = per triplet Re/Im of FFT(z_k) over wavelength at |freq| <= 16 and of rFFT(N_k) at freq <= 16 (200 values);
- c0+cx, c0+pol, c0+fft, c01+cx;
- LeJEPA (measured-only bm100 s107/s108, block-2 flat of each channel run through the encoder, mean of the two
  seeds' predictions): j_c0, j_c01, j_cx (6 channels), j_c0+cx, j_c01+cx.
Nested CV as the main evaluation (2x5 outer, 5 inner row-level folds, 12 recipes, seed 42), every arm on the same
folds.

Rule fixed before results: channels X contribute if (base + X) − base > 0 on >= 4 of the 6 sets and the mean
ΔR² >= +.01, for base = c0 and c01 (raw) and j_c0, j_c01 (LeJEPA).

    PYTHONPATH=src:. python ops/complex_channels.py --set dataset0055 [--device cuda] [--out outputs/complex-channels]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spectral_lejepa.evaluation.bank import extract_bank
from spectral_lejepa.evaluation.nested import nested_cv
from spectral_lejepa.evaluation.run import model_input
from spectral_lejepa.models.vit_1d import EvalBackbone
from spectral_lejepa.training.checkpoint import load_model

ROOT = Path('/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression')
SOURCES = {'labeled_data': '/mnt5/noy/nova_samples/labeled_data_test/v3/dataset0022',
           **{d: str(ROOT / d / d) for d in ('dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0120')}}
CKPTS = ['/mnt5/home/hadar/nova/SpectralFM-jepa-budget/outputs/measured-budget/bm100_s107_20261112-201740/checkpoint_last.pt',
         '/mnt5/home/hadar/nova/SpectralFM-jepa-budget/outputs/measured-budget/bm100_s108_20261113-072559/checkpoint_last.pt']
CHANNELS = (0, 1, 26, 27, 28, 29, 30, 31)
K_FFT = 16
SEED = 42


def load(ds):
    """({channel: [n, 245] float32}, y [n], n_rows): labeled source rows with every channel finite."""
    df = pd.read_pickle(SOURCES[ds])
    y = pd.read_pickle(SOURCES[ds] + '_ref')['parameter_0'].to_numpy()
    pre = next(c for c in df.columns if ':' in str(c)).split(':')[0].rsplit('_', 1)[0]
    X = {c: df[[f'{pre}_{c}:{i}' for i in range(245)]].to_numpy(np.float64) for c in CHANNELS}
    ok = np.isfinite(y) & np.all([np.isfinite(a).all(1) for a in X.values()], 0)
    return {c: a[ok].astype(np.float32) for c, a in X.items()}, y[ok].astype(np.float64), len(df)


def derived(X):
    """pol [n, 1470] and fft [n, 200] from the two triplets (N, C, S) = (26, 27, 28), (29, 30, 31)."""
    pol, fft = [], []
    for n_, c_, s_ in ((26, 27, 28), (29, 30, 31)):
        N, z = X[n_].astype(np.float64), X[c_].astype(np.float64) + 1j * X[s_].astype(np.float64)
        pol += [N, np.abs(z), np.unwrap(np.angle(z), axis=1)]
        F = np.fft.fft(z, axis=1) / z.shape[1]
        F = np.concatenate([F[:, :K_FFT + 1], F[:, -K_FFT:]], axis=1)
        R = np.fft.rfft(N, axis=1)[:, :K_FFT + 1] / N.shape[1]
        fft += [F.real, F.imag, R.real, R.imag]
    return np.concatenate(pol, 1).astype(np.float32), np.concatenate(fft, 1).astype(np.float32)


def raw_arms(X):
    pol, fft = derived(X)
    cx = np.concatenate([X[c] for c in CHANNELS[2:]], 1)
    c01 = np.concatenate([X[0], X[1]], 1)
    return {'c0': X[0], 'c01': c01, 'cx': cx, 'pol': pol, 'fft': fft,
            'c0+cx': np.concatenate([X[0], cx], 1), 'c0+pol': np.concatenate([X[0], pol], 1),
            'c0+fft': np.concatenate([X[0], fft], 1), 'c01+cx': np.concatenate([c01, cx], 1)}


def jepa_arms(X, ckpt_path, device):
    model, ckpt = load_model(ckpt_path)
    F = {c: extract_bank(EvalBackbone(model), model_input(X[c], ckpt['config']), device=device, batch_size=64,
                         readouts=('flat',), flat_blocks=[2])['layer2/flat'] for c in CHANNELS}
    cx = np.concatenate([F[c] for c in CHANNELS[2:]], 1)
    return {'j_c0': F[0], 'j_c01': np.concatenate([F[0], F[1]], 1), 'j_cx': cx,
            'j_c0+cx': np.concatenate([F[0], cx], 1), 'j_c01+cx': np.concatenate([F[0], F[1], cx], 1)}


def run(arms, y, device):
    res = nested_cv(arms, {a: [a] for a in arms}, y, seed=SEED, backend='torch', device=device)
    return {a: res[a]['oof'] for a in arms}, {a: res[a]['chosen'] for a in arms}, res['test_folds']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--set', required=True, choices=list(SOURCES))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, default=Path('outputs/complex-channels'))
    a = ap.parse_args(argv)
    X, y, n_rows = load(a.set)
    print(a.set, 'labeled samples with finite channels', len(y), 'of', n_rows, 'rows', flush=True)
    oof, chosen, folds = run(raw_arms(X), y, a.device)
    print('raw arms done', flush=True)
    jo = []
    for ck in CKPTS:
        o, c, f = run(jepa_arms(X, ck, a.device), y, a.device)
        if f != folds:
            raise RuntimeError('folds differ between runs')
        jo.append(o)
        chosen |= {f'{k}@{Path(ck).parent.name[:10]}': v for k, v in c.items()}
        torch.cuda.empty_cache()
    oof |= {k: np.mean([o[k] for o in jo], 0) for k in jo[0]}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / f'{a.set}_oof.npz', y=y, **oof)
    (a.out / f'{a.set}_chosen.json').write_text(json.dumps({'n': len(y), 'n_rows': n_rows, 'chosen': chosen}))
    r2 = lambda p: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    print(a.set, ' '.join(f'{k} {np.mean([r2(p) for p in P]):.3f}' for k, P in oof.items()))


if __name__ == '__main__':
    main()
