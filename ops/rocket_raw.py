"""Side experiment: ROCKET features of the raw spectra vs raw ridge vs the robust JEPA ensemble.

ROCKET (Dempster et al. 2020): random 1-D convolution kernels, two features each (proportion of
positive values and max). Kernels are drawn once from a fixed seed and never see labels. The ROCKET
arm goes through the same nested CV as raw and the JEPA members (2x5 outer, 5 inner, same recipes,
seed 42), on the same rows (read from a member's bank.npz), so every comparison is paired by fold.

    python ops/rocket_raw.py [--kernels 10000] [--out outputs/rocket-raw]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spectral_lejepa.evaluation.ensemble import ensemble_predictions
from spectral_lejepa.evaluation.nested import nested_cv, paired_delta

MEMBERS = [*sorted(Path('outputs/robust-ensemble').glob('robust_*/eval/*_step*')),
           *Path('outputs/robustness-continuation/robust_s401/eval').glob('*_step*')]
SETS = ('labeled_data', 'dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113',
        'dataset0114', 'dataset0120')


def rocket_kernels(n, length=245, seed=0):
    """Kernel parameters as in the reference ROCKET: length in {7, 9, 11}, centred N(0,1) weights,
    bias U(-1, 1), exponential dilation, padding on half the kernels."""
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n):
        k = int(rng.choice((7, 9, 11)))
        w = rng.normal(size=k)
        dil = int(2 ** rng.uniform(0, np.log2((length - 1) / (k - 1))))
        pad = (k - 1) * dil // 2 if rng.integers(2) else 0
        ks.append((k, dil, pad, w - w.mean(), rng.uniform(-1, 1)))
    return ks


@torch.no_grad()
def rocket_features(x, kernels, device='cuda', batch=1024):
    """x [N, L] -> [N, 2 * len(kernels)]: (ppv, max) per kernel. Kernels sharing (length, dilation,
    padding) run as one conv1d."""
    groups = {}
    for i, (k, dil, pad, w, b) in enumerate(kernels):
        groups.setdefault((k, dil, pad), []).append(i)
    out = np.zeros((len(x), 2 * len(kernels)), np.float32)
    xt = torch.as_tensor(x, dtype=torch.float32)
    for (k, dil, pad), idx in groups.items():
        W = torch.tensor(np.stack([kernels[i][3] for i in idx]), dtype=torch.float32, device=device)[:, None]
        B = torch.tensor([kernels[i][4] for i in idx], dtype=torch.float32, device=device)
        for s in range(0, len(x), batch):
            h = F.conv1d(xt[s:s + batch, None].to(device), W, B, padding=pad, dilation=dil)   # [b, K, L']
            out[s:s + batch, [2 * i for i in idx]] = (h > 0).float().mean(-1).cpu().numpy()
            out[s:s + batch, [2 * i + 1 for i in idx]] = h.amax(-1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernels', type=int, default=10000)
    ap.add_argument('--out', default='outputs/rocket-raw')
    ap.add_argument('--sets', nargs='*', default=SETS)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kernels = rocket_kernels(a.kernels)
    rows = {}
    for ds in a.sets:
        bank = np.load(MEMBERS[0] / ds / 'bank.npz', allow_pickle=True)
        raw, y = bank['input_raw'][:, 0], bank['y']
        X = rocket_features(raw, kernels)
        res = nested_cv({'rocket': X, 'raw': raw.astype(np.float32)}, {'rocket': ['rocket'], 'raw': ['raw']}, y,
                        seed=42, backend='torch', device='cuda')
        oofs = [np.load(m / ds / 'nested_oof.npz') for m in MEMBERS]
        if any(not np.array_equal(o['y'], y) for o in oofs):
            raise ValueError(f'{ds}: row order differs from the members')
        jepa, raw_ref = ensemble_predictions(oofs), oofs[0]['raw']
        rows[ds] = {'n': int(len(y)), 'raw': float(np.mean([_r2(y, p) for p in raw_ref])),
                    'raw_rerun': res['raw']['r2_mean'], 'rocket': res['rocket']['r2_mean'],
                    'jepa': float(np.mean([_r2(y, p) for p in jepa])),
                    'rocket_vs_raw': paired_delta(y, res['rocket']['oof'], raw_ref),
                    'rocket_vs_jepa': paired_delta(y, res['rocket']['oof'], jepa),
                    'rocket_chosen': [c['recipe'] if 'recipe' in c else c for c in res['rocket']['chosen']]}
        np.savez_compressed(out / f'{ds}_oof.npz', y=y, rocket=res['rocket']['oof'], raw=res['raw']['oof'])
        r = rows[ds]
        print(f"{ds:13s} n={r['n']:5d} raw {r['raw']:.3f} (rerun {r['raw_rerun']:.3f})  rocket {r['rocket']:.3f}"
              f"  jepa {r['jepa']:.3f}", flush=True)
        (out / 'results.json').write_text(json.dumps({'kernels': a.kernels, 'members': [str(m) for m in MEMBERS],
                                                      'sets': rows}, indent=1, default=str))


def _r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


if __name__ == '__main__':
    main()
