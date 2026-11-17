"""Evaluate the multichannel encoders (configs/multichannel_m.yaml, seeds 107/108) against LeJEPA bm100 comp0+1.

Per measured set: the labeled samples with finite comp0 and comp1 (source pickle rows, row = spec index; the
Appendix K samples). Arms, all on the same nested-CV folds (2x5 outer, 5 inner row-level, 12 recipes, seed 42):
- mc: block-2 flat of the [8, 245] input (absent channels NaN, as in pretraining; channel validity as
  ops/build_multichannel.py), mean of the two seeds' predictions;
- mc_c01: the same encoders with the complex triplets withheld (NaN) -- what the triplets add inside the model;
- mc_random: a random-weight multichannel encoder (seed 0), the control;
- j_c01: bm100 s107/s108 block-2 flat [comp0 | comp1] (Appendix K's LeJEPA concat), mean of the two seeds;
- raw_c01: raw [comp0 | comp1].
Rule (fixed before results, commit a134c9c): mc beats j_c01 if mc − j_c01 > 0 on >= 5/7 sets, mean ΔR² >= +.01,
and no set below j_c01 by more than .01.

    PYTHONPATH=src:. python ops/multichannel_eval.py --set dataset0055 --runs <mc_s107 dir> <mc_s108 dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ops.build_multichannel import load_set
from spectral_lejepa.evaluation.bank import extract_bank
from spectral_lejepa.evaluation.nested import nested_cv, paired_delta
from spectral_lejepa.evaluation.probe import r2
from spectral_lejepa.evaluation.run import model_input
from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
from spectral_lejepa.training.checkpoint import load_model

ROOT = Path('/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression')
BM100 = ['/mnt5/home/hadar/nova/SpectralFM-jepa-budget/outputs/measured-budget/bm100_s107_20261112-201740/checkpoint_last.pt',
         '/mnt5/home/hadar/nova/SpectralFM-jepa-budget/outputs/measured-budget/bm100_s108_20261113-072559/checkpoint_last.pt']


def load(ds):
    """(X [n, 8, 245] raw with NaN = absent, y [n]) for labeled rows with comp0 and comp1 present."""
    X = load_set(ROOT / ds / ds)
    y = pd.read_pickle(ROOT / ds / f'{ds}_ref')['parameter_0'].to_numpy()
    ok = np.isfinite(y) & ~np.isnan(X[:, :2]).any((1, 2))
    return X[ok], y[ok].astype(np.float64)


def block2(model, Z, device):
    return extract_bank(EvalBackbone(model), Z, device=device, batch_size=64, readouts=('flat',),
                        flat_blocks=[2])['layer2/flat']


def mc_input(X, cfg):
    s = cfg['derived']['channel_stats']
    return ((X - np.array(s['mean'], np.float32)[:, None]) / np.array(s['std'], np.float32)[:, None]).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--set', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='multichannel run dirs (checkpoint_last.pt inside)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, default=Path('outputs/multichannel/eval'))
    a = ap.parse_args(argv)
    X, y = load(a.set)
    print(a.set, 'n', len(y), 'with triplet 1', int((~np.isnan(X[:, 2]).all(1)).sum()),
          'triplet 2', int((~np.isnan(X[:, 5]).all(1)).sum()), flush=True)
    seeds, arms = [], {'raw_c01': np.concatenate([X[:, 0], X[:, 1]], 1)}
    for i, run in enumerate(a.runs):
        model, ck = load_model(Path(run) / 'checkpoint_last.pt')
        Z = mc_input(X, ck['config'])
        Zc = Z.copy()
        Zc[:, 2:] = np.nan
        arms[f'mc@{i}'], arms[f'mc_c01@{i}'] = block2(model, Z, a.device), block2(model, Zc, a.device)
        if i == 0:
            torch.manual_seed(0)
            rnd = build_model(ck['config']['model'], ck['config']['data']['sequence_length']).eval()
            arms['mc_random'] = block2(rnd, Z, a.device)
    for i, ck_path in enumerate(BM100):
        model, ck = load_model(ck_path)
        arms[f'j_c01@{i}'] = np.concatenate([block2(model, model_input(X[:, c], ck['config']), a.device) for c in (0, 1)], 1)
    res = nested_cv(arms, {k: [k] for k in arms}, y, seed=42, backend='torch', device=a.device)
    oof = {k: res[k]['oof'] for k in arms}
    for base in ('mc', 'mc_c01', 'j_c01'):
        oof[base] = np.mean([oof.pop(k) for k in list(oof) if k.startswith(base + '@')], 0)
    score = {k: float(np.mean([r2(y, p) for p in P])) for k, P in oof.items()}
    deltas = {f'{a_}-{b}': paired_delta(y, oof[a_], oof[b]) for a_, b in
              (('mc', 'j_c01'), ('mc', 'mc_c01'), ('mc', 'raw_c01'), ('mc_c01', 'j_c01'))}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / f'{a.set}_oof.npz', y=y, **oof)
    (a.out / f'{a.set}.json').write_text(json.dumps({'n': len(y), 'r2': score, 'deltas': deltas,
                                                     'chosen': {k: res[k]['chosen'] for k in arms}}))
    print(a.set, ' '.join(f'{k} {v:.3f}' for k, v in score.items()))


if __name__ == '__main__':
    main()
