"""Two label-free/fixed-probe checks on feature fragility.

Interpretation rule (fixed BEFORE any numbers were seen):
  Memorization is supported if trained-checkpoint feature drift is clearly higher (> 1.5x median ratio)
  on heavily oversampled rows (labeled_regression/*) than on low-rate rows (default mass groups) while the
  random-weight control shows no such gap (ratio <= ~1.2). Otherwise not supported.

Check 1 (probe vs features): perturbed-R2 with a FIXED probe, standardize (fit on clean train rows) +
  RidgeCV(alphas=logspace(-3,6,19)), same outer folds as primary (KFold(5, shuffle, 42+r), 2 repeats),
  clean-fit / perturbed-test, noise 1% and 3% with the robustness script's seeded noise. Raw vs seed2 block2-flat.
Check 2 (memorization signature): relative block0/2/6-flat drift under noise 1%, 3% and +-1 bin shift
  on ~2000 rows from labeled_regression/* vs ~2000 low-rate default rows; Spearman vs per-row sampling rate;
  masked-prediction MSE (predictor vs target tokens, fixed 75% random mask) as an extra.
Run: PYTHONPATH=src:. .venv/bin/python scripts/memorization_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import spectral_lejepa.evaluation.probe_torch as pt
from scripts.evaluate import model_input
from scripts.robustness_perturbations import (DATASETS, N_FOLDS, N_REPEATS, PERTURB_SEED, PERTURBATIONS, ROOT, SEED,
                                              perturb)
from spectral_lejepa.data.loader import _source_groups, packed_train_rows, source_row_probs
from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.probe import r2
from spectral_lejepa.models.masking import random_mask
from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
from spectral_lejepa.training.checkpoint import load_model

OUT = Path('outputs/production-validation/memorization')
PACK = Path('/mnt5/home/hadar/nova/data/spectra_all')
SW = {'labeled_regression': 0.5, 'default': 0.5}
NOISE = ['noise_1pct', 'noise_3pct']
pt.ALPHAS = np.logspace(-3, 6, 19)   # fixed probe grid (ridge_gcv reads the module global)
RECIPE = ('standardize', 'ridgecv')
CKPTS = {'seed2': 'outputs/paired-long/p48_s2_20261008-161508/checkpoint_last.pt',
         'seed101': 'outputs/final-model/production_p48_s101_20261010-024928/checkpoint_last.pt'}
N_PER = 2000
DEV = 'cuda'


@torch.no_grad()
def hidden_flat(bb, signals, layers=(0, 2, 6), bs=128):
    out = {l: [] for l in layers}
    for i in range(0, len(signals), bs):
        x = torch.as_tensor(signals[i:i + bs], dtype=torch.float32, device=DEV)
        hs = bb(x).hidden_states
        for l in layers:
            out[l].append(hs[l].reshape(len(x), -1).float().cpu().numpy())
    return {l: np.concatenate(v) for l, v in out.items()}


def fixed_oof(feats, y):
    """{variant: oof [R, n]}: fit standardize+ridge on clean train rows, predict every variant's test rows."""
    names = list(feats)
    oof = {k: np.zeros((N_REPEATS, len(y))) for k in names}
    for r in range(N_REPEATS):
        for f, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(y)):
            X_te = np.concatenate([feats[k][te] for k in names])
            p = pt.recipe_predictions(feats['clean'][tr], y[tr], X_te, [RECIPE], seed=SEED + 1000 + 10 * r + f,
                                      device=DEV)[RECIPE]
            for j, k in enumerate(names):
                oof[k][r, te] = p[j * len(te):(j + 1) * len(te)]
    return {k: float(np.mean([r2(y, p) for p in v])) for k, v in oof.items()}


def check1(models):
    manifest = json.loads((ROOT / 'manifest.json').read_text())['jobs']
    res = {'recipe': 'standardize+ridgecv alphas logspace(-3,6,19)', 'datasets': {}}
    for di, ds in enumerate(DATASETS):
        _, raw, y, _ = load_bank(Path(manifest['prior_s2']['bank_root']) / ds / 'bank.npz', readouts=('flat',))
        y = np.asarray(y, np.float64)
        V = {'clean': raw}
        for i, name in enumerate(PERTURBATIONS):   # same index i/seed as the robustness script
            if name in NOISE:
                V[name] = perturb(raw, *PERTURBATIONS[name], np.random.default_rng([PERTURB_SEED, di, i]))
        m, ck = models['seed2']
        bb = EvalBackbone(m).to(DEV).eval()
        F = {k: hidden_flat(bb, model_input(v, ck['config']), layers=(2,))[2] for k, v in V.items()}
        res['datasets'][ds] = {'n': len(y), 'raw': fixed_oof(V, y), 'seed2_block2_flat': fixed_oof(F, y)}
        print(ds, res['datasets'][ds], flush=True)
    lin = json.loads((ROOT / 'prior_s2/linear_summary.json').read_text())['sets']['labeled_data']['families']['layer2/flat']
    from collections import Counter
    res['labeled_data_original_inner_choice_layer2_flat'] = dict(Counter(f"{c['norm']}+{c['probe']}" for c in lin['chosen']))
    rawc = json.loads((Path(manifest['prior_s2']['bank_root']) / 'labeled_data/nested_results.json').read_text())['families']['raw']['chosen']
    res['labeled_data_original_inner_choice_raw'] = dict(Counter(f"{c['norm']}+{c['probe']}" for c in rawc))
    return res


def rel_drift(a, b):
    return np.linalg.norm(b - a, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)


def check2(models):
    stats = json.loads((PACK / 'stats.json').read_text())
    counts = stats['counts']; names = list(counts)
    ends = np.cumsum(list(counts.values())); starts = ends - np.array(list(counts.values()))
    arr = np.load(PACK / 'spectra.npy', mmap_mode='r')
    train = packed_train_rows(PACK)
    probs = source_row_probs(PACK, train, SW)
    groups, mass = _source_groups(PACK, train, SW)
    N = len(train)
    rng = np.random.default_rng(0)
    src_of = names_arr = np.array(names)[np.searchsorted(ends, train, side='right')]
    lr = np.flatnonzero(groups == 'labeled_regression')
    df = np.flatnonzero(groups == 'default')
    pick = {'lr': rng.choice(lr, N_PER, replace=False), 'default': rng.choice(df, N_PER, replace=False)}
    idx = np.concatenate([pick['lr'], pick['default']])
    rows = train[idx]
    raw = np.stack([np.asarray(arr[r], np.float32) for r in rows])
    rate = probs[idx] * N                          # oversampling factor vs uniform
    source = src_of[idx]
    grp = np.array([f"labeled_regression/*" if s.startswith('labeled_regression') else s for s in source])
    meta = {'mass': mass, 'n_train_rows': int(N), 'oversampling_factor': {
        'labeled_regression': float(probs[lr[0]] * N), 'default': float(probs[df[0]] * N)}}
    arms = {}
    for name, p in CKPTS.items():
        arms[name] = models[name]
    cfg101 = models['seed101'][1]['config']
    torch.manual_seed(0)
    arms['random'] = (build_model(cfg101['model'], 245).eval(), models['seed101'][1])
    # variants (raw units): clean, noise, shift
    V = {'clean': raw}
    for i, key in enumerate(['noise_1pct', 'noise_3pct', 'shift_+1', 'shift_-1']):
        V[key] = perturb(raw, *PERTURBATIONS[key], np.random.default_rng([PERTURB_SEED, 99, i]))
    V['shift'] = None
    out = {'meta': meta, 'arms': {}}
    gen = torch.Generator().manual_seed(0)
    mi, vi = random_mask(len(raw), 48, 0.75, gen)
    for arm, (m, ck) in arms.items():
        m = m.to(DEV).eval(); bb = EvalBackbone(m)
        H = {k: hidden_flat(bb, model_input(v, ck['config'])) for k, v in V.items() if v is not None}
        # prediction error (per row)
        with torch.no_grad():
            mse = []
            x_all = torch.as_tensor(model_input(raw, ck['config']))
            for i in range(0, len(x_all), 128):
                x = x_all[i:i + 128].to(DEV)
                o = m(x, mi[i:i + 128].to(DEV), vi[i:i + 128].to(DEV))
                mse.append(((o['predicted'].float() - o['target_masked'].float()) ** 2).mean((1, 2)).cpu().numpy())
            mse = np.concatenate(mse)
        res = {}
        for L in (0, 2, 6):
            d = {'noise_1pct': rel_drift(H['clean'][L], H['noise_1pct'][L]),
                 'noise_3pct': rel_drift(H['clean'][L], H['noise_3pct'][L]),
                 'shift_1': 0.5 * (rel_drift(H['clean'][L], H['shift_+1'][L]) + rel_drift(H['clean'][L], H['shift_-1'][L]))}
            for pn, v in d.items():
                per = {g: float(np.median(v[grp == g])) for g in dict.fromkeys(grp)}
                lrm = float(np.median(v[np.isin(np.arange(len(v)), np.arange(N_PER))]))
                dfm = float(np.median(v[N_PER:]))
                res[f'block{L}/{pn}'] = {'median_by_group': per, 'median_lr': lrm, 'median_default': dfm,
                                          'ratio_lr_over_default': lrm / dfm,
                                          'spearman_vs_rate': float(spearmanr(rate, v).statistic)}
        res['mask_pred_mse'] = {'median_lr': float(np.median(mse[:N_PER])), 'median_default': float(np.median(mse[N_PER:])),
                                'by_group': {g: float(np.median(mse[grp == g])) for g in dict.fromkeys(grp)}}
        out['arms'][arm] = res
        print(arm, json.dumps({k: (v['ratio_lr_over_default'] if 'ratio_lr_over_default' in v else v['median_lr'] / v['median_default'])
                              for k, v in res.items()}), flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = {n: load_model(p) for n, p in CKPTS.items()}
    r1 = check1(models)
    (OUT / 'check1_fixed_probe.json').write_text(json.dumps(r1, indent=2))
    r2_ = check2(models)
    (OUT / 'check2_memorization.json').write_text(json.dumps(r2_, indent=2))


if __name__ == '__main__':
    main()
