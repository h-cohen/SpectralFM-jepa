"""Stress test of one checkpoint's block-2 flat features under a fixed perturbation grid.

Grid (fixed before any result): noise 1/3/10/20 % x 3 draws, offset +-5 %, slope +-5 %, scale 0.95/1.05,
shift +-1 bin. Probes are fit on CLEAN rows only (production recipe chosen by inner CV, and a fixed
standardize + RidgeCV) and predict every variant. Per variant: development score (inner CV inside each outer
training set), outer-test R² and RMSE, and displacement = mean |pred_variant - pred_clean| on outer-test rows.
Usage: python -m spectral_lejepa stress --checkpoint CKPT --output OUT.json [--controls] [--device cuda]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectral_lejepa.evaluation import probe_torch
from spectral_lejepa.evaluation.nested import _inner_scores
from spectral_lejepa.evaluation.perturbation import DATASETS, block2_flat, perturb
from spectral_lejepa.evaluation.perturbation_eval import ALPHAS, BANKS
from spectral_lejepa.evaluation.probe import r2
from spectral_lejepa.evaluation.run import model_input
from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training.checkpoint import load_model
from spectral_lejepa.utils.io import write_json

N_REPEATS, N_FOLDS, N_INNER, SEED = 2, 5, 5, 42

# ---- FIXED before any scoring; do not tune ----
GRID = {}
for _pct in (1, 3, 10, 20):
    for _d in range(3):
        GRID[f'noise_{_pct}pct_d{_d}'] = ('noise', _pct / 100)
for _kind in ('offset', 'slope'):
    for _sgn in (1, -1):
        GRID[f'{_kind}_{"+" if _sgn > 0 else "-"}5pct'] = (_kind, _sgn * .05)
GRID['scale_0.95'] = ('scale', .95)
GRID['scale_1.05'] = ('scale', 1.05)
GRID['shift_+1'] = ('shift', 1)
GRID['shift_-1'] = ('shift', -1)
# -----------------------------------------------


def variants(raw, ds_index):
    """{'clean': raw, name: perturbed}. Draw seed [2, ds_index, i] (i = grid position); noise adds its draw index d."""
    out = {'clean': raw}
    for i, (name, (kind, mag)) in enumerate(GRID.items()):
        seed = [2, ds_index, i] + ([int(name[-1])] if kind == 'noise' else [])
        out[name] = perturb(raw, kind, mag, np.random.default_rng(seed))
    return out


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def stress_arm(feats, y, device='cuda'):
    """feats {variant: [n, d]} incl. 'clean'. Returns {variant: {dev_r2, dev_r2_ridge, outer_r2, outer_rmse,
    outer_r2_ridge, displacement, displacement_ridge}}. Every fit uses clean rows only."""
    y = np.asarray(y, np.float64)
    names = list(feats)
    n = len(y)
    oof = {k: np.zeros((N_REPEATS, n)) for k in names}
    oof_rg = {k: np.zeros((N_REPEATS, n)) for k in names}
    dev = {k: [] for k in names}
    dev_rg = {k: [] for k in names}
    for r in range(N_REPEATS):
        for f, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(y)):
            seed = SEED + 1000 + 10 * r + f
            clean = feats['clean']
            inner = _inner_scores(clean[tr], y[tr], N_INNER, seed, 'torch', device)
            rc = max(inner, key=inner.get)
            X_te = np.concatenate([feats[k][te] for k in names])
            p = probe_torch.recipe_predictions(clean[tr], y[tr], X_te, [rc], seed=seed, device=device)[rc]
            rg = _ridge().fit(clean[tr], y[tr])
            for j, k in enumerate(names):
                oof[k][r, te] = p[j * len(te):(j + 1) * len(te)]
                oof_rg[k][r, te] = rg.predict(feats[k][te])
            # development score: same probes, inner folds of the training rows, clean fit, variant inner-test rows
            ip = {k: np.zeros(len(tr)) for k in names}
            ip_rg = {k: np.zeros(len(tr)) for k in names}
            for itr, ite in KFold(N_INNER, shuffle=True, random_state=seed).split(tr):
                a, b = tr[itr], tr[ite]
                X_in = np.concatenate([feats[k][b] for k in names])
                q = probe_torch.recipe_predictions(clean[a], y[a], X_in, [rc], seed=seed, device=device)[rc]
                rgi = _ridge().fit(clean[a], y[a])
                for j, k in enumerate(names):
                    ip[k][ite] = q[j * len(b):(j + 1) * len(b)]
                    ip_rg[k][ite] = rgi.predict(feats[k][b])
            for k in names:
                dev[k].append(r2(y[tr], ip[k]))
                dev_rg[k].append(r2(y[tr], ip_rg[k]))
    out = {}
    for k in names:
        out[k] = {'dev_r2': float(np.mean(dev[k])), 'dev_r2_ridge': float(np.mean(dev_rg[k])),
                  'outer_r2': float(np.mean([r2(y, p) for p in oof[k]])),
                  'outer_rmse': float(np.mean([np.sqrt(np.mean((y - p) ** 2)) for p in oof[k]])),
                  'outer_r2_ridge': float(np.mean([r2(y, p) for p in oof_rg[k]])),
                  'displacement': float(np.mean(np.abs(oof[k] - oof['clean']))),
                  'displacement_ridge': float(np.mean(np.abs(oof_rg[k] - oof_rg['clean'])))}
    return out


def stress(checkpoint, output, device='cuda', controls=False):
    model, ckpt = load_model(checkpoint, map_location='cpu')
    cfg = ckpt['config']
    random_model = None
    if controls:
        torch.manual_seed(0)
        random_model = build_model(cfg['model'], 245).eval()
    sets = {}
    for i, ds in enumerate(DATASETS):
        bank = np.load(BANKS / ds / 'bank.npz', allow_pickle=True)
        raw, y = bank['input_raw'][:, 0].astype(np.float32), bank['y']
        inputs = variants(raw, i)
        arms = {'trained': {k: block2_flat(model, model_input(v, cfg), device) for k, v in inputs.items()}}
        if controls:
            arms['raw'] = inputs
            arms['random'] = {k: block2_flat(random_model, model_input(v, cfg), device) for k, v in inputs.items()}
        sets[ds] = {arm: stress_arm(F, y, device) for arm, F in arms.items()}
        print(ds, json.dumps({a: round(s['clean']['outer_r2'], 4) for a, s in sets[ds].items()}), flush=True)
    write_json(output, {'checkpoint': str(checkpoint), 'grid': {k: list(v) for k, v in GRID.items()}, 'sets': sets})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--controls', action='store_true', help='also score the raw-spectrum and random-weight arms')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv)
    stress(a.checkpoint, a.output, a.device, a.controls)


if __name__ == '__main__':
    main()
