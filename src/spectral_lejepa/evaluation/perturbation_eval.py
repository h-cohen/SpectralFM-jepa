"""Registered perturbation evaluation for one robustness-continuation job (or the starting checkpoint).

Fixed before any study result: A6 perturbations (src/spectral_lejepa/evaluation/perturbation.py, same seeds) plus
the held-out family below. Probe: standardize + RidgeCV(logspace(-3, 6, 19)) fit on CLEAN training rows
of the primary outer folds (seed42, 2 x 5), scored on perturbed test rows. Drift summary: per-row relative
block2-flat drift, median within each dataset, then median of the eight dataset medians.
Usage: python -m spectral_lejepa.evaluation.perturbation_eval {job|start}
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectral_lejepa.utils.io import write_json
from spectral_lejepa.evaluation.run import model_input
from spectral_lejepa.evaluation.perturbation import DATASETS, PERTURBATIONS, block2_flat, perturb
from spectral_lejepa.evaluation.probe import r2
from spectral_lejepa.training.checkpoint import load_model

ROOT = Path('outputs/robustness-continuation')
START = 'outputs/paired-long/p48_s2_20261008-161508/checkpoint_last.pt'
BANKS = Path('outputs/frozen-feature-study/prior_s2')   # same input_raw/y for every checkpoint
HELDOUT = {'shift_2': ('shift2', 2), 'noise_5pct': ('noise', .05), 'sine_baseline_5pct': ('sine', .05)}
ALPHAS = np.logspace(-3, 6, 19)
N_REPEATS, N_FOLDS, SEED = 2, 5, 42


def perturb_heldout(raw, kind, mag, rng):
    x = raw.astype(np.float64)
    if kind == 'noise':
        return perturb(raw, kind, mag, rng)
    if kind == 'shift2':
        return np.concatenate([x[:, 2:], x[:, -1:], x[:, -1:]], 1).astype(np.float32)
    spread = x.max(1, keepdims=True) - x.min(1, keepdims=True)
    return (x + mag * spread * np.sin(2 * np.pi * np.linspace(0, 1, x.shape[1]))[None]).astype(np.float32)


def variants(raw, ds_index):
    out = {'clean': raw}
    for i, (name, (kind, mag)) in enumerate(PERTURBATIONS.items()):   # identical draws to A6
        out[name] = perturb(raw, kind, mag, np.random.default_rng([0, ds_index, i]))
    for i, (name, (kind, mag)) in enumerate(HELDOUT.items()):
        out[name] = perturb_heldout(raw, kind, mag, np.random.default_rng([1, ds_index, i]))
    return out


def ridge_scores(feats, y):
    oof = {k: np.zeros((N_REPEATS, len(y))) for k in feats}
    for r in range(N_REPEATS):
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(y):
            probe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(feats['clean'][tr], y[tr])
            for k in feats:
                oof[k][r, te] = probe.predict(feats[k][te])
    return {k: float(np.mean([r2(y, p) for p in P])) for k, P in oof.items()}


def relative_drift(clean, other):
    return np.linalg.norm(other - clean, axis=1) / (np.linalg.norm(clean, axis=1) + 1e-12)


def evaluate(checkpoint, out):
    model, ckpt = load_model(checkpoint, map_location='cpu')
    sets = {}
    for i, ds in enumerate(DATASETS):
        bank = np.load(BANKS / ds / 'bank.npz', allow_pickle=True)
        raw, y = bank['input_raw'][:, 0].astype(np.float32), bank['y']
        inputs = variants(raw, i)
        trained = {k: block2_flat(model, model_input(v, ckpt['config'])) for k, v in inputs.items()}
        drift = {k: relative_drift(trained['clean'], f) for k, f in trained.items() if k != 'clean'}
        sets[ds] = {'n': len(y), 'trained_r2': ridge_scores(trained, y), 'raw_r2': ridge_scores(inputs, y),
                    'drift_median': {k: float(np.median(v)) for k, v in drift.items()},
                    'drift_mean': {k: float(np.mean(v)) for k, v in drift.items()}}
        print(ds, json.dumps(sets[ds]['drift_median']), flush=True)
    names = [k for k in sets[DATASETS[0]]['drift_median']]
    summary = {'drift_median_of_medians': {k: float(np.median([sets[d]['drift_median'][k] for d in DATASETS]))
                                           for k in names},
               'mean_r2_small_sets': {k: float(np.mean([sets[d]['trained_r2'][k] for d in DATASETS[1:]]))
                                      for k in ['clean', *names]}}
    write_json(out, {'checkpoint': str(checkpoint), 'sets': sets, 'summary': summary})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job')
    job = p.parse_args().job
    if job == 'start':
        evaluate(START, ROOT / 'start_perturbation.json')
    else:
        state = json.loads((ROOT / job / 'state.json').read_text())
        evaluate(state['checkpoint'], ROOT / job / 'perturbation.json')
