"""A6: robustness of frozen block-2 flat features (and raw spectra) to fixed spectral perturbations.

Probes are fit on CLEAN training rows with the recipe the inner CV already chose, then predict
PERTURBED test rows. Same outer folds/probe seeds as the primary benchmark.
Run: python -m spectral_lejepa perturbation (fixed checkpoint list of the 2026-10-13 production validation).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from spectral_lejepa.evaluation.run import model_input
from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.probe import r2
from spectral_lejepa.evaluation.probe_torch import recipe_predictions
from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training.checkpoint import load_model

# ---- FIXED before any scoring; do not tune ----
PERTURBATIONS = {
    'noise_1pct': ('noise', 0.01),   # sigma = 1% of each spectrum's std
    'noise_3pct': ('noise', 0.03),
    'offset_5pct': ('offset', 0.05),  # + 5% of spectrum range
    'slope_5pct': ('slope', 0.05),    # linear ramp, zero-mean, spanning 5% of range
    'scale_1.05': ('scale', 1.05),
    'shift_+1': ('shift', 1),         # roll by one bin, edge padded
    'shift_-1': ('shift', -1),
}
PERTURB_SEED = 0
# -----------------------------------------------

ROOT = Path('outputs/frozen-feature-study')
OUT = Path('outputs/production-validation/robustness')
DATASETS = ['labeled_data', 'dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113',
            'dataset0114', 'dataset0120']
TRAINED = ['prior_s0', 'prior_s1', 'prior_s2', 'baseline_s101']
READOUT = 'layer2/flat'
N_REPEATS, N_FOLDS, SEED = 2, 5, 42


def perturb(raw, kind, mag, rng):
    x = raw.astype(np.float64)
    rng_ = x.max(1, keepdims=True) - x.min(1, keepdims=True)
    if kind == 'noise':
        x = x + rng.standard_normal(x.shape) * mag * x.std(1, keepdims=True)
    elif kind == 'offset':
        x = x + mag * rng_
    elif kind == 'slope':
        x = x + mag * rng_ * np.linspace(-.5, .5, x.shape[1])[None]
    elif kind == 'scale':
        x = x * mag
    elif kind == 'shift':
        x = np.concatenate([x[:, 1:], x[:, -1:]], 1) if mag > 0 else np.concatenate([x[:, :1], x[:, :-1]], 1)
    return x.astype(np.float32)


def variants(raw, ds_index):
    out = {'clean': raw}
    for i, (name, (kind, mag)) in enumerate(PERTURBATIONS.items()):
        out[name] = perturb(raw, kind, mag, np.random.default_rng([PERTURB_SEED, ds_index, i]))
    return out


@torch.no_grad()
def block2_flat(model, signals, device='cuda', batch_size=64):
    model.to(device).eval()
    out = []
    for i in range(0, len(signals), batch_size):
        x = torch.as_tensor(signals[i:i + batch_size], dtype=torch.float32, device=device)
        _, hidden = model.encoder(model.tokenizer(x), return_all=True)
        out.append(hidden[1].reshape(len(x), -1).float().cpu().numpy())  # layer2 = hidden[1]; [B, 48*256]
    return np.concatenate(out)


def drift(f_clean, f_pert):
    return float(np.mean(np.linalg.norm(f_pert - f_clean, axis=1) / (np.linalg.norm(f_clean, axis=1) + 1e-12)))


def score_arm(feats, y, chosen, device):
    """feats {variant: [n, d]}; chosen: list of {repeat, fold, norm, probe}. Returns {variant: oof [R, n]}."""
    names = list(feats)
    n = len(y)
    oof = {k: np.zeros((N_REPEATS, n)) for k in names}
    choice = {(c['repeat'], c['fold']): (c['norm'], c['probe']) for c in chosen}
    for r in range(N_REPEATS):
        for f, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(y)):
            rc = choice[(r, f)]
            X_te = np.concatenate([feats[k][te] for k in names])
            p = recipe_predictions(feats['clean'][tr], y[tr], X_te, [rc], seed=SEED + 1000 + 10 * r + f,
                                   device=device)[rc]
            for j, k in enumerate(names):
                oof[k][r, te] = p[j * len(te):(j + 1) * len(te)]
    return oof


def metrics(y, oof, feats):
    clean = float(np.mean([r2(y, p) for p in oof['clean']]))
    res = {'clean_r2': clean}
    for k in oof:
        if k == 'clean': continue
        v = float(np.mean([r2(y, p) for p in oof[k]]))
        res[k] = {'r2': v, 'drop': clean - v, 'drift': drift(feats['clean'], feats[k])}
    return res


def main():
    device = torch.device('cuda')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'manifest.json').read_text())['jobs']
    results = {'perturbations': PERTURBATIONS, 'perturb_seed': PERTURB_SEED, 'readout': READOUT,
               'datasets': {}, 'repro': {}}
    models = {}
    for name in TRAINED:
        m, ckpt = load_model(manifest[name]['checkpoint']); models[name] = (m, ckpt)
    torch.manual_seed(0)
    rand_cfg = models['baseline_s101'][1]
    random_model = build_model(rand_cfg['config']['model'], 245).eval()

    for di, ds in enumerate(DATASETS):
        src = Path(manifest['prior_s2']['bank_root']) / ds
        _, raw, y, _ = load_bank(src / 'bank.npz', readouts=('flat',))
        y = np.asarray(y, np.float64)
        V = variants(raw, di)
        ref = np.load(ROOT / 'prior_s2' / ds / 'linear_oof.npz')
        raw_chosen = json.loads((src / 'nested_results.json').read_text())['families']['raw']['chosen']
        arms, repro, rec = {}, {}, {}
        # raw arm
        oof = score_arm(V, y, raw_chosen, device)
        arms['raw'] = metrics(y, oof, V)
        repro['raw'] = float(np.abs(oof['clean'] - ref['raw']).max())
        arms['raw']['input_drift_note'] = 'drift on input_raw'
        # random arm (s101 config, torch seed 0)
        sb = Path(manifest['baseline_s101']['bank_root']) / ds
        rc_chosen = json.loads((sb / 'random_control/nested_results.json').read_text())['families'][READOUT]['chosen']
        ref_b = np.load(ROOT / 'baseline_s101' / ds / 'linear_oof.npz')
        F = {k: block2_flat(random_model, model_input(v, rand_cfg['config']), device) for k, v in V.items()}
        oof = score_arm(F, y, rc_chosen, device)
        arms['random'] = metrics(y, oof, F)
        repro['random'] = float(np.abs(oof['clean'] - ref_b['random_' + READOUT]).max())
        del F
        for name in TRAINED:
            m, ck = models[name]
            F = {k: block2_flat(m, model_input(v, ck['config']), device) for k, v in V.items()}
            bank_saved, _, _, _ = load_bank(Path(manifest[name]['bank_root']) / ds / 'bank.npz', readouts=('flat',))
            ref_n = np.load(ROOT / name / ds / 'linear_oof.npz')
            lin = json.loads((ROOT / name / 'linear_summary.json').read_text())['sets'][ds]['families'][READOUT]['chosen']
            oof = score_arm(F, y, lin, device)
            arms[name] = metrics(y, oof, F)
            repro[name] = {'oof_max_abs_diff': float(np.abs(oof['clean'] - ref_n[READOUT]).max()),
                           'feature_max_abs_diff': float(np.abs(F['clean'] - bank_saved[READOUT]).max())}
            del F
        results['datasets'][ds] = {'n': len(y), 'arms': arms, 'repro': repro}
        print(ds, 'done', repro, flush=True)
        (OUT / 'results.json').write_text(json.dumps(results, indent=2))

    # raw drift: re-add as input-space drift (computed from V per dataset above is per-arm 'drift')
    arm_names = ['raw', 'random', *TRAINED]
    rows = []
    for ds, d in results['datasets'].items():
        for arm in arm_names:
            for p in PERTURBATIONS:
                e = d['arms'][arm][p]
                rows.append({'dataset': ds, 'perturbation': p, 'arm': arm, 'r2_clean': d['arms'][arm]['clean_r2'],
                             'r2_perturbed': e['r2'], 'drop': e['drop'], 'drift': e['drift']})
    with open(OUT / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for p in PERTURBATIONS:
        def mean_drop(arms):
            return float(np.mean([r['drop'] for r in rows if r['perturbation'] == p and r['arm'] in arms]))
        def mean_drift(arms):
            return float(np.mean([r['drift'] for r in rows if r['perturbation'] == p and r['arm'] in arms]))
        worst = {}
        for arm_group, arms in (('raw', ['raw']), ('random', ['random']), ('trained', TRAINED)):
            per_ds = {ds: float(np.mean([r['drop'] for r in rows if r['dataset'] == ds and r['perturbation'] == p
                                          and r['arm'] in arms])) for ds in DATASETS}
            worst[arm_group] = max(per_ds, key=per_ds.get), per_ds[max(per_ds, key=per_ds.get)]
        summ[p] = {'raw_drop': mean_drop(['raw']), 'random_drop': mean_drop(['random']),
                   'trained_drop': mean_drop(TRAINED), 'raw_drift': mean_drift(['raw']),
                   'random_drift': mean_drift(['random']), 'trained_drift': mean_drift(TRAINED), 'worst': worst}
    results['summary'] = summ
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
