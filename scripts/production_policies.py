"""Nested comparisons of predeclared production readout policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_results_report import DATASETS
from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.nested import nested_cv, paired_delta
from spectral_lejepa.training.checkpoint import sha256_file

MENU = ('layer2/flat', 'layer2', 'layer3', 'layer3/seg4')


def prepare_arms(bank, raw):
    missing = set(MENU) - set(bank)
    if missing:
        raise ValueError(f'Missing readouts: {sorted(missing)}')
    if any(len(bank[a]) != len(raw) for a in MENU):
        raise ValueError('Readout rows differ from raw rows')
    arms = {'raw': np.asarray(raw, dtype=np.float32),
            **{a: np.asarray(bank[a], dtype=np.float32) for a in MENU}}
    arms['multiscale'] = np.concatenate([arms[a] for a in MENU], axis=1)
    return arms


def policy_families():
    return {'raw': ['raw'], 'fixed_block2': ['layer2/flat'],
            'menu': list(MENU), 'menu_with_raw': ['raw', *MENU], 'multiscale': ['multiscale']}


def compare(bank, raw, y, **protocol):
    results = nested_cv(prepare_arms(bank, raw), policy_families(), y, **protocol)
    return results


def run(bank_root, output, backend='torch', device='cuda', publish=False, group='production-analysis'):
    output.mkdir(parents=True, exist_ok=True)
    families = policy_families()
    protocol = {'n_repeats': 2, 'n_folds': 5, 'n_inner': 5, 'seed': 42,
                'backend': backend, 'device': device, 'n_jobs': 1}
    run = None
    if publish:
        import wandb
        run = wandb.init(project='spectralfm-lejepa', group=group, job_type='readout-policy-comparison',
                         config={'protocol': protocol, 'menu': MENU, 'bank_root': str(bank_root), 'exploratory': True})
    sets = {}
    for dataset in DATASETS:
        path = bank_root / dataset / 'bank.npz'
        rc_path = bank_root / dataset / 'random_control/bank.npz'
        model, raw, y, meta = load_bank(path, readouts=('mean', 'seg4', 'flat'))
        random, rc_raw, rc_y, rc_meta = load_bank(rc_path, readouts=('mean', 'seg4', 'flat'))
        if not np.array_equal(raw, rc_raw) or not np.array_equal(y, rc_y):
            raise ValueError(f'{dataset}: model/random rows do not match')
        if meta.get('checkpoint') != rc_meta.get('checkpoint'):
            raise ValueError(f'{dataset}: model/random provenance differs')
        result, control = compare(model, raw, y, **protocol), compare(random, raw, y, **protocol)
        if not np.allclose(result['raw']['oof'], control['raw']['oof'], rtol=0, atol=1e-8):
            raise ValueError(f'{dataset}: raw predictions differ between policies')
        point = {'n': len(y), 'policies': {}, 'source_hashes': {'model': sha256_file(path), 'random': sha256_file(rc_path)}}
        for policy in families:
            r, c = result[policy], control[policy]
            point['policies'][policy] = {'r2': r['r2_mean'], 'sd': r['bootstrap_sd'],
                'random_r2': c['r2_mean'], 'random_sd': c['bootstrap_sd'], 'chosen': r['chosen'],
                'vs_raw': paired_delta(y, r['oof'], result['raw']['oof']),
                'vs_random': paired_delta(y, r['oof'], c['oof'])}
        sets[dataset] = point
        np.savez_compressed(output / f'{dataset}_oof.npz', y=y,
                            **{p: result[p]['oof'] for p in families},
                            **{f'random_{p}': control[p]['oof'] for p in families})
        (output / f'{dataset}.json').write_text(json.dumps(point, indent=2, allow_nan=False))
        print(dataset, {p: round(v['r2'], 4) for p, v in point['policies'].items()}, flush=True)
        if run:
            run.log({f'{dataset}/{p}/r2': v['r2'] for p, v in point['policies'].items()})
    summary = {'sets': sets, 'protocol': protocol, 'menu': MENU,
               'status': 'exploratory reused-benchmark analysis; original qualification unchanged',
               'scorecard': {p: {'wins': sum(s['policies'][p]['vs_raw']['delta'] > 0 for s in sets.values()),
                   'mean_delta': float(np.mean([s['policies'][p]['vs_raw']['delta'] for s in sets.values()]))} for p in families}}
    (output / 'policy_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    if run:
        from scripts.production_analysis_report import policy_figure
        from spectral_lejepa.utils.wandb import log_figure
        fig = policy_figure(summary)
        log_figure(run, 'policies/paired_gains', fig)
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/production-analysis/policies'))
    parser.add_argument('--backend', choices=['torch', 'sklearn'], default='torch')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    run(args.bank_root, args.output, args.backend, args.device, args.wandb)


if __name__ == '__main__':
    main()
