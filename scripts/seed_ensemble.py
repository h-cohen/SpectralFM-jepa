"""Registered A1/A3/A4 analysis: equal-weight seed ensemble of saved block2-flat outer predictions.

Reuses frozen-feature-study OOFs (identical seed42 outer folds for every checkpoint), so no refit.
Registration: docs/superpowers/specs/2026-10-13-robust-production-roadmap.md.
"""
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.nested import paired_delta, write_json
from spectral_lejepa.evaluation.probe import r2

SRC = Path('outputs/frozen-feature-study')
OUT = Path('outputs/production-validation/seed_ensemble')
SEEDS = ('prior_s0', 'prior_s1', 'prior_s2', 'baseline_s101')
SETS = ('labeled_data', 'dataset0055', 'dataset0106', 'dataset0109',
        'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120')
ARM = 'layer2/flat'


def score(y, P):
    return float(np.mean([r2(y, p) for p in P]))


def pooled(deltas, sds):
    """Inverse-variance pooled mean and its SD (datasets treated as independent)."""
    w = 1 / np.square(sds)
    return float((w * deltas).sum() / w.sum()), float(w.sum() ** -0.5)


def main():
    sets = {}
    for ds in SETS:
        oofs = [np.load(SRC / m / ds / 'linear_oof.npz') for m in SEEDS]
        y, raw, rand = oofs[0]['y'], oofs[0]['raw'], oofs[0][f'random_{ARM}']
        for z in oofs[1:]:  # same folds, raw and random arms must be identical
            assert np.array_equal(z['raw'], raw) and np.array_equal(z[f'random_{ARM}'], rand)
        preds = {m: z[ARM] for m, z in zip(SEEDS, oofs)} | {'ensemble': np.mean([z[ARM] for z in oofs], axis=0)}
        sets[ds] = {'n': len(y), 'raw_r2': score(y, raw), 'random_r2': score(y, rand), 'arms': {
            m: {'r2': score(y, P), 'vs_raw': paired_delta(y, P, raw), 'vs_random': paired_delta(y, P, rand)}
            for m, P in preds.items()}}

    arms = (*SEEDS, 'ensemble')
    summary = {}
    for m in arms:
        d = np.array([sets[ds]['arms'][m]['vs_raw']['delta'] for ds in SETS])
        s = np.array([sets[ds]['arms'][m]['vs_raw']['sd'] for ds in SETS])
        dr = np.array([sets[ds]['arms'][m]['vs_random']['delta'] for ds in SETS])
        sr = np.array([sets[ds]['arms'][m]['vs_random']['sd'] for ds in SETS])
        summary[m] = {'raw_wins': int((d > 0).sum()), 'mean_vs_raw': float(d.mean()),
                      'random_wins': int((dr > 0).sum()), 'mean_vs_random': float(dr.mean()),
                      'pooled_vs_raw': pooled(d, s), 'pooled_vs_random': pooled(dr, sr)}

    single_mean = float(np.mean([summary[m]['mean_vs_raw'] for m in SEEDS]))
    worst = {ds: min(sets[ds]['arms'][m]['vs_raw']['delta'] for m in SEEDS) for ds in SETS}
    crit_a = summary['ensemble']['mean_vs_raw'] >= single_mean
    crit_b = {ds: sets[ds]['arms']['ensemble']['vs_raw']['delta'] >= worst[ds] for ds in SETS}
    decision = {'single_seed_mean_vs_raw': single_mean, 'criterion_a_mean': bool(crit_a),
                'criterion_b_no_worse_than_worst_seed': crit_b, 'passed': bool(crit_a and all(crit_b.values()))}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'results.json', {'arm': ARM, 'seeds': SEEDS, 'sets': sets,
                                      'summary': summary, 'decision': decision})

    print(f"{'dataset':13s} {'n':>5s} {'raw':>6s} {'rand':>6s} " + ' '.join(f'{m[-4:]:>14s}' for m in arms))
    for ds in SETS:
        a = sets[ds]['arms']
        print(f"{ds:13s} {sets[ds]['n']:5d} {sets[ds]['raw_r2']:6.3f} {sets[ds]['random_r2']:6.3f} " + ' '.join(
            f"{a[m]['vs_raw']['delta']:+.3f}±{a[m]['vs_raw']['sd']:.3f}" for m in arms))
    for m in arms:
        print(m, json.dumps(summary[m]))
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
