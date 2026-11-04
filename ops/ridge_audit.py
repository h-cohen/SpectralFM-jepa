"""Ridge/linear-head audit.

1. Recipe audit (existing nested-CV outputs): how often each (normalizer + probe) recipe is chosen per set
   and arm (raw, random-weight control, layer2/flat), and how stable the choice is (share of the 10 outer
   folds that pick the modal recipe).
2. Head audit (outputs of `evaluate --config configs/eval_ridge_study.yaml`): per set, R² and RMSE (label
   units) of every fixed head and arm; paired dR² (+ bootstrap SD) and dRMSE vs raw under the same head;
   JEPA-with-ridge_ens vs client_raw; the nested-selected readout vs ridge_ens per arm; the distribution of
   the RidgeCV alpha behind each chosen ridge recipe.

    python ops/ridge_audit.py [--recipes MEMBER=EVAL_DIR ...] [--heads EVAL_DIR ...] [--out outputs/ridge-study/audit.json]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.nested import paired_delta
from spectral_lejepa.evaluation.probe import r2
from spectral_lejepa.evaluation.ridge_heads import rmse

ARMS = ("raw", "layer2/flat", "random")
MAIN = Path('/mnt5/home/hadar/nova/SpectralFM-jepa')
MEMBERS = {
    'nr_s107': 'outputs/accuracy-ensemble/members/nr_s107/eval/nr_s107_20261022-204051_step497990',
    'robust_s103': 'outputs/signal-squeeze/members/robust_s103/eval/robust_s103_20261021-190555_step49799',
}


def _set_dirs(run):
    return sorted(p.parent for p in Path(run).glob('*/nested_results.json'))


def _arm_files(set_dir, arm):
    """(nested_results.json, nested_oof.npz, key) of an arm; the random control's layer2/flat sits in random_control/."""
    d, key = (set_dir / 'random_control', 'layer2/flat') if arm == 'random' else (set_dir, arm)
    return d / 'nested_results.json', d / 'nested_oof.npz', key


def chosen_frequencies(chosen):
    """Counts of 'norm+probe' over the outer folds, the modal recipe and its share of the folds."""
    counts = collections.Counter(f"{c['norm']}+{c['probe']}" for c in chosen)
    modal, top = counts.most_common(1)[0]
    return {'n_folds': len(chosen), 'counts': dict(counts.most_common()), 'modal': modal,
            'stability': top / len(chosen), 'n_distinct': len(counts)}


def recipe_audit(run):
    out = {}
    for sd in _set_dirs(run):
        out[sd.name] = {}
        for arm in ARMS:
            res, _, key = _arm_files(sd, arm)
            if res.exists():
                out[sd.name][arm] = chosen_frequencies(json.loads(res.read_text())['families'][key]['chosen'])
    return out


def _scores(y, P, Praw):
    d = paired_delta(y, P, Praw)
    return {'r2': float(np.mean([r2(y, p) for p in P])), 'rmse': float(np.mean([rmse(y, p) for p in P])),
            'delta_r2': d['delta'], 'delta_r2_sd': d['sd'],
            'delta_rmse': float(np.mean([rmse(y, p) for p in P]) - np.mean([rmse(y, p) for p in Praw]))}


def heads_audit(run):
    """Per-set tables, per-head/arm summary over sets, and the chosen-alpha distribution."""
    per_set, alphas = {}, collections.defaultdict(collections.Counter)
    per_set_alpha = {}
    for sd in _set_dirs(run):
        hz = sd / 'fixed_heads_oof.npz'
        if not hz.exists():
            continue
        H = np.load(hz)
        y = H['y'].astype(np.float64)
        heads = sorted({k.split('__')[0] for k in H.files if k != 'y'})
        row = {}
        for head in heads:
            raw = H[f'{head}__raw']
            row[head] = {}
            for arm in ARMS:
                k = f"{head}__{arm.replace('/', '_')}"
                if k in H.files:
                    row[head][arm] = _scores(y, H[k], raw)
        if 'ridge_ens' in heads and 'client_raw' in heads:
            row_j = paired_delta(y, H['ridge_ens__layer2_flat'], H['client_raw__raw'])
            row_j |= {'rmse_delta': float(np.mean([rmse(y, p) for p in H['ridge_ens__layer2_flat']])
                                          - np.mean([rmse(y, p) for p in H['client_raw__raw']]))}
        else:
            row_j = None
        sel = {}
        for arm in ARMS:
            res, oof, key = _arm_files(sd, arm)
            if oof.exists() and f"ridge_ens__{arm.replace('/', '_')}" in H.files:
                S = np.load(oof)[key]
                d = paired_delta(y, S, H[f"ridge_ens__{arm.replace('/', '_')}"])
                sel[arm] = {**d, 'selected_r2': float(np.mean([r2(y, p) for p in S])),
                            'h1_r2': row['ridge_ens'][arm]['r2']}
            if res.exists():
                for c in json.loads(res.read_text())['families'][key]['chosen']:
                    if c.get('alpha') is not None:
                        alphas[arm][c['alpha']] += 1
                        per_set_alpha.setdefault(sd.name, {}).setdefault(arm, collections.Counter())[c['alpha']] += 1
        per_set[sd.name] = {**row, 'jepa_h1_vs_client_raw': row_j, 'selected_vs_h1': sel}
    summary = {}
    for head in sorted({h for s in per_set.values() for h in s if h in ('ridge_ens', 'std_ols', 'client_raw')}):
        summary[head] = {}
        for arm in ARMS:
            rows = [s[head][arm] for s in per_set.values() if arm in s.get(head, {})]
            if rows and arm != 'raw':
                d = np.array([r['delta_r2'] for r in rows])
                summary[head][arm] = {'sets': len(rows), 'wins': int((d > 0).sum()), 'mean_delta_r2': float(d.mean()),
                                      'sd_delta_r2_across_sets': float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                                      'mean_delta_rmse': float(np.mean([r['delta_rmse'] for r in rows])),
                                      'mean_r2': float(np.mean([r['r2'] for r in rows]))}
            elif rows:
                summary[head][arm] = {'sets': len(rows), 'mean_r2': float(np.mean([r['r2'] for r in rows])),
                                      'mean_rmse': float(np.mean([r['rmse'] for r in rows]))}
    return {'per_set': per_set, 'summary': summary,
            'alphas': {a: {str(k): v for k, v in sorted(c.items())} for a, c in alphas.items()},
            'alphas_per_set': {s: {a: {str(k): v for k, v in sorted(c.items())} for a, c in d.items()}
                               for s, d in per_set_alpha.items()}}


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() or p.exists() else MAIN / p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--recipes', nargs='*', metavar='MEMBER=EVAL_DIR',
                    help='member evals for the recipe audit (default: nr_s107 and robust_s103)')
    ap.add_argument('--heads', nargs='*', default=[], metavar='EVAL_DIR',
                    help='run dirs written by the ridge study (fixed_heads_oof.npz per set)')
    ap.add_argument('--out', default='outputs/ridge-study/audit.json')
    a = ap.parse_args(argv)
    members = dict(r.split('=', 1) for r in a.recipes) if a.recipes else MEMBERS
    out = {'recipes': {m: recipe_audit(_resolve(d)) for m, d in members.items()}}
    for d in a.heads:
        out.setdefault('heads', {})[Path(d).name] = heads_audit(_resolve(d))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for m, sets in out['recipes'].items():
        print(f'== {m}: modal recipe (share of 10 folds), distinct recipes')
        for s, arms in sets.items():
            print(f'  {s:14s} ' + '  '.join(f"{arm}: {v['modal']} {v['stability']:.1f} ({v['n_distinct']})"
                                            for arm, v in arms.items()))
    for run, h in out.get('heads', {}).items():
        print(f'== heads {run}')
        for head, arms in h['summary'].items():
            for arm, v in arms.items():
                print(f'  {head:11s} {arm:12s} ' + ' '.join(f'{k}={x:.4g}' for k, x in v.items()))


if __name__ == '__main__':
    main()
