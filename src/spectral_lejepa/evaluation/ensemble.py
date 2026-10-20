"""Equal-weight ensembles of members' out-of-fold probe predictions.

    python -m spectral_lejepa ensemble --banks <evaluate run dir> ... [--readout layer2/flat] [--output scores.json]

Each run dir is one member's `evaluate` output (<output_dir>/<run>_step<N>). The ensemble prediction is the
mean of the members' outer-fold predictions; it is scored against raw spectra and against the mean of the
members' random-weight controls, paired on the same folds.

    python -m spectral_lejepa ensemble --stack --banks <evaluate run dir> ... --output-dir D

`--stack` refits from the members' bank.npz files in one nested CV: raw, `avg` (each member's
block-2 flat probe, recipe chosen in inner CV, predictions averaged) and `stack` (convex NNLS weights
over every member × HEADS, fitted on inner-CV predictions of each outer training fold).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.nested import nested_cv, paired_delta
from spectral_lejepa.evaluation.probe import r2

HEADS = ('layer2/flat', 'layer2', 'layer6')


def member_oof(member_dir, dataset):
    """nested_oof.npz of a pipeline job: <member>/eval/<training run>_step<N>/<dataset>/nested_oof.npz."""
    member_dir = Path(member_dir)
    run = Path(json.loads((member_dir / 'state.json').read_text())['checkpoint']).parent.name
    (bank,) = (member_dir / 'eval').glob(f'{run}_step*')
    return np.load(bank / dataset / 'nested_oof.npz')


def ensemble_predictions(oofs, arm='layer2/flat'):
    return np.mean([o[arm] for o in oofs], 0)


def score_ensemble(banks, readout='layer2/flat'):
    banks = [Path(b) for b in banks]
    sets = sorted(set.intersection(*({p.parent.name for p in b.glob('*/nested_oof.npz')} for b in banks)))
    out = {'members': [str(b) for b in banks], 'readout': readout, 'sets': {}}
    for ds in sets:
        oofs = [np.load(b / ds / 'nested_oof.npz') for b in banks]
        rand = [np.load(b / ds / 'random_control' / 'nested_oof.npz') for b in banks]
        y, raw = oofs[0]['y'], oofs[0]['raw']
        if any(not np.array_equal(o['y'], y) for o in oofs + rand):
            raise ValueError(f'{ds}: members were evaluated on different labels or row order')
        ens = ensemble_predictions(oofs, readout)
        out['sets'][ds] = {'vs_raw': paired_delta(y, ens, raw),
                           'vs_random': paired_delta(y, ens, ensemble_predictions(rand, readout))}
    d_raw = [s['vs_raw']['delta'] for s in out['sets'].values()]
    d_rand = [s['vs_random']['delta'] for s in out['sets'].values()]
    out['summary'] = {'n_sets': len(d_raw), 'raw_wins': int(sum(d > 0 for d in d_raw)),
                      'random_wins': int(sum(d > 0 for d in d_rand)),
                      'mean_delta_raw': float(np.mean(d_raw)), 'mean_delta_random': float(np.mean(d_rand))}
    return out


def _readout(bank, name, f):
    """A stored arm, or 'layer{a}+{b}/flat' = layer{a}/flat and layer{b}/flat side by side."""
    parts = [f'layer{b}/flat' for b in name.split('/')[0][len('layer'):].split('+')] if '+' in name else [name]
    for p in parts:
        if p not in bank:
            raise ValueError(f'head {p} missing from member bank {f}')
    return np.hstack([bank[p] for p in parts]) if len(parts) > 1 else bank[parts[0]]


def fuse_set(bank_files, heads=HEADS, avg_head='layer2/flat', select=None, **probe):
    """One set: raw, avg, stack (when heads) and select (when select) on the same outer folds.
    select: candidate readouts; each member's readout and recipe are picked together by inner CV
    (one nested_cv family per member) and the members' picks are averaged."""
    arms, y = {}, None
    for i, f in enumerate(bank_files):
        bank, raw, yi, _ = load_bank(f, readouts=('mean', 'seg4', 'flat'))
        if y is None:
            y, arms['raw'] = yi, raw
        elif not np.array_equal(yi, y):
            raise ValueError(f'{f}: members were evaluated on different labels or row order')
        for h in dict.fromkeys([avg_head, *heads, *(select or ())]):
            arms[f'm{i}:{h}'] = _readout(bank, h, f)
    n = len(bank_files)
    families = {'raw': ['raw'], **{f'sel{i}': [f'm{i}:{c}' for c in select] for i in range(n) if select}}
    stacks = {'stack': [f'm{i}:{h}' for i in range(n) for h in heads]} if heads else {}
    flat = [f'm{i}:{avg_head}' for i in range(n)]
    res = nested_cv(arms, families, y, ensembles={'avg': (flat, n)}, stacks=stacks, **probe)
    keys = ['raw', 'avg', *stacks]
    out = {'y': y, 'oof': {k: res[k]['oof'] for k in keys}, 'r2': {k: res[k]['r2_mean'] for k in keys}}
    if heads:
        out['weights'] = [[{'arm': m['arm'], 'weight': m['weight']} for m in c['members']] for c in res['stack']['chosen']]
    if select:
        out['oof']['select'] = np.mean([res[f'sel{i}']['oof'] for i in range(n)], 0)
        out['r2']['select'] = float(np.mean([r2(y, p) for p in out['oof']['select']]))
        out['select_chosen'] = [[res[f'sel{i}']['chosen'][j] for i in range(n)] for j in range(len(res['sel0']['chosen']))]
    return out


def fuse(banks, output, heads=HEADS, select=None, **probe):
    """Every set all members share; random-weight controls fused the same way. Each fused arm X gets
    X_vs_raw and X_vs_random; arms other than avg also get X_vs_avg."""
    banks, output = [Path(b) for b in banks], Path(output)
    sets = sorted(set.intersection(*({p.parent.name for p in b.glob('*/bank.npz')} for b in banks)))
    out = {'members': [str(b) for b in banks], 'heads': list(heads), 'select': list(select or ()), 'sets': {}}
    for ds in sets:
        m = fuse_set([b / ds / 'bank.npz' for b in banks], heads, select=select, **probe)
        rc = fuse_set([b / ds / 'random_control' / 'bank.npz' for b in banks], heads, select=select, **probe)
        y, P, R = m['y'], m['oof'], rc['oof']
        fused = [k for k in P if k != 'raw']
        entry = {'r2': m['r2'], 'random': {'r2': {k: rc['r2'][k] for k in fused}}}
        for k in fused:
            entry[f'{k}_vs_raw'] = paired_delta(y, P[k], P['raw'])
            entry[f'{k}_vs_random'] = paired_delta(y, P[k], R[k])
            if k != 'avg':
                entry[f'{k}_vs_avg'] = paired_delta(y, P[k], P['avg'])
        if 'weights' in m:
            mean_w = {}
            for fold in m['weights']:
                for d in fold:
                    mean_w[d['arm']] = mean_w.get(d['arm'], 0.) + d['weight'] / len(m['weights'])
            entry.update(mean_weights=mean_w, fold_weights=m['weights'])
        if 'select_chosen' in m:
            entry['select_chosen'] = m['select_chosen']
        out['sets'][ds] = entry
        (output / ds).mkdir(parents=True, exist_ok=True)
        np.savez(output / ds / 'fusion_oof.npz', y=y, **P, **{f'random_{k}': v for k, v in R.items()})
    first = next(iter(out['sets'].values()), {})
    keys = [k for k in first if k.endswith(('_vs_raw', '_vs_random', '_vs_avg'))]
    d = lambda key: [s[key]['delta'] for s in out['sets'].values()]
    out['summary'] = {'n_sets': len(sets), **{f'{k}_wins': int(sum(x > 0 for x in d(k))) for k in keys},
                      **{f'{k}_mean': float(np.mean(d(k))) for k in keys}}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fusion.json').write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--banks', nargs='+', required=True, help="members' evaluate run dirs")
    ap.add_argument('--readout', default='layer2/flat')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--stack', action='store_true', help='refit raw, avg and stack from the members\' bank.npz')
    ap.add_argument('--output-dir', type=Path, help='--stack: fusion.json and <set>/fusion_oof.npz')
    ap.add_argument('--backend', default='torch')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-jobs', type=int, default=1)
    ap.add_argument('--select', nargs='+', help='--stack: candidate readouts chosen per member in inner CV')
    ap.add_argument('--no-stack-heads', action='store_true', help='--stack: skip the NNLS stack arm')
    a = ap.parse_args(argv)
    if a.stack:
        out = fuse(a.banks, a.output_dir, heads=() if a.no_stack_heads else HEADS, select=a.select,
                   backend=a.backend, device=a.device, n_jobs=a.n_jobs)
        for ds, s in out['sets'].items():
            print(f"{ds:14s} " + ' '.join(f"{k} {v:.4f}" for k, v in s['r2'].items()))
        print(json.dumps(out['summary']))
        return
    out = score_ensemble(a.banks, a.readout)
    for ds, s in out['sets'].items():
        print(f"{ds:14s} Δraw {s['vs_raw']['delta']:+.4f} ± {s['vs_raw']['sd']:.4f}   Δrandom {s['vs_random']['delta']:+.4f}")
    print(json.dumps(out['summary']))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
