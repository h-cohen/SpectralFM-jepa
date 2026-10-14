"""Equal-weight ensembles of members' out-of-fold probe predictions.

    python -m spectral_lejepa ensemble --banks <evaluate run dir> ... [--readout layer2/flat] [--output scores.json]

Each run dir is one member's `evaluate` output (<output_dir>/<run>_step<N>). The ensemble prediction is the
mean of the members' outer-fold predictions; it is scored against raw spectra and against the mean of the
members' random-weight controls, paired on the same folds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.nested import paired_delta


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--banks', nargs='+', required=True, help="members' evaluate run dirs")
    ap.add_argument('--readout', default='layer2/flat')
    ap.add_argument('--output', type=Path)
    a = ap.parse_args(argv)
    out = score_ensemble(a.banks, a.readout)
    for ds, s in out['sets'].items():
        print(f"{ds:14s} Δraw {s['vs_raw']['delta']:+.4f} ± {s['vs_raw']['sd']:.4f}   Δrandom {s['vs_random']['delta']:+.4f}")
    print(json.dumps(out['summary']))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
