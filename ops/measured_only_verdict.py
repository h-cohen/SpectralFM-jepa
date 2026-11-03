"""Verdict of the measured-only study (spec docs/superpowers/specs/2026-11-03-measured-only-pretraining-design.md),
from the three-family fusion `ensemble --stack --no-stack-heads --mix-groups a a a a s s s s m m m m` wrote. No refitting.

    PYTHONPATH=src:. python ops/measured_only_verdict.py [--dir outputs/measured-only/ens] [--output verdict.json]

Families: a = all data (nr_s107-110), s = synthetic only (so_s107-110), m = measured only (mo_s107-110); each is
the family's block-2 flat average (`avg_<label>` in fusion_oof.npz). Random is the measured family's
random-weight controls fused the same way. Per set: R², RMSE and nRMSE of every arm, ΔR²/ΔRMSE vs raw for the
three families, Δ = measured − all with bootstrap SDs (`report_rmse.delta`, the resamples of
`nested.paired_delta`), the synthetic share (all − measured) / (all − raw) where all − raw > 0, and whether
measured stays within .01 of raw. The verdict uses the seven measured sets only; labeled_data is reported but
excluded (all-data saw it).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ops.report_rmse import delta, summ

SEVEN = ('dataset0055', 'dataset0106', 'dataset0109', 'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120')
FAMILY = {'all': 'avg_a', 'synth': 'avg_s', 'measured': 'avg_m'}
DEFAULT_DIR = Path('/mnt5/home/hadar/nova/SpectralFM-jepa/outputs/measured-only/ens')


def verdict(per_set):
    """Fixed before results: 'synthetic not needed' if measured − all ≥ −.01 on ≥5/7 and the mean ≥ −.01;
    'synthetic helps' if measured − all < −.03 on ≥4/7; otherwise 'mixed'."""
    d = np.array([per_set[s]['m_minus_a']['d_r2'] for s in SEVEN])
    if (d >= -.01).sum() >= 5 and d.mean() >= -.01:
        return 'synthetic not needed'
    if (d < -.03).sum() >= 4:
        return 'synthetic helps'
    return 'mixed'


def synthetic_share(all_vs_raw, measured_vs_raw):
    """(all − measured) / (all − raw); None unless all-data beats raw."""
    return (all_vs_raw - measured_vs_raw) / all_vs_raw if all_vs_raw > 0 else None


def summarize(fusion_dir):
    fusion_dir = Path(fusion_dir)
    fusion = json.loads((fusion_dir / 'fusion.json').read_text())
    per_set = {}
    for ds in fusion['sets']:
        z = np.load(fusion_dir / ds / 'fusion_oof.npz')
        y = z['y']
        P = {'raw': z['raw'], 'random': z['random_avg_m'], **{k: z[v] for k, v in FAMILY.items()}}
        vs_raw = {k: delta(y, P[k], P['raw']) for k in FAMILY}
        per_set[ds] = {
            'n': int(len(y)),
            'arms': {k: summ(y, p) for k, p in P.items()},
            'vs_raw': vs_raw,
            'm_minus_a': delta(y, P['measured'], P['all']),
            'synthetic_share': synthetic_share(vs_raw['all']['d_r2'], vs_raw['measured']['d_r2']),
            'measured_within_01_of_raw': bool(vs_raw['measured']['d_r2'] >= -.01),
        }
    d = [per_set[s]['m_minus_a']['d_r2'] for s in SEVEN]
    return {'fusion_dir': str(fusion_dir), 'mix_groups': fusion.get('mix_groups'), 'per_set': per_set,
            'mean_m_minus_a': float(np.mean(d)), 'within_01': int(sum(x >= -.01 for x in d)),
            'below_03': int(sum(x < -.03 for x in d)), 'labeled_data_excluded': True, 'verdict': verdict(per_set)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dir', type=Path, default=DEFAULT_DIR, help='fusion output dir (fusion.json, <set>/fusion_oof.npz)')
    ap.add_argument('--output', type=Path, help='default: <dir>/../verdict.json')
    a = ap.parse_args(argv)
    out = summarize(a.dir)
    path = a.output or a.dir.parent / 'verdict.json'
    path.write_text(json.dumps(out, indent=2))
    for ds, e in out['per_set'].items():
        m = e['m_minus_a']
        print(f"{ds:13s} m-a {m['d_r2']:+.3f} ± {m['sd_r2']:.3f}  "
              + '  '.join(f"{k} {e['vs_raw'][k]['d_r2']:+.3f}" for k in FAMILY))
    print(f"mean m-a {out['mean_m_minus_a']:+.3f}, within .01 {out['within_01']}/7, below .03 {out['below_03']}/7 "
          f"-> {out['verdict']}  ({path})")


if __name__ == '__main__':
    main()
