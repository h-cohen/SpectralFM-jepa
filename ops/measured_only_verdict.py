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


def fig_delta(out, outdir):
    """fig_measured_only.svg: ΔR² vs raw (± bootstrap SD) per set for the three four-seed families."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sets = ['labeled_data', *SEVEN]
    col = {'all': '#3a6ea5', 'synth': '#999999', 'measured': '#c0504d'}
    lab = {'all': 'all data', 'synth': 'synthetic only', 'measured': 'measured only'}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, k in enumerate(FAMILY):
        v = [out['per_set'][s]['vs_raw'][k]['d_r2'] for s in sets]
        e = [out['per_set'][s]['vs_raw'][k]['sd_r2'] for s in sets]
        ax.bar(np.arange(len(sets)) + (j - 1) * .26, v, .26, yerr=e, capsize=2, color=col[k], label=lab[k])
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(range(len(sets)), [s.replace('dataset', '') for s in sets])
    ax.set_ylabel('ΔR² vs raw (± SD)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_measured_only.svg')
    plt.close(fig)


def fig_valid(hist, outdir):
    """fig_measured_only_valid.svg: measured valid/loss per run, best step marked.
    hist {run: {'step': [...], 'valid/loss': [...], 'best_step': int}}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for run, h in hist.items():
        (line,) = ax.plot(h['step'], h['valid/loss'], lw=1, label=run)
        b = h['step'].index(h['best_step'])
        ax.plot(h['step'][b], h['valid/loss'][b], 'o', color=line.get_color(), ms=5)
    ax.set_xlabel('step')
    ax.set_ylabel('valid/loss (held-out measured groups)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_measured_only_valid.svg')
    plt.close(fig)


def fetch_valid(runs_dir, path):
    """W&B valid/loss history of every mo_s* run in runs_dir (needs the W&B key: run through bash -ic)."""
    import torch
    import wandb
    import yaml
    api, hist = wandb.Api(), {}
    for d in sorted(Path(runs_dir).glob('mo_s*_*/')):
        best = torch.load(d / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
        cfg = yaml.safe_load((d / 'config.yaml').read_text())
        entity = cfg['wandb'].get('entity') or api.default_entity
        run = api.run(f"{entity}/{cfg['wandb']['project']}/{best['wandb_run_id']}")
        rows = run.history(keys=['valid/loss'], pandas=False)
        hist[d.name.split('_2')[0]] = {'step': [int(r['_step']) for r in rows],
                                       'valid/loss': [float(r['valid/loss']) for r in rows],
                                       'best_step': int(best['step'])}
    Path(path).write_text(json.dumps(hist, indent=2))
    return hist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dir', type=Path, default=DEFAULT_DIR, help='fusion output dir (fusion.json, <set>/fusion_oof.npz)')
    ap.add_argument('--output', type=Path, help='default: <dir>/../verdict.json')
    ap.add_argument('--figures', type=Path, help='write fig_measured_only.svg (and the valid figure if history exists)')
    ap.add_argument('--fetch', action='store_true', help='fetch valid/loss histories from W&B into <dir>/../valid_history.json')
    a = ap.parse_args(argv)
    hist_path = a.dir.parent / 'valid_history.json'
    if a.fetch:
        fetch_valid(a.dir.parent, hist_path)
    out = summarize(a.dir)
    path = a.output or a.dir.parent / 'verdict.json'
    path.write_text(json.dumps(out, indent=2))
    if a.figures:
        fig_delta(out, a.figures)
        if hist_path.exists():
            fig_valid(json.loads(hist_path.read_text()), a.figures)
    for ds, e in out['per_set'].items():
        m = e['m_minus_a']
        print(f"{ds:13s} m-a {m['d_r2']:+.3f} ± {m['sd_r2']:.3f}  "
              + '  '.join(f"{k} {e['vs_raw'][k]['d_r2']:+.3f}" for k in FAMILY))
    print(f"mean m-a {out['mean_m_minus_a']:+.3f}, within .01 {out['within_01']}/7, below .03 {out['below_03']}/7 "
          f"-> {out['verdict']}  ({path})")


if __name__ == '__main__':
    main()
