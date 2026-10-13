"""Apply the registered robustness-continuation criteria (manifest `criteria`) to completed jobs."""
import json
from pathlib import Path

import numpy as np

from spectral_lejepa.utils.io import write_json
from scripts.robustness_perturbations import DATASETS
from spectral_lejepa.evaluation.nested import paired_delta

ROOT = Path('outputs/robustness-continuation')
ARM = 'layer2/flat'


def clean_scores(job):
    state = json.loads((ROOT / job / 'state.json').read_text())
    bank = ROOT / job / 'eval' / f"{Path(state['checkpoint']).parent.name}_step49799"
    out = {}
    for ds in DATASETS:
        fam = json.loads((bank / ds / 'nested_results.json').read_text())['families']
        oof = np.load(bank / ds / 'nested_oof.npz')
        out[ds] = {'r2': fam[ARM]['r2_mean'], 'raw': fam['raw']['r2_mean'],
                   'delta_raw': fam[ARM]['r2_mean'] - fam['raw']['r2_mean'], 'y': oof['y'], 'oof': oof[ARM]}
    return out


def main():
    m = json.loads((ROOT / 'manifest.json').read_text())
    c = m['criteria']
    seeds = sorted({j['seed'] for j in m['jobs'].values()})
    result, regressions = {'seeds': {}}, {}
    for seed in seeds:
        ctl, rob = clean_scores(f'control_s{seed}'), clean_scores(f'robust_s{seed}')
        pert = {arm: json.loads((ROOT / f'{arm}_s{seed}' / 'perturbation.json').read_text())
                for arm in ('control', 'robust')}
        paired = {ds: paired_delta(rob[ds]['y'], rob[ds]['oof'], ctl[ds]['oof']) for ds in DATASETS}
        for ds, p in paired.items():
            regressions.setdefault(ds, []).append(p['delta'])
        clean = {arm: {'wins': int(sum(s[ds]['delta_raw'] > 0 for ds in DATASETS)),
                       'mean_delta_raw': float(np.mean([s[ds]['delta_raw'] for ds in DATASETS]))}
                 for arm, s in (('control', ctl), ('robust', rob))}
        drift = {arm: pert[arm]['summary']['drift_median_of_medians'] for arm in pert}
        noisy = {arm: {k: pert[arm]['summary']['mean_r2_small_sets'][k] for k in ['clean', *c['noisy_r2_suite']]}
                 for arm in pert}
        labeled = {arm: {k: pert[arm]['sets']['labeled_data']['trained_r2'][k] for k in ['clean', *c['noisy_r2_suite']]}
                   for arm in pert}
        checks = {
            'drift': drift['robust']['noise_1pct'] <= c['drift_noise1pct_max_ratio_vs_control'] * drift['control']['noise_1pct'],
            'noisy_r2': all(noisy['robust'][k] > noisy['control'][k] for k in c['noisy_r2_suite']),
            'clean_mean': clean['robust']['mean_delta_raw'] >= clean['control']['mean_delta_raw'] + c['clean_mean_delta_raw_min_vs_control'],
            'clean_wins': clean['robust']['wins'] >= clean['control']['wins'] + c['clean_raw_wins_min_vs_control']}
        result['seeds'][seed] = {'checks': checks, 'clean': clean, 'drift': drift, 'noisy_small_sets': noisy,
                                 'labeled_data_noisy': labeled, 'paired_robust_minus_control': paired,
                                 'per_dataset_delta_raw': {arm: {ds: s[ds]['delta_raw'] for ds in DATASETS}
                                                          for arm, s in (('control', ctl), ('robust', rob))}}
    repeated = [ds for ds, v in regressions.items() if all(d < c['repeated_dataset_regression_limit'] for d in v)]
    result['repeated_regressions'] = repeated
    result['primary_pass'] = bool(all(all(s['checks'].values()) for s in result['seeds'].values()) and not repeated)
    write_json(ROOT / 'results_analysis.json', result)
    for seed, s in result['seeds'].items():
        print(seed, json.dumps(s['checks']), json.dumps(s['clean']),
              'drift1%', {a: round(d['noise_1pct'], 4) for a, d in s['drift'].items()},
              'noisy', {a: {k: round(v, 3) for k, v in d.items()} for a, d in s['noisy_small_sets'].items()})
        print('  paired', {ds[-4:]: f"{p['delta']:+.3f}±{p['sd']:.3f}" for ds, p in s['paired_robust_minus_control'].items()})
    print('repeated regressions', repeated, 'PRIMARY PASS' if result['primary_pass'] else 'PRIMARY FAIL')


if __name__ == '__main__':
    main()
