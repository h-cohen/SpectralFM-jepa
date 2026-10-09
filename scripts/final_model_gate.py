"""Predeclared benchmark gate for the fresh fixed-readout candidate."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.build_results_report import DATASETS


def assess(summary):
    sets = summary.get('sets', {})
    complete = all(d in sets for d in DATASETS)
    raw, random, canaries = [], [], []
    for d in DATASETS:
        if d not in sets:
            continue
        p = sets[d]
        values = (p.get('embedding_r2'), p.get('raw_r2'),
                  (p.get('random_control') or {}).get('embedding_r2'))
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            complete = False
            continue
        raw.append(values[0] - values[1]); random.append(values[0] - values[2])
        canary = p.get('canary', {})
        canaries.append(bool(canary.get('block', {}).get('passed')) and bool(canary.get('raw', {}).get('passed')))
    wins = sum(v > 0 for v in raw)
    mean_raw = statistics.mean(raw) if raw else None
    mean_random = statistics.mean(random) if random else None
    checks = {'eight_complete_finite_sets': complete, 'majority_raw_wins': wins >= 5,
              'positive_mean_gain_vs_raw': mean_raw is not None and mean_raw > 0,
              'positive_mean_gain_vs_random': mean_random is not None and mean_random > 0,
              'all_canaries_pass': len(canaries) == 8 and all(canaries)}
    return {'benchmark_qualified': all(checks.values()), 'checks': checks, 'wins': wins,
            'mean_delta_vs_raw': mean_raw, 'mean_delta_vs_random': mean_random,
            'deployment_release_ready': False,
            'release_note': 'Fixed-readout benchmark replication only. Independent deployment-domain validation is still required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    root = Path('outputs/final-model')
    summary_path = root / 'eval' / f'{args.checkpoint.parent.name}_step497990' / 'summary.json'
    summary = json.loads(summary_path.read_text())
    result = assess(summary)
    result.update(checkpoint=str(args.checkpoint), summary=str(summary_path), readout='layer2/flat', training_seed=101)
    export_path = root / 'production/metadata.json'
    metadata = json.loads(export_path.read_text())
    if metadata['step'] != 497990 or metadata['readout'] != 'layer2/flat' or metadata['training_seed'] != 101:
        raise ValueError('Export does not match the registered final candidate')
    evaluation = summary['lineage']['evaluation_config']['evaluation']
    if evaluation['readouts'] != ['flat'] or evaluation['flat_blocks'] != [2]:
        raise ValueError('Benchmark used a different readout')
    if summary['lineage']['pretraining_checkpoint_sha256'] != metadata['checkpoint_sha256']:
        raise ValueError('Evaluation and export checkpoints differ')
    metadata['benchmark_status'] = 'qualified' if result['benchmark_qualified'] else 'not qualified'
    metadata['benchmark_assessment'] = result
    export_path.write_text(json.dumps(metadata, indent=2))
    (root / 'assessment.json').write_text(json.dumps(result, indent=2))
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(state='benchmark-qualified' if result['benchmark_qualified'] else 'benchmark-not-qualified',
                    step=497990, assessment=str(root / 'assessment.json'), checkpoint=str(args.checkpoint))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    from spectral_lejepa.config import load_config
    from spectral_lejepa.utils import wandb as wb
    cfg = load_config('configs/final_model.yaml')
    run = wb.init_run(cfg, job_type='assessment', name='production_p48_s101_fixed_readout_gate')
    wb.log(run, {'gate/qualified': int(result['benchmark_qualified']), 'gate/wins': result['wins'],
                 'gate/mean_delta_vs_raw': result['mean_delta_vs_raw'],
                 'gate/mean_delta_vs_random': result['mean_delta_vs_random'], 'gate/deployment_release_ready': 0})
    if run is not None:
        run.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
