"""Robust continuations of the other 48p seeds for a robust four-seed ensemble.

Spec: docs/superpowers/specs/2026-10-13-robust-ensemble.md. Same recipe as robust_s401 (weight 0.1,
calibrated perturbations, 49,799 steps from each saved checkpoint).
  register   job configs, hashes and criteria (refuses to run twice)
  <job>      train -> gradient audit -> all-block eval -> perturbation eval (job and its start)
  analyze    registered criteria for the ensemble robust_s401 + the three new jobs
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from spectral_lejepa.config import load_config
from spectral_lejepa.training.checkpoint import sha256_file
from scripts.continuation_study import write_json
from scripts.robustness_continuation import ROOT as PILOT, STEPS, training_config

ROOT = Path('outputs/robust-ensemble')
STARTS = {   # name: (checkpoint, sha256, training seed); seed2 is covered by robust_s401
    'robust_p0': ('outputs/paired-long/p48_s0_20261008-161502/checkpoint_last.pt',
                  'a71f06f0e20b6e0073304a2f925b9ebcbc15f4a3cbe171b936c4450812b0b7aa', 410),
    'robust_p1': ('outputs/paired-long/p48_s1_20261008-161507/checkpoint_last.pt',
                  'f9d9d4fd9be43903a1c7a6fd51738d25475f09bdc78e4ccc68fa48f1b038f481', 411),
    'robust_s101': ('outputs/final-model/production_p48_s101_20261010-024928/checkpoint_last.pt',
                    '30486b3e78c354596d7bb5f09ebf351805309b0e9505c88d7ed698145df7fbbe', 412),
}
GPUS = (0, 1, 2)               # physical; GPU3 is broken
SEED2_JOB = 'robust_s401'      # fixed before results: lower seed of the two seed2 robust runs
CRITERIA = {'ensemble_raw_wins_min': 5, 'ensemble_mean_delta_raw_gt': 0.0, 'ensemble_mean_delta_random_gt': 0.0,
            'preferred_over_original_ensemble': {'raw_wins_min': 7, 'random_wins_min': 7},
            'job_drift_noise1pct_max_ratio_vs_start': .5}


def register():
    if (ROOT / 'manifest.json').exists():
        raise FileExistsError('Study already registered')
    weight = json.loads((PILOT / 'calibration.json').read_text())['weight']
    uuids = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, GPUS)), '--query-gpu=uuid',
                                     '--format=csv,noheader'], text=True).strip().splitlines()
    jobs = {}
    for slot, (name, (checkpoint, sha, seed)) in enumerate(STARTS.items()):
        if sha256_file(checkpoint) != sha:
            raise ValueError(f'{checkpoint} changed')
        base = ROOT / name
        base.mkdir(parents=True, exist_ok=False)
        c = training_config(seed, 'robust', weight, out=ROOT / 'training')
        c['experiment']['name'] = name
        c['training']['init_checkpoint'] = checkpoint
        c['wandb'].update(group='robust-ensemble', tags=['robust-ensemble', 'exploratory', name])
        train = base / 'train.yaml'
        train.write_text(yaml.safe_dump(c, sort_keys=False))
        e = load_config('configs/eval_production_all_blocks.yaml')
        e['experiment'].update(name=name, output_dir=str(base / 'eval'))
        e['wandb'].update(group='robust-ensemble', tags=['robust-ensemble', 'all-blocks', name])
        ev = base / 'eval.yaml'
        ev.write_text(yaml.safe_dump(e, sort_keys=False))
        jobs[name] = {'seed': seed, 'start_checkpoint': checkpoint, 'start_sha256': sha,
                      'physical_gpu': GPUS[slot], 'gpu_uuid': uuids[slot],
                      'train_config': str(train), 'eval_config': str(ev), 'train_sha256': sha256_file(train),
                      'eval_sha256': sha256_file(ev), 'service': f'spectralfm-{name.replace("_", "-")}'}
    m = {'registered_at': datetime.now(timezone.utc).isoformat(), 'steps': STEPS, 'robust_weight': weight,
         'jobs': jobs, 'seed2_member': str(PILOT / SEED2_JOB), 'primary_readout': 'layer2/flat',
         'criteria': CRITERIA,
         'status': 'exploratory; contract-level candidate only; export, gate and holdout plan unchanged'}
    write_json(ROOT / 'manifest.json', m)
    Path('docs/superpowers/registrations/2026-10-13-robust-ensemble.json').write_text(json.dumps(m, indent=2))
    return m


def run_job(name):
    from spectral_lejepa.training.trainer import train
    from scripts.evaluate import main as evaluate
    from scripts.gradient_audit import run_audit, write_result
    from scripts.robustness_eval import evaluate as perturbation_eval
    job = json.loads((ROOT / 'manifest.json').read_text())['jobs'][name]
    for kind in ('train', 'eval'):
        if sha256_file(job[kind + '_config']) != job[kind + '_sha256']:
            raise ValueError('Registered config changed')
    if sha256_file(job['start_checkpoint']) != job['start_sha256']:
        raise ValueError('Starting checkpoint changed')
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != job['gpu_uuid'].removeprefix('GPU-'):
        raise ValueError('GPU differs from registration')
    base = ROOT / name
    if (base / 'state.json').exists():
        raise FileExistsError('Job already started; recovery must preserve training state')
    state = {'name': name, 'state': 'training', **job}
    write_json(base / 'state.json', state)
    try:
        checkpoint = train(load_config(job['train_config']))['final_checkpoint']
        state.update(state='final-gradient-audit', checkpoint=checkpoint, checkpoint_sha256=sha256_file(checkpoint))
        write_json(base / 'state.json', state)
        write_result(run_audit(Path(checkpoint), 'cuda'), base / 'gradient_audit.json')
        state['state'] = 'all-block-evaluation'
        write_json(base / 'state.json', state)
        evaluate(['--checkpoint', checkpoint, '--config', job['eval_config']])
        state['state'] = 'perturbation-evaluation'
        write_json(base / 'state.json', state)
        perturbation_eval(checkpoint, base / 'perturbation.json')
        perturbation_eval(job['start_checkpoint'], base / 'start_perturbation.json')
        state['state'] = 'completed'
        write_json(base / 'state.json', state)
    except Exception as e:
        state.update(state='failed', error=repr(e))
        write_json(base / 'state.json', state)
        raise


def job_oof(base, ds):
    state = json.loads((base / 'state.json').read_text())
    bank = base / 'eval' / f"{Path(state['checkpoint']).parent.name}_step{STEPS}"
    return np.load(bank / ds / 'nested_oof.npz')


def analyze():
    from scripts.robustness_perturbations import DATASETS
    from spectral_lejepa.evaluation.nested import paired_delta
    m = json.loads((ROOT / 'manifest.json').read_text())
    c = m['criteria']
    members = [Path(m['seed2_member'])] + [ROOT / n for n in m['jobs']]
    ref = Path('outputs/frozen-feature-study/prior_s2')
    sets, per_job = {}, {str(b): {} for b in members}
    for ds in DATASETS:
        oofs = [job_oof(b, ds) for b in members]
        y, raw = oofs[0]['y'], oofs[0]['raw']
        random = np.load(ref / ds / 'linear_oof.npz')['random_layer2/flat']
        for o in oofs:
            assert np.array_equal(o['y'], y) and np.allclose(o['raw'], raw, atol=1e-3)
        for b, o in zip(members, oofs):
            per_job[str(b)][ds] = paired_delta(y, o['layer2/flat'], raw)['delta']
        ens = np.mean([o['layer2/flat'] for o in oofs], 0)
        sets[ds] = {'vs_raw': paired_delta(y, ens, raw), 'vs_random': paired_delta(y, ens, random)}
    dr = [s['vs_raw']['delta'] for s in sets.values()]
    dn = [s['vs_random']['delta'] for s in sets.values()]
    summary = {'raw_wins': int(sum(d > 0 for d in dr)), 'random_wins': int(sum(d > 0 for d in dn)),
               'mean_delta_raw': float(np.mean(dr)), 'mean_delta_random': float(np.mean(dn))}
    drift = {}
    for n in m['jobs']:
        p = json.loads((ROOT / n / 'perturbation.json').read_text())['summary']
        s = json.loads((ROOT / n / 'start_perturbation.json').read_text())['summary']
        drift[n] = {'job': p['drift_median_of_medians']['noise_1pct'], 'start': s['drift_median_of_medians']['noise_1pct'],
                    'noisy_small_sets': p['mean_r2_small_sets'], 'start_noisy_small_sets': s['mean_r2_small_sets']}
    pref = c['preferred_over_original_ensemble']
    checks = {'contract': summary['raw_wins'] >= c['ensemble_raw_wins_min']
              and summary['mean_delta_raw'] > c['ensemble_mean_delta_raw_gt']
              and summary['mean_delta_random'] > c['ensemble_mean_delta_random_gt'],
              'preferred_over_original_ensemble': summary['raw_wins'] >= pref['raw_wins_min']
              and summary['random_wins'] >= pref['random_wins_min'],
              'job_drift': all(d['job'] <= c['job_drift_noise1pct_max_ratio_vs_start'] * d['start'] for d in drift.values())}
    result = {'members': [str(b) for b in members], 'summary': summary, 'checks': checks, 'sets': sets,
              'per_job_delta_raw': per_job, 'drift': drift}
    write_json(ROOT / 'results_analysis.json', result)
    print(json.dumps({'summary': summary, 'checks': checks}, indent=2))
    for b, d in per_job.items():
        print(b, 'wins', sum(v > 0 for v in d.values()), {k[-4:]: round(v, 3) for k, v in d.items()})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job')
    a = p.parse_args()
    if a.job in ('register', 'analyze'):
        out = globals()[a.job]()
        if a.job == 'register':
            print(json.dumps(out, indent=2, default=str))
    else:
        run_job(a.job)
