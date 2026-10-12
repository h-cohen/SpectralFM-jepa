"""Calibrate, register and run paired block-2 robustness continuations from prior48pseed2.

Design: docs/superpowers/specs/2026-10-13-robustness-continuation.md.
  calibrate  1,000-step weight-0 continuation (seed 400) + gradient audit -> robust weight
  register   four job configs, criteria and hashes (refuses to run twice)
  <job>      train -> gradient audit -> all-block eval -> fixed-menu policies -> perturbation eval
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from spectral_lejepa.config import load_config
from spectral_lejepa.training.checkpoint import sha256_file
from scripts.continuation_study import write_json

ROOT = Path('outputs/robustness-continuation')
CHECKPOINT = 'outputs/paired-long/p48_s2_20261008-161508/checkpoint_last.pt'
SHA = 'ec4c041dc4f83aa04e9ec568a335cdf7e27dbc7aa7fac965f2fd1f686b566ff0'
STEPS, CALIBRATION_STEPS, CALIBRATION_SEED = 49799, 1000, 400
WEIGHTS = (.1, .3, 1.0)        # smallest with weighted block2 norm >= RATIO x prediction norm
RATIO = .5
GPUS = (0, 1, 2, 4)            # physical; GPU3 is broken


def robust_settings(weight):
    rec = json.loads((ROOT / 'noise_calibration.json').read_text())['recommended']
    return {'weight': weight, 'block': 2, 'prob': .5, **{k: rec[k] for k in
            ('sigma_max', 'offset_max', 'slope_max', 'scale_max', 'shift_max')}}


def training_config(seed, arm, weight, steps=STEPS, out=ROOT / 'training'):
    c = load_config('configs/final_model.yaml')
    c['experiment'].update(name=f'robust_{arm}_s{seed}', seed=seed, output_dir=str(out))
    c['training'].update(init_checkpoint=CHECKPOINT, resume=None, epochs=1, max_steps=steps,
                         warmup_epochs=1000.5 / STEPS, learning_rate=5e-5, final_lr_ratio=.01,
                         val_every=5000, diag_every=5000, ckpt_every=5000,
                         robust=robust_settings(weight if arm == 'robust' else 0.0))
    c['wandb'].update(group='robustness-continuation',
                      tags=['robustness-continuation', 'exploratory', arm, f'seed{seed}'])
    return c


def check_start():
    if sha256_file(CHECKPOINT) != SHA:
        raise ValueError('Starting checkpoint changed')


def calibrate():
    from spectral_lejepa.training.trainer import train
    from scripts.gradient_audit import run_audit, write_result
    if (ROOT / 'calibration.json').exists():
        raise FileExistsError('Calibration already done')
    check_start()
    cfg = training_config(CALIBRATION_SEED, 'calibration', 0.0, CALIBRATION_STEPS, ROOT / 'calibration')
    cfg['training'].update(val_every=CALIBRATION_STEPS, diag_every=CALIBRATION_STEPS, ckpt_every=CALIBRATION_STEPS)
    checkpoint = train(cfg)['final_checkpoint']
    audit = run_audit(Path(checkpoint), 'cuda')
    write_result(audit, ROOT / 'calibration' / 'gradient_audit.json')
    norms = audit['means']['blocks']['block2']['gradient_norms']
    chosen = next((w for w in WEIGHTS if w * norms['robustness_unweighted'] >= RATIO * norms['mse']), WEIGHTS[-1])
    result = {'checkpoint': checkpoint, 'checkpoint_sha256': sha256_file(checkpoint), 'block2_norms': norms,
              'rule': f'smallest of {WEIGHTS} with w*robust >= {RATIO}*mse at block2, else {WEIGHTS[-1]}',
              'weight': chosen}
    write_json(ROOT / 'calibration.json', result)
    return result


def register():
    if (ROOT / 'manifest.json').exists():
        raise FileExistsError('Study already registered')
    check_start()
    calibration = json.loads((ROOT / 'calibration.json').read_text())
    uuids = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, GPUS)), '--query-gpu=uuid',
                                     '--format=csv,noheader'], text=True).strip().splitlines()
    jobs = {}
    for slot, (seed, arm) in enumerate((s, a) for s in (401, 402) for a in ('control', 'robust')):
        name = f'{arm}_s{seed}'
        base = ROOT / name
        base.mkdir(parents=True, exist_ok=False)
        train = base / 'train.yaml'
        train.write_text(yaml.safe_dump(training_config(seed, arm, calibration['weight']), sort_keys=False))
        e = load_config('configs/eval_production_all_blocks.yaml')
        e['experiment'].update(name=name, output_dir=str(base / 'eval'))
        e['wandb'].update(group='robustness-continuation', tags=['robustness-continuation', 'all-blocks', name])
        ev = base / 'eval.yaml'
        ev.write_text(yaml.safe_dump(e, sort_keys=False))
        jobs[name] = {'seed': seed, 'arm': arm, 'physical_gpu': GPUS[slot], 'gpu_uuid': uuids[slot],
                      'train_config': str(train), 'eval_config': str(ev), 'train_sha256': sha256_file(train),
                      'eval_sha256': sha256_file(ev), 'service': f'spectralfm-robust-{arm}-s{seed}'}
    m = {'registered_at': datetime.now(timezone.utc).isoformat(), 'checkpoint': CHECKPOINT,
         'checkpoint_sha256': SHA, 'steps': STEPS, 'jobs': jobs, 'robust': robust_settings(calibration['weight']),
         'noise_calibration_sha256': sha256_file(ROOT / 'noise_calibration.json'), 'calibration': calibration,
         'primary_readout': 'layer2/flat',
         'criteria': {'drift_noise1pct_max_ratio_vs_control': .5,
                      'noisy_r2_suite': ['noise_1pct', 'noise_3pct'], 'noisy_r2_probe': 'standardize+ridgecv',
                      'noisy_r2_must_beat_control': True, 'labeled_data_noisy_reported_separately': True,
                      'clean_mean_delta_raw_min_vs_control': -.005, 'clean_raw_wins_min_vs_control': -1,
                      'repeated_dataset_regression_limit': -.02, 'both_seeds_required': True},
         'heldout_perturbations': {'shift_2': ('shift', 2), 'noise_5pct': ('noise', .05),
                                   'sine_baseline_5pct': ('sine', .05)},
         'status': 'exploratory; original export, gate and holdout candidates unchanged'}
    write_json(ROOT / 'manifest.json', m)
    Path('docs/superpowers/registrations/2026-10-13-robustness-continuation.json').write_text(json.dumps(m, indent=2))
    return m


def run_job(name):
    from spectral_lejepa.training.trainer import train
    from scripts.evaluate import main as evaluate
    from scripts.gradient_audit import run_audit, write_result
    from scripts.production_policies import run as policies
    job = json.loads((ROOT / 'manifest.json').read_text())['jobs'][name]
    for kind in ('train', 'eval'):
        if sha256_file(job[kind + '_config']) != job[kind + '_sha256']:
            raise ValueError('Registered config changed')
    check_start()
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
        state['state'] = 'fixed-menu-evaluation'
        write_json(base / 'state.json', state)
        policies(base / 'eval' / f'{Path(checkpoint).parent.name}_step{STEPS}', base / 'policies',
                 publish=True, group='robustness-continuation')
        state['state'] = 'perturbation-evaluation'
        write_json(base / 'state.json', state)
        subprocess.run([sys.executable, '-m', 'scripts.robustness_eval', name], check=True)
        state['state'] = 'completed'
        write_json(base / 'state.json', state)
    except Exception as e:
        state.update(state='failed', error=repr(e))
        write_json(base / 'state.json', state)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job')
    a = p.parse_args()
    if a.job in ('calibrate', 'register'):
        print(json.dumps(globals()[a.job](), indent=2, default=str))
    else:
        run_job(a.job)
