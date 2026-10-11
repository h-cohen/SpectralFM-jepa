"""Register and execute paired objective-balance continuations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

import yaml

from spectral_lejepa.config import load_config
from spectral_lejepa.training.checkpoint import sha256_file

ROOT = Path('outputs/continuation-study')
CHECKPOINT = 'outputs/final-model/production_p48_s101_20261010-024928/checkpoint_last.pt'
SOURCE_SHA = '30486b3e78c354596d7bb5f09ebf351805309b0e9505c88d7ed698145df7fbbe'
ARMS = {'control': .05, 'low_sigreg': .01}


def training_config(seed, arm):
    c = load_config('configs/final_model.yaml')
    c['experiment'].update(name=f'continuation_{arm}_s{seed}', seed=seed, output_dir=str(ROOT / 'training'))
    c['training'].update(init_checkpoint=CHECKPOINT, resume=None, epochs=2, max_steps=99598,
                         warmup_epochs=1000.5 / 49799, learning_rate=5e-5, final_lr_ratio=.01,
                         lambda_sigreg=ARMS[arm], val_every=10000, diag_every=10000, ckpt_every=10000)
    c['wandb'].update(group='continuation-study', tags=['continuation', 'exploratory', arm, f'seed{seed}'])
    return c


def evaluation_config(name):
    c = load_config('configs/eval_production_all_blocks.yaml')
    c['experiment'].update(name=name, output_dir=str(ROOT / name / 'eval'))
    c['evaluation']['flat_blocks'] = [2, 6]
    c['wandb'].update(group='continuation-study', tags=['continuation', 'exploratory', name])
    return c


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def register():
    if sha256_file(CHECKPOINT) != SOURCE_SHA:
        raise ValueError('Starting checkpoint hash differs from the approved source')
    path = ROOT / 'manifest.json'
    if path.exists():
        raise FileExistsError('Study is already registered')
    jobs = {}
    uuids = subprocess.check_output(['nvidia-smi', '-i', '0,1,2,4', '--query-gpu=uuid',
                                     '--format=csv,noheader'], text=True).strip().splitlines()
    for gpu, (seed, arm) in enumerate((s, a) for s in (202, 203) for a in ARMS):
        name = f'{arm}_s{seed}'
        directory = ROOT / name
        directory.mkdir(parents=True, exist_ok=False)
        cfg, ev = directory / 'train.yaml', directory / 'eval.yaml'
        cfg.write_text(yaml.safe_dump(training_config(seed, arm), sort_keys=False))
        ev.write_text(yaml.safe_dump(evaluation_config(name), sort_keys=False))
        jobs[name] = {'seed': seed, 'arm': arm, 'lambda_sigreg': ARMS[arm], 'logical_gpu': gpu,
                      'gpu_uuid': uuids[gpu], 'physical_gpu': (0, 1, 2, 4)[gpu],
                      'train_config': str(cfg), 'eval_config': str(ev),
                      'train_sha256': sha256_file(cfg), 'eval_sha256': sha256_file(ev),
                      'service': f'spectralfm-continuation-{name.replace("_", "-")}'}
    write_json(path, {'registered_at': datetime.now(timezone.utc).isoformat(),
                     'checkpoint': CHECKPOINT, 'checkpoint_sha256': SOURCE_SHA, 'jobs': jobs,
                     'steps': 99598, 'status': 'registered; exploratory; no deployment promotion',
                     'baseline_policies': 'outputs/production-analysis/policies/policy_summary.json',
                     'preference': 'Each seed: mean raw gain exceeds frozen and matched control; raw win count does not decline; block2 mean R2 within .005 of frozen. Not a release gate.'})
    return jobs


def run_job(name):
    from spectral_lejepa.training.trainer import train
    from scripts.evaluate import main as evaluate
    from scripts.production_policies import run as policies
    from scripts.gradient_audit import run_audit, write_result
    job = json.loads((ROOT / 'manifest.json').read_text())['jobs'][name]
    import torch
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != job['gpu_uuid'].removeprefix('GPU-'):
        raise ValueError('CUDA device differs from registered healthy GPU UUID')
    for key in ('train', 'eval'):
        if sha256_file(job[f'{key}_config']) != job[f'{key}_sha256']:
            raise ValueError(f'{key} config changed after registration')
    if sha256_file(CHECKPOINT) != SOURCE_SHA:
        raise ValueError('Starting checkpoint changed')
    state_path = ROOT / name / 'state.json'
    if state_path.exists():
        raise FileExistsError('Run already started; resume must preserve optimizer and RNG explicitly')
    state = {'name': name, 'state': 'training', **job}
    write_json(state_path, state)
    try:
        result = train(load_config(job['train_config']))
        ckpt = result['final_checkpoint']
        state.update(checkpoint=ckpt, checkpoint_sha256=sha256_file(ckpt), state='gradient-audit')
        write_json(state_path, state)
        audit = run_audit(Path(ckpt), 'cuda')
        write_result(audit, ROOT / name / 'gradient_audit.json')
        import wandb
        from scripts.production_analysis_report import gradient_figure
        from spectral_lejepa.utils.wandb import log_figure
        with wandb.init(project='spectralfm-lejepa', group='continuation-study', job_type='gradient-audit',
                        name=name + '-gradients', config={'checkpoint': ckpt, 'seed': job['seed'], 'lambda': job['lambda_sigreg']}) as w:
            w.summary.update(audit['means'])
            log_figure(w, 'audit/weighted_term_gradients', gradient_figure(audit))
        state['state'] = 'evaluation'; write_json(state_path, state)
        evaluate(['--checkpoint', ckpt, '--config', job['eval_config']])
        bank_root = ROOT / name / 'eval' / f'{Path(ckpt).parent.name}_step99598'
        state['state'] = 'policy-comparison'; write_json(state_path, state)
        policies(bank_root, ROOT / name / 'policies', publish=True, group='continuation-study')
        state['state'] = 'completed'; write_json(state_path, state)
        subprocess.run(['flock', '-x', str(ROOT / 'report.lock'), sys.executable,
                        '-m', 'scripts.build_results_report'], check=True)
    except Exception as e:
        state.update(state='failed', error=repr(e)); write_json(state_path, state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', help='register or control_s202/low_sigreg_s202/control_s203/low_sigreg_s203')
    a = p.parse_args()
    if a.job == 'register':
        print(json.dumps(register(), indent=2))
    else:
        run_job(a.job)


if __name__ == '__main__':
    main()
