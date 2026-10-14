"""SIGReg and depth ablation (local spec 2026-10-14-sigreg-depth-ablation-design): job table, per-job pipeline, launch.

    python ops/sigreg_depth_ablation.py launch            # one systemd service per GPU queue (bash -ic for W&B)
    python ops/sigreg_depth_ablation.py run JOB [JOB...]  # train -> evaluate -> isotropy, resumable
Study orchestration: delete once the results are in the report.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('outputs/sigreg-depth-ablation')
JOBS = {'lam0_d6_s0': (6, 0.0, 0), 'lam001_d6_s0': (6, 0.01, 0), 'lam005_d6_s0': (6, 0.05, 0),
        'lam02_d6_s0': (6, 0.2, 0), 'lam005_d6_s1': (6, 0.05, 1), 'lam005_d4_s0': (4, 0.05, 0),
        'lam005_d4_s1': (4, 0.05, 1), 'lam005_d2_s0': (2, 0.05, 0), 'lam005_d2_s1': (2, 0.05, 1)}
GPUS = (0, 1, 2, 4, 5, 6, 7)   # physical; GPU3 is broken
PY = [str(REPO / '.venv/bin/python'), '-m', 'spectral_lejepa']


def queues():
    order = ['lam0_d6_s0', 'lam001_d6_s0', 'lam005_d6_s0', 'lam02_d6_s0', 'lam005_d6_s1', 'lam005_d4_s0',
             'lam005_d4_s1', 'lam005_d2_s0', 'lam005_d2_s1']
    q = {g: [j] for g, j in zip(GPUS, order)}
    for g, j in zip(GPUS, order[len(GPUS):]):
        q[g].append(j)
    return q


def train_command(job, resume=None):
    depth, lam, seed = JOBS[job]
    cmd = PY + ['pretrain', '--config', 'configs/final_model.yaml', f'model.depth={depth}',
                f'training.lambda_sigreg={lam}', f'experiment.seed={seed}', f'experiment.name={job}',
                f'experiment.output_dir={ROOT}/training', 'training.epochs=2', 'training.warmup_epochs=0.2',
                'wandb.group=sigreg-depth-ablation']
    return cmd + ([f'training.resume={resume}'] if resume else [])


def stage(job, root=ROOT):
    d = Path(root) / job
    if (d / 'isotropy.json').exists():
        return 'done'
    state = json.loads((d / 'state.json').read_text()) if (d / 'state.json').exists() else {}
    if 'checkpoint' in state:
        return 'isotropy' if (d / 'eval_done').exists() else 'evaluate'
    runs = sorted((Path(root) / 'training').glob(f'{job}_*/checkpoint_last.pt'))
    return f'resume:{runs[-1]}' if runs else 'train'


def run(job):
    d = ROOT / job
    d.mkdir(parents=True, exist_ok=True)
    while (s := stage(job)) != 'done':
        print(f'[{job}] {s}', flush=True)
        if s == 'train' or s.startswith('resume:'):
            log = subprocess.run(train_command(job, s.removeprefix('resume:') if ':' in s else None),
                                 check=True, capture_output=True, text=True).stdout
            (d / 'training.log').write_text(log)
            ck = [l.split(': ', 1)[1] for l in log.splitlines() if l.startswith('final checkpoint: ')][-1]
            (d / 'state.json').write_text(json.dumps({'job': job, 'design': JOBS[job], 'checkpoint': ck}))
        elif s == 'evaluate':
            ck = json.loads((d / 'state.json').read_text())['checkpoint']
            subprocess.run(PY + ['evaluate', '--checkpoint', ck, '--config', 'configs/eval_production_all_blocks.yaml',
                                 f'experiment.output_dir={d}/eval', f'experiment.name={job}',
                                 'wandb.group=sigreg-depth-ablation'], check=True)
            (d / 'eval_done').write_text('')
        elif s == 'isotropy':
            ck = json.loads((d / 'state.json').read_text())['checkpoint']
            subprocess.run(PY + ['isotropy', '--checkpoint', ck, '--output', str(d / 'isotropy.json')], check=True)


def launch():
    ROOT.mkdir(parents=True, exist_ok=True)
    for gpu, jobs in queues().items():
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'],
                                       text=True).strip()
        inner = (f'cd {REPO} && export CUDA_VISIBLE_DEVICES={uuid} OMP_NUM_THREADS=4 JOBLIB_TEMP_FOLDER=/dev/shm '
                 f'TMPDIR={REPO}/.tmp MPLCONFIGDIR={REPO}/.tmp/mpl && mkdir -p .tmp/mpl && '
                 f'.venv/bin/python ops/sigreg_depth_ablation.py run {" ".join(jobs)} >> {ROOT}/gpu{gpu}.log 2>&1')
        subprocess.run(['systemd-run', '--user', f'--unit=spectralfm-abl-gpu{gpu}', 'bash', '-ic', inner], check=True)
        print(f'gpu{gpu} ({uuid}): {jobs}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['launch']:
        launch()
    elif sys.argv[1:2] == ['run']:
        for j in sys.argv[2:]:
            run(j)
    else:
        sys.exit(__doc__)
