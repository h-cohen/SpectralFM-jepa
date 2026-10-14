import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('abl', Path(__file__).resolve().parents[1] / 'ops/sigreg_depth_ablation.py')
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)


def test_job_table_is_the_registered_design():
    assert abl.JOBS == {'lam0_d6_s0': (6, 0.0, 0), 'lam001_d6_s0': (6, 0.01, 0), 'lam005_d6_s0': (6, 0.05, 0),
                        'lam02_d6_s0': (6, 0.2, 0), 'lam005_d6_s1': (6, 0.05, 1), 'lam005_d4_s0': (4, 0.05, 0),
                        'lam005_d4_s1': (4, 0.05, 1), 'lam005_d2_s0': (2, 0.05, 0), 'lam005_d2_s1': (2, 0.05, 1)}


def test_queues_cover_every_job_once_on_healthy_gpus():
    q = abl.queues()
    assert sorted(q) == [0, 1, 2, 4, 5, 6, 7]
    assert q[0] == ['lam0_d6_s0', 'lam005_d2_s0'] and q[1] == ['lam001_d6_s0', 'lam005_d2_s1']
    assert sorted(j for js in q.values() for j in js) == sorted(abl.JOBS)


def test_train_command_overrides():
    cmd = ' '.join(abl.train_command('lam005_d2_s1'))
    for part in ('model.depth=2', 'training.lambda_sigreg=0.05', 'experiment.seed=1', 'training.epochs=2',
                 'training.warmup_epochs=0.2', 'experiment.name=lam005_d2_s1', 'wandb.group=sigreg-depth-ablation'):
        assert part in cmd
    assert 'training.resume=' in ' '.join(abl.train_command('lam005_d2_s1', resume='x.pt'))


def test_stage_progression(tmp_path):
    root, job = tmp_path / 'abl', 'lam005_d2_s0'
    d = root / job
    assert abl.stage(job, root) == 'train'
    run = root / 'training' / f'{job}_20261014-000000'
    run.mkdir(parents=True)
    (run / 'checkpoint_last.pt').write_bytes(b'x')
    assert abl.stage(job, root) == f'resume:{run / "checkpoint_last.pt"}'
    d.mkdir(parents=True)
    (d / 'state.json').write_text(json.dumps({'checkpoint': str(run / 'checkpoint_last.pt')}))
    assert abl.stage(job, root) == 'evaluate'
    (d / 'eval_done').write_text('')
    assert abl.stage(job, root) == 'isotropy'
    (d / 'isotropy.json').write_text('{}')
    assert abl.stage(job, root) == 'done'
