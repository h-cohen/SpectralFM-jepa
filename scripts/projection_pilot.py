"""Register and run paired SIGReg projection-space training pilots."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from spectral_lejepa.config import load_config
from spectral_lejepa.training.checkpoint import sha256_file
from scripts.continuation_study import write_json

ROOT=Path('outputs/projection-pilot')
CHECKPOINT='outputs/paired-long/p48_s2_20261008-161508/checkpoint_last.pt'
SHA='ec4c041dc4f83aa04e9ec568a335cdf7e27dbc7aa7fac965f2fd1f686b566ff0'


def training_config(seed,arm):
    c=load_config('configs/final_model.yaml')
    c['experiment'].update(name=f'projection_{arm}_s{seed}',seed=seed,output_dir=str(ROOT/'training'))
    c['model'].update(sigreg_projector='control' if arm=='control' else 'mlp',
                      sigreg_projector_hidden_dim=512,sigreg_projector_dim=128)
    c['training'].update(init_checkpoint=CHECKPOINT,resume=None,epochs=1,max_steps=49799,
                         warmup_epochs=1000.5/49799,learning_rate=5e-5,final_lr_ratio=.01,
                         val_every=5000,diag_every=5000,ckpt_every=5000)
    c['wandb'].update(group='projection-pilot',tags=['projection-pilot','exploratory',arm,f'seed{seed}'])
    return c


def register():
    if (ROOT/'manifest.json').exists():raise FileExistsError('Pilot already registered')
    if sha256_file(CHECKPOINT)!=SHA:raise ValueError('Starting checkpoint changed')
    uuids=subprocess.check_output(['nvidia-smi','-i','0,1,2,4','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    jobs={}
    for slot,(seed,arm) in enumerate((s,a) for s in (301,302) for a in ('control','projected')):
        name=f'{arm}_s{seed}';base=ROOT/name;base.mkdir(parents=True,exist_ok=False)
        train=base/'train.yaml';train.write_text(yaml.safe_dump(training_config(seed,arm),sort_keys=False))
        e=load_config('configs/eval_production_all_blocks.yaml')
        e['experiment'].update(name=name,output_dir=str(base/'eval'))
        e['wandb'].update(group='projection-pilot',tags=['projection-pilot','all-blocks',name])
        ev=base/'eval.yaml';ev.write_text(yaml.safe_dump(e,sort_keys=False))
        jobs[name]={'seed':seed,'arm':arm,'physical_gpu':(0,1,2,4)[slot],'gpu_uuid':uuids[slot],
                    'train_config':str(train),'eval_config':str(ev),'train_sha256':sha256_file(train),
                    'eval_sha256':sha256_file(ev),'service':f'spectralfm-projection-{arm}-s{seed}'}
    m={'registered_at':datetime.now(timezone.utc).isoformat(),'checkpoint':CHECKPOINT,'checkpoint_sha256':SHA,
       'jobs':jobs,'steps':49799,'primary_readout':'layer2/flat','secondary_readout':'layer6/flat',
       'research_preference':{'min_gain_vs_control_each_seed':.005,'min_raw_wins':7,
                              'repeated_material_regression_limit':-.02},
       'status':'exploratory; original export and qualification unchanged'}
    write_json(ROOT/'manifest.json',m)
    Path('docs/superpowers/registrations/2026-10-12-projection-pilot.json').write_text(json.dumps(m,indent=2))
    return m


def run_job(name):
    from spectral_lejepa.training.trainer import train
    from scripts.evaluate import main as evaluate
    from scripts.gradient_audit import run_audit,write_result
    from scripts.production_policies import run as policies
    job=json.loads((ROOT/'manifest.json').read_text())['jobs'][name]
    for kind in ('train','eval'):
        if sha256_file(job[kind+'_config'])!=job[kind+'_sha256']:raise ValueError('Registered config changed')
    if sha256_file(CHECKPOINT)!=SHA:raise ValueError('Starting checkpoint changed')
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')!=job['gpu_uuid'].removeprefix('GPU-'):
        raise ValueError('GPU differs from registration')
    base=ROOT/name
    if (base/'state.json').exists():raise FileExistsError('Pilot already started; recovery must preserve training state')
    state={'name':name,'state':'training',**job};write_json(base/'state.json',state)
    try:
        result=train(load_config(job['train_config']));checkpoint=result['final_checkpoint']
        state.update(state='final-gradient-audit',checkpoint=checkpoint,checkpoint_sha256=sha256_file(checkpoint))
        write_json(base/'state.json',state)
        write_result(run_audit(Path(checkpoint),'cuda'),base/'gradient_audit.json')
        state['state']='all-block-evaluation';write_json(base/'state.json',state)
        evaluate(['--checkpoint',checkpoint,'--config',job['eval_config']])
        bank=base/'eval'/f'{Path(checkpoint).parent.name}_step49799'
        state['state']='fixed-menu-evaluation';write_json(base/'state.json',state)
        policies(bank,base/'policies',publish=True,group='projection-pilot')
        state['state']='completed';write_json(base/'state.json',state)
        subprocess.run(['flock','-x',str(ROOT/'report.lock'),sys.executable,'-m','scripts.build_results_report'],check=True)
    except Exception as e:
        state.update(state='failed',error=repr(e));write_json(base/'state.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job');a=p.parse_args()
    print(json.dumps(register(),indent=2)) if a.job=='register' else run_job(a.job)
