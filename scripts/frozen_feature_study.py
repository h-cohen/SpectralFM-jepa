"""Frozen block coverage, final-normalization ablation and fair downstream heads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.build_results_report import DATASETS
from scripts.continuation_study import write_json
from scripts.evaluate import model_input
from spectral_lejepa.evaluation.bank import readout_arms, load_bank, save_bank
from spectral_lejepa.evaluation.nested import nested_cv, paired_delta
from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training.checkpoint import load_model, sha256_file

ROOT = Path('outputs/frozen-feature-study')
POOLS = ('mean', 'seg4', 'flat')


@torch.no_grad()
def extract_features(model, signals, device='cuda', batch_size=64):
    model.to(device).eval()
    out = {}
    for i in range(0, len(signals), batch_size):
        x = torch.as_tensor(signals[i:i+batch_size], dtype=torch.float32, device=device)
        tokens = model.tokenizer(x)
        post, hidden = model.encoder(tokens, return_all=True)
        arms = readout_arms((tokens, *hidden[:-1], post), POOLS)
        pre = readout_arms((hidden[-1],), POOLS)
        arms.update({k.replace('layer0', 'layer6_pre_norm', 1): v for k, v in pre.items()})
        for k, v in arms.items(): out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def verify_bank(old, new):
    for key, value in old.items():
        if key not in new or value.shape != new[key].shape or not np.allclose(value, new[key], atol=5e-5, rtol=1e-5):
            raise ValueError(f'Cached feature rows/values differ: {key}')


def register():
    if (ROOT / 'manifest.json').exists(): raise FileExistsError('Study already registered')
    sources = {}
    for seed in range(3):
        d = next(p for p in Path('outputs/paired-long').glob(f'p48_s{seed}_*') if p.is_dir())
        sources[f'prior_s{seed}'] = (d/'checkpoint_last.pt', Path('outputs/long-1/eval')/f'{d.name}_step497990')
    m = json.loads(Path('outputs/production-analysis/manifest.json').read_text())
    sources['baseline_s101'] = (Path(m['checkpoint']), Path('outputs/production-analysis/all-blocks/eval') / f'{Path(m["checkpoint"]).parent.name}_step497990')
    for p in sorted(Path('outputs/continuation-study').glob('*_s20*/state.json')):
        s = json.loads(p.read_text());c = Path(s['checkpoint'])
        sources[p.parent.name] = (c, p.parent/'eval'/f'{c.parent.name}_step99598')
    uuids = subprocess.check_output(['nvidia-smi','-i','0,1,2,4,5,6,7','--query-gpu=uuid','--format=csv,noheader'], text=True).strip().splitlines()
    jobs = {}
    for i, (name, (c, bank)) in enumerate(sources.items()):
        jobs[name] = {'checkpoint': str(c), 'checkpoint_sha256': sha256_file(c), 'bank_root': str(bank),
                      'slot': i % 7, 'gpu_uuid': uuids[i % 7]}
        for ds in DATASETS:
            if not (bank/ds/'bank.npz').exists(): raise FileNotFoundError(bank/ds/'bank.npz')
    data = {'registered_at': datetime.now(timezone.utc).isoformat(), 'jobs': jobs,
            'status': 'exploratory frozen feature/head study; original production gate unchanged',
            'protocol': {'n_repeats': 2, 'n_folds': 5, 'n_inner': 5, 'seed': 42},
            'heads': {'readout': 'layer2/flat', 'ridge': 'RidgeCV original20alphas',
                      'pls_components': [2,8,16], 'krr_alpha': [.01,.1,1], 'krr_gamma_multiplier': [.1,1,10]}}
    write_json(ROOT/'manifest.json', data)
    Path('docs/superpowers/registrations/2026-10-12-frozen-feature-study.json').write_text(json.dumps(data,indent=2))
    return data


def linear_scores(bank, random, raw, y, directory, source, checkpoint_sha):
    old = json.loads((source/'nested_results.json').read_text())
    rc_old = json.loads((source/'random_control/nested_results.json').read_text())
    if old['meta']['pretraining_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Cached scorecard checkpoint differs')
    needed = list(dict.fromkeys(['raw', 'layer2/flat', *[k for k in bank if k not in old['families']]]))
    families = {k: [k] for k in needed}
    kwargs = dict(n_repeats=2,n_folds=5,n_inner=5,seed=42,n_jobs=1,backend='torch',device='cuda')
    trained = nested_cv({'raw':raw,**bank},families,y,**kwargs)
    control = nested_cv({'raw':raw,**random},families,y,**kwargs)
    with np.load(source/'nested_oof.npz') as f:
        legacy_oof = {k:f[k] for k in old['families']}
    with np.load(source/'random_control/nested_oof.npz') as f:
        rc_oof = {k:f[k] for k in rc_old['families']}
    scale = max(1.,float(np.std(y)))
    for k in ('raw','layer2/flat'):
        if not np.allclose(trained[k]['oof'],legacy_oof[k],atol=1e-6*scale,rtol=1e-6):
            raise ValueError(f'Cached trained OOFs do not match: {k}')
        if not np.allclose(control[k]['oof'],rc_oof[k],atol=1e-6*scale,rtol=1e-6):
            raise ValueError(f'Cached random OOFs do not match: {k}')
    point = {'n':len(y),'families':{},'norm_gains':{}}
    predictions, random_predictions = {},{}
    for key in bank:
        t,c = (trained[key],control[key]) if key in trained else (old['families'][key],rc_old['families'][key])
        a,b = (t['oof'],c['oof']) if key in trained else (legacy_oof[key],rc_oof[key])
        predictions[key],random_predictions[key] = a,b
        point['families'][key] = {k:t[k] for k in ('r2_mean','bootstrap_sd','chosen')}
        point['families'][key].update(random_r2=c['r2_mean'],random_sd=c['bootstrap_sd'],
            vs_raw=paired_delta(y,a,trained['raw']['oof']),vs_random=paired_delta(y,a,b),
            reused=key not in trained)
    point['raw'] = {k:trained['raw'][k] for k in ('r2_mean','bootstrap_sd')}
    for suffix in ('','/seg4','/flat'):
        point['norm_gains'][suffix or 'mean'] = paired_delta(y,predictions['layer6_pre_norm'+suffix],predictions['layer6'+suffix])
    np.savez_compressed(directory/'linear_oof.npz',y=y,raw=trained['raw']['oof'],**predictions,
                        **{f'random_{k}':v for k,v in random_predictions.items()})
    return point


def run_job(name):
    from scripts.frozen_heads import compare_heads
    import wandb
    from scripts.frozen_feature_report import publish_figures
    job = json.loads((ROOT/'manifest.json').read_text())['jobs'][name]
    if sha256_file(job['checkpoint']) != job['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != job['gpu_uuid'].removeprefix('GPU-'):
        raise ValueError('Wrong GPU UUID')
    model, ckpt = load_model(job['checkpoint'])
    torch.manual_seed(0);random_model = build_model(ckpt['config']['model'],245).eval()
    base = ROOT/name;state={'name':name,'state':'linear-evaluation','completed_sets':[],**job}
    write_json(base/'state.json',state)
    with wandb.init(project='spectralfm-lejepa',group='frozen-feature-study',job_type='frozen-features',name=name,
                    config={'source':job,'primary':'existing linear benchmark','secondary':'standardized nonlinear head menu'}) as run:
        state['run_id']=run.id;write_json(base/'state.json',state)
        try:
            sets={}
            for ds in sorted(DATASETS,key=lambda d:(d=='labeled_data',d)):
                source=Path(job['bank_root'])/ds;d=base/ds;d.mkdir(parents=True,exist_ok=True)
                cached,raw,y,meta=load_bank(source/'bank.npz',readouts=POOLS)
                if Path(meta['checkpoint']).resolve()!=Path(job['checkpoint']).resolve():raise ValueError('Bank checkpoint differs')
                bank=extract_features(model,model_input(raw,ckpt['config']))
                verify_bank(cached,bank)
                rc=extract_features(random_model,model_input(raw,ckpt['config']))
                if (source/'random_control/bank.npz').exists():
                    old_rc,rr,ry,_=load_bank(source/'random_control/bank.npz',readouts=POOLS)
                    if not np.array_equal(raw,rr) or not np.array_equal(y,ry):raise ValueError('Random rows differ')
                    verify_bank(old_rc,rc)
                metadata={'checkpoint':job['checkpoint'],'checkpoint_sha256':job['checkpoint_sha256'],'set':ds,
                          'source_bank_sha256':sha256_file(source/'bank.npz'),'n':len(y),'stages':list(bank)}
                save_bank(d/'bank.npz',bank,raw,y,metadata);save_bank(d/'random_bank.npz',rc,raw,y,{**metadata,'model':'random_seed0'})
                point=linear_scores(bank,rc,raw,y,d,source,job['checkpoint_sha256']);point['source']=metadata
                write_json(d/'linear.json',point);sets[ds]=point
                state['completed_sets'].append(ds);write_json(base/'state.json',state)
                print(name,ds,'linear complete',flush=True)
                run.log({'progress/linear_sets':len(sets)})
                del cached,bank,rc
            write_json(base/'linear_summary.json',{'sets':sets,'source':job})
            publish_figures(run,{'sets':sets},base)
            state['state']='secondary-heads';write_json(base/'state.json',state)
            for ds in sorted(DATASETS,key=lambda d:(d=='labeled_data',d)):
                d=base/ds;b,raw,y,_=load_bank(d/'bank.npz',readouts=('flat',));r,_,_,_=load_bank(d/'random_bank.npz',readouts=('flat',))
                result=compare_heads(b['layer2/flat'],r['layer2/flat'],raw,y)
                result['source']=sets[ds]['source']
                with np.load(d/'linear_oof.npz') as reference:
                    if not np.array_equal(y,reference['y']):raise ValueError('Head/linear label rows differ')
                    result['trained_vs_primary_linear']=paired_delta(y,result['arms']['trained']['oof'],reference['layer2/flat'])
                    result['trained_vs_primary_raw']=paired_delta(y,result['arms']['trained']['oof'],reference['raw'])
                    result['raw_menu_vs_primary_raw']=paired_delta(y,result['arms']['raw']['oof'],reference['raw'])
                arrays={k:result['arms'][k].pop('oof') for k in result['arms']}
                np.savez_compressed(d/'head_oof.npz',y=y,**arrays)
                write_json(d/'heads.json',result);print(name,ds,'heads complete',flush=True)
                run.log({'progress/head_sets':len(list(base.glob('*/heads.json')))})
                del b,r
            state['state']='completed';write_json(base/'state.json',state)
            publish_figures(run,{'sets':sets},base,heads=True)
            subprocess.run(['flock','-x',str(ROOT/'report.lock'),sys.executable,'-m','scripts.build_results_report'],check=True)
        except Exception as e:
            state.update(state='failed',error=repr(e));write_json(base/'state.json',state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job');a=p.parse_args()
    print(json.dumps(register(),indent=2)) if a.job=='register' else run_job(a.job)


if __name__=='__main__':main()
