"""Compile completed projection-pilot comparisons from saved paired predictions."""
import json
import argparse
from pathlib import Path

import numpy as np

from scripts.build_results_report import DATASETS
from scripts.local_wandb import read_history
from scripts.projection_pilot_report import collect_pilot
from spectral_lejepa.evaluation.nested import paired_delta
from spectral_lejepa.training.checkpoint import sha256_file


def compile_analysis(root=Path('.')):
    root=Path(root);base=root/'outputs/projection-pilot';data=collect_pilot(root)
    if sha256_file(root/data['checkpoint'])!=data['checkpoint_sha256']:raise ValueError('Starting checkpoint changed')
    rows={};paired={}
    for name,job in data['jobs'].items():
        state=job['progress'];score=job['results']['scorecard'];depth=job['depth']
        if state['state']!='completed' or len(depth)!=8:raise ValueError(f'{name}: incomplete pilot')
        if sha256_file(root/state['checkpoint'])!=state['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        for kind in ('train','eval'):
            if sha256_file(root/job[kind+'_config'])!=job[kind+'_sha256']:raise ValueError('Registered config changed')
        for ds,p in depth.items():
            if len([k for k in p['families'] if k.startswith('layer')])!=21:
                raise ValueError(f'{name}/{ds}: incomplete block coverage')
            if not all(np.isfinite([f['r2_mean'],f['bootstrap_sd']]).all() for f in p['families'].values()):
                raise ValueError('Nonfinite evaluation result')
        audit=json.loads((base/name/'gradient_audit.json').read_text())
        norms=audit['means']['blocks']['block6']['gradient_norms']
        histories=read_history(next((root/'wandb').glob(f'run-*-{job["run_id"]}/run-{job["run_id"]}.wandb')))
        latest={k:next((r[k] for r in reversed(histories) if k in r),None) for k in
                ('valid/mse_loss','representation/effective_rank','train/global_step','train/amp_skipped_updates',
                 'train/loss','train/grad_norm')}
        if latest['train/global_step']!=49799:raise ValueError('Training step count differs')
        if not np.isfinite([latest['train/loss'],latest['train/grad_norm']]).all():raise ValueError('Nonfinite training result')
        rows[name]={'arm':job['arm'],'checkpoint_sha256':state['checkpoint_sha256'],
                    'fixed2_wins':score['fixed_block2']['wins'],'fixed2_delta':score['fixed_block2']['mean_delta'],
                    'menu_wins':score['menu']['wins'],'menu_delta':score['menu']['mean_delta'],
                    'flat6_wins':sum(p['families']['layer6/flat']['r2_mean']>p['summary']['raw_r2'] for p in depth.values()),
                    'flat6_delta':float(np.mean([p['families']['layer6/flat']['r2_mean']-p['summary']['raw_r2'] for p in depth.values()])),
                    'block6_token_sig_to_mse':norms['token_sigreg']/norms['mse'],'final_metrics':latest}
    for seed in (301,302):
        a,b=f'projected_s{seed}',f'control_s{seed}';paired[str(seed)]={}
        rows[a]['primary_gain_vs_control']=rows[a]['fixed2_delta']-rows[b]['fixed2_delta']
        for ds in DATASETS:
            with np.load(base/a/'policies'/f'{ds}_oof.npz') as pa,np.load(base/b/'policies'/f'{ds}_oof.npz') as pb:
                if not np.array_equal(pa['y'],pb['y']) or not np.allclose(pa['raw'],pb['raw'],rtol=0,atol=1e-8):
                    raise ValueError('Paired evaluation rows/raw predictions differ')
                if not np.allclose(pa['random_fixed_block2'],pb['random_fixed_block2'],rtol=0,atol=1e-8):
                    raise ValueError('Paired random controls differ')
                paired[str(seed)][ds]=paired_delta(pa['y'],pa['fixed_block2'],pb['fixed_block2'])
    result={'assessment':data['assessment'],'rows':rows,'paired_fixed2':paired,
            'starting_fixed2_mean_delta':float(np.mean([p['vs_raw']['delta'] for p in data['starting_fixed2'].values()])),
            'uncertainty':'Paired row bootstrap SD; does not capture new-population or training-seed uncertainty.',
            'recommendation':'Stop this recipe; retain prior48pseed2/block2-flat as a retrospective validation shortlist. No export/gate change.'}
    for key in ('wandb_summary_run_id','wandb_summary_url'):
        if key in data.get('analysis',{}):result[key]=data['analysis'][key]
    (base/'results_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def publish_analysis(result):
    import wandb
    from scripts.production_analysis_report import depth_figure
    from scripts.projection_pilot_report import comparison_figure,history_figure
    from spectral_lejepa.utils.wandb import log_figure
    data=collect_pilot(Path('.'))
    run=wandb.init(project='spectralfm-lejepa',group='projection-pilot',job_type='pilot-summary',
                   name='projection-pilot-final-comparison',config={'checkpoint_sha256':data['checkpoint_sha256'],
                   'criteria':data['research_preference'],'status':'exploratory; original production gate unchanged'})
    metrics={f'{name}/{key}':value for name,row in result['rows'].items() for key,value in row.items()
             if isinstance(value,(int,float))}
    metrics['decision/primary_advance']=int(result['assessment']['status']=='research-preference-supported')
    run.log(metrics)
    log_figure(run,'comparison/primary_and_all_blocks',comparison_figure(data))
    log_figure(run,'training/geometry_prediction_gradients',history_figure(data))
    for name,job in data['jobs'].items():
        log_figure(run,'all_blocks/'+name,depth_figure(job['depth'],'Where the signal lives · '+name))
    result['wandb_summary_run_id']=run.id
    result['wandb_summary_url']=run.url
    run.finish()
    Path('outputs/projection-pilot/results_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wandb',action='store_true');args=parser.parse_args()
    result=compile_analysis()
    if args.wandb:publish_analysis(result)
    print(json.dumps(result,indent=2))
