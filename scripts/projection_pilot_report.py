"""Measured pilot status, geometry and gradient trajectories."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.local_wandb import read_history
from scripts.report_story import _image
from scripts.build_results_report import DATASETS,DATASET_LABELS


def assess_pilot(data):
    jobs=data['jobs'];details={}
    if any('results' not in j or len(j['results'].get('sets',{}))!=8 for j in jobs.values()):
        return {'status':'pending'}
    for seed in (301,302):
        control=jobs[f'control_s{seed}']['results']['scorecard']['fixed_block2']
        projected=jobs[f'projected_s{seed}']['results']['scorecard']['fixed_block2']
        gain=projected['mean_delta']-control['mean_delta']
        details[str(seed)]={'mean_gain_vs_control':gain,'projected_wins':projected['wins'],
                            'control_wins':control['wins'],
                            'passes':gain>=.005 and projected['wins']>=max(7,control['wins'])}
    bad=[]
    for ds in DATASETS:
        diffs=[jobs[f'projected_s{s}']['results']['sets'][ds]['policies']['fixed_block2']['r2']-
               jobs[f'control_s{s}']['results']['sets'][ds]['policies']['fixed_block2']['r2'] for s in (301,302)]
        if all(d<-.02 for d in diffs):bad.append(ds)
    return {'status':'research-preference-supported' if all(d['passes'] for d in details.values()) and not bad else 'stop-or-inconclusive',
            'paired_seeds':details,'repeated_regressions_vs_control':bad}


def collect_pilot(root):
    base=Path(root)/'outputs/projection-pilot'
    if not (base/'manifest.json').exists():return {}
    data=json.loads((base/'manifest.json').read_text())
    for name,job in data['jobs'].items():
        p=base/name/'state.json'
        if p.exists():job['progress']=json.loads(p.read_text())
        p=base/name/'policies/policy_summary.json'
        if p.exists():job['results']=json.loads(p.read_text())
        job['depth']={}
        for ds in DATASETS:
            paths=list((base/name/'eval').glob(f'*/{ds}/nested_results.json'))
            if paths:
                point=json.loads(paths[0].read_text())
                expected=job.get('progress',{}).get('checkpoint_sha256')
                if point['meta']['pretraining_checkpoint_sha256']!=expected:
                    raise ValueError(f'{name}/{ds}: pilot result checkpoint differs')
                job['depth'][ds]={'families':point['families'],
                                 'summary':{'raw_r2':point['families']['raw']['r2_mean']}}
        log=base/name/'pipeline.log'
        if log.exists():
            match=re.search(r'https://wandb.ai/hcohen/spectralfm-lejepa/runs/(\w+)',log.read_text())
            if match:job['run_id']=match.group(1)
        job['history']=[]
        if job.get('run_id'):
            paths=sorted((Path(root)/'wandb').glob(f'run-*-{job["run_id"]}/run-{job["run_id"]}.wandb'))
            if paths:
                try:
                    rows=read_history(paths[-1])
                    keys=('valid/mse_loss','representation/effective_rank','projection/pooled/effective_rank',
                          'gradient_terms/block6/token_sigreg','gradient_terms/block6/mse')
                    job['history']=[{'step':r.get('optimizer_step',r.get('_step')),
                                     **{k:r[k] for k in keys if k in r}} for r in rows if any(k in r for k in keys)]
                    latest=next((r for r in reversed(rows) if 'train/global_step' in r),{})
                    job['latest_step']=latest.get('train/global_step')
                except ValueError as e:job['history_error']=str(e)
    data['assessment']=assess_pilot(data)
    p=base/'results_analysis.json'
    if p.exists():
        analysis=json.loads(p.read_text())
        for name,row in analysis['rows'].items():
            if row['checkpoint_sha256']!=data['jobs'][name].get('progress',{}).get('checkpoint_sha256'):
                raise ValueError('Cached pilot analysis checkpoint differs')
        data['analysis']=analysis
    p=Path(root)/'outputs/frozen-feature-study/prior_s2'
    data['starting_fixed2']={}
    for ds in DATASETS:
        q=p/ds/'linear.json'
        if q.exists():
            s=json.loads(q.read_text())
            if s['source']['checkpoint_sha256']!=data['checkpoint_sha256']:
                raise ValueError('Pilot starting reference checkpoint differs')
            data['starting_fixed2'][ds]=s['families']['layer2/flat']
    return data


def comparison_figure(data):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    paired=data['analysis']['paired_fixed2']
    for offset,seed,color in [(-.08,301,'#3976b8'),(.08,302,'#d18a25')]:
        p=[paired[str(seed)][d] for d in DATASETS]
        axes[0].errorbar([x['delta'] for x in p],np.arange(8)+offset,
                         xerr=[x['sd'] for x in p],fmt='o',capsize=2,label=f'Seed {seed}',color=color)
    axes[0].set_yticks(range(8),[DATASET_LABELS[d] for d in DATASETS]);axes[0].invert_yaxis()
    axes[0].axvline(0,color='#596b74');axes[0].legend(fontsize=8)
    axes[0].set(title='Primary: projected − control · block2-flat',xlabel='Paired ΔR² ± bootstrap SD')
    for name,job in data['jobs'].items():
        values=[np.mean([p['families'][f'layer{s}/flat']['r2_mean']-p['summary']['raw_r2']
                         for p in job['depth'].values()]) for s in range(7)]
        axes[1].plot(range(7),values,'o-',ms=4,label=name)
    axes[1].axhline(0,color='#596b74');axes[1].legend(fontsize=8)
    axes[1].set(title='Secondary: late-layer recovery',xlabel='Block · flat readout',ylabel='Mean ΔR² vs raw · eight datasets')
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Projection pilot · later features recover, primary features do not improve')
    return fig


def history_figure(data):
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    keys=['valid/mse_loss','representation/effective_rank','projection/pooled/effective_rank','gradient_ratio']
    titles=['Backbone prediction MSE','Backbone mean-pooled rank (256d)',
            'Projected pooled rank (128d; control head unused)','Weighted token SIG / prediction gradient · block6']
    for ax,key,title in zip(axes.flat,keys,titles):
        for name,job in data['jobs'].items():
            points=[]
            for p in job['history']:
                if key=='gradient_ratio':
                    a,b=p.get('gradient_terms/block6/token_sigreg'),p.get('gradient_terms/block6/mse')
                    if a is not None and b:points.append((p['step'],a/b))
                elif key in p:points.append((p['step'],p[key]))
            if points:ax.plot([x for x,y in points],[y for x,y in points],'o-',ms=3,label=name)
        ax.set(title=title,xlabel='Additional optimizer steps');ax.grid(alpha=.15)
    if axes[0,0].lines:axes[0,0].legend(fontsize=8)
    fig.suptitle('Projection-space pilot · observed validation diagnostics')
    return fig


def render_pilot(data,export_dir=None):
    if not data:return ''
    body=['<section class="section" id="projectionPilot"><h2>Training pilot · SIGReg in projection space</h2><p>Four paired runs from earlier48pseed2:control versus a training-only256→512→128 MLP, seeds301/302,49,799additionalsteps. Both arms instantiate the same head; control does not use it in the objective. Prediction MSE and pooled invariance remain in256dbackbone space; only projected-mode token/pooled SIGReg use128d. No BatchNorm,dropout,teacher or stop-gradient. Serving features remain before the head.</p><table class="table"><thead><tr><th>Run</th><th>State</th><th>Additional steps</th></tr></thead><tbody>']
    for name,job in data['jobs'].items():
        label=name
        if job.get('run_id'):label=f'<a href="https://wandb.ai/hcohen/spectralfm-lejepa/runs/{job["run_id"]}">{name}</a>'
        body.append(f'<tr><td>{label}</td><td>{job.get("progress",{}).get("state","registered")}</td><td>{job.get("latest_step") or "pending"} /49,799</td></tr>')
    body.append('</tbody></table>')
    analysis=data.get('analysis')
    if analysis:
        if analysis.get('wandb_summary_url'):
            body.append(f'<p><a href="{analysis["wandb_summary_url"]}">W&amp;B · final paired comparison and all-block figures</a></p>')
        body.append('<h3>Conclusion · stop this projector recipe</h3><p>All four runs and eight-dataset evaluations completed. Projected fixed block2 loses mean R² against its control on both seeds; all four retain 7/8 raw wins and lose 0106. None exceeds the starting seed2 fixed embedding. The predeclared primary criterion fails.</p><table class="table"><thead><tr><th>Run</th><th>Fixed2 wins / mean Δraw</th><th>Change vs control</th><th>Flat6 wins / mean Δraw</th><th>Declared menu wins / mean Δraw</th></tr></thead><tbody>')
        for name,row in analysis['rows'].items():
            change='—' if row['arm']=='control' else f'{row["primary_gain_vs_control"]:+.5f}'
            body.append(f'<tr><td>{name}</td><td>{row["fixed2_wins"]}/8 / {row["fixed2_delta"]:+.5f}</td><td>{change}</td><td>{row["flat6_wins"]}/8 / {row["flat6_delta"]:+.5f}</td><td>{row["menu_wins"]}/8 / {row["menu_delta"]:+.5f}</td></tr>')
        body.append('</tbody></table><p>Projected final-block flat features recover to 3/8 wins on both seeds, versus 1/8 and 0/8 for controls, while block2 remains the strongest fixed family in every run. Prediction MSE falls about 32%, but measured final-block token SIGReg / prediction gradient ratios rise to about 39–44× versus 15–16× for controls. Backbone pooled rank also decreases. The projector has not demonstrated that it absorbs the regularization burden or improves isotropy. This result separates better reconstruction and late-layer recovery from a better production embedding.</p>')
        body.append(f'<img class="story-plot" src="{_image(comparison_figure(data),"projection_pilot_comparison",export_dir)}" alt="Paired primary gains with uncertainty and all-block flat gain curves">')
    finished={n:j for n,j in data['jobs'].items() if 'results' in j}
    if finished:
        from scripts.frozen_feature_report import result_table
        first=next(iter(finished.values()))['results']['sets']
        raw={d:{'r2_mean':p['policies']['raw']['r2'],'bootstrap_sd':p['policies']['raw']['sd']} for d,p in first.items()}
        counts={d:p['n'] for d,p in first.items()};rows=[('Primary raw',raw,raw)]
        if len(data.get('starting_fixed2',{}))==8:
            rows.append(('Starting prior48pseed2 · fixed2',data['starting_fixed2'],raw))
        for name,job in finished.items():
            for random in (False,True):
                cells={}
                for ds,p in job['results']['sets'].items():
                    s=p['policies']['fixed_block2'];cells[ds]={'r2_mean':s['random_r2'] if random else s['r2'],
                                                            'bootstrap_sd':s['random_sd'] if random else s['sd']}
                rows.append((name+' · fixed2'+(' · random' if random else ''),cells,raw))
        body.append('<h3>Completed final fixed2 comparisons</h3>'+result_table(rows,counts))
    body.append('<p>Predeclared research assessment: '+data.get('assessment',{'status':'pending'})['status']+'. Original production export and gate unchanged.</p>')
    if analysis:
        from scripts.production_analysis_report import depth_figure
        for name,job in finished.items():
            body.append(f'<details><summary>Where the signal lives · {name} · every block and readout</summary><img class="story-plot" src="{_image(depth_figure(job["depth"],"Where the signal lives · projection pilot · "+name),"projection_depth_"+name,export_dir)}" alt="All seven states, mean, segmented and flat R² with raw references and winning final outputs"></details>')
    if any(j['history'] for j in data['jobs'].values()):
        body.append(f'<img class="story-plot" src="{_image(history_figure(data),"projection_pilot_trajectories",export_dir)}" alt="Measured pilot prediction, geometry and per-term gradient trajectories">')
    body.append('<p class="note">Rank is singular-value entropy rank; projector geometry is not the production embedding and does not establish improved downstream performance. Gradient diagnostics use FP32 validation-mode snapshots and preserve weights,RNG and stored gradients; they are not AdamW update magnitudes. Total objectives act in different regularization spaces, so lower total loss alone does not select a winner.</p><p>Final all-block raw/random evaluations and fixed2 comparisons run automatically. A research advance requires at least+.005meanR² versus matched control on both seeds,at least7raw wins without reducing control coverage,and no repeated dataset regression worse than.02. The original production gate/export remain unchanged; independent labelled release validation is still required.</p></section>')
    return ''.join(body)
