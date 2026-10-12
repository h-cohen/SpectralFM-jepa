"""Measured frozen-feature coverage and final LayerNorm comparisons."""
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.build_results_report import DATASETS, DATASET_LABELS
from scripts.production_analysis_report import depth_figure
from scripts.report_story import _image


def collect_frozen(root):
    base=Path(root)/'outputs/frozen-feature-study'
    if not (base/'manifest.json').exists():return {}
    data=json.loads((base/'manifest.json').read_text())
    for name,job in data['jobs'].items():
        path=base/name/'state.json'
        if path.exists():job['progress']=json.loads(path.read_text())
        job['sets']={};job['heads']={}
        for ds in DATASETS:
            path=base/name/ds/'linear.json'
            if path.exists():
                point=json.loads(path.read_text())
                if point['source']['checkpoint_sha256']!=job['checkpoint_sha256']:raise ValueError('Frozen result checkpoint differs')
                job['sets'][ds]=point
            path=base/name/ds/'heads.json'
            if path.exists():
                point=json.loads(path.read_text())
                if point['source']['checkpoint_sha256']!=job['checkpoint_sha256']:raise ValueError('Head result checkpoint differs')
                job['heads'][ds]=point
    return data


def norm_figure(sets,title):
    fig,ax=plt.subplots(figsize=(10,5))
    for j,(key,label,color) in enumerate([('mean','mean','#3976b8'),('/seg4','seg4','#087e83'),('/flat','flat','#d18a25')]):
        ys=[];xs=[];errors=[]
        for i,ds in enumerate(DATASETS):
            if ds not in sets:continue
            p=sets[ds]['norm_gains'][key];ys.append(i+(j-1)*.18);xs.append(p['delta']);errors.append(p['sd'])
        ax.errorbar(xs,ys,xerr=errors,fmt='o',capsize=2,color=color,label=label)
    ax.axvline(0,color='#596b74');ax.set_yticks(range(8),[DATASET_LABELS[d] for d in DATASETS]);ax.invert_yaxis()
    ax.set(xlabel='R² before final LayerNorm − after LayerNorm ± paired bootstrap SD',title=title)
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    return fig


def heads_figure(sets,title):
    fig,ax=plt.subplots(figsize=(10,5))
    for j,(key,label,color) in enumerate([('vs_raw','Trained vs same-head raw','#087e83'),('vs_random','Trained vs same-head random','#3976b8')]):
        points=[(i,sets[d][key]) for i,d in enumerate(DATASETS) if d in sets]
        ax.errorbar([p['delta'] for i,p in points],[i+(j-.5)*.18 for i,p in points],
                    xerr=[p['sd'] for i,p in points],fmt='o',capsize=2,color=color,label=label)
    ax.axvline(0,color='#596b74');ax.set_yticks(range(8),[DATASET_LABELS[d] for d in DATASETS]);ax.invert_yaxis()
    ax.set(title=title,xlabel='Secondary head-menu ΔR² ± paired bootstrap SD');ax.legend(fontsize=8);ax.grid(alpha=.15)
    return fig


def depth_points(sets):
    return {d:{'families':p['families'],'summary':{'raw_r2':p['raw']['r2_mean']}} for d,p in sets.items()}


def publish_figures(run,data,base,heads=False):
    from spectral_lejepa.utils.wandb import log_figure
    if heads:
        scores={p.parent.name:json.loads(p.read_text()) for p in base.glob('*/heads.json')}
        figures=[('heads/paired_gains',heads_figure(scores,base.name+' · secondary heads'))]
    else:
        figures=[('features/all_blocks',depth_figure(depth_points(data['sets']),base.name+' · all fixed outputs')),
                 ('features/final_norm_effect',norm_figure(data['sets'],base.name+' · final normalization ablation'))]
    for key,fig in figures:
        fig.tight_layout();fig.savefig(base/(key.rsplit('/',1)[-1]+'.png'),dpi=150,bbox_inches='tight')
        log_figure(run,key,fig)


def render_frozen(data,export_dir=None):
    if not data:return ''
    body=['<section class="section" id="frozenFeatureStudy"><h2>Frozen-model tests · complete block coverage and final normalization</h2><p>No weight updates. Mean/seg4/flat at states0–6, plus the input to block6’s final LayerNorm. Existing fixed-family scores are reused only after feature and OOF pairing checks. New heads are a secondary product comparison; the original linear benchmark, export and qualification remain unchanged.</p><table class="table"><thead><tr><th>Checkpoint</th><th>State</th><th>Linear sets</th><th>Head sets</th></tr></thead><tbody>']
    for name,job in data['jobs'].items():
        state=job.get('progress',{}).get('state','queued');rid=job.get('progress',{}).get('run_id')
        label=html.escape(name)
        if rid:label=f'<a href="https://wandb.ai/hcohen/spectralfm-lejepa/runs/{rid}">{label}</a>'
        body.append(f'<tr><td>{label}</td><td>{html.escape(state)}</td><td>{len(job["sets"])}/8</td><td>{len(job["heads"])}/8</td></tr>')
    body.append('</tbody></table><p class="note">All comparisons reuse the existing benchmark and remain exploratory. Readout and head recipes are selected inside inner CV. This study completes previously missing flat measurements; it does not retrospectively expand the production menu or select a checkpoint for release.</p>')
    for name,job in data['jobs'].items():
        if len(job['sets'])==8:
            for fig,key in [(depth_figure(depth_points(job['sets']),f'Where the signal lives · {name} · all outputs measured'),f'frozen_all_blocks_{name}'),
                            (norm_figure(job['sets'],name+' · same block before/after LayerNorm'),f'frozen_norm_{name}')]:
                body.append(f'<h3>{name}</h3><img class="story-plot" src="{_image(fig,key,export_dir)}" alt="Measured frozen feature test for {name}">')
        if len(job['heads'])==8:
            body.append(f'<img class="story-plot" src="{_image(heads_figure(job["heads"],name+" · secondary head gains"),"frozen_heads_"+name,export_dir)}" alt="Fair head-menu gains for {name}">')
    body.append('<p class="small">Positive normalization gain means pre-LayerNorm features give higher nested-CV R²; it does not prove more mutual information. Head menu: standardized RidgeCV, PLS2/8/16, RBF kernel ridge (α.01/.1/1; γ.1/1/10 divided by feature count). Raw and random features get the same menu; learned transforms fit inside training folds. Projector training remains conditional on these results.</p></section>')
    return ''.join(body)
