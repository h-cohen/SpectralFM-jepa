"""Measured frozen-feature coverage and final LayerNorm comparisons."""
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectral_lejepa.reporting.results import DATASETS, DATASET_LABELS
from spectral_lejepa.reporting.sections.production_analysis import depth_figure
from spectral_lejepa.reporting.story import _image


def collect_frozen(root):
    base=Path(root)/'outputs/frozen-feature-study'
    if not (base/'manifest.json').exists():return {}
    data=json.loads((base/'manifest.json').read_text())
    for key,file in [('results_summary','results_summary.json'),('menu_replication','menu_replication.json'),
                     ('union_heads','union_heads.json')]:
        if (base/file).exists():data[key]=json.loads((base/file).read_text())
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
    for j,(key,label,color) in enumerate([('vs_raw','Trained vs secondary head-menu raw','#087e83'),
                                        ('vs_random','Trained vs same-head random','#3976b8'),
                                        ('trained_vs_primary_raw','Trained vs primary linear raw','#d18a25')]):
        points=[(i,sets[d][key]) for i,d in enumerate(DATASETS) if d in sets]
        ax.errorbar([p['delta'] for i,p in points],[i+(j-1)*.18 for i,p in points],
                    xerr=[p['sd'] for i,p in points],fmt='o',capsize=2,color=color,label=label)
    ax.axvline(0,color='#596b74');ax.set_yticks(range(8),[DATASET_LABELS[d] for d in DATASETS]);ax.invert_yaxis()
    ax.set(title=title,xlabel='Secondary head-menu ΔR² ± paired bootstrap SD');ax.legend(fontsize=8);ax.grid(alpha=.15)
    return fig


def depth_points(sets):
    return {d:{'families':p['families'],'summary':{'raw_r2':p['raw']['r2_mean']}} for d,p in sets.items()}


def macro_flat_figure(data):
    fig,ax=plt.subplots(figsize=(11,5))
    for name,job in data['jobs'].items():
        if len(job.get('sets',{}))!=8:continue
        values=[np.mean([p['families'][f'layer{s}/flat']['vs_raw']['delta'] for p in job['sets'].values()]) for s in range(7)]
        ax.plot(range(7),values,'o-',ms=4,label=name)
    ax.axhline(0,color='#596b74',ls='--');ax.axvspan(1.8,2.2,color='#087e83',alpha=.1)
    ax.set_xticks(range(7),['0\ntokens','1','2','3','4','5','6\nfinal LN'])
    ax.set(title='Complete flat readouts · all eight checkpoints',xlabel='Encoder output',
           ylabel='Mean ΔR² across eight datasets vs primary raw')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15)
    return fig


def result_table(rows,counts):
    body=['<div class="table-wrap"><table class="table results-matrix"><thead><tr><th>Interface · R² ± bootstrap SD</th>'+''.join(f'<th>{DATASET_LABELS[d]}<br><small class="sample-count">n={counts[d]}</small></th>' for d in DATASETS)+'</tr></thead><tbody>']
    for label,cells,raw in rows:
        values=[]
        for ds in DATASETS:
            p=cells[ds];win=p['r2_mean']>raw[ds]['r2_mean']
            cls=' class="win-cell"' if win else ''
            values.append(f'<td{cls}>{p["r2_mean"]:.4f} ± {p["bootstrap_sd"]:.4f}</td>')
        body.append(f'<tr><td>{html.escape(label)}</td>{"".join(values)}</tr>')
    return ''.join(body)+'</tbody></table></div>'


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
    summary=data.get('results_summary')
    if summary:
        body.append(f'<img class="story-plot" src="{_image(macro_flat_figure(data),"frozen_flat_all_checkpoints",export_dir)}" alt="Complete flat readout macro gains for all eight checkpoints"><p class="small">These are macro point estimates, not confidence intervals. A negative mean at block6 does not mean it loses every dataset; per-dataset block6 wins remain starred in the detailed plots.</p>')
        body.append('<h3>Completed findings · no replacement established</h3><p><strong>Block2-flat ranks first among all24 fixed readouts in every checkpoint</strong> by raw win count, then mean gain. Earlier48pseed2 remains the leading fixed candidate at7/8 wins and mean ΔR²+.08937. Missing late flat readouts did not uncover a stronger fixed interface. Pre/postfinalLayerNorm effects vary by seed/readout; removing it is not a consistent fix.</p><p>The secondary head menu uses standardization only and can weaken the raw baseline compared with the primary linear recipes. On labeled_data, secondary raw scores.7625 versus primary raw.8301; the seed101 trained score stays.9588. That larger trained-versus-secondary-raw gap is not a new encoder gain. The plots below now show both raw references. Keep primary and secondary scorecards separate.</p>')
        body.append('<table class="table"><thead><tr><th>Checkpoint</th><th>Fixed2 wins / mean Δraw</th><th>Declared menu wins / mean Δraw</th><th>Secondary head wins / mean Δraw</th><th>Head change vs its primary fixed2</th></tr></thead><tbody>')
        for row in summary['rows']:
            body.append(f'<tr><td>{row["model"]}</td><td>{row["fixed_wins"]}/8 / {row["fixed_mean_delta"]:+.5f}</td><td>{row["menu_wins"]}/8 / {row["menu_mean_delta"]:+.5f}</td><td>{row["secondary_head_wins"]}/8 / {row["secondary_head_mean_delta"]:+.5f}</td><td>{row["head_change_vs_primary_fixed"]:+.5f}</td></tr>')
        body.append('</tbody></table><p class="small">Declared menu replay uses each fixed arm’s saved inner choice and paired outer-fold predictions; seed101 exactly reproduces its earlier menu evaluation. No outer-test label chooses a readout. Replication across earlier seeds yields7/6/6wins, so the menu does not consistently dominate fixed block2. No8/8 solution is established. The subsequent projection-space pilot and its final decision appear in the next chapter; this frozen study itself made no weight updates.</p>')
        first=next(iter(data['jobs'].values()))
        raw={d:p['raw'] for d,p in first['sets'].items()};counts={d:p['n'] for d,p in first['sets'].items()}
        rows=[('Primary raw',raw,raw)]
        for name,job in data['jobs'].items():
            cells={d:p['families']['layer2/flat'] for d,p in job['sets'].items()}
            rows.append((name+' · block2-flat',cells,raw))
        control={d:{'r2_mean':p['families']['layer2/flat']['random_r2'],'bootstrap_sd':p['families']['layer2/flat']['random_sd']} for d,p in first['sets'].items()}
        rows.append(('Matched random · block2-flat',control,raw))
        body.append('<h4>Primary fixed block2 · complete score matrix</h4>'+result_table(rows,counts))
        rows=[]
        for name,job in data['jobs'].items():
            reference={d:h['arms']['raw'] for d,h in job['heads'].items()}
            for arm in ('raw','trained','random'):
                rows.append((name+' · secondary '+arm,{d:h['arms'][arm] for d,h in job['heads'].items()},reference))
        body.append('<details><summary>Secondary standardized head menu · complete R² matrix</summary>'+result_table(rows,counts)+'</details>')
        union=data.get('union_heads',{}).get('models',{})
        if union:
            body.append('<h4>Exploratory recipe union · original linear candidates retained</h4><p>Replay the union of the primary linear recipes and secondary heads using their identical inner folds. Each arm selects its branch from inner R²; outer labels only score predictions. Raw,trained andrandom all receive the union. More candidate recipes do not guarantee better held-out performance.</p><table class="table"><thead><tr><th>Checkpoint</th><th>Wins over union raw</th><th>Mean ΔR² over union raw</th><th>Trained change vs primary fixed2</th></tr></thead><tbody>')
            for name,point in union.items():
                body.append(f'<tr><td>{name}</td><td>{point["wins"]}/8</td><td>{point["mean_delta"]:+.5f}</td><td>{point["change_vs_primary_fixed"]:+.5f}</td></tr>')
            body.append('</tbody></table><p class="small">Union wins are5–6/8. Earlierseed2 trained predictions regress by.00994meanR² versus its primary fixed2 readout. This expanded-menu analysis is retrospective and does not replace the registered benchmark or establish a better production policy.</p>')
            rows=[]
            for name,point in union.items():
                reference={d:p['arms']['raw'] for d,p in point['sets'].items()}
                for arm in ('raw','trained','random'):
                    rows.append((name+' · union '+arm,{d:p['arms'][arm] for d,p in point['sets'].items()},reference))
            body.append('<details><summary>Expanded recipe union · complete R² matrix</summary>'+result_table(rows,counts)+'</details>')
    for name,job in data['jobs'].items():
        if len(job['sets'])==8:
            for fig,key in [(depth_figure(depth_points(job['sets']),f'Where the signal lives · {name} · all outputs measured'),f'frozen_all_blocks_{name}'),
                            (norm_figure(job['sets'],name+' · same block before/after LayerNorm'),f'frozen_norm_{name}')]:
                body.append(f'<h3>{name}</h3><img class="story-plot" src="{_image(fig,key,export_dir)}" alt="Measured frozen feature test for {name}">')
        if len(job['heads'])==8:
            body.append(f'<img class="story-plot" src="{_image(heads_figure(job["heads"],name+" · secondary head gains"),"frozen_heads_"+name,export_dir)}" alt="Fair head-menu gains for {name}">')
    body.append('<p class="small">Positive normalization gain means pre-LayerNorm features give higher nested-CV R²; it does not prove more mutual information. Head menu: standardized RidgeCV, PLS2/8/16, RBF kernel ridge (α.01/.1/1; γ.1/1/10 divided by feature count). Raw and random features get the same menu; learned transforms fit inside training folds. The subsequent projection pilot is reported separately.</p></section>')
    return ''.join(body)
