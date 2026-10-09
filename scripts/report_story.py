"""Measured presentation figures, embedded offline and exported for slides."""
from __future__ import annotations

import base64
import html
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TEAL, GOLD, BLUE = '#087e83', '#d18a25', '#3976b8'


def _image(fig, name, export_dir):
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=135, bbox_inches='tight')
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_dir / f'{name}.svg', bbox_inches='tight')
        (export_dir / f'{name}.png').write_bytes(buffer.getvalue())
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def sigreg_figure(rows):
    arms = {r['arm']: r for r in rows}
    names = [k for k in ('lambda02', 'control_s0', 'control_s1') if k in arms]
    labels = ['λ=.02\nseed 0' if k == 'lambda02' else f"λ=.05\nseed {k[-1]}" for k in names]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for a, key, title in zip(axes, ['valid_mse_loss', 'effective_rank', 'embedding_r2'],
                            ['Masked prediction MSE ↓', 'Singular-value entropy rank ↑', 'Downstream nested-CV R² ↑']):
        values = [arms[k].get(key) for k in names]
        a.bar(range(len(names)), values, color=[GOLD if k == 'lambda02' else TEAL for k in names], alpha=.85)
        a.set_xticks(range(len(names)), labels)
        a.set_title(title, fontsize=11)
        if key == 'embedding_r2':
            a.errorbar(range(len(names)), values, yerr=[arms[k]['embedding_sd'] for k in names], fmt='none', color='#14212b', capsize=3)
        for i, v in enumerate(values):
            a.text(i, v + max(values) * .03, f'{v:.3f}' if key != 'effective_rank' else f'{v:.1f}', ha='center', fontsize=9)
        a.set_ylim(0, max(values) * 1.23)
    fig.suptitle('Early SIGReg sensitivity: lower MSE did not mean better features\nScreen 1 · one downstream dataset · earlier normalization/data regime', fontsize=12)
    return fig


def history_figure(runs, completed=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for run in runs:
        for a, key, title in zip(axes, ['valid/mse_loss', 'valid/sigreg_loss', 'representation/effective_rank'],
                                ['Validation masked prediction MSE', 'Validation token SIGReg', 'Pooled singular-value entropy rank']):
            points = [p for p in run['history'] if p.get(key) is not None]
            if points:
                a.plot([p['step'] / 1000 for p in points], [p[key] for p in points],
                       color=[TEAL, BLUE, '#9a5a9b'][run['seed']], ls='--' if run['patches'] == 48 else '-',
                       marker='o', markersize=4,
                       label=f"{run['patches']}p s{run['seed']}")
            a.set(xlabel='Optimizer steps (thousands)', title=title)
            a.grid(alpha=.15)
    axes[-1].legend(fontsize=8)
    fig.suptitle(('Completed' if completed else 'Ongoing') + ' paired study · observed training diagnostics (not downstream scores)', fontsize=12)
    return fig


def render_story(data, export_dir=None):
    sensitivity = data.get('sigreg_sensitivity') or []
    histories = data.get('local_histories') or []
    plots = {}
    if sensitivity:
        plots['sigreg'] = _image(sigreg_figure(sensitivity), 'sigreg_early_sensitivity', export_dir)
    if histories:
        plots['history'] = _image(history_figure(histories, completed=data.get('matrix', {}).get('complete', False)), 'paired_training_geometry_history', export_dir)
    depth_html = ''
    variants = data.get('variants') or {}
    if variants.get('rows'):
        from scripts.readout_variants import depth_figure
        for patches in (24, 48):
            if any(r['patches'] == patches for r in variants['rows']):
                plots[f'depth{patches}'] = _image(depth_figure(variants, patches), f'signal_by_block_{patches}patches', export_dir)
        depth_html = '<section class="section" id="signalDepth"><h2>Where the signal lives</h2><p class="sub">Downstream R² at each encoder block, for each dataset and readout.</p><div class="card">' + ''.join(
            f'<img class="story-plot" src="{plots[key]}" alt="R squared by encoder block for {key.removeprefix("depth")} patches across eight datasets">' for key in ('depth24', 'depth48') if key in plots) + '<p class="note">Lines are three-seed means; bands are sample SD across training seeds, not bootstrap uncertainty. Faint dots are individual seeds. Compare depth within the same readout first: mean and seg4 are measured through block6; flat only through block2. Readout dimensions differ, so a higher flat score does not isolate a depth effect. Block6 includes final LayerNorm; its flattened output was not measured.</p><p>The fixed block2-flat 48-patch readout beats raw on7/8 datasets in each existing seed;0106 remains a loss. This motivates testing a fixed intermediate representation. It does not prove that final token features contain less information: these results concern linear recoverability under the tested readouts and probe protocol, not all possible decoders. The production choice is retrospective and needs a predeclared follow-up.</p></div></section>'
    def img(key, alt):
        return f'<img class="story-plot" src="{plots[key]}" alt="{alt}">' if key in plots else ''
    history_note = ''
    if histories:
        from scripts.local_wandb import timestamp_text
        stamps = [r['latest_timestamp'] for r in histories if r.get('latest_timestamp')]
        if stamps:
            history_note = f'<p class="small">Local history snapshot through {html.escape(timestamp_text(max(stamps)))}. Dots are measurements; connecting lines do not represent additional observations.</p>'
    return f'''{depth_html}<nav class="story-nav" aria-label="Report chapters"><a href="#paradigm">1 · Training idea</a><a href="#sigreg">2 · Measured evidence</a><a href="#completeResults">3 · Downstream results</a><a href="#signalDepth">Signal by block</a></nav>
<section class="section" id="paradigm"><h2>1. Learn transferable features without labels</h2><p class="sub">Predict missing spectral information while regularizing the feature distribution.</p>
<div class="card"><div class="pipeline"><div><b>12.76M spectra</b><small>Global affine input normalization preserves amplitude</small></div><span>→</span><div><b>Shared encoder</b><small>Full spectrum + masked context + subset views</small></div><span>→</span><div><b>Joint objective</b><small>Masked prediction + SIGReg + pooled-view consistency</small></div><span>→</span><div><b>Frozen features</b><small>Evaluate regression on held-out folds</small></div></div>
<p class="note"><b>SIGReg acts during training.</b> Masked prediction teaches the encoder spectral structure. SIGReg encourages an isotropic Gaussian feature distribution so that representation variance is spread across directions rather than concentrated in a few.</p>
<p class="small">Loss: masked MSE + .05 × token SIGReg + .95 × pooled-view invariance + .05 × pooled SIGReg. Source-balanced sampling allocates half the batch to regression-source spectra, without their labels. Evaluation spectra participate in pretraining without labels: evaluation is transductive.</p></div></section>
<section class="section" id="sigreg"><h2>2. Training-time regularization: measured evidence</h2><p class="sub">Separate the intended mechanism from what the experiments actually show.</p><div class="card">{img('sigreg', 'Early SIGReg weight sensitivity: prediction error, embedding rank, and downstream R squared')}
<p>The early λ=.05 controls retained more embedding rank and achieved higher downstream R² than the λ=.02 arm, despite higher masked prediction error. This is sensitivity to SIGReg weight, under an earlier recipe on one dataset.</p>
<p class="note"><b>Evidence limit.</b> No matched SIGReg-off control has been measured. The weaker-weight arm has one seed; its paired difference from the seed-0 control was −.044 ± .029 bootstrap SD and was classified neutral. The current runs combine SIGReg with prediction and view consistency, so their changes cannot be attributed to SIGReg alone.</p>
{img('history', 'Observed validation prediction error, SIGReg, and pooled embedding rank for all six ongoing runs')}{history_note}
<p class="small">The logged rank is singular-value entropy rank. Increasing rank indicates variance spread across more directions; it does not prove an isotropic Gaussian or improved downstream R². Downstream evidence appears in the scorecards below.</p></div></section><div id="results"></div>'''
