"""Offline plots and summaries for exploratory production analyses."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.build_results_report import DATASETS, DATASET_LABELS
from scripts.report_story import _image


def collect_analysis(root):
    base = Path(root) / 'outputs/production-analysis'
    data = {}
    for key, name in [('manifest', 'manifest.json'), ('permutations', 'permutation_audit.json'),
                      ('gradients', 'gradient_audit.json'), ('policies', 'policies/policy_summary.json')]:
        p = base / name
        if p.exists():
            data[key] = json.loads(p.read_text())
    depth = {}
    for directory in (base / 'all-blocks/eval').glob('*_step497990'):
        for ds in DATASETS:
            nested, summary = directory / ds / 'nested_results.json', directory / ds / 'summary.json'
            if nested.exists() and summary.exists():
                measured = json.loads(nested.read_text())
                expected = data.get('manifest', {}).get('checkpoint_sha256')
                if expected and measured.get('meta', {}).get('pretraining_checkpoint_sha256') != expected:
                    raise ValueError(f'{ds}: analysis checkpoint does not match registered source')
                depth[ds] = {'families': measured['families'],
                             'summary': json.loads(summary.read_text())}
    data['depth'] = depth
    return data


def permutation_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (arm, result) in zip(axes, data['arms'].items()):
        ax.hist(result['null_scores'], bins=15, color='#3976b8', alpha=.7)
        ax.axvline(result['real_r2'], color='#087e83', lw=2, label=f"Real labels: {result['real_r2']:.3f}")
        ax.axvline(result['threshold'], color='#d18a25', ls='--', label='Original limit: .020')
        canary = data['canary']['arm_r2'][arm]
        ax.axvline(canary, color='#ba5551', ls=':', label=f'Shuffle42: {canary:.5f}')
        ax.set(title=f'{arm} · {result["permutation_count"]} permutations', xlabel='Frozen-probe R²', ylabel='Count')
        ax.legend(fontsize=8)
    fig.suptitle('Label-permutation audit · original gate unchanged')
    return fig


def gradient_figure(data):
    keys = ['mse', 'token_sigreg', 'global_invariance', 'global_sigreg']
    labels = ['Prediction', 'Token SIG', 'View', 'Pooled SIG']
    colors = ['#087e83', '#d18a25', '#596b74', '#3976b8']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(labels, [data['means']['terms'][k] for k in keys], color=colors)
    axes[0].set(title='Weighted objective values', ylabel='Scalar contribution')
    for ax, block in zip(axes[1:], ['block2', 'block6']):
        ax.bar(labels, [data['means']['blocks'][block]['gradient_norms'][k] for k in keys], color=colors)
        ax.set(title=block + ' · weighted-term gradients', ylabel='Parameter gradient norm')
    for ax in axes:
        ax.set_yscale('log'); ax.tick_params(axis='x', labelrotation=25)
    fig.suptitle('Frozen-checkpoint diagnostic · four B256 batches · no parameter updates')
    return fig


def depth_figure(depth, title='Where the signal lives · original seed101 · all blocks/readouts'):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, ds in zip(axes.flat, DATASETS):
        if ds not in depth:
            ax.set_title(DATASET_LABELS[ds]); ax.text(.5, .5, 'Pending', ha='center', transform=ax.transAxes)
            ax.set_axis_off(); continue
        point = depth[ds]
        ax.axhline(point['summary']['raw_r2'], color='#596b74', ls='--', label='Raw')
        for pool, suffix, color in [('mean', '', '#3976b8'), ('seg4', '/seg4', '#087e83'), ('flat', '/flat', '#d18a25')]:
            stages = np.arange(7)
            families = [point['families'].get(f'layer{s}{suffix}') for s in stages]
            values = np.array([f['r2_mean'] if f else np.nan for f in families])
            sd = np.array([f['bootstrap_sd'] if f else np.nan for f in families])
            ax.plot(stages, values, 'o-', ms=4, color=color,
                    label=f'{pool} ({256 if pool == "mean" else 1024 if pool == "seg4" else 12288:,}d)')
            ax.fill_between(stages, values-sd, values+sd, alpha=.12, color=color)
        ax.axvspan(5.75, 6.25, color='#d18a25', alpha=.09)
        ax.set_xticks(range(7), ['0\ntokens', '1', '2', '3', '4', '5', '6\nfinal'])
        ax.set(title=DATASET_LABELS[ds], xlabel='Encoder block', ylabel='Nested-CV R²'); ax.grid(alpha=.15)
    if DATASETS[0] in depth:
        axes[0, 0].legend(fontsize=8)
    else:
        next((ax for ax, ds in zip(axes.flat, DATASETS) if ds in depth), axes[0, 0]).legend(fontsize=8)
    fig.suptitle(title + '\nBands: bootstrap SD · missing measurements leave gaps')
    return fig


def policy_figure(data):
    names = [p for p in data['scorecard'] if p != 'raw']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for j, (policy, color) in enumerate(zip(names, ['#3976b8', '#087e83', '#d18a25', '#9a5a9b'])):
        y = np.arange(8) + (j - 1.5) * .13
        for ax, key, title in zip(axes, ['vs_raw', 'vs_random'], ['Gain over raw', 'Gain over matched random']):
            points = [data['sets'][d]['policies'][policy][key] for d in DATASETS]
            ax.errorbar([p['delta'] for p in points], y, xerr=[p['sd'] for p in points],
                        fmt='o', color=color, ms=4, capsize=2, label=policy)
            ax.set_yticks(range(8), [DATASET_LABELS[d] for d in DATASETS]); ax.set(title=title, xlabel='Paired ΔR² ± bootstrap SD')
            ax.axvline(0, color='#596b74', lw=1)
    axes[0].invert_yaxis(); axes[1].invert_yaxis(); axes[1].legend(fontsize=8)
    fig.suptitle('Controlled readout policies · same outer folds · selection inside inner CV')
    return fig


def render_analysis(data, export_dir=None):
    if not data or not any(data.get(k) for k in ('manifest', 'permutations', 'gradients', 'depth')):
        return ''
    body = ['<section class="section" id="productionAnalysis"><h2>Production analysis · frozen seed101 encoder</h2><p class="note">Exploratory reuse of the existing benchmark. The original qualification gate and model weights are unchanged; independent development/deployment data has not been supplied.</p>']
    if data.get('policies'):
        p = data['policies']
        body.append('<h3>What the completed comparison tells us</h3><table class="table"><thead><tr><th>Interface</th><th>Wins over raw</th><th>Mean ΔR² vs raw</th><th>Mean ΔR² vs random</th></tr></thead><tbody>')
        for name in ('fixed_block2', 'menu', 'menu_with_raw', 'multiscale'):
            score = p['scorecard'][name]
            random_gain = np.mean([s['policies'][name]['vs_random']['delta'] for s in p['sets'].values()])
            body.append(f'<tr><td>{name}</td><td>{score["wins"]}/8</td><td>{score["mean_delta"]:+.4f}</td><td>{random_gain:+.4f}</td></tr>')
        body.append('</tbody></table><p>The four-readout menu is the leading development interface. It improves coverage without retraining, but does not dominate fixed block2 on every task. The menu raises 0055 and 0106 while reducing 0112 from 0.7276 to 0.6509; one of ten outer folds chooses a different readout. A small-sample selection can hurt held-out performance even though selection is correctly confined to inner CV. Keep the fixed export unchanged until menu replication or independent validation.</p>')
        body.append('<p>The new raw wins on 0106 (+0.0031 ± 0.0282 paired bootstrap SD) and 0114 (+0.0018 ± 0.0015) are small. A positive point estimate counts as a win under the declared rule, but is not evidence of a reliable advantage. 0109 still loses. Raw fallback and fixed concatenation do not improve the aggregate menu result; fallback cannot guarantee raw performance on unseen rows.</p>')
        body.append('<p><strong>Hypothesis tested:</strong> these depth and gradient measurements motivated matched continuations with unchanged versus lower SIGReg. The completed continuation chapter below compares those finals with this frozen baseline. Lower self-supervised loss alone is not a promotion criterion.</p>')
        body.append('<p class="small">The measurements do not prove SIGReg caused late-layer degradation. Block2 is not required by the architecture, and no new readout is added retrospectively to claim 8/8. Matched duration/objective continuations are now complete, with no compelling replacement for this checkpoint. Before model promotion, supply independent spectra/parameter_0 labels with source, simulation or batch IDs for group-disjoint development and a reserved release holdout. The original canary failure remains unresolved.</p>')
    def add(fig, name, alt):
        body.append(f'<img class="story-plot" src="{_image(fig, name, export_dir)}" alt="{alt}">')
    if data.get('depth'):
        add(depth_figure(data['depth']), 'seed101_all_block_depth', 'Seed101 R squared by every block and readout')
        body.append(f'<p class="small">{len(data["depth"])}/8 dataset evaluations available. All seven states include flat, mean and seg4 readouts; tokenizer=0, final normalized output=6. Bands describe bootstrap uncertainty for this single trained encoder.</p>')
    if data.get('gradients'):
        d = data['gradients']; add(gradient_figure(d), 'production_gradient_audit', 'Scalar objective and actual per-term gradients')
        b2, b6 = d['means']['blocks']['block2']['gradient_norms'], d['means']['blocks']['block6']['gradient_norms']
        fraction = (d['means']['terms']['token_sigreg'] + d['means']['terms']['global_sigreg']) / d['means']['total_loss']
        body.append(f'<p>Block2 gradient norms: prediction {b2["mse"]:.4f}, token SIGReg {b2["token_sigreg"]:.4f}, pooled SIGReg {b2["global_sigreg"]:.4f}. Block6: prediction {b6["mse"]:.4f}, token SIGReg {b6["token_sigreg"]:.4f}, pooled SIGReg {b6["global_sigreg"]:.4f}. Regularization contributes {fraction:.1%} of scalar loss, but this does not imply the same gradient dominance at the production block. This motivates a controlled late-regularization experiment, not an automatic objective change.</p><p class="small">FP32 evaluation-mode diagnostic on generic corpus validation rows. Gradient norms are not AdamW update magnitudes and do not measure downstream-label alignment. Parameters and stored gradients remain unchanged.</p>')
    if data.get('permutations'):
        d = data['permutations']; add(permutation_figure(d), 'production_permutation_audit', 'Real and shuffled-label R squared distributions')
        block, raw = d['arms']['block'], d['arms']['raw']
        body.append(f'<p>Registered permutations: {block["permutation_count"]}. Conditional empirical p for real-label scores: block {block["empirical_p_value"]:.4f}, raw {raw["empirical_p_value"]:.4f}. The original shuffle42 failure is reproduced. New scores above the original .020 limit: block {block["count_above_threshold"]}/{block["permutation_count"]}, raw {raw["count_above_threshold"]}/{raw["permutation_count"]}. The original block canary remains an unusual outcome within this sample. This audit does not clear leakage or provide an independent production test; the original failed gate remains failed.</p>')
        body.append(f'<p class="small">Exact raw duplicates: {d["data_audit"]["duplicate_spectrum_groups"]}; conflicting duplicate labels: {d["data_audit"]["conflicting_duplicate_label_groups"]}. Train/test folds are disjoint. This is an exact-row audit, not proof of independent simulation groups or absence of near duplicates.</p>')
    policies = data.get('policies')
    if policies:
        add(policy_figure(policies), 'production_policy_gains', 'Policy gains over raw and architecture-matched random controls')
        body.append('<h3>Controlled interfaces · inner-selected recipes only</h3><div class="table-wrap"><table class="table results-matrix"><thead><tr><th>Policy · bootstrap SD</th>' + ''.join(f'<th>{DATASET_LABELS[d]}<br><small class="sample-count">n={policies["sets"][d]["n"]}</small></th>' for d in DATASETS) + '</tr></thead><tbody>')
        for policy, score in policies['scorecard'].items():
            cells = []
            for ds in DATASETS:
                p = policies['sets'][ds]['policies'][policy]; cls = ' class="win-cell"' if p['vs_raw']['delta'] > 0 else ''
                cells.append(f'<td{cls}>{p["r2"]:.4f} ± {p["sd"]:.4f}</td>')
            body.append(f'<tr><td>{policy} · {score["wins"]}/8 wins</td>{"".join(cells)}</tr>')
            if policy != 'raw':
                control = []
                for ds in DATASETS:
                    p = policies['sets'][ds]['policies'][policy]
                    baseline = policies['sets'][ds]['policies']['raw']['r2']
                    cls = ' class="win-cell"' if p['random_r2'] > baseline else ''
                    control.append(f'<td{cls}>{p["random_r2"]:.4f} ± {p["random_sd"]:.4f}</td>')
                body.append(f'<tr><td>{policy} · matched random</td>{"".join(control)}</tr>')
        body.append('</tbody></table></div><p class="small">Menu: block2-flat,block2-mean,block3-mean,block3-seg4. Menu+raw adds raw as a selectable fallback. Multiscale concatenates the four readouts(13,824features). Same rows/folds and matched random controls; selection stays inside inner CV. Reused benchmark scores are development evidence.</p>')
    else:
        body.append('<p>Controlled readout-policy comparisons are queued after the all-block evaluation.</p>')
    body.append('</section>')
    return ''.join(body)
