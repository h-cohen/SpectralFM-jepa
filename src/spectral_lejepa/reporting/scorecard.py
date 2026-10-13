"""Consistent, dataset-level figures for nested-CV scorecards."""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _dataset_n(records, dataset):
    values = [record['sets'][dataset].get('n') for record in records
              if dataset in record.get('sets', {}) and record['sets'][dataset].get('n') is not None]
    if not values:
        return 'n unavailable'
    if len(set(values)) == 1:
        return f'n={values[0]}'
    return f'n={min(values)}–{max(values)}'


def _axes(datasets, width_per_panel=3.8, height_per_panel=2.5):
    count = len(datasets)
    if not count:
        raise ValueError('at least one dataset is required')
    cols = min(4, count)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(width_per_panel * cols, height_per_panel * rows),
                             squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def dataset_comparison_figure(records, datasets, title='Per-dataset nested-CV R²'):
    """Compare raw, pretrained and random-init R² for each record and dataset."""
    fig, axes = _axes(datasets)
    colors = {'raw_r2': '#59636e', 'model_r2': '#2878b5', 'random_r2': '#e28e2c'}
    offsets = {'raw_r2': -.18, 'model_r2': 0., 'random_r2': .18}
    labels = {'raw_r2': 'Raw spectrum', 'model_r2': 'Pretrained FM', 'random_r2': 'Random init'}
    for ax, dataset in zip(axes, datasets):
        for i, record in enumerate(records):
            point = record.get('sets', {}).get(dataset, {})
            for key in colors:
                value = point.get(key)
                if value is not None and math.isfinite(value):
                    ax.scatter(value, i + offsets[key], color=colors[key], s=28, zorder=3)
        ax.set_yticks(range(len(records)), [record['label'] for record in records], fontsize=7)
        ax.set_title(f"{dataset.replace('dataset', '')} ({_dataset_n(records, dataset)})", fontsize=9)
        ax.axvline(0, color='0.8', lw=.7, zorder=0)
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Nested-CV R²', fontsize=8)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[k],
                      markeredgecolor='none', label=labels[k]) for k in colors]
    fig.suptitle(title, y=1.01)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .985), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


def paired_gains_figure(records, datasets, title='Paired nested-CV gains'):
    """Show measured paired gains and SDs; missing values remain absent."""
    fig, axes = _axes(datasets)
    comparisons = [
        ('delta_vs_raw', 'sd_vs_raw', 'Model − raw', '#2878b5'),
        ('delta_vs_random', 'sd_vs_random', 'Model − random init', '#e28e2c'),
        ('delta_vs_control', 'sd_vs_control', 'Model − matched control', '#6a9f58'),
    ]
    offsets = np.linspace(-.22, .22, len(comparisons))
    for ax, dataset in zip(axes, datasets):
        for i, record in enumerate(records):
            point = record.get('sets', {}).get(dataset, {})
            for offset, (delta_key, sd_key, _, color) in zip(offsets, comparisons):
                delta = point.get(delta_key)
                if delta is None or not math.isfinite(delta):
                    continue
                sd = point.get(sd_key)
                error = sd if sd is not None and math.isfinite(sd) else None
                ax.errorbar(delta, i + offset, xerr=error, fmt='o', color=color, markersize=4,
                            capsize=2 if error is not None else 0, linewidth=.9, zorder=3)
        ax.set_yticks(range(len(records)), [record['label'] for record in records], fontsize=7)
        ax.set_title(f"{dataset.replace('dataset', '')} ({_dataset_n(records, dataset)})", fontsize=9)
        ax.axvline(0, color='0.25', lw=1, zorder=0)
        ax.axvline(.05, color='#a43c3c', lw=1, ls='--', zorder=0)
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('R² gain (paired bootstrap SD)', fontsize=8)
    handles = [Line2D([0], [0], marker='o', color=color, label=label, lw=0)
               for _, _, label, color in comparisons]
    fig.suptitle(title + ' (dashed line: +0.05 strong-win threshold)', y=1.01)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .985), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig
