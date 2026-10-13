"""Per-dataset figures for the embedding suite (outputs/embedding-eval)."""
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

READOUTS = ('block2_mean', 'block2_flat')
COLORS = {'input': '#999999', 'block2_mean': '#3a6ea5', 'block2_flat': '#c0504d'}
PANEL_MODEL = 'robust_s401'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'svg.hashsalt': 'spectralfm'})   # reproducible SVG ids


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    return path


def _summary(root):
    with open(root / 'summary.csv') as f:
        return [(m, s, r, k, float(v)) for m, s, r, k, v in list(csv.reader(f))[1:]]


def _value(rows, model, s, r, key):
    return next(v for m, ss, rr, k, v in rows if (m, ss, rr, k) == (model, s, r, key))


def _kinds(rows):
    """Noise types in run order."""
    return list(dict.fromkeys(k.split('/', 1)[1] for *_, k, _ in rows if k.startswith('noise_cosine/')))


def noise_figure(rows, s, path):
    kinds = _kinds(rows)
    members = sorted({m for m, ss, *_ in rows if ss == s and m != 'ensemble'})
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(kinds))
    for i, r in enumerate(READOUTS):
        pos = x + (i - .5) * .38
        ax.bar(pos, [_value(rows, 'ensemble', s, r, f'noise_cosine/{k}') for k in kinds], .38,
               label=f'{r} (ensemble mean)', color=COLORS[r])
        for m in members:
            ax.plot(pos, [_value(rows, m, s, r, f'noise_cosine/{k}') for k in kinds], 'k.', ms=3)
    ax.set(xticks=x, xticklabels=kinds, ylim=(min(0, ax.get_ylim()[0]), 1.02), ylabel='cosine(clean, noisy)',
           title=f'{s}: noise robustness (dots = members)')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    return _save(fig, path)


def match_figure(rows, s, path):
    models = sorted({m for m, ss, _, k, _ in rows if ss == s and k == 'embedding_stack_match_rate'})
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(models))
    ax.bar(x - .27, [_value(rows, m, s, 'block2_flat', 'input_stack_match_rate') for m in models], .27,
           label='input', color=COLORS['input'])
    for i, r in enumerate(READOUTS):
        ax.bar(x + i * .27, [_value(rows, m, s, r, 'embedding_stack_match_rate') for m in models], .27,
               label=r, color=COLORS[r])
    ax.set(xticks=x, xticklabels=models, ylim=(0, 1), ylabel='top-5 same-stack rate', title=f'{s}: stack retrieval')
    ax.legend(frameon=False, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    return _save(fig, path)


def distribution_figure(root, members, s, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    keys = [('intra_stack', 'same stack'), ('random_inter', 'random pair'), ('preset_inter', 'same position')]
    arrays = [np.load(root / m / 'arrays.npz') for m in members]
    for ax, space in zip(axes, ('input',) + READOUTS):
        suffix, r = ('inp', READOUTS[1]) if space == 'input' else ('emb', space)
        for key, label in keys:
            vals = np.concatenate([a[f'dist/{s}/{r}/{key}_{suffix}'] for a in arrays])
            ax.hist(vals, bins=40, range=(-1, 1), histtype='step', density=True, label=label)
        ax.set(title=space, xlabel='cosine')
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f'{s}: pairwise cosine, members pooled', fontsize=9)
    return _save(fig, path)


def panel_figure(root, path):
    a = np.load(root / PANEL_MODEL / 'arrays.npz')
    names = ['panel/input'] + [f'panel/{r}{c}' for r in READOUTS for c in ('', '_centered')]
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3.3))
    for ax, n in zip(axes, names):
        ax.imshow(a[n], vmin=-1, vmax=1, cmap='RdBu_r')
        for b in (30, 60, 80):   # single_channel_all | multi_channel | sampled_data | labeled_data
            ax.axhline(b - .5, color='k', lw=.5)
            ax.axvline(b - .5, color='k', lw=.5)
        ax.set(title=n.split('/', 1)[1], xticks=[], yticks=[])
    fig.suptitle(f'Structured 100-spectrum panel ({PANEL_MODEL}); blocks: single_channel_all, multi_channel, '
                 'sampled_data, labeled_data', fontsize=9)
    return _save(fig, path)


def summary_figure(rows, path):
    sets = list(dict.fromkeys(s for m, s, _, k, _ in rows if m == 'ensemble' and k.startswith('noise_cosine/')))
    kinds = _kinds(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, .45 * len(sets) + 1.8))
    for ax, r in zip(axes, READOUTS):
        M = np.array([[_value(rows, 'ensemble', s, r, f'noise_cosine/{k}') for k in kinds] for s in sets])
        ax.imshow(M, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='w' if v < .6 else 'k')
        ax.set(xticks=range(len(kinds)), xticklabels=kinds, yticks=range(len(sets)), yticklabels=sets,
               title=f'{r}: ensemble cosine(clean, noisy)')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    return _save(fig, path)


def figures(root, docs=None):
    root = Path(root)
    rows = _summary(root)
    members = sorted({m for m, *_ in rows if m != 'ensemble'})
    out = root / 'figures'
    paths = []
    for s in dict.fromkeys(s for _, s, _, k, _ in rows if k.startswith('noise_cosine/')):
        paths.append(noise_figure(rows, s, out / s / 'noise_cosine.svg'))
    for s in dict.fromkeys(s for _, s, _, k, _ in rows if k == 'embedding_stack_match_rate'):
        paths.append(match_figure(rows, s, out / s / 'match_rate.svg'))
        paths.append(distribution_figure(root, members, s, out / s / 'cosine_distributions.svg'))
    paths.append(panel_figure(root, out / 'panel.svg'))
    paths.append(summary_figure(rows, out / 'summary_noise.svg'))
    if docs is not None:
        Path(docs).mkdir(parents=True, exist_ok=True)
        for p in (out / 'panel.svg', out / 'summary_noise.svg'):
            shutil.copy(p, Path(docs) / p.name)
    return paths


def main():
    figures(Path('outputs/embedding-eval'), Path('docs/figures/embedding-eval'))


if __name__ == '__main__':
    main()
