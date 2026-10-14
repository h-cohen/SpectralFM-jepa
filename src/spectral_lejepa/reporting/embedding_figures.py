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


def _save(fig, path, top=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, top))
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


def noise_grid_figure(rows, path):
    """Every noise set in one figure: ensemble bars per noise type and readout, member values as dots."""
    sets = list(dict.fromkeys(s for m, s, _, k, _ in rows if m == 'ensemble' and k.startswith('noise_cosine/')))
    kinds = _kinds(rows)
    members = sorted({m for m, *_ in rows if m != 'ensemble'})
    cols = 4
    fig, axes = plt.subplots(-(-len(sets) // cols), cols, figsize=(13, 2.3 * -(-len(sets) // cols)), sharey=True,
                             squeeze=False)
    x = np.arange(len(kinds))
    for ax, s in zip(axes.flat, sets):
        for i, r in enumerate(READOUTS):
            pos = x + (i - .5) * .38
            ax.bar(pos, [_value(rows, 'ensemble', s, r, f'noise_cosine/{k}') for k in kinds], .38, color=COLORS[r],
                   label=r)
            for m in members:
                ax.plot(pos, [_value(rows, m, s, r, f'noise_cosine/{k}') for k in kinds], 'k.', ms=2)
        ax.set(title=s, xticks=x, ylim=(0, 1.02))
        ax.set_xticklabels([k.replace('gaussian_', 'g_') for k in kinds], rotation=40, ha='right', fontsize=7)
    for ax in axes.flat[len(sets):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle('cosine(clean, noisy) per dataset: bars = ensemble mean, dots = members', fontsize=9)
    return _save(fig, path, top=.94)


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
    return _save(fig, path, top=.9)


def panel_figure(root, path):
    """Row 1: cosine similarity of the input and both block-2 readouts; row 2: the readouts after mean-centring."""
    a = np.load(root / PANEL_MODEL / 'arrays.npz')
    bounds, groups = (0, 30, 60, 80, 100), ('single_channel_all', 'multi_channel', 'sampled_data', 'labeled_data')
    mids = [(lo + hi) / 2 - .5 for lo, hi in zip(bounds, bounds[1:])]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    cells = [(axes[0, 0], 'input', 'input'), (axes[0, 1], 'block2_mean', 'block-2 mean'),
             (axes[0, 2], 'block2_flat', 'block-2 flat'),
             (axes[1, 1], 'block2_mean_centered', 'block-2 mean, after mean-centring'),
             (axes[1, 2], 'block2_flat_centered', 'block-2 flat, after mean-centring')]
    for ax, key, title in cells:
        im = ax.imshow(a[f'panel/{key}'], vmin=-1, vmax=1, cmap='RdBu_r')
        for b in bounds[1:-1]:
            ax.axhline(b - .5, color='k', lw=.5)
            ax.axvline(b - .5, color='k', lw=.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(mids, [g.split('_')[0] for g in groups], fontsize=7)
        ax.set_yticks(mids, [g.split('_')[0] for g in groups], fontsize=7, rotation=90, va='center')
        ax.tick_params(length=0)
    key = axes[1, 0]
    key.axis('off')
    key.text(0, .5, f'{PANEL_MODEL}, 100 spectra in four groups:\n\n' + '\n'.join(
        f'{lo}–{hi - 1}: {g}' for lo, hi, g in zip(bounds, bounds[1:], groups)) +
        '\n\nRow 1: cosine similarity.\nRow 2: the same after subtracting\nthe mean feature vector.',
        fontsize=8, va='center', transform=key.transAxes)
    fig.subplots_adjust(left=.04, right=.88, top=.95, bottom=.04, wspace=.15, hspace=.2)
    bar = fig.colorbar(im, cax=fig.add_axes((.91, .2, .015, .6)))
    bar.set_label('cosine similarity')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    return path


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
    paths.append(noise_grid_figure(rows, out / 'noise_grid.svg'))
    if docs is not None:   # the figures the training report embeds
        Path(docs).mkdir(parents=True, exist_ok=True)
        for p in (out / 'panel.svg', out / 'summary_noise.svg', out / 'noise_grid.svg'):
            shutil.copy(p, Path(docs) / p.name)
        if (out / 'sanity' / 'cosine_distributions.svg').exists():
            shutil.copy(out / 'sanity' / 'cosine_distributions.svg', Path(docs) / 'sanity_cosine_distributions.svg')
    return paths


def main():
    figures(Path('outputs/embedding-eval'), Path('docs/figures/embedding-eval'))


if __name__ == '__main__':
    main()
