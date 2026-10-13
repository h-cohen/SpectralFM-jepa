import csv
import json

import numpy as np

from spectral_lejepa.reporting.embedding_figures import figures

KINDS = ['gaussian_std', 'gaussian_mean', 'shot_low', 'shot_high', 'gain_low', 'gain_high']
READOUTS = ['block2_mean', 'block2_flat']
GROUPS = ['single_channel_all'] * 30 + ['multi_channel'] * 30 + ['sampled_data'] * 20 + ['labeled_data'] * 20


def fake_run(root):
    rows = [('model', 'set', 'readout', 'metric', 'value')]
    for m in ['robust_s401', 'robust_p0', 'ensemble']:
        for s in ['sanity', 'dataset0055']:
            for r in READOUTS:
                rows += [(m, s, r, f'noise_cosine/{k}', .9) for k in KINDS]
        for r in READOUTS:
            rows += [(m, 'sanity', r, key, .5) for key in ('embedding_stack_match_rate', 'input_stack_match_rate', 'match_score_avg')]
        if m == 'ensemble':
            continue
        d = root / m
        d.mkdir(parents=True)
        arrays = {f'panel/{r}{c}': np.eye(100, dtype=np.float32) for r in READOUTS for c in ('', '_centered')}
        arrays['panel/input'] = np.eye(100, dtype=np.float32)
        for r in READOUTS:
            for key in ('intra_stack_emb', 'intra_stack_inp', 'random_inter_emb', 'random_inter_inp', 'preset_inter_emb', 'preset_inter_inp'):
                arrays[f'dist/sanity/{r}/{key}'] = np.linspace(-1, 1, 50)
        np.savez(d / 'arrays.npz', **arrays)
        (d / 'results.json').write_text(json.dumps({'panel': {'groups': GROUPS}}))
    with open(root / 'summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows(rows)


def test_figures_cover_every_set_and_copy_summaries(tmp_path):
    fake_run(tmp_path / 'run')
    paths = figures(tmp_path / 'run', tmp_path / 'docs')
    names = {p.relative_to(tmp_path / 'run' / 'figures').as_posix() for p in paths}
    assert {'sanity/noise_cosine.svg', 'sanity/match_rate.svg', 'sanity/cosine_distributions.svg',
            'dataset0055/noise_cosine.svg', 'panel.svg', 'summary_noise.svg', 'noise_grid.svg'} <= names
    assert sorted(p.name for p in (tmp_path / 'docs').iterdir()) == [
        'noise_grid.svg', 'panel.svg', 'sanity_cosine_distributions.svg', 'summary_noise.svg']


def test_rerun_overwrites_instead_of_accumulating(tmp_path):
    fake_run(tmp_path / 'run')
    a = figures(tmp_path / 'run', None)
    b = figures(tmp_path / 'run', None)
    assert sorted(a) == sorted(b) and all(p.read_bytes() for p in b)
