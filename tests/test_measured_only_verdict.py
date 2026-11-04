import json

import numpy as np
import pytest

from ops.measured_only_verdict import SEVEN, fig_delta, fig_valid, summarize, synthetic_share, verdict


def _per_set(deltas):
    return {s: {'m_minus_a': {'d_r2': d}} for s, d in zip(SEVEN, deltas)}


def test_verdict_not_needed():
    assert verdict(_per_set([0, .005, -.005, .02, 0, 0, -.02])) == 'synthetic not needed'


def test_verdict_helps():
    assert verdict(_per_set([-.04, -.05, -.04, -.06, 0, 0, 0])) == 'synthetic helps'


def test_verdict_mixed():
    assert verdict(_per_set([-.02] * 7)) == 'mixed'


def test_verdict_not_needed_needs_mean_above_minus_01():
    # 5/7 within .01, but two large losses pull the mean below -.01
    assert verdict(_per_set([0, 0, 0, 0, 0, -.2, -.2])) == 'mixed'


def test_verdict_ignores_labeled_data():
    per_set = _per_set([0] * 7) | {'labeled_data': {'m_minus_a': {'d_r2': -1.0}}}
    assert verdict(per_set) == 'synthetic not needed'


def test_verdict_needs_all_seven_sets():
    with pytest.raises(KeyError):
        verdict(_per_set([0] * 6))


def test_synthetic_share():
    assert synthetic_share(all_vs_raw=.10, measured_vs_raw=.04) == pytest.approx(.6)
    assert synthetic_share(all_vs_raw=0.0, measured_vs_raw=.04) is None
    assert synthetic_share(all_vs_raw=-.05, measured_vs_raw=.04) is None


def test_summarize_reads_fusion_dir(tmp_path):
    rng = np.random.default_rng(0)
    sets = [*SEVEN, 'labeled_data']
    fusion = {'mix_groups': ['a'] * 4 + ['s'] * 4 + ['m'] * 4, 'sets': {}}
    for ds in sets:
        y = rng.standard_normal(40)
        P = {'raw': y + rng.normal(0, .8, (2, 40)), 'avg_a': y + rng.normal(0, .3, (2, 40)),
             'avg_s': y + rng.normal(0, .6, (2, 40)), 'avg_m': y + rng.normal(0, .3, (2, 40))}
        P |= {f'random_{k}': y + rng.normal(0, .9, (2, 40)) for k in ('avg_a', 'avg_s', 'avg_m')}
        (tmp_path / ds).mkdir()
        np.savez(tmp_path / ds / 'fusion_oof.npz', y=y, **P)
        fusion['sets'][ds] = {}
    (tmp_path / 'fusion.json').write_text(json.dumps(fusion))

    out = summarize(tmp_path)

    assert out['verdict'] in {'synthetic not needed', 'synthetic helps', 'mixed'}
    assert set(out['per_set']) == set(sets)
    e = out['per_set']['dataset0055']
    assert set(e['arms']) == {'raw', 'random', 'all', 'synth', 'measured'}
    assert {'r2', 'rmse', 'nrmse'} <= set(e['arms']['measured'])
    for arm in ('all', 'synth', 'measured'):
        assert e['vs_raw'][arm]['d_r2'] == pytest.approx(e['arms'][arm]['r2'] - e['arms']['raw']['r2'])
    assert e['m_minus_a']['d_r2'] == pytest.approx(e['arms']['measured']['r2'] - e['arms']['all']['r2'])
    assert e['m_minus_a']['sd_r2'] > 0
    assert 'd_rmse' in e['m_minus_a']
    assert isinstance(e['measured_within_01_of_raw'], bool)
    assert out['labeled_data_excluded'] is True
    assert out['mean_m_minus_a'] == pytest.approx(np.mean([out['per_set'][s]['m_minus_a']['d_r2'] for s in SEVEN]))


def _fusion_dir(tmp_path):
    rng = np.random.default_rng(1)
    fusion = {'mix_groups': None, 'sets': {}}
    for ds in [*SEVEN, 'labeled_data']:
        y = rng.standard_normal(30)
        P = {k: y + rng.normal(0, .5, (2, 30)) for k in ('raw', 'avg_a', 'avg_s', 'avg_m', 'random_avg_m')}
        (tmp_path / ds).mkdir()
        np.savez(tmp_path / ds / 'fusion_oof.npz', y=y, **P)
        fusion['sets'][ds] = {}
    (tmp_path / 'fusion.json').write_text(json.dumps(fusion))
    return tmp_path


def test_fig_delta_writes_svg(tmp_path):
    out = summarize(_fusion_dir(tmp_path))
    fig_delta(out, tmp_path / 'figs')
    assert (tmp_path / 'figs' / 'fig_measured_only.svg').read_text().startswith('<?xml')


def test_fig_valid_writes_svg_with_best_marked(tmp_path):
    hist = {'mo_s107': {'step': [10, 20, 30], 'valid/loss': [2.0, 1.5, 1.6], 'best_step': 20},
            'mo_s108': {'step': [10, 20, 30], 'valid/loss': [2.1, 1.7, 1.4], 'best_step': 30}}
    fig_valid(hist, tmp_path)
    svg = (tmp_path / 'fig_measured_only_valid.svg').read_text()
    assert 'mo_s107' in svg and 'mo_s108' in svg
