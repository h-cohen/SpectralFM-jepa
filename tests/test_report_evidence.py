import numpy as np
import pytest

from scripts.report_evidence import build_evidence, geometry_diagnostic, whitening_comparison


def test_geometry_whitens_training_rows_and_preserves_heldout_boundary():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 6)) * np.array([20, 6, 2, 1, .5, .2])
    result = geometry_diagnostic(x, seed=42)
    assert result['before']['top1_variance'] > .7
    assert result['after_train']['top1_variance'] == pytest.approx(1 / 6)
    assert result['after_train']['rank_fraction'] == pytest.approx(1)
    assert result['n_train'] + result['n_test'] == len(x)
    assert result['n_test'] > 0
    changed = x.copy()
    changed[result['test_rows']] *= 10
    other = geometry_diagnostic(changed, seed=42)
    assert other['after_train'] == result['after_train']
    assert other['after_test']['covariance_trace'] != result['after_test']['covariance_trace']


def test_whitening_policies_use_same_outer_folds_and_inner_selection():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 5)) * np.array([15, 3, 1, .3, .1])
    y = x[:, 2] + rng.normal(size=40) * .1
    result = whitening_comparison(x, y, n_repeats=1, n_folds=2, n_inner=2)
    assert len(result['folds']) == 2
    for fold in result['folds']:
        assert fold['with']['normalizer'].startswith('whiten')
        assert fold['without']['normalizer'] in ('none', 'standardize')
        assert set(fold['train_rows']).isdisjoint(fold['test_rows'])
    assert result['delta'] == pytest.approx(result['with_r2'] - result['without_r2'])
    assert result['n'] == 40


def test_heldout_labels_cannot_change_first_outer_fold_recipe():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 4))
    y = x[:, 0] + rng.normal(size=30) * .2
    first = whitening_comparison(x, y, n_repeats=1, n_folds=2, n_inner=2)
    altered = y.copy()
    altered[first['folds'][0]['test_rows']] += 1000
    second = whitening_comparison(x, altered, n_repeats=1, n_folds=2, n_inner=2)
    assert first['folds'][0]['with'] == second['folds'][0]['with']
    assert first['folds'][0]['without'] == second['folds'][0]['without']


def test_cache_does_not_bypass_a_lower_cpu_sample_limit(tmp_path, monkeypatch):
    path = tmp_path / 'outputs/long-1/eval/lr50_s0_run_step497990/dataset0055/bank.npz'
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(3)
    np.savez(path, bank__layer6=rng.normal(size=(40, 1, 1, 5)), y=rng.normal(size=40))
    monkeypatch.setattr('scripts.report_evidence.whitening_comparison', lambda x, y: {'n': len(y)})
    output = tmp_path / 'outputs/training-progress/story_evidence.json'
    first = build_evidence(tmp_path, output, max_probe_n=100)
    assert first['rows'][0]['whitening'] == {'n': 40}
    second = build_evidence(tmp_path, output, max_probe_n=20)
    assert second['rows'][0]['whitening'] is None
    assert 'exceeds CPU diagnostic limit 20' in second['rows'][0]['whitening_skip']
