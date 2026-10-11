import numpy as np

import json

from scripts.permutation_audit import summarize_null, permutation_audit, write_result


def test_empirical_null_uses_plus_one_and_strict_threshold_count():
    out = summarize_null(real_score=0.3, null_scores=[0.1, 0.3, 0.5], threshold=0.3)
    assert out['empirical_p_value'] == 0.75
    assert out['fraction_above_threshold'] == 1 / 3
    assert out['count_at_or_above_real'] == 2


def test_output_persists_full_scores_and_provenance(tmp_path):
    result = {'source_hashes': {'bank': 'abc'}, 'method': {'permutation_seeds': [1000]},
              'arms': {'block': {'null_scores': [0.1, 0.2]}}}
    path = tmp_path / 'nested/audit.json'
    write_result(result, path)
    assert json.loads(path.read_text()) == result


def test_permutation_audit_is_repeatable_and_audits_rows_and_folds():
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(40, 6)).astype(np.float32)
    block = np.column_stack([raw, raw[:, :2] ** 2]).astype(np.float32)
    y = raw[:, 0] - 0.3 * raw[:, 1] + rng.normal(scale=0.2, size=40)
    raw[1] = raw[0]
    y[1] = y[0] + 1

    kwargs = dict(permutation_seeds=[1000, 1001, 1002], n_folds=5, fold_seed=42)
    first = permutation_audit(block, raw, y, **kwargs)
    second = permutation_audit(block, raw, y, **kwargs)

    assert first['arms']['block']['null_scores'] == second['arms']['block']['null_scores']
    assert first['data_audit']['duplicate_spectrum_groups'] == 1
    assert first['data_audit']['duplicate_spectrum_excess_rows'] == 1
    assert first['data_audit']['conflicting_duplicate_label_groups'] == 1
    assert len(first['folds']) == 5
    tests = [row['test_indices'] for row in first['folds']]
    assert sorted(i for fold in tests for i in fold) == list(range(40))
    for fold in first['folds']:
        assert set(fold['train_indices']).isdisjoint(fold['test_indices'])
    for arm in ('block', 'raw'):
        assert first['arms'][arm]['threshold'] == 0.02
        assert first['arms'][arm]['permutation_count'] == 3
        assert first['arms'][arm]['empirical_p_value'] >= 0.25
