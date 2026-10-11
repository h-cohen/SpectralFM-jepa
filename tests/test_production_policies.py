import numpy as np
import pytest

from scripts.production_policies import prepare_arms, policy_families, compare


def test_policy_menu_is_fixed_and_multiscale_has_declared_dimension():
    bank = {'layer2/flat': np.ones((3, 12288)), 'layer2': np.ones((3, 256)),
            'layer3': np.ones((3, 256)), 'layer3/seg4': np.ones((3, 1024))}
    arms = prepare_arms(bank, np.ones((3, 245)))
    assert arms['multiscale'].shape == (3, 13824)
    families = policy_families()
    assert families['fixed_block2'] == ['layer2/flat']
    assert 'raw' not in families['menu']
    assert families['menu_with_raw'] == ['raw', *families['menu']]
    assert families['multiscale'] == ['multiscale']


def test_missing_readout_and_mismatched_rows_fail_before_scoring():
    with pytest.raises(ValueError, match='Missing'):
        prepare_arms({}, np.ones((3, 245)))
    bank = {'layer2/flat': np.ones((3, 12)), 'layer2': np.ones((3, 4)),
            'layer3': np.ones((2, 4)), 'layer3/seg4': np.ones((3, 16))}
    with pytest.raises(ValueError, match='rows'):
        prepare_arms(bank, np.ones((3, 245)))


def test_outer_test_labels_cannot_choose_the_first_menu_readout():
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(40, 6)).astype(np.float32)
    bank = {arm: raw[:, :4] * (i + 1) for i, arm in enumerate(policy_families()['menu'])}
    y = raw[:, 0] + rng.normal(size=40) * .1
    protocol = dict(n_repeats=1, n_folds=2, n_inner=2, seed=42)
    first = compare(bank, raw, y, **protocol)
    changed = y.copy()
    changed[first['test_folds'][0][0]] += 1000
    second = compare(bank, raw, changed, **protocol)
    for policy in ('menu', 'menu_with_raw'):
        assert first[policy]['chosen'][0] == second[policy]['chosen'][0]
