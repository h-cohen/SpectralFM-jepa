import json

import numpy as np
import pytest

from spectral_lejepa.evaluation import benchmark


def test_minirocket_features_are_label_free_and_seeded():
    pytest.importorskip('sktime')
    x = np.random.default_rng(0).normal(size=(12, 245)).astype(np.float32)
    a = benchmark.features('minirocket', x, seed=0)
    assert a.shape[0] == 12 and a.shape[1] > 9000
    assert np.array_equal(a, benchmark.features('minirocket', x, seed=0))
    assert not np.array_equal(a, benchmark.features('minirocket', x, seed=1))


def _fake(root, members, n=20):
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    for m in members:
        d = root / m / 'eval' / 'r_step1' / 'dataset0055'
        d.mkdir(parents=True)
        np.savez(d / 'nested_oof.npz', y=y, raw=np.stack([y + rng.normal(size=n)] * 2),
                 **{'layer2/flat': np.stack([y + .3 * rng.normal(size=n)] * 2)})
    out = root / 'bench' / 'dataset0055'
    out.mkdir(parents=True)
    for arm, noise in (('minirocket_s0', .5), ('rocket', .8)):
        np.savez(out / f'{arm}.npz', y=y, oof=np.stack([y + noise * rng.normal(size=n)] * 2))
    for m in members:
        np.savez(out / f'concat_{m}.npz', y=y, oof=np.stack([y + .2 * rng.normal(size=n)] * 2))
    return y


def test_score_builds_combinations_and_criteria(tmp_path):
    members = ['m1', 'm2']
    _fake(tmp_path, members)
    res = benchmark.score(tmp_path / 'bench', [tmp_path / m / 'eval' / 'r_step1' for m in members])
    s = res['sets']['dataset0055']
    assert {'raw', 'jepa', 'minirocket_s0', 'rocket', 'pred_avg', 'concat'} <= set(s['r2'])
    assert s['r2']['jepa'] > s['r2']['raw']
    assert set(res['criteria']) == {'B1', 'B2', 'B3'}
    json.dumps(res)
