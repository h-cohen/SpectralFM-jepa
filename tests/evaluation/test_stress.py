import numpy as np

from spectral_lejepa.evaluation import probe_torch, stress as S


def _raw(n=40, L=245, seed=0):
    return np.random.default_rng(seed).standard_normal((n, L)).astype(np.float32) + 3


def test_variants_deterministic_and_clean_identity():
    raw = _raw()
    a, b = S.variants(raw, 3), S.variants(raw, 3)
    assert list(a) == ['clean', *S.GRID] and len(S.GRID) == 12 + 8
    assert a['clean'] is raw
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a['noise_1pct_d0'], a['noise_1pct_d1'])
    assert not np.array_equal(S.variants(raw, 4)['noise_3pct_d0'], a['noise_3pct_d0'])


def _feats(n=40, d=12, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, 0] * 2 + .1 * rng.standard_normal(n)
    return {'clean': X, 'noise': X + .3 * rng.standard_normal(X.shape)}, y


def test_probe_never_fits_corrupted_rows(monkeypatch):
    feats, y = _feats()
    seen = []
    orig = probe_torch.recipe_predictions

    def rec(X_tr, *a, **k):
        seen.append(np.asarray(X_tr))
        return orig(X_tr, *a, **k)
    monkeypatch.setattr(probe_torch, 'recipe_predictions', rec)
    S.stress_arm(feats, y, device='cpu')
    clean = {tuple(r) for r in np.asarray(feats['clean'], np.float32)} | {tuple(r) for r in feats['clean']}
    assert seen
    for X in seen:
        assert all(tuple(r) in clean for r in X)
    noisy = {tuple(r) for r in feats['noise']}
    assert not any(tuple(r) in noisy for X in seen for r in X)


def test_stress_arm_schema_and_identity():
    feats, y = _feats()
    feats['same'] = feats['clean'].copy()
    out = S.stress_arm(feats, y, device='cpu')
    keys = {'dev_r2', 'dev_r2_ridge', 'outer_r2', 'outer_rmse', 'outer_r2_ridge', 'displacement'}
    assert keys <= set(out['same']) and out['same']['displacement'] == 0
    assert out['same']['dev_r2'] == out['clean']['dev_r2']
    assert out['same']['dev_r2_ridge'] == out['clean']['dev_r2_ridge']
    assert out['noise']['displacement'] > 0
