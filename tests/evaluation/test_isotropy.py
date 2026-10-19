import json

import numpy as np
import torch

from spectral_lejepa.evaluation.isotropy import geometry, isotropy


def test_isotropic_features_score_as_isotropic():
    X = np.random.default_rng(0).normal(size=(2000, 32))
    g = geometry(X)
    assert g['effective_rank'] > .8 * 32 and abs(g['mean_cosine']) < .05


def test_collapsed_features_score_as_collapsed():
    rng = np.random.default_rng(0)
    X = np.outer(rng.normal(size=2000), rng.normal(size=32)) + 5.0
    g = geometry(X)
    assert g['effective_rank'] < 1.5 and g['mean_cosine'] > .9
    iso = geometry(np.random.default_rng(0).normal(size=(2000, 32)))['sigreg']   # lightly's statistic grows with batch size
    assert g['sigreg'] > 50 * iso


def test_rank_and_cosine_are_scale_free():
    X = np.random.default_rng(1).normal(size=(500, 16)) @ np.diag(np.linspace(1, 3, 16))
    a, b = geometry(X), geometry(10 * X)
    assert np.isclose(a['effective_rank'], b['effective_rank']) and np.isclose(a['mean_cosine'], b['mean_cosine'])


def test_isotropy_on_a_depth2_checkpoint(make_checkpoint, tmp_path):
    ck = torch.load(make_checkpoint('a'), weights_only=False)
    packed = tmp_path / 'packed'
    packed.mkdir()
    np.save(packed / 'spectra.npy', np.random.default_rng(0).normal(.4, .5, (300, 245)).astype(np.float32))
    np.save(packed / 'valid_rows.npy', np.arange(200, 300))
    from spectral_lejepa.models.vit_1d import build_model
    ck['config']['model']['depth'] = 2
    ck['config']['data']['packed_dir'] = str(packed)
    ck['model'] = build_model(ck['config']['model'], 245).state_dict()
    torch.save(ck, tmp_path / 'd2.pt')
    out = isotropy(tmp_path / 'd2.pt', tmp_path / 'iso.json', n=64, device='cpu')
    assert sorted(out['blocks']) == ['0', '1', '2'] and out['depth'] == 2
    v = out['blocks']['2']
    assert all(np.isfinite(v[k][m]) for k in ('trained', 'random') for m in v[k])
    assert json.loads((tmp_path / 'iso.json').read_text())['n'] == 64


def test_feature_correlation_is_identity_for_isotropic_and_full_for_collapsed():
    from spectral_lejepa.evaluation.isotropy import feature_correlation
    rng = np.random.default_rng(0)
    C, s = feature_correlation(rng.normal(size=(20000, 8)))
    assert C.shape == (8, 8) and np.allclose(np.diag(C), 1) and s['mean_abs_offdiag'] < .02
    z = rng.normal(size=(20000, 1))
    C, s = feature_correlation(z @ rng.normal(size=(1, 8)) + 1e-3 * rng.normal(size=(20000, 8)))
    assert s['mean_abs_offdiag'] > .99 and s['dist_to_identity'] > .99


def test_feature_correlation_ignores_constant_dimensions():
    from spectral_lejepa.evaluation.isotropy import feature_correlation
    X = np.random.default_rng(0).normal(size=(500, 4))
    X[:, 2] = 3.0
    C, s = feature_correlation(X)
    assert np.isfinite(C).all() and C[2, 2] == 1 and C[2, 0] == 0 and s['dead_dims'] == 1
