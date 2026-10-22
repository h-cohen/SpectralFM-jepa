import numpy as np
import pytest
import torch

from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training.gradient_terms import audit_batch
from spectral_lejepa.training.loss import LeJEPAObjective
from spectral_lejepa.training.stage2_diag import diag

MODEL = {'num_patches': 48, 'dim': 16, 'depth': 6, 'heads': 4, 'mlp_dim': 32, 'dropout': 0.0,
         'predictor_dim': 16, 'predictor_depth': 1, 'predictor_heads': 4, 'predictor_mlp_dim': 32}
ROBUST = {'weight': .1, 'block': 2, 'prob': .5, 'sigma_max': .2, 'offset_max': .05, 'slope_max': .05,
          'scale_max': .05, 'shift_max': 1}
STATS = {'mean': .4, 'std': .5}


def _ckpt(path, packed, seed, robust):
    torch.manual_seed(seed)
    cfg = {'model': MODEL, 'data': {'sequence_length': 245, 'normalization': 'global', 'packed_dir': str(packed)},
           'derived': {'global_stats': STATS}, 'masking': {'strategy': 'random', 'mask_ratio': .75, 'num_blocks': 3},
           'training': {'lambda_sigreg': .05, 'sigreg_num_slices': 8, 'sigreg_knots': 5, 'sigreg_t_max': 3,
                        'global_weight': 1.0, 'global_views': 2, 'global_keep': .5, 'robust': robust}}
    torch.save({'model': build_model(MODEL, 245).state_dict(), 'step': 1, 'config': cfg}, path)
    return path


@pytest.fixture
def packed(tmp_path):
    d = tmp_path / 'packed'
    d.mkdir()
    np.save(d / 'spectra.npy', np.random.default_rng(0).normal(.4, .5, (300, 245)).astype(np.float32))
    np.save(d / 'valid_rows.npy', np.arange(0, 300))
    return d


def test_audit_default_groups_unchanged():
    model = build_model(MODEL, 245).eval()
    obj = LeJEPAObjective(.05, num_slices=8, knots=5, t_max=3, global_weight=1.0)
    out = audit_batch(model, obj, torch.randn(8, 245), {'strategy': 'random', 'mask_ratio': .75, 'num_blocks': 3},
                      {'views': 2, 'keep': .5}, seed=4000, device='cpu')
    assert set(out['blocks']) == {'block2', 'block6'}
    assert 'pairwise_cosine' not in out['blocks']['block2']


def test_pairwise_cosine_with_groups():
    model = build_model(MODEL, 245).eval()
    obj = LeJEPAObjective(.05, num_slices=8, knots=5, t_max=3, global_weight=1.0, robust_weight=.1)
    out = audit_batch(model, obj, torch.randn(8, 245), {'strategy': 'random', 'mask_ratio': .75, 'num_blocks': 3},
                      {'views': 2, 'keep': .5}, seed=4000, device='cpu', robust_cfg=ROBUST, stats=STATS,
                      groups={'early': list(model.encoder.blocks[0].parameters())})
    terms = [*out['terms'], 'robustness_unweighted']
    pair = out['blocks']['early']['pairwise_cosine']
    assert len(terms) == 6 and len(pair) == 15
    assert set(pair) == {f'{a}|{b}' for i, a in enumerate(terms) for b in terms[i + 1:]}
    assert all(-1 - 1e-6 <= v <= 1 + 1e-6 for v in pair.values() if v is not None)


@pytest.mark.parametrize('robust', [ROBUST, None])
def test_diag_self_reference_zero_displacement(tmp_path, packed, robust):
    ck = _ckpt(tmp_path / 'a.pt', packed, 0, robust)
    out = diag(ck, ck, tmp_path / 'o.json', device='cpu')
    assert out['features']['displacement'] == 0
    assert out['features']['effective_rank'] > 1 and out['features']['var_min_decile'] >= 0
    assert 'pairwise_cosine' in out['gradients']['early']
    other = diag(ck, _ckpt(tmp_path / 'b.pt', packed, 1, robust), tmp_path / 'o2.json', device='cpu')
    assert other['features']['displacement'] > 0
    early = out['gradients']['early']
    assert early['gradient_norms']['robustness_unweighted'] > 0
    unweighted = {k: v for k, v in early['pairwise_cosine'].items() if 'robustness_unweighted' in k and 'robustness|' not in k}
    assert len(unweighted) == 4 and all(isinstance(v, float) and np.isfinite(v) for v in unweighted.values())
    if robust is None:   # weight 0: the weighted robustness pairs are zero-norm, reported as null, not 0.0
        assert early['gradient_norms']['robustness'] == 0
        assert all(v is None for k, v in early['pairwise_cosine'].items() if 'robustness|' in k or k.endswith('|robustness'))
