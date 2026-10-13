import numpy as np
import torch

from spectral_lejepa.training.audit import audit_batch
from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training.loss import LeJEPAObjective


def test_gradient_audit_reports_finite_block_terms_without_mutating_parameters_or_grads():
    cfg = {
        'num_patches': 48, 'dim': 16, 'depth': 6, 'heads': 4, 'mlp_dim': 32,
        'dropout': 0.0, 'predictor_dim': 16, 'predictor_depth': 1,
        'predictor_heads': 4, 'predictor_mlp_dim': 32,
    }
    model = build_model(cfg, 245).eval()
    objective = LeJEPAObjective(.05, num_slices=8, knots=5, t_max=3, global_weight=1.0)
    watched = model.encoder.blocks[1].linear1.weight
    watched.grad = torch.full_like(watched, .125)
    params = list(model.parameters())
    weights_before = [p.detach().clone() for p in params]
    grads_before = [None if p.grad is None else p.grad.detach().clone() for p in params]
    x = torch.randn(16, 245)
    out = audit_batch(model, objective, x, {'strategy': 'random', 'mask_ratio': .75, 'num_blocks': 3},
                      {'views': 2, 'keep': .5}, seed=4000, device='cpu')

    assert set(out['terms']) == {'mse', 'token_sigreg', 'global_invariance', 'global_sigreg'}
    assert set(out['blocks']) == {'block2', 'block6'}
    for block in out['blocks'].values():
        assert all(np.isfinite(v) for v in block['gradient_norms'].values())
        assert np.isfinite(block['cosine_prediction_vs_other'])
    assert np.isfinite(out['total_loss'])
    assert all(torch.equal(p, before) for p, before in zip(params, weights_before))
    for p, before in zip(params, grads_before):
        assert (p.grad is None) == (before is None)
        if before is not None:
            assert torch.equal(p.grad, before)
