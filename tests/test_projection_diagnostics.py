import numpy as np
import torch

from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training.loss import LeJEPAObjective
from spectral_lejepa.training.projection_diagnostics import run_projection_diagnostics


SMALL = {
    "num_patches": 8, "dim": 16, "depth": 6, "heads": 4, "mlp_dim": 32,
    "dropout": 0.0, "predictor_dim": 16, "predictor_depth": 1,
    "predictor_heads": 4, "predictor_mlp_dim": 32, "projector": "none",
    "projector_hidden_dim": 32, "projector_dim": 8,
    "sigreg_projector": "mlp", "sigreg_projector_hidden_dim": 32,
    "sigreg_projector_dim": 8,
}


class RecordingRun:
    def __init__(self):
        self.logged = []

    def log(self, values, step=None):
        self.logged.append((values, step))


def test_projection_diagnostics_are_finite_and_leave_model_and_rng_unchanged(monkeypatch):
    from matplotlib.figure import Figure
    from spectral_lejepa.training import projection_diagnostics as pd

    torch.manual_seed(21)
    model = build_model(SMALL, 245).train()
    objective = LeJEPAObjective(.05, num_slices=8, knots=5, t_max=3, global_weight=1.0)
    watched = model.encoder.blocks[1].linear1.weight
    watched.grad = torch.full_like(watched, 0.125)
    params = list(model.parameters())
    weights_before = [p.detach().clone() for p in params]
    grads_before = [None if p.grad is None else p.grad.detach().clone() for p in params]
    signals = torch.randn(8, 245)
    rng_before = torch.random.get_rng_state().clone()
    run = RecordingRun()
    figures = []
    monkeypatch.setattr(pd.wb, "log_figure", lambda run, key, fig, step: figures.append((key, fig, step)))

    metrics = run_projection_diagnostics(
        run, model, objective, signals,
        {"strategy": "random", "mask_ratio": .75, "num_blocks": 3},
        {"views": 2, "keep": .5}, step=17, seed=4000, device="cpu",
    )

    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert "projection/token/effective_rank" in metrics
    assert "projection/pooled/effective_rank" in metrics
    assert "gradient_terms/block2/token_sigreg" in metrics
    assert "gradient_terms/block6/global_sigreg" in metrics
    # The caller merges these metrics with the standard diagnostic payload and logs once.
    assert run.logged == []
    assert {key for key, _, _ in figures} == {
        "projection/token_singular_values", "projection/pooled_singular_values"
    }
    assert all(isinstance(fig, Figure) and figure_step == 17 for _, fig, figure_step in figures)
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(torch.equal(p, before) for p, before in zip(params, weights_before))
    for parameter, before in zip(params, grads_before):
        assert (parameter.grad is None) == (before is None)
        if before is not None:
            assert torch.equal(parameter.grad, before)


def test_projection_diagnostics_skip_when_projector_is_absent():
    model = build_model({**SMALL, "sigreg_projector": "none"}, 245)
    objective = LeJEPAObjective(.05, num_slices=8, knots=5, global_weight=1.0)
    assert run_projection_diagnostics(None, model, objective, torch.randn(8, 245), {}, {}, 0, 0, "cpu") == {}
