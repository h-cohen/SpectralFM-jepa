import math

import numpy as np
import torch
from matplotlib.figure import Figure

from spectral_lejepa.models.tokenizer import patch_bounds
from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training import diagnostics as diag
from tests.test_model import SMALL


def test_isotropic_gaussian():
    torch.manual_seed(0)
    s, sv = diag.representation_stats(torch.randn(2000, 16))
    assert s["effective_rank"] > 14
    assert s["covariance_condition"] < 2
    assert abs(s["mean_pairwise_cosine"]) < 0.05
    assert abs(s["std"] - 1) < 0.05
    assert abs(s["covariance_trace"] - 16) < 1.5
    assert sv.shape == (16,)


def test_low_rank_is_detected():
    torch.manual_seed(0)
    s, _ = diag.representation_stats(torch.randn(2000, 2) @ torch.randn(2, 16))
    assert s["effective_rank"] < 3


def test_nearly_low_rank_has_large_condition():
    # exactly-dead directions are excluded from the condition number; tiny-but-alive ones are not
    torch.manual_seed(0)
    Z = torch.randn(2000, 2) @ torch.randn(2, 16) + 1e-4 * torch.randn(2000, 16)
    s, _ = diag.representation_stats(Z)
    assert s["covariance_condition"] > 1e6


def test_fully_collapsed_is_finite_and_alerts():
    s, _ = diag.representation_stats(torch.zeros(500, 16))
    assert s["effective_rank"] == 0.0 and s["covariance_condition"] == math.inf
    assert all(math.isfinite(v) for k, v in s.items() if k != "covariance_condition")
    assert diag.collapse_alert(s, reference_rank=10.0)


def test_collapse_alert_thresholds():
    healthy = {"std": 1.0, "effective_rank": 10.0}
    assert not diag.collapse_alert(healthy, reference_rank=12.0)
    assert diag.collapse_alert({"std": 0.05, "effective_rank": 10.0}, reference_rank=12.0)
    assert diag.collapse_alert({"std": 1.0, "effective_rank": 2.0}, reference_rank=12.0)


def test_encode_shape():
    model = build_model(SMALL, 245).eval()
    out = diag.encode(model, torch.randn(10, 245), device="cpu", batch_size=4)
    assert out.shape == (10, 24, 32) and out.dtype == torch.float32


def test_figures():
    sv = torch.linspace(5, 0.1, 16)
    assert isinstance(diag.singular_value_figure(sv), Figure)
    signals = np.random.default_rng(0).normal(size=(4, 245))
    masked = np.stack([np.sort(np.random.default_rng(i).choice(24, 12, replace=False)) for i in range(4)])
    fig = diag.masking_figure(signals, patch_bounds(), masked)
    assert isinstance(fig, Figure) and len(fig.axes) == 4
