import torch

from spectral_lejepa.models.vit_1d import gather_tokens
from spectral_lejepa.training.loss import LeJEPAObjective


def test_components_and_total():
    torch.manual_seed(0)
    obj = LeJEPAObjective(lambda_sigreg=0.05, num_slices=64)
    p, tm, t = torch.randn(8, 12, 16), torch.randn(8, 12, 16), torch.randn(8, 24, 16)
    out = obj(p, tm, t)
    assert set(out) == {"loss", "mse_loss", "sigreg_loss"}
    assert torch.allclose(out["loss"], out["mse_loss"] + 0.05 * out["sigreg_loss"])


def test_mse_is_exact():
    out = LeJEPAObjective(0.0, num_slices=8)(torch.zeros(2, 3, 4), torch.ones(2, 3, 4), torch.randn(2, 24, 4))
    assert out["mse_loss"].item() == 1.0
    assert out["loss"].item() == 1.0


def test_sigreg_small_for_gaussian_large_for_collapse():
    torch.manual_seed(0)
    obj = LeJEPAObjective(1.0, num_slices=256)
    zeros = torch.zeros(256, 12, 32)
    gaussian = obj(zeros, zeros, torch.randn(256, 24, 32))["sigreg_loss"].item()
    collapsed = obj(zeros, zeros, 0.01 * torch.randn(256, 24, 32))["sigreg_loss"].item()
    assert gaussian < 3.0
    assert collapsed > 20.0


def test_gradients_reach_target_through_both_terms():
    torch.manual_seed(0)
    target = torch.randn(16, 24, 8, requires_grad=True)
    masked = torch.arange(12).repeat(16, 1)
    out = LeJEPAObjective(0.05, num_slices=32)(torch.zeros(16, 12, 8), gather_tokens(target, masked), target)
    out["loss"].backward()
    assert target.grad[:, :12].abs().sum() > 0   # MSE + SIGReg
    assert target.grad[:, 12:].abs().sum() > 0   # SIGReg only (unmasked positions)


def test_sigreg_runs_in_fp32_under_autocast():
    obj = LeJEPAObjective(0.05, num_slices=16)
    t = torch.randn(32, 24, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = obj(t[:, :12], t[:, :12], t)
    assert out["sigreg_loss"].dtype == torch.float32 and out["loss"].dtype == torch.float32
