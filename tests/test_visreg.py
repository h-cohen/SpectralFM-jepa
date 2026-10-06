import torch

from spectral_lejepa.training.loss import LeJEPAObjective
from spectral_lejepa.training.visreg import VISReg


def test_gaussian_has_small_terms():
    torch.manual_seed(0)
    out = VISReg(num_slices=64)(torch.randn(4096, 32))
    assert out["shape"] < 0.05 and out["scale"] < 0.01 and out["center"] < 0.01
    assert torch.allclose(out["reg"], out["center"] + out["scale"] + out["shape"])


def test_collapse_is_large_finite_with_gradient():
    torch.manual_seed(0)
    z = (torch.ones(256, 16) + 1e-3 * torch.randn(256, 16)).requires_grad_()
    out = VISReg(num_slices=32)(z)
    assert torch.isfinite(out["reg"]) and out["reg"] > 0.5
    out["reg"].backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_scaled_gaussian_scale_term_one_shape_small():
    torch.manual_seed(0)
    out = VISReg(num_slices=64)(2 * torch.randn(4096, 32))
    assert abs(out["scale"].item() - 1.0) < 0.05 and out["shape"] < 0.05


def test_batched_equals_mean_of_unbatched():
    z = torch.randn(3, 512, 16)
    reg = VISReg(num_slices=32)
    torch.manual_seed(1)
    b = reg(z)
    parts = []
    for i in range(3):
        torch.manual_seed(1)
        parts.append(reg(z[i]))
    for k in b:
        assert torch.allclose(b[k], torch.stack([p[k] for p in parts]).mean(), atol=1e-5)


def test_objective_visreg_combination_and_keys():
    torch.manual_seed(0)
    obj = LeJEPAObjective(0.05, num_slices=16, global_weight=0.5, regularizer="visreg", visreg_lambda=0.6)
    p, tm, t, v = torch.randn(8, 12, 16), torch.randn(8, 12, 16), torch.randn(8, 24, 16), torch.randn(3, 8, 16)
    out = obj(p, tm, t, v)
    assert {"sigreg_loss", "global_sigreg_loss", "reg_center", "reg_scale", "reg_shape"} <= set(out)
    tok = 0.4 * out["mse_loss"] + 0.6 * out["sigreg_loss"]
    glob = 0.4 * out["global_inv_loss"] + 0.6 * out["global_sigreg_loss"]
    assert torch.allclose(out["global_loss"], glob)
    assert torch.allclose(out["loss"], tok + 0.5 * glob)


def test_objective_sigreg_unchanged_and_no_reg_keys():
    p, tm, t = torch.randn(8, 12, 16), torch.randn(8, 12, 16), torch.randn(8, 24, 16)
    default = LeJEPAObjective(0.05, num_slices=16)
    explicit = LeJEPAObjective(0.05, num_slices=16, regularizer="sigreg", visreg_lambda=0.9, visreg_slices=8)
    torch.manual_seed(3); a = default(p, tm, t)
    torch.manual_seed(3); b = explicit(p, tm, t)
    assert set(a) == set(b) == {"loss", "mse_loss", "sigreg_loss"}
    assert all(torch.equal(a[k], b[k]) for k in a)
