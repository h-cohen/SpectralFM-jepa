import torch

from spectral_lejepa.data.perturb import perturb
from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
from spectral_lejepa.training.loss import LeJEPAObjective

STATS = {"mean": 5.0, "std": 2.0}
MODEL = {'num_patches': 48, 'dim': 16, 'depth': 6, 'heads': 4, 'mlp_dim': 32, 'dropout': 0.0,
         'predictor_dim': 16, 'predictor_depth': 1, 'predictor_heads': 4, 'predictor_mlp_dim': 32}
K = 2


def tiny(train=False):
    torch.manual_seed(0)
    model = build_model(MODEL, 245)
    return model.train() if train else model.eval()


def idx(b):
    return torch.arange(36).repeat(b, 1), torch.arange(36, 48).repeat(b, 1)


def test_identity_perturbation():
    x = torch.randn(32, 245)
    cfg = {"sigma_max": .2, "offset_max": .05, "slope_max": .05, "scale_max": .05, "shift_max": 1, "prob": 0}
    out = perturb(x, cfg, STATS, torch.Generator().manual_seed(0))
    assert (out - x).abs().max() < 1e-6


def test_prefix_matches_full_at_block_k():
    model = tiny()
    tokens = model.tokenizer(torch.randn(8, 245))
    with torch.no_grad():
        full = model.encoder(tokens, return_all=True)[1][K - 1]
        pre = model.encoder.prefix(tokens, K)
    assert (full - pre).abs().max() < 1e-6


def test_training_block_index_is_exported_readout():
    model = tiny()
    x = torch.randn(8, 245)
    with torch.no_grad():
        train_side = model.encoder(model.tokenizer(x), return_all=True)[1][K - 1]
        exported = EvalBackbone(model)(input_values=x, output_hidden_states=True).hidden_states[K]
    assert (train_side - exported).abs().max() < 1e-6


def test_identical_inputs_zero_robust_loss():
    h = torch.randn(16, 48, 16)
    obj = LeJEPAObjective(.05, num_slices=8, knots=5, robust_weight=.1)
    p, t = torch.randn(16, 36, 16), torch.randn(16, 48, 16)
    out = obj(p, t[:, :36], t, robust=(h, h.clone()))
    assert out["robust_loss"] < 1e-10


def test_weight_zero_matches_base():
    model = tiny()
    x = torch.randn(16, 245)
    masked, visible = idx(16)
    obj = LeJEPAObjective(.05, num_slices=8, knots=5, robust_weight=0.0)

    def run(with_robust):
        model.zero_grad()
        out = model(x, masked, visible, robust_x=x + .1 if with_robust else None)
        torch.manual_seed(1)
        res = obj(out["predicted"], out["target_masked"], out["target"],
                  robust=out["robust"] if with_robust else None)
        res["loss"].backward()
        return res["loss"].detach(), [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                       for p in model.parameters()]
    l1, g1 = run(True)
    l0, g0 = run(False)
    assert (l1 - l0).abs() < 1e-6
    assert all((a - b).abs().max() < 1e-6 for a, b in zip(g1, g0))


def test_train_mode_discrepancy_measured():
    model = tiny(train=True)
    x = torch.randn(8, 245)
    with torch.no_grad():
        a = model.encoder(model.tokenizer(x), return_all=True)[1][K - 1]
        b = model.encoder(model.tokenizer(x), return_all=True)[1][K - 1]
    diff = (a - b).abs().max().item()
    print(f"train-mode max abs discrepancy: {diff:.3e}")
    assert torch.isfinite(torch.tensor(diff))
