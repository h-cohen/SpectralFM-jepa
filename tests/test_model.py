import pytest
import torch

from spectral_lejepa.models.masking import random_mask
from spectral_lejepa.models.vit_1d import EvalBackbone, Encoder, build_model, gather_tokens

SMALL = dict(num_patches=24, dim=32, depth=2, heads=4, mlp_dim=64, dropout=0.0,
             predictor_dim=24, predictor_depth=2, predictor_heads=4, predictor_mlp_dim=48,
             projector="none", projector_hidden_dim=64, projector_dim=16)


def setup(projector="none", B=4, seed=0):
    torch.manual_seed(seed)
    model = build_model({**SMALL, "projector": projector}, 245).eval()
    x = torch.randn(B, 245)
    m, v = random_mask(B, 24, 0.5, torch.Generator().manual_seed(seed))
    return model, x, m, v


def test_forward_shapes():
    model, x, m, v = setup()
    out = model(x, m, v)
    assert out["predicted"].shape == (4, 12, 32)
    assert out["target"].shape == (4, 24, 32)
    assert out["target_masked"].shape == (4, 12, 32)


def test_target_masked_is_target_at_masked_positions():
    model, x, m, v = setup()
    out = model(x, m, v)
    for b in range(4):
        assert torch.equal(out["target_masked"][b], out["target"][b, m[b]])
    assert torch.equal(gather_tokens(out["target"], m), out["target_masked"])


def test_one_shared_encoder_no_ema():
    model, x, m, v = setup()
    assert sum(isinstance(mod, Encoder) for mod in model.modules()) == 1
    assert not any("ema" in name or "momentum" in name or "teacher" in name
                   for name, _ in model.named_parameters())
    assert len(list(model.buffers())) == 1  # only the tokenizer's (non-persistent) gather index
    calls = []
    model.encoder.register_forward_hook(lambda mod, inp, out: calls.append(tuple(inp[0].shape)))
    model(x, m, v)
    assert calls == [(4, 12, 32), (4, 24, 32)]  # context (visible tokens) then target (all tokens)


def test_gradient_flows_through_target_path():
    model, x, m, v = setup()
    model.train()
    out = model(x, m, v)
    assert out["target"].requires_grad
    out["target"].square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.predictor.parameters())


def test_gradient_flows_through_context_path():
    model, x, m, v = setup()
    model.train()
    model(x, m, v)["predicted"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.predictor.parameters())


def test_prediction_never_sees_masked_samples():
    model, x, m, v = setup()
    x2 = x.clone()
    for b in range(len(x)):
        for p in m[b].tolist():
            lo, hi = model.tokenizer.bounds[p]
            x2[b, lo:hi] = torch.randn(hi - lo) * 10
    with torch.no_grad():
        assert torch.allclose(model(x, m, v)["predicted"], model(x2, m, v)["predicted"], atol=1e-6)


def test_projector_mlp():
    model, x, m, v = setup(projector="mlp")
    model.train()
    out = model(x, m, v)
    assert out["predicted"].shape == (4, 12, 16) and out["target"].shape == (4, 24, 16)


def test_unknown_projector_rejected():
    with pytest.raises(ValueError, match="projector"):
        build_model({**SMALL, "projector": "linear"}, 245)


def test_eval_backbone_hidden_states():
    model, x, _, _ = setup()
    backbone = EvalBackbone(model)
    # Both calls under no_grad to ensure eval-mode TransformerEncoderLayer uses fused fast path
    with torch.no_grad():
        out = backbone(input_values=x, output_hidden_states=True)
        hs = out.hidden_states
        assert len(hs) == SMALL["depth"] + 1
        assert all(h.shape == (4, 24, 32) for h in hs)
        assert torch.allclose(hs[-1], model.encoder(model.tokenizer(x)))
    assert not hasattr(backbone, "feature_extractor") and not hasattr(backbone, "feature_projection")
