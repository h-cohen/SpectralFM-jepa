import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.gradient_audit import audit_batch
from spectral_lejepa.config import load_config
from spectral_lejepa.data.perturb import perturb, shift_one
from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training.checkpoint import load_checkpoint
from spectral_lejepa.training.loss import LeJEPAObjective
from spectral_lejepa.training.trainer import train
from tests.test_train_smoke import REPO, TINY_OVERRIDES

STATS = {"mean": 5.0, "std": 2.0}
ROBUST = {"weight": 0.3, "block": 2, "sigma_max": .2, "offset_max": .05, "slope_max": .05,
          "scale_max": .05, "shift_max": 1, "prob": .5}
MODEL = {'num_patches': 48, 'dim': 16, 'depth': 6, 'heads': 4, 'mlp_dim': 32, 'dropout': 0.0,
         'predictor_dim': 16, 'predictor_depth': 1, 'predictor_heads': 4, 'predictor_mlp_dim': 32,
         'projector': 'none', 'projector_hidden_dim': 32, 'projector_dim': 8}
MASK = {'strategy': 'random', 'mask_ratio': .75, 'num_blocks': 3}


def test_perturb_is_identity_at_zero_and_deterministic():
    z = torch.randn(32, 245)
    zero = {**{k: 0 for k in ROBUST}, "prob": 1}
    assert torch.allclose(perturb(z, zero, STATS, torch.Generator().manual_seed(0)), z, atol=1e-5)
    assert torch.allclose(perturb(z, {**ROBUST, "prob": 0}, STATS, torch.Generator().manual_seed(0)), z, atol=1e-5)
    a = perturb(z, ROBUST, STATS, torch.Generator().manual_seed(1))
    b = perturb(z, ROBUST, STATS, torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and not torch.equal(a, z)


def test_shift_pads_edges():
    x = torch.arange(5.).repeat(3, 1)
    out = shift_one(x, torch.tensor([1, 0, -1]))
    assert out.tolist() == [[1, 2, 3, 4, 4], [0, 1, 2, 3, 4], [0, 0, 1, 2, 3]]


def test_robust_tap_keeps_legacy_outputs_and_adds_weighted_term():
    torch.manual_seed(0)
    model = build_model(MODEL, 245).eval()
    x = torch.randn(16, 245)
    masked, visible = torch.arange(36).repeat(16, 1), torch.arange(36, 48).repeat(16, 1)
    legacy = model(x, masked, visible)
    tapped = model(x, masked, visible, robust_x=x + .1)
    assert torch.equal(legacy["target"], tapped["target"])
    clean, _ = tapped["robust"]
    assert torch.allclose(clean, model.encoder(model.tokenizer(x), return_all=True)[1][1])
    base = LeJEPAObjective(.05, num_slices=8, knots=5)
    weighted = LeJEPAObjective(.05, num_slices=8, knots=5, robust_weight=.3)
    args = (tapped["predicted"], tapped["target_masked"], tapped["target"])
    torch.manual_seed(1); plain = base(*args)
    torch.manual_seed(1); rob = weighted(*args, robust=tapped["robust"])
    assert rob["robust_loss"] > 0
    assert torch.allclose(rob["loss"], plain["loss"] + .3 * rob["robust_loss"])
    same = model(x, masked, visible, robust_x=x)["robust"]
    assert torch.allclose(same[0], same[1], atol=1e-5)


def test_gradient_audit_reports_robust_terms():
    model = build_model(MODEL, 245).eval()
    objective = LeJEPAObjective(.05, num_slices=8, knots=5, global_weight=1.0, robust_weight=.3)
    out = audit_batch(model, objective, torch.randn(16, 245), MASK, {'views': 2, 'keep': .5},
                      seed=4000, device='cpu', robust_cfg=ROBUST, stats=STATS)
    assert 'robustness' in out['terms']
    norms = out['blocks']['block2']['gradient_norms']
    assert norms['robustness'] == pytest.approx(.3 * norms['robustness_unweighted'], rel=1e-4)


def packed_cfg(tmp_path, name, *extra):
    packed = tmp_path / "packed"
    if not packed.exists():
        packed.mkdir()
        rng = np.random.default_rng(0)
        np.save(packed / "spectra.npy", rng.normal(5.0, 2.0, size=(100, 245)).astype(np.float32))
        (packed / "stats.json").write_text(json.dumps({**STATS, "n_rows": 100, "n_dropped": 0}))
        np.save(packed / "drop_rows.npy", np.array([], dtype=np.int64))
        np.save(packed / "valid_rows.npy", np.arange(10, 30))
    cfg = load_config(REPO / "configs" / "pretrain.yaml", [
        f"experiment.name={name}", f"experiment.output_dir={tmp_path / name}", "data.source=packed",
        f"data.packed_dir={packed}", "data.normalization=global", *TINY_OVERRIDES,
        "training.max_steps=6", "training.log_every=1", "training.ckpt_every=3", "training.val_every=6",
        "training.diag_every=6", *extra])
    return cfg


def test_robust_training_logs_term_and_resumes_exactly(tmp_path):
    def run(name, weight, **training):
        cfg = packed_cfg(tmp_path, name)
        cfg["training"].update(robust={**ROBUST, "weight": weight}, **training)
        with pytest.warns(UserWarning):
            return train(cfg)
    control, robust = run("control", 0.0), run("robust", .3)
    assert all("train/robust_loss" in h and math.isfinite(h["train/loss"]) for h in robust["history"])
    assert [h["train/mse_loss"] for h in control["history"]][0] == robust["history"][0]["train/mse_loss"]
    last = load_checkpoint(robust["final_checkpoint"])
    assert "valid/robust_loss" in last["metrics"] and "robust" in last["rng_state"]
    resumed = run("resumed", .3, resume=str(Path(robust["output_dir"]) / "checkpoint_step3.pt"))
    assert [h["train/loss"] for h in resumed["history"]] == [h["train/loss"] for h in robust["history"][3:]]


def test_robust_requires_global_normalization(tmp_path):
    cfg = packed_cfg(tmp_path, "zscore")
    cfg["data"]["normalization"] = "sample_zscore"
    cfg["training"]["robust"] = ROBUST
    with pytest.raises(ValueError, match="global normalization"):
        train(cfg)
