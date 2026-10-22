from pathlib import Path

import pytest
import torch

from spectral_lejepa.training.checkpoint import load_checkpoint
from spectral_lejepa.training.loss import LeJEPAObjective
from spectral_lejepa.training.trainer import train

from .test_robust_paths import idx, tiny
from .test_train_smoke import data_dir, tiny_cfg  # noqa: F401

K = 2


def batch():
    return torch.randn(16, 245)


def test_zero_when_student_equals_teacher():
    student, teacher = tiny(train=True), tiny()
    teacher.load_state_dict(student.state_dict())
    x = batch()
    masked, visible = idx(16)
    out = student(x, masked, visible, return_block=K)
    with torch.no_grad():
        ref = teacher.encoder.prefix(teacher.tokenizer(x), K)
    obj = LeJEPAObjective(.05, num_slices=8, knots=5, preserve_weight=1.0)
    res = obj(out["predicted"], out["target_masked"], out["target"], preserve=(out["clean_block"], ref))
    assert res["preserve_loss"] < 1e-10


def test_weight_zero_matches_none_loss_and_grads():
    x = batch()
    masked, visible = idx(16)
    teacher = tiny().requires_grad_(False)
    grads = []
    for use in (False, True):
        model = tiny(train=True)
        for p in model.parameters():   # move student off the teacher so the term is nonzero
            p.data += .01
        torch.manual_seed(1)
        out = model(x, masked, visible, return_block=K if use else None)
        pres = (out["clean_block"], teacher.encoder.prefix(teacher.tokenizer(x), K)) if use else None
        obj = LeJEPAObjective(.05, num_slices=8, knots=5, preserve_weight=0.0)
        torch.manual_seed(2)
        res = obj(out["predicted"], out["target_masked"], out["target"], preserve=pres)
        res["loss"].backward()
        grads.append((res["loss"].item(), [p.grad.clone() for p in model.parameters()]))
    assert grads[0][0] == grads[1][0]
    assert all(torch.equal(a, b) for a, b in zip(grads[0][1], grads[1][1]))


def _base(data_dir, tmp_path):
    with pytest.warns(UserWarning):
        return train(tiny_cfg(data_dir, tmp_path / "base", "base"))["final_checkpoint"]


def _cfg(data_dir, tmp_path, name, base, **training):
    cfg = tiny_cfg(data_dir, tmp_path / name, name)
    cfg["training"].update(max_steps=6, ckpt_every=3, log_every=1, val_every=6, diag_every=6,
                           init_checkpoint=base, preserve={"weight": 5.0, "block": K}, **training)
    return cfg


def test_training_with_preserve_logs_and_teacher_untouched(data_dir, tmp_path):
    base = _base(data_dir, tmp_path)
    cfg = _cfg(data_dir, tmp_path, "pres", base)
    with pytest.warns(UserWarning):
        result = train(cfg)
    assert all("train/preserve_loss" in h for h in result["history"])
    assert result["history"][0]["train/preserve_loss"] < 1e-6   # step 1 is computed at the init weights
    ckpt = load_checkpoint(result["final_checkpoint"])
    assert not any(k.startswith("teacher") for k in ckpt["model"])
    assert "preserve_loss" not in str(ckpt["model"].keys())
    assert load_checkpoint(base)["step"] == 60   # reference file is read only


def test_teacher_gets_no_grad():
    from spectral_lejepa.training.trainer import teacher_pair
    student, teacher = tiny(train=True), tiny().requires_grad_(False)
    x = batch()
    masked, visible = idx(16)
    out = student(x, masked, visible, return_block=K)
    h_s, h_t = teacher_pair(teacher, x, out, {"block": K})
    assert not h_t.requires_grad and h_s.requires_grad
    ((h_s - h_t) ** 2).mean().backward()
    assert all(p.grad is None for p in teacher.parameters())


def test_preserve_resumes_to_same_final_step(data_dir, tmp_path):
    base = _base(data_dir, tmp_path)
    with pytest.warns(UserWarning):
        full = train(_cfg(data_dir, tmp_path, "full", base))
    mid = Path(full["output_dir"]) / "checkpoint_step3.pt"
    cfg = _cfg(data_dir, tmp_path, "res", None, resume=str(mid))
    with pytest.warns(UserWarning):
        res = train(cfg)
    assert [h["train/global_step"] for h in res["history"]] == [4, 5, 6]
    a, b = load_checkpoint(full["final_checkpoint"]), load_checkpoint(res["final_checkpoint"])
    assert b["step"] == 6
    for k, v in a["model"].items():
        assert torch.allclose(v, b["model"][k], atol=1e-5), k


@pytest.mark.parametrize("bad,msg", [({"weight": -1, "block": 2}, "weight"), ({"weight": 1, "block": 99}, "block")])
def test_invalid_preserve_fails(data_dir, tmp_path, bad, msg):
    base = _base(data_dir, tmp_path)
    cfg = _cfg(data_dir, tmp_path, "bad", base)
    cfg["training"]["preserve"] = bad
    with pytest.raises(ValueError, match=msg):
        train(cfg)


def test_preserve_needs_init_checkpoint(data_dir, tmp_path):
    cfg = tiny_cfg(data_dir, tmp_path / "x", "x")
    cfg["training"]["preserve"] = {"weight": 1.0, "block": K}
    with pytest.raises(ValueError, match="init_checkpoint"):
        train(cfg)
