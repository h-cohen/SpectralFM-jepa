import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from spectral_lejepa.config import load_config
from spectral_lejepa.training.checkpoint import load_checkpoint, load_model
from spectral_lejepa.training.trainer import lr_factor, schedule_lengths, train

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_dir(tmp_path):
    """Sinusoids with random frequency/phase: learnable structure for a tiny overfit run."""
    rng = np.random.default_rng(0)
    t = np.arange(245) / 245
    wav = tmp_path / "data" / "wav"
    wav.mkdir(parents=True)
    names = []
    for i in range(80):
        x = np.sin(2 * np.pi * rng.uniform(1, 5) * t + rng.uniform(0, 2 * np.pi)) + 0.05 * rng.normal(size=245)
        sf.write(wav / f"spec_{i}.wav", x.astype(np.float32), 16000, subtype="FLOAT")
        names.append(f"spec_{i}.wav")
    for split, part in (("train", names[:64]), ("valid", names[64:])):
        (tmp_path / "data" / f"{split}.tsv").write_text(f"{wav}\n" + "".join(f"{n}\t245\n" for n in part))
    return tmp_path / "data"


TINY_OVERRIDES = [
    "data.num_workers=0",
    "model.dim=32", "model.depth=2", "model.heads=4", "model.mlp_dim=64",
    "model.predictor_dim=24", "model.predictor_depth=2", "model.predictor_heads=4",
    "model.predictor_mlp_dim=48",
    "training.batch_size=8", "training.max_steps=60", "training.learning_rate=1.0e-3",
    "training.log_every=5", "training.val_every=20", "training.diag_every=20",
    "training.ckpt_every=30", "training.device=cpu", "training.amp=false",
    "training.sigreg_num_slices=64", "wandb.enabled=false",
]


def tiny_cfg(data_dir, out_dir, name="smoke"):
    return load_config(REPO / "configs" / "pretrain.yaml", [
        f"experiment.name={name}", f"experiment.output_dir={out_dir}",
        f"data.manifest_dir={data_dir}", *TINY_OVERRIDES])


def test_schedule():
    assert schedule_lengths(100, {"max_steps": None, "epochs": 20, "warmup_epochs": 1.0}) == (2000, 100)
    assert schedule_lengths(8, {"max_steps": 60, "epochs": 20, "warmup_epochs": 1.0}) == (60, 6)
    assert lr_factor(0, 10, 100, 0.001) == pytest.approx(0.1)
    assert lr_factor(10, 10, 100, 0.001) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100, 0.001) == pytest.approx(0.001)


def test_tiny_training_run(data_dir, tmp_path):
    with pytest.warns(UserWarning, match="batch_size=8"):
        result = train(tiny_cfg(data_dir, tmp_path / "out"))
    out = Path(result["output_dir"])
    assert (out / "config.yaml").exists()
    for name in ("checkpoint_step30.pt", "checkpoint_step60.pt", "checkpoint_last.pt"):
        assert (out / name).exists(), name

    history = result["history"]
    assert len(history) == 12
    assert all(math.isfinite(h["train/loss"]) for h in history)
    first = np.mean([h["train/loss"] for h in history[:3]])
    last = np.mean([h["train/loss"] for h in history[-3:]])
    assert last < first
    assert {"train/mse_loss", "train/sigreg_loss", "train/learning_rate", "train/epoch",
            "train/global_step", "train/samples_seen"} <= set(history[0])

    ckpt = load_checkpoint(out / "checkpoint_last.pt")
    assert ckpt["step"] == 60
    assert ckpt["config"]["training"]["lambda_sigreg"] == 0.05
    assert len(ckpt["git_commit"]) == 40 and ckpt["wandb_run_id"] is None
    assert "valid/loss" in ckpt["metrics"] and "representation/effective_rank" in ckpt["metrics"]
    model, _ = load_model(out / "checkpoint_last.pt")
    assert model.encoder(model.tokenizer(torch.randn(2, 245))).shape == (2, 24, 32)


def test_runs_are_reproducible(data_dir, tmp_path):
    with pytest.warns(UserWarning):
        a = train(tiny_cfg(data_dir, tmp_path / "a", name="rep_a"))
        b = train(tiny_cfg(data_dir, tmp_path / "b", name="rep_b"))
    assert [h["train/loss"] for h in a["history"]] == [h["train/loss"] for h in b["history"]]


def test_cuda_requested_without_gpu_fails(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = tiny_cfg(data_dir, tmp_path / "out")
    cfg["training"]["device"] = "cuda"
    with pytest.raises(RuntimeError, match="no GPU"):
        train(cfg)


def test_impossible_block_mask_fails_before_any_work(data_dir, tmp_path):
    cfg = tiny_cfg(data_dir, tmp_path / "out")
    cfg["masking"].update(strategy="block", num_blocks=13)
    with pytest.raises(ValueError, match="num_blocks"):
        train(cfg)
    assert not (tmp_path / "out").exists()


def test_block_masking_trains(data_dir, tmp_path):
    cfg = tiny_cfg(data_dir, tmp_path / "out", name="block")
    cfg["masking"]["strategy"] = "block"
    cfg["training"]["max_steps"] = 10
    with pytest.warns(UserWarning):
        result = train(cfg)
    assert all(math.isfinite(h["train/loss"]) for h in result["history"])


class FakeRun:
    id = "fake"

    def log(self, *a, **k):
        pass

    def alert(self, *a, **k):
        pass

    def finish(self):
        pass


@pytest.mark.parametrize("flag,expected", [(None, 0), (False, 0), (True, 2)])
def test_checkpoint_upload_is_opt_in(data_dir, tmp_path, monkeypatch, flag, expected):
    from spectral_lejepa.utils import wandb as wb
    calls = []
    monkeypatch.setattr(wb, "init_run", lambda *a, **k: FakeRun())
    monkeypatch.setattr(wb, "log_figure", lambda *a, **k: None)
    monkeypatch.setattr(wb, "log_checkpoint", lambda *a, **k: calls.append(a))
    cfg = tiny_cfg(data_dir, tmp_path / "out", name=f"ckpt_{flag}")
    cfg["training"].update(max_steps=4, ckpt_every=2, val_every=4, diag_every=4, log_every=2)
    if flag is None:
        cfg["wandb"].pop("log_checkpoints", None)       # an old config without the key
    else:
        cfg["wandb"]["log_checkpoints"] = flag
    with pytest.warns(UserWarning):
        train(cfg)
    assert len(calls) == expected
