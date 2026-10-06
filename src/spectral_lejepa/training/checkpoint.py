"""Checkpoints carry everything needed to identify and rebuild a model: weights, optimizer
state, step/epoch, the resolved config, git commit and the W&B run id."""
from __future__ import annotations

import hashlib
import os

import torch

from ..models.vit_1d import build_model


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(path, *, model, optimizer, scaler, step, epoch, config, metadata, metrics):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "config": config,
               "metrics": metrics, **metadata}
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # never leave a truncated checkpoint behind


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model(path, map_location="cpu"):
    ckpt = load_checkpoint(path, map_location)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["data"]["sequence_length"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt
