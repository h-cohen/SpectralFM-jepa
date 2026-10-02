"""Representation bank: one mean-pooled vector per encoder stage per spectrum.

Extraction uses the clean-eval interface: model(input_values=x, output_hidden_states=True)
.hidden_states -> tuple of [B, T, D]; stage i is named f"layer{i}" and pooled by the mean
over T (every headline clean-eval number uses mean pooling).

bank.npz keeps the parent's layout so either side can read the other's files:
bank__<stage> [N, K=1 component, S stats, D], input_raw / input_z [N, 1, 245], y [N],
_meta = repr(dict). Ours store S=1 statistic ("mean"); the parent's store 10.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import torch

from .data import normalize_like_fairseq


@torch.no_grad()
def extract_mean_bank(model, signals_z, device="cuda", batch_size=64) -> dict:
    model.eval()
    model.to(device)
    t = torch.from_numpy(np.asarray(signals_z, dtype=np.float32))
    out = None
    for i in range(0, len(t), batch_size):
        hidden = model(input_values=t[i:i + batch_size].to(device), output_hidden_states=True).hidden_states
        if out is None:
            out = {f"layer{li}": [] for li in range(len(hidden))}
        if len(hidden) != len(out):
            raise RuntimeError(f"hidden_states length changed mid-run: {len(hidden)} vs {len(out)}")
        for li, h in enumerate(hidden):
            out[f"layer{li}"].append(h.mean(dim=1).float().cpu().numpy())
    return {stage: np.concatenate(v).astype(np.float32) for stage, v in out.items()}


def save_bank(path, bank, input_raw, y, meta: dict):
    n = len(y)
    payload = {f"bank__{s}": arr.reshape(n, 1, 1, arr.shape[-1]) for s, arr in bank.items()}
    payload["input_raw"] = np.asarray(input_raw, dtype=np.float32).reshape(n, 1, -1)
    payload["input_z"] = normalize_like_fairseq(payload["input_raw"][:, 0]).reshape(n, 1, -1)
    payload["y"] = np.asarray(y, dtype=np.float64)
    payload["_meta"] = np.array([repr({"comps": (0,), "pool_stats": ("mean",), **meta})])
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, path)


def load_bank(path):
    data = np.load(path, allow_pickle=False)
    meta = ast.literal_eval(str(data["_meta"][0]))
    if tuple(meta.get("comps", (0,)))[0] != 0:
        raise ValueError(f"{path}: first stored component is {meta['comps'][0]}, expected 0")
    mean_i = tuple(meta["pool_stats"]).index("mean")
    bank = {k[len("bank__"):]: np.ascontiguousarray(data[k][:, 0, mean_i, :], dtype=np.float32)
            for k in data.files if k.startswith("bank__")}   # file order = parent's stage order
    input_raw = np.ascontiguousarray(data["input_raw"][:, 0, :], dtype=np.float32)
    return bank, input_raw, np.asarray(data["y"], dtype=np.float64), meta
