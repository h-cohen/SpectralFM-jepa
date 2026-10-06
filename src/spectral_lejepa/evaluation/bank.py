"""Representation bank: one vector per (encoder stage × readout) per spectrum.

Three readouts available:
- "mean": mean over T tokens.
- "seg4": mean of 4 equal token segments, concatenated to [B, 4D].
- "flat": all tokens concatenated to [B, T*D].

Extraction uses the clean-eval interface: model(input_values=x, output_hidden_states=True)
.hidden_states -> tuple of [B, T, D]; stage i is f"layer{i}", with arms f"layer{i}/<readout>".

bank.npz keeps the parent's layout so either side can read the other's files:
bank__<arm> [N, K=1 component, S stats, D], input_raw / input_z [N, 1, 245], y [N],
_meta = repr(dict). Our mean banks store S=1 statistic ("mean"); parent's store 10.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import torch

from .data import normalize_like_fairseq


READOUTS = ("mean", "seg4", "flat")


def _seg4(h):
    """Means of 4 equal token segments, concatenated: [B, T, D] -> [B, 4D] (parent's edges)."""
    e = np.linspace(0, h.shape[1], 5).astype(int)
    return torch.cat([h[:, e[s]:max(e[s] + 1, e[s + 1])].mean(dim=1) for s in range(4)], dim=1)


def readout_arms(hidden_states, readouts, flat_blocks=None) -> dict:
    """{arm: [B, d]}: `layer{i}` = mean over tokens, `layer{i}/seg4`, `layer{i}/flat` = all tokens.

    `flat_blocks` (None = all) restricts the flat readout to those block indices."""
    fns = {"mean": lambda h: h.mean(dim=1), "seg4": _seg4, "flat": lambda h: h.reshape(len(h), -1)}
    out = {}
    for i, h in enumerate(hidden_states):
        for r in readouts:
            if r == "flat" and flat_blocks is not None and i not in flat_blocks:
                continue
            out[f"layer{i}" if r == "mean" else f"layer{i}/{r}"] = fns[r](h).float().cpu().numpy()
    return out


@torch.no_grad()
def extract_bank(model, signals_z, device="cuda", batch_size=64, readouts=("mean",), flat_blocks=None) -> dict:
    model.eval()
    model.to(device)
    t = torch.from_numpy(np.asarray(signals_z, dtype=np.float32))
    out = None
    for i in range(0, len(t), batch_size):
        hidden = model(input_values=t[i:i + batch_size].to(device), output_hidden_states=True).hidden_states
        arms = readout_arms(hidden, readouts, flat_blocks)
        if out is None:
            out = {k: [] for k in arms}
        if list(arms) != list(out):
            raise RuntimeError("hidden_states changed mid-run")
        for k, v in arms.items():
            out[k].append(v)
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def save_bank(path, bank, input_raw, y, meta: dict):
    n = len(y)
    payload = {f"bank__{s.replace('/', '__')}": arr.reshape(n, 1, 1, arr.shape[-1]) for s, arr in bank.items()}
    payload["input_raw"] = np.asarray(input_raw, dtype=np.float32).reshape(n, 1, -1)
    payload["input_z"] = normalize_like_fairseq(payload["input_raw"][:, 0]).reshape(n, 1, -1)
    payload["y"] = np.asarray(y, dtype=np.float64)
    payload["_meta"] = np.array([repr({"comps": (0,), "pool_stats": ("mean",), **meta})])
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, path)


def load_bank(path, readouts=("mean",)):
    """(bank {arm: float32 [N, d]} in file order, input_raw [N, 245], y [N], meta).
    Our banks hold the arms they were extracted with; the parent's 10-statistic banks give
    `layer{i}` (mean) and `layer{i}/seg4` (seg0..seg3) — never `flat`."""
    data = np.load(path, allow_pickle=False)
    meta = ast.literal_eval(str(data["_meta"][0]))
    if tuple(meta.get("comps", (0,)))[0] != 0:
        raise ValueError(f"{path}: first stored component is {meta['comps'][0]}, expected 0")
    stats = tuple(meta["pool_stats"])
    bank = {}
    for key in (k for k in data.files if k.startswith("bank__")):
        name = key[len("bank__"):].replace("__", "/")
        readout = name.split("/")[1] if "/" in name else "mean"
        arr = data[key][:, 0]                                  # [N, S, D]
        if readout in readouts:
            bank[name] = np.ascontiguousarray(arr[:, stats.index("mean")] if readout == "mean"
                                              else arr[:, 0], dtype=np.float32)
        if len(stats) > 1 and "seg4" in readouts:              # parent bank: derive seg4 from seg0..seg3
            seg = [stats.index(f"seg{s}") for s in range(4)]
            bank[f"{name}/seg4"] = np.ascontiguousarray(arr[:, seg].reshape(len(arr), -1), dtype=np.float32)
    input_raw = np.ascontiguousarray(data["input_raw"][:, 0, :], dtype=np.float32)
    return bank, input_raw, np.asarray(data["y"], dtype=np.float64), meta
