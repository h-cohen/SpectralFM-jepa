import json

import numpy as np
import pytest

from spectral_lejepa.evaluation.nested import pair_with_baseline

PROTO = {"n": 30, "n_repeats": 2, "n_folds": 5, "n_inner": 5, "seed": 42}


def make_baseline(tmp_path, y, P, proto=PROTO):
    np.savez(tmp_path / "nested_oof.npz", y=y, embedding=P, raw=P, embedding_top3=P)
    (tmp_path / "nested_results.json").write_text(json.dumps({"protocol": proto}))


def test_pairing_requires_identical_rows(tmp_path):
    y = np.arange(30, dtype=np.float64)
    P = np.stack([y + 1, y - 1])
    make_baseline(tmp_path, y, P)
    ours = {"embedding": P + 0.5, "raw": P, "embedding_top3": P}
    out = pair_with_baseline(y, ours, tmp_path, PROTO)
    assert out["raw"]["delta"] == 0.0               # identical raw arms must give exactly zero
    assert out["embedding"]["delta"] < 0
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y[::-1].copy(), ours, tmp_path, PROTO)
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y[:20], ours, tmp_path, PROTO)


def test_pairing_requires_identical_protocol_and_folds(tmp_path):
    y = np.arange(30, dtype=np.float64)
    P = np.stack([y + 1, y - 1])
    make_baseline(tmp_path, y, P)
    ours = {"embedding": P + 0.5, "raw": P, "embedding_top3": P}
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y, ours, tmp_path, {**PROTO, "seed": 43})
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y, {**ours, "raw": P + 1e-3}, tmp_path, PROTO)


def test_pairing_raw_tolerance_for_another_backend(tmp_path):
    y = np.arange(30, dtype=np.float64)
    P = np.stack([y + 1, y - 1])
    make_baseline(tmp_path, y, P)
    ours = {"embedding": P + 0.5, "raw": P + 1e-4, "embedding_top3": P}
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y, ours, tmp_path, PROTO)
    assert "embedding" in pair_with_baseline(y, ours, tmp_path, PROTO, raw_atol=1e-2 * np.std(y))
    with pytest.raises(ValueError, match="refusing to pair"):   # other folds: large differences
        pair_with_baseline(y, {**ours, "raw": P[:, ::-1]}, tmp_path, PROTO, raw_atol=1e-2 * np.std(y))
