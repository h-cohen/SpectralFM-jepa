import numpy as np
import pytest

from spectral_lejepa.evaluation.nested import pair_with_baseline


def test_pairing_requires_identical_rows(tmp_path):
    y = np.arange(30, dtype=np.float64)
    P = np.stack([y + 1, y - 1])
    np.savez(tmp_path / "nested_oof.npz", y=y, embedding=P, raw=P, embedding_top3=P)
    ours = {"embedding": P + 0.5, "raw": P, "embedding_top3": P}
    out = pair_with_baseline(y, ours, tmp_path)
    assert out["raw"]["delta"] == 0.0               # identical raw arms must give exactly zero
    assert out["embedding"]["delta"] < 0
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y[::-1].copy(), ours, tmp_path)
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y[:20], ours, tmp_path)
