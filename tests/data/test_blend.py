import numpy as np
import torch

from spectral_lejepa.data.loader import BlendSpectra, make_loader


class _Base(torch.utils.data.Dataset):
    """Rows 0-9 measured (value i), rows 10-19 synthetic (value 100+i); length-4 signals."""

    def __len__(self):
        return 20

    def __getitem__(self, i):
        return torch.full((4,), float(i if i < 10 else 100 + i))


MASK = np.arange(20) < 10
SYNTH = np.arange(10, 20)


def test_blend_prob_zero_is_identity():
    ds = BlendSpectra(_Base(), MASK, SYNTH, prob=0.0, lam_low=0.5)
    assert all(torch.equal(ds[i], _Base()[i]) for i in range(20))


def test_synthetic_rows_never_blended():
    ds = BlendSpectra(_Base(), MASK, SYNTH, prob=1.0, lam_low=0.5)
    assert all(torch.equal(ds[i], _Base()[i]) for i in range(10, 20))


def test_blend_is_convex_mix_with_lambda_in_range():
    ds = BlendSpectra(_Base(), MASK, SYNTH, prob=1.0, lam_low=0.5)
    torch.manual_seed(0)
    for _ in range(50):
        m = 3
        v = float(ds[m][0])
        lam_ok = [s for s in range(110, 120) if 0.5 - 1e-6 <= (v - s) / (m - s) <= 1 + 1e-6]
        assert lam_ok and v != m


def test_blend_reproducible_under_seed():
    ds = BlendSpectra(_Base(), MASK, SYNTH, prob=0.5, lam_low=0.5)
    a = [b.clone() for b in make_loader(ds, 4, True, 7, 2, True)]
    b = [b.clone() for b in make_loader(ds, 4, True, 7, 2, True)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
