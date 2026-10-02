import pytest
import torch

from spectral_lejepa.models.masking import num_masked, random_mask


def gen(seed):
    return torch.Generator().manual_seed(seed)


def test_shapes_and_counts():
    m, v = random_mask(8, 24, 0.5, gen(0))
    assert m.shape == (8, 12) and v.shape == (8, 12)
    assert m.dtype == torch.long and v.dtype == torch.long
    assert num_masked(24, 0.45) == 11 and num_masked(24, 0.4) == 10


def test_disjoint_complete_sorted():
    m, v = random_mask(16, 24, 0.5, gen(1))
    for mi, vi in zip(m.tolist(), v.tolist()):
        assert sorted(mi + vi) == list(range(24))
        assert not set(mi) & set(vi)
        assert mi == sorted(mi) and vi == sorted(vi)


def test_masks_differ_between_samples():
    m, _ = random_mask(16, 24, 0.5, gen(2))
    assert len({tuple(row) for row in m.tolist()}) > 1


def test_reproducible_with_seed():
    a = random_mask(4, 24, 0.5, gen(3))
    b = random_mask(4, 24, 0.5, gen(3))
    c = random_mask(4, 24, 0.5, gen(4))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.99, 1.0])
def test_degenerate_ratios_rejected(ratio):
    with pytest.raises(ValueError, match="mask_ratio"):
        random_mask(2, 24, ratio, gen(0))
