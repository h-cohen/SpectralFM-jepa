import pytest
import torch

from spectral_lejepa.models.masking import block_mask, make_mask, num_masked, random_mask


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


def runs(idx):
    """Number of contiguous runs in a sorted index list."""
    return 1 + sum(b - a > 1 for a, b in zip(idx, idx[1:]))


def test_block_mask_counts_and_partition():
    m, v = block_mask(64, 24, 0.5, 3, gen(0))
    assert m.shape == (64, 12) and v.shape == (64, 12)
    for mi, vi in zip(m.tolist(), v.tolist()):
        assert sorted(mi + vi) == list(range(24)) and not set(mi) & set(vi)
        assert mi == sorted(mi) and runs(mi) == 3


def test_block_mask_other_settings():
    m, _ = block_mask(32, 24, 0.75, 2, gen(1))
    assert m.shape == (32, 18) and all(runs(r) == 2 for r in m.tolist())
    m, _ = block_mask(8, 49, 0.5, 1, gen(2))
    assert all(runs(r) == 1 for r in m.tolist())


def test_block_mask_reproducible_and_varied():
    a, b, c = block_mask(16, 24, 0.5, 3, gen(5)), block_mask(16, 24, 0.5, 3, gen(5)), block_mask(16, 24, 0.5, 3, gen(6))
    assert torch.equal(a[0], b[0]) and not torch.equal(a[0], c[0])
    assert len({tuple(r) for r in a[0].tolist()}) > 1


@pytest.mark.parametrize("num_blocks", [0, 13])
def test_block_mask_impossible_layout_rejected(num_blocks):
    with pytest.raises(ValueError, match="num_blocks"):
        block_mask(2, 24, 0.5, num_blocks, gen(0))


def test_block_mask_too_few_visible_rejected():
    # 24 tokens, 0.9 -> 22 masked, 2 visible: 4 blocks need >= 3 separating visible tokens
    with pytest.raises(ValueError, match="num_blocks"):
        block_mask(2, 24, 0.9, 4, gen(0))


def test_make_mask_dispatch():
    cfg = {"strategy": "random", "mask_ratio": 0.5, "num_blocks": 3}
    assert torch.equal(make_mask(cfg, 4, 24, gen(0))[0], random_mask(4, 24, 0.5, gen(0))[0])
    cfg["strategy"] = "block"
    assert torch.equal(make_mask(cfg, 4, 24, gen(0))[0], block_mask(4, 24, 0.5, 3, gen(0))[0])
    with pytest.raises(ValueError, match="strategy"):
        make_mask({**cfg, "strategy": "multiblock"}, 4, 24, gen(0))
