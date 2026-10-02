"""Token masking over the patch sequence: random (unstructured) or block (contiguous runs).

Each sample independently gets round(mask_ratio * 24) masked positions, drawn uniformly
without replacement (argsort of uniform noise). Indices are returned sorted, so token
order is preserved. The count is the same for every sample, so no padding masks are
needed. Pass a seeded CPU generator for reproducibility on any device.
"""
from __future__ import annotations

import torch


def num_masked(num_patches: int, mask_ratio: float) -> int:
    n = round(mask_ratio * num_patches)
    if not 1 <= n <= num_patches - 1:
        raise ValueError(f"mask_ratio={mask_ratio} masks {n} of {num_patches} tokens; "
                         "need at least one masked and one visible token")
    return n


def random_mask(batch_size: int, num_patches: int, mask_ratio: float,
                generator: torch.Generator | None = None, device=None):
    """Returns (masked_idx [B, n_mask], visible_idx [B, num_patches - n_mask]), both sorted."""
    n_mask = num_masked(num_patches, mask_ratio)
    order = torch.rand(batch_size, num_patches, generator=generator).argsort(dim=1)
    masked_idx = order[:, :n_mask].sort(dim=1).values
    visible_idx = order[:, n_mask:].sort(dim=1).values
    return masked_idx.to(device), visible_idx.to(device)


def _composition(total: int, parts: int, generator) -> list[int]:
    """Uniformly random non-negative integer sizes [parts] summing to total (stars and bars)."""
    bars = torch.randperm(total + parts - 1, generator=generator)[: parts - 1].sort().values.tolist()
    edges = [-1, *bars, total + parts - 1]
    return [edges[i + 1] - edges[i] - 1 for i in range(parts)]


def block_mask(batch_size: int, num_patches: int, mask_ratio: float, num_blocks: int,
               generator: torch.Generator | None = None, device=None):
    """Mask exactly round(mask_ratio * P) tokens per sample, grouped into `num_blocks`
    contiguous runs separated by visible tokens. Run lengths and gaps are random
    compositions, so masked spans cannot be filled in from a visible neighbour on each side."""
    n_mask = num_masked(num_patches, mask_ratio)
    n_visible = num_patches - n_mask
    if not 1 <= num_blocks <= n_mask or n_visible < num_blocks - 1:
        raise ValueError(f"num_blocks={num_blocks} impossible with {n_mask} masked and "
                         f"{n_visible} visible tokens")
    masked, visible = [], []
    for _ in range(batch_size):
        run_lengths = [s + 1 for s in _composition(n_mask - num_blocks, num_blocks, generator)]
        gaps = _composition(n_visible - (num_blocks - 1), num_blocks + 1, generator)
        gaps = [g + (1 if 0 < i < num_blocks else 0) for i, g in enumerate(gaps)]  # interior gaps >= 1
        is_masked = torch.zeros(num_patches, dtype=torch.bool)
        pos = 0
        for gap, length in zip(gaps, run_lengths):
            pos += gap
            is_masked[pos:pos + length] = True
            pos += length
        masked.append(is_masked.nonzero().squeeze(1))
        visible.append((~is_masked).nonzero().squeeze(1))
    return torch.stack(masked).to(device), torch.stack(visible).to(device)


def make_mask(masking_cfg: dict, batch_size: int, num_patches: int,
              generator: torch.Generator | None = None, device=None):
    """The configured masking strategy: masking_cfg = {strategy, mask_ratio, num_blocks}."""
    strategy = masking_cfg["strategy"]
    if strategy == "random":
        return random_mask(batch_size, num_patches, masking_cfg["mask_ratio"], generator, device)
    if strategy == "block":
        return block_mask(batch_size, num_patches, masking_cfg["mask_ratio"], masking_cfg["num_blocks"],
                          generator, device)
    raise ValueError(f"unknown masking.strategy {strategy!r}; use 'random' or 'block'")
