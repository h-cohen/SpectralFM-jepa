"""Random unstructured masking over the 24 tokens.

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
