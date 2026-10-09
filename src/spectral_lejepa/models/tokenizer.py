"""Contiguous patch projection with zero padding and learned positions."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_bounds(sequence_length: int = 245, num_patches: int = 24) -> list[tuple[int, int]]:
    parts = np.array_split(np.arange(sequence_length), num_patches)
    return [(int(p[0]), int(p[-1]) + 1) for p in parts]


class PatchTokenizer(nn.Module):
    def __init__(self, sequence_length: int = 245, num_patches: int = 24, dim: int = 256):
        super().__init__()
        self.sequence_length = sequence_length
        self.bounds = patch_bounds(sequence_length, num_patches)
        self.patch_width = max(hi - lo for lo, hi in self.bounds)
        # index[p, j] = sample index of position j in patch p; positions past a patch's end
        # point at `sequence_length`, i.e. at the zero appended in patchify().
        index = torch.full((num_patches, self.patch_width), sequence_length, dtype=torch.long)
        for p, (lo, hi) in enumerate(self.bounds):
            index[p, : hi - lo] = torch.arange(lo, hi)
        self.register_buffer("index", index, persistent=False)
        self.proj = nn.Linear(self.patch_width, dim)
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L] -> [B, T, ceil(L/T)]."""
        if x.shape[-1] != self.sequence_length:
            raise ValueError(f"expected signals of length {self.sequence_length}, got {x.shape[-1]}")
        return F.pad(x, (0, 1))[:, self.index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L] -> [B, T, dim] tokens, positional embedding included."""
        return self.proj(self.patchify(x)) + self.pos_embed
