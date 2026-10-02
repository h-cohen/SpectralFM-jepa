"""245-point signal -> 24 patch tokens.

245 = 24 x 10 + 5, so equal patches are impossible without dropping or inventing samples.
We split the signal into 24 contiguous, non-overlapping patches with np.array_split:
5 patches of 11 samples (indices 0..54), then 19 patches of 10 (55..244). Every sample
lands in exactly one patch, so masking a patch hides exactly its samples and no
neighbouring token can leak them. Each 10-sample patch is right-padded with one zero to
width 11 so one shared Linear(11 -> dim) embeds all patches.
"""
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
        """[B, 245] -> [B, 24, 11]."""
        if x.shape[-1] != self.sequence_length:
            raise ValueError(f"expected signals of length {self.sequence_length}, got {x.shape[-1]}")
        return F.pad(x, (0, 1))[:, self.index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 245] -> [B, 24, dim] tokens, positional embedding included."""
        return self.proj(self.patchify(x)) + self.pos_embed
