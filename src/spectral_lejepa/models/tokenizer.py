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
    """in_channels > 1: input [B, C, L], each token projects the same patch of every channel jointly; an absent
    channel is NaN (zeroed here) and a learned per-channel presence embedding is added for each present one."""

    def __init__(self, sequence_length: int = 245, num_patches: int = 24, dim: int = 256, in_channels: int = 1):
        super().__init__()
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.bounds = patch_bounds(sequence_length, num_patches)
        self.patch_width = max(hi - lo for lo, hi in self.bounds)
        # index[p, j] = sample index of position j in patch p; positions past a patch's end
        # point at `sequence_length`, i.e. at the zero appended in patchify().
        index = torch.full((num_patches, self.patch_width), sequence_length, dtype=torch.long)
        for p, (lo, hi) in enumerate(self.bounds):
            index[p, : hi - lo] = torch.arange(lo, hi)
        self.register_buffer("index", index, persistent=False)
        self.proj = nn.Linear(self.patch_width * in_channels, dim)
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if in_channels > 1:
            self.channel_embed = nn.Parameter(torch.zeros(in_channels, dim))
            nn.init.trunc_normal_(self.channel_embed, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L] -> [B, T, ceil(L/T)]; [B, C, L] -> [B, T, C * ceil(L/T)] (channel-major within a token)."""
        if x.shape[-1] != self.sequence_length:
            raise ValueError(f"expected signals of length {self.sequence_length}, got {x.shape[-1]}")
        if x.dim() == 2:
            return F.pad(x, (0, 1))[:, self.index]
        p = F.pad(x, (0, 1))[:, :, self.index]                    # [B, C, T, W]
        return p.permute(0, 2, 1, 3).reshape(x.shape[0], self.index.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L] (or [B, C, L], NaN = absent channel) -> [B, T, dim] tokens, positional embedding included."""
        if self.in_channels == 1:
            return self.proj(self.patchify(x)) + self.pos_embed
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected [B, {self.in_channels}, L] input, got {tuple(x.shape)}")
        present = ~torch.isnan(x).all(-1)                          # [B, C]
        tokens = self.proj(self.patchify(torch.nan_to_num(x, nan=0.0)))
        return tokens + self.pos_embed + (present.to(tokens.dtype) @ self.channel_embed)[:, None]
