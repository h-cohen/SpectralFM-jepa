"""Shared 1-D transformer encoder and masked-token predictor. Both encoder paths receive gradients."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from .tokenizer import PatchTokenizer


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, N, D], idx [B, K] -> [B, K, D]: per-sample token selection."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def transformer_blocks(dim, depth, heads, mlp_dim, dropout) -> nn.ModuleList:
    return nn.ModuleList([
        nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout, activation="gelu",
                                   batch_first=True, norm_first=True)
        for _ in range(depth)])


class Encoder(nn.Module):
    """Pre-LN transformer over any subset of tokens (positions already embedded)."""

    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.blocks = transformer_blocks(dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, return_all=False):
        hidden = []
        x = tokens
        for block in self.blocks:
            x = block(x)
            hidden.append(x)
        out = self.norm(x)
        return (out, hidden) if return_all else out


class Predictor(nn.Module):
    """Predicts encoder embeddings at masked positions from the context embeddings.

    Context tokens and one learned mask token per masked position (plus that position's
    embedding) are processed jointly; the outputs at the mask-token slots are the predictions.
    """

    def __init__(self, num_patches, in_dim, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = transformer_blocks(dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_dim)

    def forward(self, context, visible_idx, masked_idx):
        B, n_mask = masked_idx.shape
        ctx = self.embed(context) + self.pos_embed[visible_idx]                      # [B, n_vis, dim]
        queries = self.mask_token.expand(B, n_mask, -1) + self.pos_embed[masked_idx]  # [B, n_mask, dim]
        x = torch.cat([ctx, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, -n_mask:]))                                   # [B, n_mask, in_dim]


class LeJEPA(nn.Module):
    def __init__(self, sequence_length, num_patches, dim, depth, heads, mlp_dim, dropout,
                 predictor_dim, predictor_depth, predictor_heads, predictor_mlp_dim,
                 projector="none", projector_hidden_dim=1024, projector_dim=128):
        super().__init__()
        self.tokenizer = PatchTokenizer(sequence_length, num_patches, dim)
        self.encoder = Encoder(dim, depth, heads, mlp_dim, dropout)
        self.predictor = Predictor(num_patches, dim, predictor_dim, predictor_depth,
                                   predictor_heads, predictor_mlp_dim, dropout)
        if projector == "none":
            self.projector = None
        elif projector == "mlp":
            # imported here so the baseline (projector: none) does not need it
            from lightly.models.modules import LeJEPAProjectionHead
            self.projector = LeJEPAProjectionHead(dim, projector_hidden_dim, projector_dim)
        else:
            raise ValueError(f"unknown projector {projector!r}; use 'none' or 'mlp'")

    def forward(self, x, masked_idx, visible_idx, view_idxs=None):
        tokens = self.tokenizer(x)                                            # [B, T, D]
        context = self.encoder(gather_tokens(tokens, visible_idx))            # [B, n_vis, D]
        predicted = self.predictor(context, visible_idx, masked_idx)          # [B, n_mask, D]
        target = self.encoder(tokens)                                         # [B, T, D]
        views = None
        if view_idxs is not None:   # pooled encoder outputs (before any projector): full view, then subsets
            views = torch.stack([target.mean(1)] + [self.encoder(gather_tokens(tokens, idx)).mean(1)
                                                    for idx in view_idxs])   # [V+1, B, D]
        if self.projector is not None:
            predicted, target = self._project(predicted, target)
        out = {"predicted": predicted, "target": target, "target_masked": gather_tokens(target, masked_idx)}
        if views is not None:
            out["views"] = views
        return out

    def _project(self, predicted, target):
        """Project predictions and targets in ONE call so BatchNorm sees one set of batch statistics."""
        B, n_mask, D = predicted.shape
        out = self.projector(torch.cat([predicted.reshape(-1, D), target.reshape(-1, D)]))
        return out[: B * n_mask].reshape(B, n_mask, -1), out[B * n_mask:].reshape(B, target.shape[1], -1)


class EvalBackbone(nn.Module):
    """Expose tokenizer/block states; the last state includes the final LayerNorm."""

    def __init__(self, model: LeJEPA):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.encoder = model.encoder

    def forward(self, input_values, output_hidden_states=True):
        tokens = self.tokenizer(input_values)
        out, hidden = self.encoder(tokens, return_all=True)
        return SimpleNamespace(hidden_states=(tokens, *hidden[:-1], out), last_hidden_state=out)


def build_model(model_cfg: dict, sequence_length: int) -> LeJEPA:
    keys = ("num_patches", "dim", "depth", "heads", "mlp_dim", "dropout", "predictor_dim",
            "predictor_depth", "predictor_heads", "predictor_mlp_dim", "projector",
            "projector_hidden_dim", "projector_dim")
    return LeJEPA(sequence_length=sequence_length, **{k: model_cfg[k] for k in keys})
