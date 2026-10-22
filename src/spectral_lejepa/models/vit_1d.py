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

    def prefix(self, tokens, k):
        """State after the first k blocks (no final norm): the block-k readout."""
        for block in self.blocks[:k]:
            tokens = block(tokens)
        return tokens


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
                 predictor_dim, predictor_depth, predictor_heads, predictor_mlp_dim):
        super().__init__()
        self.tokenizer = PatchTokenizer(sequence_length, num_patches, dim)
        self.encoder = Encoder(dim, depth, heads, mlp_dim, dropout)
        self.predictor = Predictor(num_patches, dim, predictor_dim, predictor_depth,
                                   predictor_heads, predictor_mlp_dim, dropout)

    def forward(self, x, masked_idx, visible_idx, view_idxs=None, robust_x=None, robust_block=2, return_block=None):
        tokens = self.tokenizer(x)                                            # [B, T, D]
        context = self.encoder(gather_tokens(tokens, visible_idx))            # [B, n_vis, D]
        predicted = self.predictor(context, visible_idx, masked_idx)          # [B, n_mask, D]
        if robust_x is None and return_block is None:
            target = self.encoder(tokens)                                     # [B, T, D]
        else:   # block-k states of the clean (and perturbed) full sequences, [B, T, D] each
            target, hidden = self.encoder(tokens, return_all=True)
            if robust_x is not None:
                robust = (hidden[robust_block - 1], self.encoder.prefix(self.tokenizer(robust_x), robust_block))
        views = None
        if view_idxs is not None:   # pooled encoder outputs: full view, then subsets
            views = torch.stack([target.mean(1)] + [self.encoder(gather_tokens(tokens, idx)).mean(1)
                                                    for idx in view_idxs])   # [V+1, B, D]
        out = {"predicted": predicted, "target": target, "target_masked": gather_tokens(target, masked_idx)}
        if views is not None:
            out["views"] = views
        if robust_x is not None:
            out["robust"] = robust
        if return_block is not None:   # clean block-k tokens, [B, T, D]; the same tensor the robust term uses
            out["clean_block"] = hidden[return_block - 1]
        return out


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


# Retired experiment options (prediction projector, SIGReg-input projector). Saved configs still carry
# them; only their disabled values are accepted.
RETIRED_MODEL_KEYS = ("projector", "projector_hidden_dim", "projector_dim", "sigreg_projector",
                      "sigreg_projector_hidden_dim", "sigreg_projector_dim")


def core_model_config(model_cfg: dict) -> dict:
    return {k: v for k, v in model_cfg.items() if k not in RETIRED_MODEL_KEYS}


def build_model(model_cfg: dict, sequence_length: int) -> LeJEPA:
    for key in ("projector", "sigreg_projector"):
        if model_cfg.get(key, "none") != "none":
            raise ValueError(f"model.{key}={model_cfg[key]!r} is retired; only 'none' is supported")
    keys = ("num_patches", "dim", "depth", "heads", "mlp_dim", "dropout", "predictor_dim",
            "predictor_depth", "predictor_heads", "predictor_mlp_dim")
    return LeJEPA(sequence_length=sequence_length, **{k: model_cfg[k] for k in keys})
