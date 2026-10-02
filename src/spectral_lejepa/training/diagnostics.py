"""Diagnostics used to decide lambda_sigreg from measurements rather than assumptions.

representation_stats() summarizes a set of embeddings [N, D]:
  mean_norm, std (mean per-dimension std), effective_rank (exp of the entropy of the
  normalized singular values; D for isotropic data, ~1 for collapse), covariance_trace,
  covariance_condition (largest / smallest non-negligible covariance eigenvalue; inf when
  everything collapsed) and mean_pairwise_cosine (1.0 when all embeddings point the same way).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def encode(model, signals, device, batch_size=256) -> torch.Tensor:
    """Unmasked encoder output for every signal: [N, 245] -> [N, num_patches, D], float32 on CPU."""
    outs = []
    for i in range(0, len(signals), batch_size):
        x = signals[i:i + batch_size].to(device)
        outs.append(model.encoder(model.tokenizer(x)).float().cpu())
    return torch.cat(outs)


@torch.no_grad()
def representation_stats(Z: torch.Tensor, max_cosine_rows: int = 1024):
    Z = Z.double()
    centered = Z - Z.mean(dim=0)
    sv = torch.linalg.svdvals(centered)                     # descending
    eig = sv.square() / max(len(Z) - 1, 1)                   # covariance eigenvalues
    total = sv.sum()
    if total > 0:
        p = sv[sv > 0] / total
        effective_rank = float(torch.exp(-(p * p.log()).sum()))
    else:
        effective_rank = 0.0
    alive = eig[eig > 1e-12 * eig[0]] if eig[0] > 0 else eig[:0]
    condition = float(alive[0] / alive[-1]) if len(alive) else float("inf")
    rows = F.normalize(Z[:max_cosine_rows], dim=1)
    cos = rows @ rows.T
    n = len(rows)
    mean_cos = float((cos.sum() - cos.diagonal().sum()) / max(n * (n - 1), 1))
    stats = {"mean_norm": float(Z.norm(dim=1).mean()), "std": float(Z.std(dim=0).mean()),
             "effective_rank": effective_rank, "covariance_trace": float(eig.sum()),
             "covariance_condition": condition, "mean_pairwise_cosine": mean_cos}
    return stats, sv.float()


def collapse_alert(stats, reference_rank, std_floor=0.1, rank_fraction=0.25) -> bool:
    """True when embeddings shrink (std) or lose rank relative to the step-0 reference."""
    return stats["std"] < std_floor or stats["effective_rank"] < rank_fraction * reference_rank


def singular_value_figure(sv):
    sv = np.asarray(sv, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(np.arange(1, len(sv) + 1), sv / max(sv[0], 1e-30))
    ax.set_xlabel("component")
    ax.set_ylabel("singular value / largest")
    ax.set_title("mean-pooled valid embeddings")
    fig.tight_layout()
    return fig


def masking_figure(signals, bounds, masked_idx):
    """Signals with patch boundaries; masked patches orange, visible patches blue."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 1.8 * len(signals)), sharex=True, squeeze=False)
    for ax, x, masked in zip(axes[:, 0], signals, masked_idx):
        masked = {int(i) for i in masked}
        for p, (lo, hi) in enumerate(bounds):
            is_masked = p in masked
            ax.axvspan(lo - 0.5, hi - 0.5, color="tab:orange" if is_masked else "tab:blue",
                       alpha=0.3 if is_masked else 0.07, lw=0)
            ax.axvline(lo - 0.5, color="0.75", lw=0.5)
        ax.plot(np.arange(len(x)), x, color="k", lw=0.8)
        ax.set_ylabel("z-score")
    axes[0, 0].set_title(f"orange = masked ({len(masked_idx[0])} of {len(bounds)} patches), blue = visible")
    axes[-1, 0].set_xlabel(f"sample index ({len(signals[0])} points, {len(bounds)} patches)")
    fig.tight_layout()
    return fig
