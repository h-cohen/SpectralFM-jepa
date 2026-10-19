"""Per-block representation geometry of a checkpoint and a random-weight model of the same architecture.

    python -m spectral_lejepa isotropy --checkpoint CKPT --output OUT.json [--n 10000] [--device cuda]

Features are encoder states on held-out corpus rows (valid_rows.npy), block k = hidden_states[k] as in the
probe banks (block 0 = tokens, last block = after the final LayerNorm). Effective rank and mean cosine are
scale-free; the SIGReg (Epps-Pulley) statistic tests against a standard Gaussian, so it is scale-sensitive by design.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lightly.loss import SIGReg

from spectral_lejepa.data.loader import valid_subset
from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
from spectral_lejepa.training.checkpoint import load_model
from spectral_lejepa.utils.io import write_json


def _spectrum(X):
    X = np.asarray(X, np.float64)
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return s / s.sum()


def geometry(X, batch=256, seed=0):
    """effective_rank: exp(entropy) of the centred PCA spectrum; top_pc_share; mean pairwise cosine
    (on <= 2000 rows); sigreg: mean SIGReg statistic over row batches of `batch` (training's batch size)."""
    X = np.asarray(X, np.float64)
    p = _spectrum(X)
    sub = X[np.random.default_rng(seed).permutation(len(X))[:2000]]
    u = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12)
    torch.manual_seed(seed)
    reg = SIGReg()
    t = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        sig = [float(reg(t[i:i + batch])) for i in range(0, len(t) - batch + 1, batch)] or [float(reg(t))]
    return {'effective_rank': float(np.exp(-(p * np.log(p + 1e-12)).sum())), 'top_pc_share': float(p[0]),
            'mean_cosine': float((u @ u.T).mean()), 'sigreg': float(np.mean(sig))}


def feature_correlation(X):
    """Pearson correlation between feature dimensions across spectra, [d, d]; an isotropic Gaussian (SIGReg's
    target) gives the identity. Constant dimensions get correlation 0 with the rest and 1 with themselves.
    Summary: mean |off-diagonal|; dist_to_identity = ||C - I||_F / sqrt(d(d-1)) (0 identity, 1 one direction)."""
    X = np.asarray(X, np.float64)
    Z = X - X.mean(0)
    sd = Z.std(0)
    dead = sd < 1e-12
    Z = np.where(dead, 0., Z / np.where(dead, 1., sd))
    C = Z.T @ Z / len(Z)
    np.fill_diagonal(C, 1.)
    d = len(C)
    off = C[~np.eye(d, dtype=bool)]
    return C, {'mean_abs_offdiag': float(np.abs(off).mean()),
               'dist_to_identity': float(np.sqrt((off ** 2).sum() / (d * (d - 1)))), 'dead_dims': int(dead.sum())}


def label_spectrum(X, y, k=50):
    """(variance share of each of the top-k principal directions, in-sample label R² of a linear fit on the
    top 1..k directions). The directions are orthogonal, so the R² is the running sum of squared correlations.
    Descriptive only: no held-out data."""
    Z = np.asarray(X, np.float64) - np.mean(X, 0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    yc = np.asarray(y, np.float64) - np.mean(y)
    k = min(k, len(S))
    r = U[:, :k].T @ yc / (np.linalg.norm(U[:, :k], axis=0) * np.linalg.norm(yc))
    return (S ** 2 / (S ** 2).sum())[:k], np.cumsum(r ** 2)


@torch.no_grad()
def _block_states(model, x, device, batch=256):
    """{block k: (pooled (N, D), tokens (N, T, D))} for k = 0..depth."""
    backbone = EvalBackbone(model.to(device).eval())
    pooled, tokens = {}, {}
    for i in range(0, len(x), batch):
        hs = backbone(torch.as_tensor(x[i:i + batch], device=device)).hidden_states
        for k, h in enumerate(hs):
            pooled.setdefault(k, []).append(h.float().mean(1).cpu().numpy())
            tokens.setdefault(k, []).append(h.float().cpu().numpy())
    return {k: (np.concatenate(pooled[k]), np.concatenate(tokens[k])) for k in pooled}


def isotropy(checkpoint, output, n=10000, seed=0, device='cuda'):
    model, ck = load_model(str(checkpoint))
    cfg = ck['config']
    packed = Path(cfg['data']['packed_dir'])
    stats = cfg['derived']['global_stats']
    rows = valid_subset(np.load(packed / 'valid_rows.npy'), n, seed)
    x = ((np.load(packed / 'spectra.npy', mmap_mode='r')[rows].astype(np.float32) - stats['mean'])
         / stats['std']).astype(np.float32)
    torch.manual_seed(seed)
    random = build_model(cfg['model'], cfg['data']['sequence_length'])
    trained, rand = _block_states(model, x, device), _block_states(random, x, device)
    rng = np.random.default_rng(seed)
    blocks = {}
    for k in trained:
        tok = trained[k][1].reshape(-1, trained[k][1].shape[-1])
        tok = tok[rng.permutation(len(tok))[:20000]]
        p = _spectrum(tok)
        blocks[str(k)] = {'trained': geometry(trained[k][0], seed=seed), 'random': geometry(rand[k][0], seed=seed),
                          'trained_tokens': {'effective_rank': float(np.exp(-(p * np.log(p + 1e-12)).sum())),
                                             'top_pc_share': float(p[0])}}
    out = {'checkpoint': str(checkpoint), 'n': int(len(rows)), 'depth': int(cfg['model']['depth']), 'blocks': blocks}
    write_json(Path(output), out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv)
    out = isotropy(a.checkpoint, a.output, a.n, a.seed, a.device)
    for k, v in out['blocks'].items():
        print(k, {m: round(x, 3) for m, x in v['trained'].items()})


if __name__ == '__main__':
    main()
