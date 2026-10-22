"""Stage-2 diagnostics: block-2 feature geometry, displacement from the stage-1 reference, and early-encoder
gradient cosines between loss terms (no training, eval mode).

    python -m spectral_lejepa stage2-diag --checkpoint C --reference R --output O.json [--device cuda]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from spectral_lejepa.data.loader import valid_subset
from spectral_lejepa.evaluation.isotropy import _spectrum
from spectral_lejepa.training.checkpoint import load_model
from spectral_lejepa.training.gradient_terms import audit_batch
from spectral_lejepa.training.loss import LeJEPAObjective
from spectral_lejepa.utils.io import write_json

N_ROWS, N_BATCHES, BATCH, SEED = 2048, 4, 64, 4000


@torch.no_grad()
def _block2(model, x, device, batch=64):
    model.to(device).eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], device=device)
        _, hidden = model.encoder(model.tokenizer(xb), return_all=True)
        out.append(hidden[1].reshape(len(xb), -1).float().cpu().numpy())   # [B, num_patches * dim]
    return np.concatenate(out)


def _mean_nested(items):
    if isinstance(items[0], dict):
        return {k: _mean_nested([i[k] for i in items]) for k in items[0]}
    items = [i for i in items if i is not None]   # None = zero-norm cosine, left out of the mean
    return float(np.mean(items)) if items else None


def diag(checkpoint, reference, output, device='cuda'):
    model, ck = load_model(str(checkpoint))
    ref, ref_ck = load_model(str(reference))
    cfg = ck['config']
    stats = cfg['derived']['global_stats']
    if ref_ck['config']['derived']['global_stats'] != stats:
        raise ValueError('checkpoint and reference have different global stats')
    packed = Path(cfg['data']['packed_dir'])
    rows = valid_subset(np.load(packed / 'valid_rows.npy'), N_ROWS, 0)
    x = ((np.load(packed / 'spectra.npy', mmap_mode='r')[rows].astype(np.float32) - stats['mean'])
         / stats['std']).astype(np.float32)
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        h, h_ref = _block2(model, x, device), _block2(ref, x, device)
        var = h.astype(np.float64).var(0)
        p = _spectrum(h)
        ref_scale = float(np.sqrt(h_ref.astype(np.float64).var(0).sum()))
        features = {'n': int(len(rows)), 'var_mean': float(var.mean()), 'var_min_decile': float(np.percentile(var, 10)),
                    'effective_rank': float(np.exp(-(p * np.log(p + 1e-12)).sum())),
                    'displacement': float(np.mean(np.linalg.norm(h.astype(np.float64) - h_ref, axis=1)) / ref_scale)}
        t, robust = cfg['training'], cfg['training'].get('robust')
        robust_cfg = robust or {'weight': 0.0, 'block': 2, 'prob': .5, 'sigma_max': .2, 'offset_max': .05,
                                'slope_max': .05, 'scale_max': .05, 'shift_max': 1}   # arm A: measure the unweighted term
        objective = LeJEPAObjective(*(t[k] for k in ('lambda_sigreg', 'sigreg_num_slices', 'sigreg_knots',
            'sigreg_t_max', 'global_weight')), robust_cfg['weight']).to(device).eval()
        model.to(device).eval()
        early = list(model.tokenizer.parameters()) + [q for b in model.encoder.blocks[:2] for q in b.parameters()]
        global_cfg = {'views': t['global_views'], 'keep': t['global_keep']}
        per_batch = [audit_batch(model, objective, torch.as_tensor(x[i * BATCH:(i + 1) * BATCH]), cfg['masking'],
                                 global_cfg, seed=SEED + i, device=device, robust_cfg=robust_cfg, stats=stats,
                                 groups={'early': early}) for i in range(N_BATCHES)]
    out = {'checkpoint': str(checkpoint), 'reference': str(reference), 'features': features,
           'gradients': {'batches': N_BATCHES, 'batch_size': BATCH, 'seed': SEED,
                         'early': _mean_nested([b['blocks']['early'] for b in per_batch]),
                         'terms': _mean_nested([b['terms'] for b in per_batch])}}
    write_json(Path(output), out)
    return out


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--reference', required=True)
    a.add_argument('--output', required=True, type=Path)
    a.add_argument('--device', default='cuda')
    a = a.parse_args(argv)
    out = diag(a.checkpoint, a.reference, a.output, a.device)
    print(out['features'])


if __name__ == '__main__':
    main()
