"""How to add the complex channels 26-31 to LeJEPA comp0+1 without ever hurting it (exploratory, after
ops/complex_channels.py showed that joining them into one ridge vector dilutes the probe).

Base: LeJEPA bm100 block-2 flat [comp0 | comp1] (`j_c01`). Complex blocks X, from the two triplets (N, C, S) =
(26, 27, 28), (29, 30, 31), z = C + iS = sin 2ψ e^{iΔ}:
- cx: raw 26-31; j_cx: LeJEPA block-2 flat of 26-31;
- fft8 / fft16 / fft32: Re/Im of FFT(z) at |freq| <= K and of rFFT(N) at freq <= K, both triplets;
- polar: N, |z|, cos Δ, sin Δ (no unwrapping); psidelta: ψ = ½ atan2(|z|, N), cos Δ, sin Δ;
- rho: Re/Im of ρ = tanψ e^{iΔ} = z / (1 + N) (the ellipsometric ratio); rho_fft16: its FFT at |freq| <= 16;
- fftmag16 / fftmag32: |FFT(z)| at |freq| <= K and |rFFT(N)| (phase-free, shift-invariant; round 2);
- cross: Re/Im of z1 conj(z2) and N1 - N2 (the two measurements' relative amplitude and phase).
Every block gets its own nested probe (12 recipes, inner 5-fold CV, outer 2x5, seed 42; the main evaluation's
folds). Fusions, all decided on inner-CV predictions of the outer training rows only:
- mix: weight w on the base in {0, .25, .5, .75, 1}, best inner R²;
- safe: the mix, kept only if its bootstrap gain over the base on the inner predictions is > 2 SD (200
  resamples), else the base alone;
- safe0 / safe0.5 / safe1: the safe rule at 0, .5 and 1 SD (round 2);
- stack: non-negative least squares of y on the two inner predictions (weights sum to 1);
- cat: one probe on [base | X] for the low-dimensional blocks (not cx / j_cx);
- auto: inside each outer fold, the safe fusion of the block with the best inner R² (honest selection).
LeJEPA arms run per seed (107, 108); predictions are averaged over the two seeds.

    PYTHONPATH=src:. python ops/complex_fusion.py --set dataset0055 [--out outputs/complex-fusion]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

import ops.complex_channels as cc
from spectral_lejepa.evaluation.nested import _inner_predictions, _recipe_predictions, nnls_weights
from spectral_lejepa.evaluation.probe import RECIPES, r2

SEED, N_REP, N_FOLD, N_INNER = 42, 2, 5, 5
GRID = (0, .25, .5, .75, 1)
SAFE_SD = (0, .5, 1)            # looser fallback thresholds (round 2)


def complex_blocks(X):
    out = {'cx': np.concatenate([X[c] for c in (26, 27, 28, 29, 30, 31)], 1)}
    trip = [(X[n].astype(np.float64), X[c] + 1j * X[s].astype(np.float64)) for n, c, s in ((26, 27, 28), (29, 30, 31))]
    def fft(sig, k, real=False):
        F = (np.fft.rfft(sig, axis=1)[:, :k + 1] if real else
             np.concatenate([np.fft.fft(sig, axis=1)[:, :k + 1], np.fft.fft(sig, axis=1)[:, -k:]], 1)) / sig.shape[1]
        return [F.real, F.imag]
    for k in (8, 16, 32):
        out[f'fft{k}'] = np.concatenate([a for N, z in trip for a in fft(z, k) + fft(N, k, True)], 1)
    for k in (16, 32):
        out[f'fftmag{k}'] = np.concatenate([np.abs(np.fft.fft(z, axis=1))[:, np.r_[0:k + 1, -k:0]] / z.shape[1] for _, z in trip]
                                           + [np.abs(np.fft.rfft(N, axis=1))[:, :k + 1] / N.shape[1] for N, _ in trip], 1)
    ph = [np.angle(z) for _, z in trip]
    out['polar'] = np.concatenate([a for (N, z), p in zip(trip, ph) for a in (N, np.abs(z), np.cos(p), np.sin(p))], 1)
    out['psidelta'] = np.concatenate([a for (N, z), p in zip(trip, ph)
                                      for a in (.5 * np.arctan2(np.abs(z), N), np.cos(p), np.sin(p))], 1)
    rho = [z / (1 + N) for N, z in trip]
    out['rho'] = np.concatenate([a for r in rho for a in (r.real, r.imag)], 1)
    out['rho_fft16'] = np.concatenate([a for r in rho for a in fft(r, 16)], 1)
    x = trip[0][1] * np.conj(trip[1][1])
    out['cross'] = np.concatenate([x.real, x.imag, trip[0][0] - trip[1][0]], 1)
    return {k: np.nan_to_num(v).astype(np.float32) for k, v in out.items()}


def best_inner(X, y, seed, device):
    P = _inner_predictions(X, y, N_INNER, seed, 'torch', device, RECIPES)
    rc = max(RECIPES, key=lambda r: r2(y, P[r]))
    return rc, P[rc]


def safe_mix(pb, px, y, seed, n_sd=2):
    """(weight on base, inner R²) of the best grid mix, falling back to the base unless the gain is > 2 SD."""
    scores = {w: r2(y, w * pb + (1 - w) * px) for w in GRID}
    w = max(scores, key=lambda v: (scores[v], v))
    if w == 1:
        return 1, scores[1], w
    rng = np.random.default_rng(seed)
    g = np.array([r2(y[i], w * pb[i] + (1 - w) * px[i]) - r2(y[i], pb[i])
                  for i in (rng.integers(0, len(y), len(y)) for _ in range(200))])
    return (w, scores[w], w) if g.mean() > n_sd * g.std() else (1, scores[1], w)


def run_seed(base, blocks, y, device):
    """{arm: oof [R, n]} for base, every block alone and every fusion, plus per-fold choices."""
    n = len(y)
    oof, log = {}, []
    put = lambda k, r, te, p: oof.setdefault(k, np.zeros((N_REP, n))).__setitem__((r, te), p)
    for r in range(N_REP):
        for f, (tr, te) in enumerate(KFold(N_FOLD, shuffle=True, random_state=SEED + r).split(np.zeros(n))):
            seed = SEED + 1000 + 10 * r + f
            ytr = y[tr]
            rb, pb = best_inner(base[tr], ytr, seed, device)
            tb = _recipe_predictions(base[tr], ytr, base[te], [rb], seed, 'torch', device)[rb]
            put('base', r, te, tb)
            rec = {'repeat': r, 'fold': f, 'base': list(rb), 'blocks': {}}
            safe_inner = {}
            for k, Xk in blocks.items():
                rx, px = best_inner(Xk[tr], ytr, seed, device)
                tx = _recipe_predictions(Xk[tr], ytr, Xk[te], [rx], seed, 'torch', device)[rx]
                put(k, r, te, tx)
                w_mix = max(GRID, key=lambda v: (r2(ytr, v * pb + (1 - v) * px), v))
                put(f'mix:{k}', r, te, w_mix * tb + (1 - w_mix) * tx)
                w_safe, s_safe, _ = safe_mix(pb, px, ytr, seed)
                put(f'safe:{k}', r, te, w_safe * tb + (1 - w_safe) * tx)
                safe_inner[k] = (s_safe, w_safe * tb + (1 - w_safe) * tx)
                for sd in SAFE_SD:
                    w_sd = safe_mix(pb, px, ytr, seed, sd)[0]
                    put(f'safe{sd}:{k}', r, te, w_sd * tb + (1 - w_sd) * tx)
                ws = nnls_weights(np.stack([pb, px], 1), ytr)
                put(f'stack:{k}', r, te, ws[0] * tb + ws[1] * tx)
                rec['blocks'][k] = {'recipe': list(rx), 'inner_r2': r2(ytr, px), 'w_mix': w_mix, 'w_safe': w_safe,
                                    'w_stack': ws.tolist()}
                if k not in ('cx', 'j_cx'):
                    Z = np.concatenate([base, Xk], 1)
                    rz, _ = best_inner(Z[tr], ytr, seed, device)
                    put(f'cat:{k}', r, te, _recipe_predictions(Z[tr], ytr, Z[te], [rz], seed, 'torch', device)[rz])
            # ties (every block falling back to the base) resolve to the base itself
            k_auto = max(safe_inner, key=lambda k: safe_inner[k][0] - 1e-9 * (safe_inner[k][0] == r2(ytr, pb)))
            put('auto', r, te, safe_inner[k_auto][1])
            rec['auto'] = k_auto
            log.append(rec)
    return oof, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--set', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, default=Path('outputs/complex-fusion'))
    ap.add_argument('--blocks', nargs='+', help='complex blocks to run (default: all; j_cx always)')
    a = ap.parse_args(argv)
    X, y, _ = cc.load(a.set)
    blocks = complex_blocks(X)
    if a.blocks:
        blocks = {k: blocks[k] for k in a.blocks}
    runs, logs = [], {}
    for ck in cc.CKPTS:
        J = cc.jepa_arms(X, ck, a.device)
        oof, log = run_seed(J['j_c01'], {**blocks, 'j_cx': J['j_cx']}, y, a.device)
        runs.append(oof)
        logs[Path(ck).parent.name[:10]] = log
        print(a.set, Path(ck).parent.name[:10], 'done', flush=True)
    oof = {k: np.mean([o[k] for o in runs], 0) for k in runs[0]}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / f'{a.set}_oof.npz', y=y, **oof)
    (a.out / f'{a.set}_log.json').write_text(json.dumps(logs))
    score = lambda P: float(np.mean([r2(y, p) for p in P]))
    print(a.set, 'n', len(y), ' '.join(f'{k} {score(P):.3f}' for k, P in oof.items()))


if __name__ == '__main__':
    main()
