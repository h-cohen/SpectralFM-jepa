"""Error metrics beyond R², from out-of-fold predictions already on disk (no refitting).

Every arm is a [R, n] array of OOF predictions (R repeats of the outer CV); a metric is computed per
repeat and averaged. Comparisons to the reference arm (raw ridge) are paired on the same spectra.
"""
import numpy as np


def _rmse(y, P):
    return float(np.mean(np.sqrt(((P - y) ** 2).mean(1))))


def error_metrics(y, P, ref='raw', n_boot=1000, seed=0):
    """{arm: metrics}. rmse in label units; nrmse = rmse / sd(y); mae, median and 90th-percentile
    absolute error; bias = mean residual; residual_slope = slope of (pred - y) on y (0 ideal, negative
    means shrinkage toward the mean); rmse_by_quintile over quintiles of y. Non-reference arms also get
    win_rate (share of spectra with lower |error| than the reference) and vs_raw: paired ΔRMSE with
    bootstrap SD and the bootstrap share of resamples where the arm's RMSE is lower."""
    y = np.asarray(y, dtype=np.float64)
    edges = np.quantile(y, np.linspace(0, 1, 6))
    q = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, 4)
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    R = np.asarray(P[ref], dtype=np.float64)
    out = {}
    for arm, A in P.items():
        A = np.asarray(A, dtype=np.float64)
        out[arm] = _arm(y, A, q)
        if arm == ref:
            continue
        d = np.array([_rmse(y[i], A[:, i]) - _rmse(y[i], R[:, i]) for i in boots])
        out[arm]['win_rate'] = float((np.abs(A - y) < np.abs(R - y)).mean())
        out[arm]['vs_raw'] = {'delta_rmse': out[arm]['rmse'] - _rmse(y, R), 'sd': float(d.std()),
                              'p_lower': float((d < 0).mean())}
    return out


def _arm(y, P, q):
    P = np.asarray(P, dtype=np.float64)
    e = P - y
    a = np.abs(e)
    return {'rmse': _rmse(y, P), 'nrmse': _rmse(y, P) / float(y.std()), 'mae': float(a.mean()),
            'median_ae': float(np.mean(np.median(a, 1))), 'p90_ae': float(np.mean(np.quantile(a, .9, axis=1))),
            'bias': float(e.mean()), 'residual_slope': float(np.polyfit(np.tile(y, len(P)), e.ravel(), 1)[0]),
            'rmse_by_quintile': [_rmse(y[q == k], P[:, q == k]) for k in range(5)]}
