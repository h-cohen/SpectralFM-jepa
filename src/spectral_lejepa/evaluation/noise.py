"""The previous evaluation's noise, applied to per-spectrum layer-normed rows, then mapped back to raw units."""
import numpy as np

EPS = 1e-5   # torch.nn.functional.layer_norm default, as the previous dataloader used


def layer_norm_rows(raw):
    raw = np.asarray(raw, np.float64)
    mu = raw.mean(1, keepdims=True)
    sd = np.sqrt(raw.var(1, keepdims=True) + EPS)
    return (raw - mu) / sd, mu, sd


def denormalize(z, mu, sd):
    return (z * sd + mu).astype(np.float32)


def noisy_raw(raw, kind, seed, configs):
    """Seeded per call; the previous code drew from the unseeded global NumPy state."""
    z, mu, sd = layer_norm_rows(raw)
    np.random.seed(seed)
    return denormalize(np.stack([configs[kind](row) for row in z]), mu, sd)


def row_cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
