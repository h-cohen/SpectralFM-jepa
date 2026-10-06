"""Torch (float64, GPU) port of probe.py's per-fold normalizers and probes, numerically matching
the sklearn 1.6.1 reference. probe.py stays the reference and the default backend.

Exactness, per piece (dtype decisions follow what sklearn computes in):
- none / standardize: float32 arrays exactly as probe.py makes them (standardize IS
  probe._standardize, run on the host: numpy's float32 axis-0 reductions are sequential, a
  GPU reduction would differ in the last bits).
- whiten: PCA(whiten=True, svd_solver="full") in float64: thin SVD of the centred training
  rows, svd_flip(u_based_decision=False) signs, explained_variance = S²/(n-1), transform as
  sklearn's `_transform` (X @ Cᵀ - mean @ Cᵀ, then / max(sqrt(ev), eps)), probe.py's k / k_keep,
  cast to float32. Only the exact ("full") solver is ported: min(n, d) <= EXACT_SVD_MAX_DIM is
  asserted, else probe._whiten runs.
- ridgecv: RidgeCV(alphas=logspace(-3, 3, 20)), cv=None, i.e. _RidgeGCV. sklearn validates X
  and y to float64 (dtype=[np.float64]), so everything here is float64: centring
  (_preprocess_data), gcv_mode "svd" (SVD of [Xc, 1]) when n > p else "eigen" (eigh of
  Xc Xcᵀ + 11ᵀ), intercept dimension by _find_smallest_angle, squared LOO errors c / diag(G⁻¹),
  first strict minimum of their mean over the alpha grid, coef = Xcᵀ c, intercept
  y_offset - X_offset·coef. One decomposition serves all 20 alphas.
- ols: LinearRegression keeps float32 input in float32 (y is cast to X.dtype too) and calls
  scipy.linalg.lstsq(Xc, yc, cond=max(n, p)·eps(float32)) -- gelsd, which zeroes singular
  values <= cond·s_max and returns the minimum-norm solution. Here: y is cast to float32 and
  centred in float32 exactly as sklearn (host numpy), the solve is the same minimum-norm
  pseudo-inverse with the same cutoff but in float64, reusing the ridge decomposition (its
  non-intercept components are those of Xc), and predictions are returned in float32 as
  sklearn's. The float32 rounding inside sklearn's own sgelsd is therefore the only difference
  (documented tolerance: tests/test_probe_torch.py).
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .probe import (EXACT_SVD_MAX_DIM, NORMALIZERS, RECIPES, WHITEN_SAMPLES_PER_DIRECTION,
                    _standardize)
from .probe import _whiten as _whiten_sklearn

ALPHAS = np.logspace(-3, 3, 20)          # = probe.make_regressor("ridgecv").alphas
F64 = torch.float64


def _devices(device):
    """`device` as a list of torch devices: None -> [cuda], a name, or a list of names."""
    names = ["cuda"] if device is None else [device] if isinstance(device, (str, torch.device)) else list(device)
    if not names:
        raise ValueError("empty probe device list")
    return [torch.device(d) for d in names]


def _device(device):
    """The single device a fit runs on (the first one of a list)."""
    return _devices(device)[0]


def _tall_svd(A):
    """Thin SVD (U, S, Vh) of A. cuSOLVER's gesvd is used on the tall orientation (it is
    several times faster there than the default Jacobi driver)."""
    if A.shape[0] >= A.shape[1]:
        driver = "gesvd" if A.is_cuda else None
        return torch.linalg.svd(A, full_matrices=False, driver=driver)
    V, S, Uh = _tall_svd(A.T)
    return Uh.T, S, V.T


def _whiten(X_tr, X_te, seed, device):
    """probe._whiten on the device: (Z_tr, Z_te) float32 with the k_keep alive directions."""
    n, d = X_tr.shape
    if min(n, d) > EXACT_SVD_MAX_DIM:     # sklearn would switch to the randomized solver
        return [torch.as_tensor(z, device=device) for z in _whiten_sklearn(X_tr, X_te, seed)]
    X64 = torch.as_tensor(np.asarray(X_tr), device=device).to(F64)
    k = min(d, max(1, n // WHITEN_SAMPLES_PER_DIRECTION))
    mean = X64.mean(0)
    _, S, Vt = _tall_svd(X64 - mean)
    # svd_flip(u_based_decision=False): the largest-|.| entry of each row of Vt is positive
    pick = Vt.gather(1, Vt.abs().argmax(1, keepdim=True))
    Vt = Vt * torch.sign(pick)
    ev = S[:k] ** 2 / (n - 1)
    tol = (np.finfo(np.float64).eps * max(n, d)) ** 2
    k_keep = max(1, int((ev > tol * ev[0]).sum()) if ev.numel() else 0)
    C = Vt[:k_keep]
    scale = ev[:k_keep].sqrt()
    scale = torch.where(scale < np.finfo(np.float64).eps, np.finfo(np.float64).eps, scale)
    shift = mean @ C.T

    def transform(X):
        X = torch.as_tensor(np.asarray(X), device=device).to(F64)
        return ((X @ C.T - shift) / scale).to(torch.float32)
    return transform(X_tr), transform(X_te)


def fold_transforms(X_tr, X_te, seed=42, device=None) -> dict:
    """{normalizer: (X_tr', X_te')} as float32 tensors on `device`, fit on X_tr only;
    value-for-value probe.fold_transforms (whitened columns up to PCA sign)."""
    device = _device(device)
    X_tr32, X_te32 = np.asarray(X_tr, dtype=np.float32), np.asarray(X_te, dtype=np.float32)
    up = lambda a: torch.as_tensor(a, device=device)   # noqa: E731
    out = {"none": (up(X_tr32), up(X_te32)),
           "standardize": tuple(up(a) for a in _standardize(X_tr32, X_te32))}
    Ztr, Zte = _whiten(X_tr32, X_te32, seed, device)
    for name in NORMALIZERS:
        if name.startswith("whiten"):
            k = Ztr.shape[1] if name == "whiten" else min(Ztr.shape[1], int(name[len("whiten"):]))
            out[name] = (Ztr[:, :k], Zte[:, :k])
    return out


def _decompose(X, intercept=True):
    """Centre X (float64) and decompose as _RidgeGCV does: gcv_mode "svd" -> thin SVD of
    [Xc, 1]; "eigen" -> eigh of Xc Xcᵀ + 11ᵀ. Returns the centring and (Q, eig, idim): Q's
    columns span the samples, eig the squared singular values / eigenvalues, idim the intercept
    component (_find_smallest_angle against the normalized ones vector). intercept=False
    decomposes Xc alone (idim None)."""
    n, p = X.shape
    x_off = X.mean(0)
    Xc = X - x_off
    ones = torch.ones(n, 1, dtype=F64, device=X.device)
    if n > p:                                  # _check_gcv_mode: "svd"
        Q, s, _ = _tall_svd(torch.cat([Xc, ones], 1) if intercept else Xc)
        eig = s ** 2
    else:                                      # "eigen"
        eig, Q = torch.linalg.eigh(Xc @ Xc.T + ones @ ones.T if intercept else Xc @ Xc.T)
    dec = {"Xc": Xc, "x_off": x_off, "Q": Q, "eig": eig, "idim": None, "mode": "svd" if n > p else "eigen"}
    if intercept:
        cos = ((ones[:, 0] / np.sqrt(n)) @ Q).abs()
        dec["idim"] = int(cos.argmax())
        dec["intercept_cos"] = float(cos[dec["idim"]])
    return dec


def ridge_gcv(X, y, dec=None):
    """_RidgeGCV.fit (fit_intercept, no sample weights, scoring=None) for every alpha at once.
    X: [n, p] (float32 values), y: [n] float64. Returns {coef, intercept, alpha}."""
    X = X.to(F64)
    y = torch.as_tensor(y, device=X.device).to(F64)
    dec = dec or _decompose(X)
    y_off = y.mean()
    yc = y - y_off
    Q, eig, idim = dec["Q"], dec["eig"], dec["idim"]
    al = torch.as_tensor(ALPHAS, dtype=F64, device=X.device)[:, None]      # [A, 1]
    QTy = Q.T @ yc
    if dec["mode"] == "svd":                  # _solve_svd_design_matrix
        w = (eig[None] + al) ** -1 - al ** -1
        w[:, idim] = -(al[:, 0] ** -1)
        c = Q @ (w * QTy).T + yc[:, None] * al.T ** -1
        G = (Q ** 2) @ w.T + al.T ** -1
    else:                                      # _solve_eigen_gram
        w = 1.0 / (eig[None] + al)
        w[:, idim] = 0
        c = Q @ (w * QTy).T
        G = (Q ** 2) @ w.T
    scores = (-((c / G) ** 2).mean(0)).cpu().numpy()
    best = 0                                   # first strict improvement, as the alpha loop
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    coef = dec["Xc"].T @ c[:, best]
    return {"coef": coef, "intercept": y_off - dec["x_off"] @ coef, "alpha": float(ALPHAS[best])}


def ols(X, y, dec=None):
    """LinearRegression().fit: minimum-norm least squares on the centred design, singular
    values <= max(n, p)·eps32·s_max dropped (lstsq/gelsd cond), y in float32 as sklearn."""
    X = X.to(F64)
    n, p = X.shape
    if dec is None or dec["intercept_cos"] < 1 - 1e-8:
        # the ridge decomposition's intercept component is not cleanly the ones column (an Xc
        # singular value ~ sqrt(n) mixes with it): decompose Xc on its own
        dec = _decompose(X, intercept=False)
    y32 = np.asarray(y).astype(np.float32)
    y_off = y32.mean(axis=0)                   # _preprocess_data: _average in float32
    yc = torch.as_tensor(y32 - y_off, device=X.device).to(F64)
    keep = torch.ones_like(dec["eig"], dtype=torch.bool)
    if dec["idim"] is not None:
        keep[dec["idim"]] = False              # the added ones column is not part of Xc
    eig, Q = dec["eig"][keep], dec["Q"][:, keep]
    cond = max(n, p) * np.finfo(np.float32).eps
    alive = eig > (cond ** 2) * eig.max()      # s > cond·s_max  <=>  s² > cond²·s_max²
    Qa = Q[:, alive]
    coef = dec["Xc"].T @ (Qa @ ((Qa.T @ yc) / eig[alive]))
    return {"coef": coef, "intercept": float(y_off) - dec["x_off"] @ coef}


def _fit(Z_tr, y_tr, Z_te, probes):
    dec = _decompose(Z_tr.to(F64))
    Zte = Z_te.to(F64)
    out = {}
    for pr in probes:
        if pr == "ridgecv":
            m = ridge_gcv(Z_tr, y_tr, dec)
            out[pr] = (Zte @ m["coef"] + m["intercept"]).cpu().numpy()
        elif pr == "ols":
            m = ols(Z_tr, y_tr, dec)
            out[pr] = (Zte @ m["coef"] + m["intercept"]).to(torch.float32).cpu().numpy()
        else:
            raise ValueError(f"unknown probe {pr!r}")
    return out


def recipe_predictions(X_tr, y_tr, X_te, recipes=RECIPES, seed=42, device=None) -> dict:
    """{(norm, probe): prediction on X_te} for the requested recipes, every normalizer and
    decomposition computed once (= probe.fit_predict on probe.fold_transforms)."""
    recipes = list(recipes)
    if bad := [rc for rc in recipes if rc not in RECIPES]:
        raise ValueError(f"unknown recipe {bad[0]!r}")
    T = fold_transforms(X_tr, X_te, seed=seed, device=device)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    out = {}
    for norm in dict.fromkeys(n for n, _ in recipes):
        Z_tr, Z_te = T[norm]
        fits = _fit(Z_tr, y_tr, Z_te, [pr for n, pr in recipes if n == norm])
        out.update({(norm, pr): p for pr, p in fits.items()})
    return {rc: out[rc] for rc in recipes}


def fit_predict(X_tr, y_tr, X_te, probe, seed=42, device=None):
    """probe.fit_predict on the device (numpy in, numpy out)."""
    if probe not in ("ridgecv", "ols"):
        raise ValueError(f"unknown probe {probe!r}")
    device = _device(device)
    Z_tr = torch.as_tensor(np.asarray(X_tr, dtype=np.float32), device=device)
    Z_te = torch.as_tensor(np.asarray(X_te, dtype=np.float32), device=device)
    return _fit(Z_tr, y_tr, Z_te, [probe])[probe]


def map_on_streams(fn, args_list, workers=1, device=None):
    """[fn(*args, device=<its device>) for args in args_list]. Jobs are dealt round-robin over
    the device(s); each device runs up to `workers` jobs at once, each worker thread on its own
    CUDA stream (cuSOLVER's SVD/eigh are latency-bound, so concurrent folds overlap). Peak GPU
    memory grows with `workers` (~2 GB per labeled_data-sized fold). Every job computes
    exactly what it would alone: only the scheduling changes, results are bit-identical."""
    devices = _devices(device)
    assign = [devices[i % len(devices)] for i in range(len(args_list))]
    if workers <= 1 and len(devices) == 1 or devices[0].type != "cuda":
        return [fn(*a, device=d) for a, d in zip(args_list, assign)]
    local = threading.local()

    def run(job):
        args, dev = job
        if getattr(local, "device", None) != dev:
            local.device, local.stream = dev, torch.cuda.Stream(device=dev)
        with torch.cuda.device(dev), torch.cuda.stream(local.stream):
            out = fn(*args, device=dev)
            local.stream.synchronize()
        return out
    pools = {d: ThreadPoolExecutor(max(1, workers)) for d in devices}
    try:
        futures = [pools[d].submit(run, (a, d)) for a, d in zip(args_list, assign)]
        return [f.result() for f in futures]
    finally:
        for pool in pools.values():
            pool.shutdown(cancel_futures=True)
        for d in devices:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
