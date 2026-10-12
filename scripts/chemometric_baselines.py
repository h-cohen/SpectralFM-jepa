"""A2: chemometric raw-spectrum references under the primary nested-CV protocol.

Same outer folds as nested.nested_cv (KFold shuffle, random_state=seed+r), same inner-fold seeds
(seed+1000+10r+f), raw input = bank input_raw. Hyperparameters are chosen by inner-CV R² on
training rows only; every transform is fit on training rows only. Arms:
  linear_raw  contract raw arm re-run on CPU/sklearn (reproduction check vs linear_oof.npz['raw'])
  pls         PLS1, mean-centred (no scaling), k in 1..min(20, n_inner_train-2)
  snv_d1_pls  row-wise SNV, Savitzky-Golay d1 (window 11, poly 2), then PLS
  krr         RBF kernel ridge on train-standardized raw, y centred; alpha x gamma/n_features grid
  union3      per outer fold, the arm of {pls, snv_d1_pls, krr} with best inner R²
  union4      same, plus the contract raw arm
  pls_z       note only: PLS on input_z
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from spectral_lejepa.evaluation.nested import _inner_scores, _recipe_predictions, bootstrap_sd, paired_delta
from spectral_lejepa.evaluation.probe import RECIPES, r2

DATASETS = ["labeled_data", "dataset0055", "dataset0106", "dataset0109", "dataset0112", "dataset0113",
            "dataset0114", "dataset0120"]
CHECKPOINTS = ["prior_s0", "prior_s1", "prior_s2", "baseline_s101"]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GAMMAS = [0.1, 0.3, 1.0, 3.0]            # times 1 / n_features
K_MAX = 20
PLS_ARMS = ("pls", "snv_d1_pls", "pls_z")
ARMS = ("linear_raw", "pls", "snv_d1_pls", "krr", "union3", "union4", "pls_z")


def snv_d1(X):
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)


def pls_all_k(X_tr, y_tr, X_te, kmax):
    """Predictions [kmax, n_te] for k=1..kmax components (PLS1 components are sequential, so the
    first k columns of one fit equal a k-component fit)."""
    mx, my = X_tr.mean(0), y_tr.mean()
    m = PLSRegression(n_components=kmax, scale=False).fit(X_tr - mx, y_tr - my)
    out = np.empty((kmax, len(X_te)))
    Z = (X_te - mx) @ m.x_rotations_
    for k in range(1, kmax + 1):
        out[k - 1] = Z[:, :k] @ m.y_loadings_[0, :k] + my
    return out


def pls_select(X, y, tr, te, inner_seed, n_inner):
    folds = list(KFold(n_inner, shuffle=True, random_state=inner_seed).split(tr))
    kmax = min(K_MAX, min(len(a) for a, _ in folds) - 2)
    P = np.zeros((kmax, len(tr)))
    for a, b in folds:
        P[:, b] = pls_all_k(X[tr][a], y[tr][a], X[tr][b], kmax)
    scores = [r2(y[tr], p) for p in P]
    k = int(np.argmax(scores)) + 1
    return pls_all_k(X[tr], y[tr], X[te], k)[k - 1], {"k": k}, scores[k - 1]


def _krr_grid(X_tr, y_tr, X_te):
    """{(alpha, gamma_mult): prediction} -- one eigendecomposition per gamma."""
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, B = (X_tr - mu) / sd, (X_te - mu) / sd
    sa, sb = (A * A).sum(1), (B * B).sum(1)
    D_tr = np.maximum(sa[:, None] + sa[None] - 2 * A @ A.T, 0)
    D_te = np.maximum(sb[:, None] + sa[None] - 2 * B @ A.T, 0)
    my = y_tr.mean()
    out = {}
    for g in GAMMAS:
        gamma = g / X_tr.shape[1]
        w, U = eigh(np.exp(-gamma * D_tr))
        Kte_U = np.exp(-gamma * D_te) @ U
        Uty = U.T @ (y_tr - my)
        for a in ALPHAS:
            out[(a, g)] = Kte_U @ (Uty / (np.maximum(w, 0) + a)) + my
    return out


def krr_select(X, y, tr, te, inner_seed, n_inner):
    P = {k: np.zeros(len(tr)) for k in [(a, g) for a in ALPHAS for g in GAMMAS]}
    for a, b in KFold(n_inner, shuffle=True, random_state=inner_seed).split(tr):
        for k, p in _krr_grid(X[tr][a], y[tr][a], X[tr][b]).items():
            P[k][b] = p
    scores = {k: r2(y[tr], p) for k, p in P.items()}
    best = max(scores, key=scores.get)
    return _krr_grid(X[tr], y[tr], X[te])[best], {"alpha": best[0], "gamma_mult": best[1]}, scores[best]


def outer_fold(ds, r, f, tr, te, X, Z, y, n_inner, inner_seed):
    with threadpool_limits(limits=1):
        res = {}
        S = _inner_scores(X[tr], y[tr], n_inner, inner_seed)            # contract raw arm, sklearn
        norm, probe = max(RECIPES, key=lambda rc: S[rc])
        pred = _recipe_predictions(X[tr], y[tr], X[te], [(norm, probe)], inner_seed, "sklearn")[(norm, probe)]
        res["linear_raw"] = (pred, {"norm": norm, "probe": probe}, S[(norm, probe)])
        D = snv_d1(X)
        res["pls"] = pls_select(X, y, tr, te, inner_seed, n_inner)
        res["snv_d1_pls"] = pls_select(D, y, tr, te, inner_seed, n_inner)
        res["krr"] = krr_select(X, y, tr, te, inner_seed, n_inner)
        res["pls_z"] = pls_select(Z, y, tr, te, inner_seed, n_inner)
        for name, members in (("union3", ("pls", "snv_d1_pls", "krr")),
                              ("union4", ("linear_raw", "pls", "snv_d1_pls", "krr"))):
            m = max(members, key=lambda a: res[a][2])
            res[name] = (res[m][0], {"arm": m, **res[m][1]}, res[m][2])
    return ds, r, f, te, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/frozen-feature-study")
    ap.add_argument("--out", default="outputs/production-validation/chemometric")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for ds in a.datasets:
        b = np.load(root / "prior_s2" / ds / "bank.npz")
        data[ds] = (b["input_raw"][:, 0].astype(np.float32), b["input_z"][:, 0].astype(np.float32),
                    b["y"].astype(np.float64))
    jobs = []
    for ds in a.datasets:                                              # biggest set first
        X, Z, y = data[ds]
        for r in range(2):
            for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=a.seed + r).split(y)):
                jobs.append((ds, r, f, tr, te, X, Z, y, 5, a.seed + 1000 + 10 * r + f))
    t0 = time.time()
    outs = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(outer_fold)(*j) for j in jobs)
    print(f"folds done in {time.time() - t0:.0f}s", flush=True)

    oof = {ds: {arm: np.zeros((2, len(data[ds][2]))) for arm in ARMS} for ds in a.datasets}
    chosen = {ds: {arm: [] for arm in ARMS} for ds in a.datasets}
    for ds, r, f, te, res in outs:
        for arm, (p, info, inner) in res.items():
            oof[ds][arm][r, te] = p
            chosen[ds][arm].append({"repeat": r, "fold": f, **info, "inner_r2": inner})

    results, flat = {}, {}
    for ds in a.datasets:
        y = data[ds][2]
        lin = np.load(root / "prior_s2" / ds / "linear_oof.npz")
        raw = lin["raw"]
        refs = {}
        for ck in CHECKPOINTS:
            L = np.load(root / ck / ds / "linear_oof.npz")
            if not np.array_equal(L["y"], y) or not np.array_equal(L["raw"], raw):
                raise ValueError(f"{ck}/{ds}: rows or raw OOF differ")
            refs[ck] = L["layer2/flat"]
        R = {"n": len(y),
             "raw_repro_max_abs_diff": float(np.abs(oof[ds]["linear_raw"] - raw).max()),
             "contract_raw": {"r2_mean": float(np.mean([r2(y, p) for p in raw])),
                              "bootstrap_sd": bootstrap_sd(y, raw)},
             "trained_block2_flat": {ck: {"r2_mean": float(np.mean([r2(y, p) for p in P])),
                                          "bootstrap_sd": bootstrap_sd(y, P)} for ck, P in refs.items()},
             "arms": {}}
        for arm in ARMS:
            P = oof[ds][arm]
            R["arms"][arm] = {
                "r2_per_repeat": [r2(y, p) for p in P], "r2_mean": float(np.mean([r2(y, p) for p in P])),
                "bootstrap_sd": bootstrap_sd(y, P),
                "vs_contract_raw": paired_delta(y, P, raw),
                "trained_minus_arm": {ck: paired_delta(y, refs[ck], P) for ck in CHECKPOINTS},
                "chosen": chosen[ds][arm]}
            flat[f"{ds}/{arm}"] = P
        results[ds] = R
        print(ds, "repro", R["raw_repro_max_abs_diff"],
              {k: round(v["r2_mean"], 4) for k, v in R["arms"].items()}, "raw", round(R["contract_raw"]["r2_mean"], 4),
              flush=True)
    meta = {"protocol": {"n_repeats": 2, "n_folds": 5, "n_inner": 5, "seed": a.seed},
            "pls": "mean-centred, no scaling, k=1..min(20,n_inner_train-2)",
            "krr": {"alphas": ALPHAS, "gamma_mult_over_n_features": GAMMAS, "standardize": "train", "y": "centred"},
            "snv_d1": "row SNV then savgol window 11 poly 2 deriv 1", "input": "input_raw (pls_z: input_z)"}
    tmp = out / "results.json.tmp"
    tmp.write_text(json.dumps({"meta": meta, "datasets": results}, indent=2, default=str))
    tmp.replace(out / "results.json")
    np.savez_compressed(out / "oof.npz", **{k.replace("/", "__"): v for k, v in flat.items()},
                        **{f"{ds}__y": data[ds][2] for ds in a.datasets})


if __name__ == "__main__":
    main()
