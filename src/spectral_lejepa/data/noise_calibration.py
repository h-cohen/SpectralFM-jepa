"""Label-free measurement-perturbation calibration (Step 0 of the robustness continuation).

Rules fixed BEFORE computing (recommended training ranges):
  sigma_max  = mass-weighted p90 of savgol noise_rel (window 11, order 2; std(resid)/std(x)).
  offset_max, slope_max, scale_max = min(0.05, mass-weighted p90 of the near-duplicate
      estimate |b|/range, |c|/range, |a-1| respectively), floored at 0.005.
  shift_max  = 1 if >= 5% of near-duplicate pairs prefer a nonzero integer shift in -3..3, else 0.
Mass = training sampling mass per source group (labeled_regression .5 / default .5).
Raw values (pre-normalization), length 245. Labels are never read.
Near-duplicates: per group, nearest neighbour (per-row z-scored Euclidean) within a pool;
top 1% closest NN pairs; fit x_j ~ a*x_i + b + c*t, t=linspace(-.5,.5).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_lejepa.data.loader import packed_train_rows, source_mass, _source_groups  # noqa: E402

PACKED = "/mnt5/home/hadar/nova/data/spectra_all"
SW = {"labeled_regression": 0.5, "default": 0.5}
N_NOISE, N_POOL, SEED = 2000, 10000, 0
QS = [10, 50, 90, 99]
OUT = Path("outputs/robustness-continuation/noise_calibration.json")


def wquant(v, w, q):
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q / 100, c, v))


def quants(v):
    return {f"p{q}": float(np.percentile(v, q)) for q in QS}


def noise_stats(x):
    sd = x.std(1)
    ok = sd > 0
    x, sd = x[ok], sd[ok]
    sg = (x - savgol_filter(x, 11, 2, axis=1)).std(1) / sd
    d = np.diff(x, axis=1)
    mad = np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1) * 1.4826 / np.sqrt(2) / sd
    return sg, mad


def fit_pair(xi, xj, s):
    """x_j[shifted by s] ~ a x_i + b + c t on the overlap; returns (a, b, c, resid_rms)."""
    n = len(xi)
    if s >= 0:
        u, v = xi[: n - s], xj[s:]
    else:
        u, v = xi[-s:], xj[: n + s]
    t = np.linspace(-.5, .5, n)[: len(u)] if True else None
    A = np.stack([u, np.ones_like(u), t], 1)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    r = v - A @ coef
    return coef, float(np.sqrt((r ** 2).mean()))


def main():
    arr = np.load(f"{PACKED}/spectra.npy", mmap_mode="r")
    rows = packed_train_rows(PACKED)
    groups, mass = _source_groups(PACKED, rows, SW)
    assert mass == source_mass(PACKED, rows, SW)
    rng = np.random.default_rng(SEED)
    res = {"mass": mass, "groups": {}}
    pool_vals = {k: {"sg": [], "mad": [], "w": []} for k in ("noise",)}
    nd = {"b": [], "c": [], "a": [], "w": [], "shift": []}
    sg_all, mad_all, w_all = [], [], []
    for g in mass:
        idx = rows[groups == g]
        sel = np.sort(rng.choice(idx, min(N_NOISE, len(idx)), replace=False))
        x = np.asarray(arr[sel], dtype=np.float64)
        sg, mad = noise_stats(x)
        sg_all.append(sg); mad_all.append(mad); w_all.append(np.full(len(sg), mass[g] / len(sg)))
        # near-duplicate pool (superset of the noise sample)
        psel = np.sort(rng.choice(idx, min(N_POOL, len(idx)), replace=False))
        p = np.asarray(arr[psel], dtype=np.float64)
        p = p[p.std(1) > 0]
        z = (p - p.mean(1, keepdims=True)) / p.std(1, keepdims=True)
        sq = (z ** 2).sum(1)
        d2 = sq[:, None] + sq[None] - 2 * z @ z.T
        np.fill_diagonal(d2, np.inf)
        nn = d2.argmin(1)
        dn = d2[np.arange(len(p)), nn]
        k = max(1, int(0.01 * len(p)))
        top = np.argsort(dn)[:k]
        pairs = {tuple(sorted((int(i), int(nn[i])))) for i in top}
        B, C, A, S = [], [], [], []
        for i, j in pairs:
            xi, xj = p[i], p[j]
            fits = {s: fit_pair(xi, xj, s) for s in range(-3, 4)}
            best = min(fits, key=lambda s: fits[s][1])
            rg = np.ptp(xi)
            if rg == 0:
                continue
            (a, b, c), _ = fits[0]  # shift-0 fit for baseline/scale numbers
            A.append(abs(a - 1)); B.append(abs(b) / rg); C.append(abs(c) / rg); S.append(best)
        S = np.array(S)
        for key, v in (("b", B), ("c", C), ("a", A)):
            nd[key] += list(v)
        nd["w"] += [mass[g] / len(B)] * len(B)
        nd["shift"] += list(S)
        res["groups"][g] = {
            "n_noise_rows": int(len(sg)), "n_pool_rows": int(len(p)), "n_pairs": int(len(B)),
            "noise_rel_savgol": quants(sg), "noise_rel_mad": quants(mad),
            "nd_offset_rel_range": quants(B), "nd_slope_rel_range": quants(C), "nd_scale_abs_dev": quants(A),
            "nd_frac_nonzero_shift": float((S != 0).mean()),
            "nd_shift_hist": {int(s): int((S == s).sum()) for s in range(-3, 4)},
            "nd_median_pair_z_dist2": float(np.median(dn[top])),
        }
    w = np.concatenate(w_all)
    sg_all, mad_all = np.concatenate(sg_all), np.concatenate(mad_all)
    res["corpus_weighted"] = {
        "noise_rel_savgol": {f"p{q}": wquant(sg_all, w, q) for q in QS},
        "noise_rel_mad": {f"p{q}": wquant(mad_all, w, q) for q in QS},
    }
    ndw = np.array(nd["w"])
    res["near_dup_weighted"] = {key: {f"p{q}": wquant(np.array(nd[key]), ndw, q) for q in QS}
                                for key in ("b", "c", "a")}
    frac_shift = float(np.average(np.array(nd["shift"]) != 0, weights=ndw))
    res["near_dup_weighted"]["frac_nonzero_shift"] = frac_shift
    p90 = {k: res["near_dup_weighted"][k]["p90"] for k in "bca"}
    f = lambda v: float(max(0.005, min(0.05, v)))
    res["recommended"] = {
        "sigma_max": res["corpus_weighted"]["noise_rel_savgol"]["p90"],
        "offset_max": f(p90["b"]), "slope_max": f(p90["c"]), "scale_max": f(p90["a"]),
        "shift_max": 1 if frac_shift >= 0.05 else 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["recommended"]))
    for g, r in res["groups"].items():
        print(g, r["n_noise_rows"], r["n_pairs"], r["noise_rel_savgol"], r["noise_rel_mad"],
              {k: r[k] for k in ("nd_offset_rel_range", "nd_slope_rel_range", "nd_scale_abs_dev", "nd_frac_nonzero_shift")})
    print(res["corpus_weighted"], res["near_dup_weighted"])


if __name__ == "__main__":
    main()
