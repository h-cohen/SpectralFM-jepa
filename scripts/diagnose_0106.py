"""Label-free diagnosis of why trained block2-flat features lose to raw on dataset0106 (roadmap A5).

No labels are read. Compares 0106 with the other seven eval sets on: input statistics, corpus coverage,
nearest-neighbour novelty, token-variance structure, effective rank, CKA and RSA (trained vs random).
Run: OMP_NUM_THREADS=4 .venv/bin/python scripts/diagnose_0106.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spectral_lejepa.data.loader import packed_train_rows, source_row_probs, _source_groups  # noqa: E402
from spectral_lejepa.models.tokenizer import patch_bounds  # noqa: E402

PACK = Path("/mnt5/home/hadar/nova/data/spectra_all")
STUDY = ROOT / "outputs/frozen-feature-study"
OUT = ROOT / "outputs/production-validation/diag0106"
SETS = ["labeled_data", "dataset0055", "dataset0106", "dataset0109", "dataset0112", "dataset0113", "dataset0114", "dataset0120"]
CKPTS = ["prior_s0", "prior_s1", "prior_s2", "baseline_s101"]
SRC_OF = {s: ("labeled_data" if s == "labeled_data" else f"labeled_regression/{s}") for s in SETS}
SW = {"labeled_regression": 0.5, "default": 0.5}
RNG = np.random.default_rng(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOUNDS = patch_bounds(245, 48)


def load_bank(ck, s, key=None, random=False):
    f = np.load(STUDY / ck / s / ("random_bank.npz" if random else "bank.npz"))
    return f if key is None else f[key]


def participation(ev):
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def erank(ev):
    p = np.clip(ev, 0, None); p = p / p.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def centered(X):
    return X - X.mean(0, keepdims=True)


def spectrum(X):
    Xc = centered(X)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s ** 2


def cka(X, Y):
    X, Y = centered(X), centered(Y)
    Kx, Ky = X @ X.T, Y @ Y.T
    return float((Kx * Ky).sum() / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()))


def rsa(X, Y):
    dx = torch.pdist(torch.tensor(X)).numpy(); dy = torch.pdist(torch.tensor(Y)).numpy()
    return float(spearmanr(dx, dy).statistic)


def patch_mean_recon(Z):
    out = np.empty_like(Z)
    for lo, hi in BOUNDS:
        out[:, lo:hi] = Z[:, lo:hi].mean(1, keepdims=True)
    return out


def interp_recon(Z):
    t = torch.tensor(Z)[:, None]
    return torch.nn.functional.interpolate(torch.nn.functional.interpolate(t, size=48, mode="linear", align_corners=False),
                                           size=245, mode="linear", align_corners=False)[:, 0].numpy()


def token_profile(X):
    """[n, 48*256] -> per-token variance over samples, summed over channels, [48]."""
    T = X.reshape(len(X), 48, 256)
    return T.var(0).sum(1)


def input_patch_var(Zc):
    return np.array([Zc[:, lo:hi].var(0).sum() for lo, hi in BOUNDS])


def nn_min(Q, R, Rn, mask_out=None, drop_self=False, chunk=256):
    """Chunked min Euclidean distance from each row of Q to rows of R (optionally excluding masked columns / exact self)."""
    out = []
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + Rn[None] - 2 * q @ R.T
        if mask_out is not None:
            d2[:, mask_out] = float("inf")
        if drop_self:
            d2[d2 < 1e-6] = float("inf")
        out.append(d2.clamp_min(0).min(1).values.sqrt().cpu())
    return torch.cat(out).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = json.loads((PACK / "stats.json").read_text())
    gm, gs = stats["mean"], stats["std"]
    counts = stats["counts"]; names = list(counts)
    ends = np.cumsum(list(counts.values())); starts = ends - np.array(list(counts.values()))
    off = {n: (int(a), int(b)) for n, a, b in zip(names, starts, ends)}
    arr = np.load(PACK / "spectra.npy", mmap_mode="r")
    train = packed_train_rows(PACK)
    probs = source_row_probs(PACK, train, SW)
    groups, mass = _source_groups(PACK, train, SW)
    res = {"global_stats": {"mean": gm, "std": gs}, "group_mass": mass, "n_train_rows": int(len(train))}

    # ---- corpus coverage ----
    src_mass = {}
    for s in SETS:
        a, b = off[SRC_OF[s]]
        sel = (train >= a) & (train < b)
        src_mass[s] = {"source": SRC_OF[s], "corpus_rows": int(sel.sum()), "mass_fraction": float(probs[sel].sum()),
                       "expected_samples_per_epoch": float(probs[sel].sum() * len(train)),
                       "per_row_prob_vs_uniform": float(probs[sel].mean() * len(train)) if sel.any() else 0.0}
    res["coverage"] = src_mass

    # reference samples: A uniform (percentiles), B drawn with the training 50/50 mass
    iA = np.sort(RNG.choice(train, 200_000, replace=False))
    iB = np.sort(RNG.choice(len(train), 200_000, replace=True, p=probs / probs.sum()))
    iB = train[iB]
    A = np.asarray(arr[iA], dtype=np.float32); B = np.asarray(arr[iB], dtype=np.float32)
    ZA, ZB = (A - gm) / gs, (B - gm) / gs
    srcB = np.searchsorted(ends, iB, side="right")
    res["corpus_sample"] = {}
    for nm, Z in (("uniform", ZA), ("mass_weighted", ZB)):
        res["corpus_sample"][nm] = {"z_pct_0.5_99.5": [float(np.percentile(Z, .5)), float(np.percentile(Z, 99.5))],
                                    "z_min_max": [float(Z.min()), float(Z.max())], "std_per_spectrum_median": float(np.median(Z.std(1))),
                                    "raw_amp_range_median": float(np.median(A.max(1) - A.min(1))) if nm == "uniform" else None}
    lo_pos, hi_pos = np.percentile(ZB, 1, axis=0), np.percentile(ZB, 99, axis=0)
    lo_g, hi_g = np.percentile(ZB, .5), np.percentile(ZB, 99.5)
    hf_corpus = lambda Z: float((np.diff(Z, axis=1) ** 2).mean() / (Z.var(1).mean() + 1e-12))
    res["corpus_sample"]["hf_ratio_B"] = hf_corpus(ZB)

    # ---- eval sets: input stats, match to corpus, NN novelty ----
    ZBt = torch.tensor(ZB, device=DEV)
    ZBn = (ZBt ** 2).sum(1)
    res["sets"] = {}
    raw_all, z_all = {}, {}
    for s in SETS:
        d = load_bank("prior_s2", s)
        raw = d["input_raw"][:, 0].astype(np.float32)
        Z = (raw - gm) / gs
        raw_all[s], z_all[s] = raw, Z
        a, b = off[SRC_OF[s]]
        block = np.asarray(arr[a:b], dtype=np.float32)
        # exact-row membership of eval spectra in their packed source (and anywhere in packed source)
        hs = {r.tobytes(): i for i, r in enumerate(block)}
        hit = [hs.get(r.tobytes()) for r in raw]
        n_exact = sum(h is not None for h in hit)
        # tolerance match fallback
        if n_exact < len(raw):
            mn = np.concatenate([torch.cdist(torch.tensor(raw[i:i+256]), torch.tensor(block)).min(1).values.numpy() for i in range(0, len(raw), 256)]); n_tol = int((mn < 1e-5).sum())
        else:
            n_tol = n_exact
        valid_rows = np.load(PACK / "valid_rows.npy")
        hit_idx = np.array([a + h for h in hit if h is not None])
        n_valid = int(np.isin(hit_idx, valid_rows).sum()) if len(hit_idx) else 0
        # NN to mass-weighted sample, own source removed, and to own source block (self excluded)
        Zt = torch.tensor(Z, device=DEV)
        own = torch.tensor(srcB == names.index(SRC_OF[s]), device=DEV)
        nn_other = nn_min(Zt, ZBt, ZBn, mask_out=own)
        nn_B = nn_min(Zt, ZBt, ZBn)
        blk = torch.tensor((block - gm) / gs, device=DEV)
        nn_own = nn_min(Zt, blk, (blk ** 2).sum(1), drop_self=True)
        # input stats
        zc = Z - Z.mean(0)
        hf = float((np.diff(Z, axis=1) ** 2).mean() / Z.var(1).mean())
        F = np.abs(np.fft.rfft(Z - Z.mean(1, keepdims=True), axis=1)) ** 2
        hf_frac = float(F[:, 61:].sum() / F.sum())  # above half of Nyquist
        outside_g = float(((Z < lo_g) | (Z > hi_g)).mean())
        outside_pos = float(((Z < lo_pos) | (Z > hi_pos)).mean())
        anyout = float((((Z < lo_pos) | (Z > hi_pos)).any(1)).mean())
        # peak width (FWHM of the tallest peak above spectrum minimum, samples)
        w = []
        for r in raw:
            base = r.min(); h = (r.max() - base) / 2 + base; k = int(r.argmax())
            l = k
            while l > 0 and r[l] > h: l -= 1
            u = k
            while u < 244 and r[u] > h: u += 1
            w.append(u - l)
        # resampling losses on within-set (between-sample) variation and on total shape
        pm, ip = patch_mean_recon(Z), interp_recon(Z)
        loss = lambda R: float(((Z - R) ** 2).sum() / ((Z - Z.mean(1, keepdims=True)) ** 2).sum())
        lossc = lambda R: float((((Z - Z.mean(0)) - (R - R.mean(0))) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum())
        # spectrum-level shape diversity: how much of within-set variance is amplitude (mean level) vs shape
        lvl = Z.mean(1)
        res["sets"][s] = {
            "n": len(raw), "raw_min": float(raw.min()), "raw_max": float(raw.max()), "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
            "raw_amp_range_median": float(np.median(raw.max(1) - raw.min(1))),
            "z_mean": float(Z.mean()), "z_std": float(Z.std()), "z_min": float(Z.min()), "z_max": float(Z.max()),
            "frac_z_outside_corpus_0.5_99.5": outside_g, "frac_values_outside_per_position_1_99": outside_pos,
            "frac_spectra_any_position_outside_1_99": anyout,
            "hf_ratio_d1sq_over_var": hf, "hf_power_frac_above_half_nyquist": hf_frac,
            "fwhm_samples_median": float(np.median(w)), "fwhm_samples_p10_p90": [float(np.percentile(w, 10)), float(np.percentile(w, 90))],
            "resample_loss_patchmean_total": loss(pm), "resample_loss_interp48_total": loss(ip),
            "resample_loss_patchmean_withinset": lossc(pm), "resample_loss_interp48_withinset": lossc(ip),
            "within_set_var_frac_from_level(mean of spectrum)": float(lvl.var() * 1 / (zc ** 2).sum(1).mean() * 1) if False else float(lvl.var() * 245 / (zc ** 2).sum(1).mean()),
            "exact_rows_in_packed_source": n_exact, "tol_match_rows_in_packed_source": n_tol, "matched_rows_in_valid_split": n_valid,
            "nn_dist_to_mass_sample_median": float(np.median(nn_B)), "nn_dist_other_sources_median": float(np.median(nn_other)),
            "nn_dist_own_source_siblings_median": float(np.median(nn_own)),
            "nn_other_over_corpus_typical": None,
        }
        print(s, res["sets"][s]["exact_rows_in_packed_source"], res["sets"][s]["nn_dist_to_mass_sample_median"], flush=True)
    # corpus-typical NN scale: NN of random corpus rows (other-source) as reference
    refZ = torch.tensor(ZB[:2000], device=DEV)
    res["corpus_sample"]["nn_dist_corpus_to_corpus_median"] = float(np.median(nn_min(refZ, ZBt, ZBn, drop_self=True)))
    # source composition of nearest neighbours of 0106 in whole mass sample
    Zt = torch.tensor(z_all["dataset0106"], device=DEV)
    d2 = (Zt ** 2).sum(1)[:, None] + ZBn[None] - 2 * Zt @ ZBt.T
    k = d2.topk(10, largest=False).indices.cpu().numpy()
    del d2
    comp = {}
    for nm in names:
        c = float((srcB[k] == names.index(nm)).mean())
        if c > 0: comp[nm] = c
    res["nn10_source_composition_0106"] = comp

    # ---- representation ----
    rep = {}
    for ck in CKPTS:
        rep[ck] = {}
        for s in SETS:
            sub = np.sort(np.random.default_rng(1).choice(len(z_all[s]), 256, replace=False)) if len(z_all[s]) > 256 else slice(None)  # labeled_data n=4716 -> 256 rows (bounds O(n^2) work)
            Z = z_all[s][sub]
            r = {}
            for mode in ("trained", "random"):
                for L in (0, 1, 2):
                    X = load_bank(ck, s, f"bank__layer{L}__flat", random=(mode == "random"))
                    X = X.reshape(len(X), -1)[sub].astype(np.float64)
                    ev = spectrum(X)
                    T = X.reshape(len(X), 48, 256)
                    tokmean = T.mean(1)  # what mean pooling keeps
                    resid_frac = float((T - tokmean[:, None]).var(0).sum() / T.var(0).sum() * 0 + 1 - (centered(tokmean) ** 2).sum() * 48 / (centered(X) ** 2).sum())
                    fv = X.var(0)
                    # leverage of samples in top-k PC space
                    U, S, _ = np.linalg.svd(centered(X), full_matrices=False)
                    kk = 5
                    lev = (U[:, :kk] ** 2).sum(1)
                    pc1 = U[:, 0] * S[0]
                    pr = token_profile(X)
                    r[f"{mode}_L{L}"] = {
                        "participation_ratio": participation(ev), "effective_rank_entropy": erank(ev),
                        "top1_var_frac": float(ev[0] / ev.sum()), "top5_var_frac": float(ev[:5].sum() / ev.sum()),
                        "feature_var_cv": float(fv.std() / fv.mean()), "frac_features_var_lt_1e-3_of_mean": float((fv < 1e-3 * fv.mean()).mean()),
                        "max_leverage_top5pc_over_uniform": float(lev.max() / (kk / len(X))),
                        "pc1_score_skew": float(((pc1 - pc1.mean()) ** 3).mean() / pc1.std() ** 3),
                        "token_mean_var_frac": float(1 - resid_frac) if False else float((centered(tokmean) ** 2).sum() * 48 / (centered(X) ** 2).sum()),
                        "token_profile_top5_share": float(np.sort(pr)[::-1][:5].sum() / pr.sum()),
                        "token_profile_participation": participation(pr),
                        "token_profile_corr_input": float(np.corrcoef(pr, input_patch_var(Z - Z.mean(0)))[0, 1]),
                        "cka_input_global_z": cka(Z.astype(np.float64), X), "rsa_spearman_input": rsa(Z.astype(np.float32), X.astype(np.float32)),
                        "token_profile": pr.tolist(),
                    }
                    # per-block mean-pooled equivalent (bank__layerL) CKA/RSA for comparison
                    Xm = load_bank(ck, s, f"bank__layer{L}", random=(mode == "random")).reshape(-1, 256)[sub].astype(np.float64)
                    r[f"{mode}_L{L}"]["meanpool_cka_input"] = cka(Z.astype(np.float64), Xm)
                    r[f"{mode}_L{L}"]["meanpool_rsa_input"] = rsa(Z.astype(np.float32), Xm.astype(np.float32))
            r["input"] = {"participation_ratio": participation(spectrum(Z.astype(np.float64))),
                          "effective_rank_entropy": erank(spectrum(Z.astype(np.float64))),
                          "top1_var_frac": float(spectrum(Z.astype(np.float64))[0] / spectrum(Z.astype(np.float64)).sum()),
                          "patch_var_profile": input_patch_var(Z - Z.mean(0)).tolist()}
            rep[ck][s] = r
        print("rep", ck, flush=True)
    res["representation"] = rep

    # trained-vs-random summary across seeds
    summ = {}
    for s in SETS:
        row = {}
        for key in ("participation_ratio", "effective_rank_entropy", "top1_var_frac", "cka_input_global_z", "rsa_spearman_input",
                    "token_mean_var_frac", "feature_var_cv", "max_leverage_top5pc_over_uniform", "token_profile_corr_input",
                    "token_profile_top5_share", "meanpool_cka_input", "meanpool_rsa_input", "pc1_score_skew"):
            for mode in ("trained", "random"):
                row[f"L2_{mode}_{key}"] = float(np.mean([rep[ck][s][f"{mode}_L2"][key] for ck in CKPTS]))
        row["L0_trained_cka"] = float(np.mean([rep[ck][s]["trained_L0"]["cka_input_global_z"] for ck in CKPTS]))
        row["L1_trained_cka"] = float(np.mean([rep[ck][s]["trained_L1"]["cka_input_global_z"] for ck in CKPTS]))
        row["L0_trained_rsa"] = float(np.mean([rep[ck][s]["trained_L0"]["rsa_spearman_input"] for ck in CKPTS]))
        row["L1_trained_rsa"] = float(np.mean([rep[ck][s]["trained_L1"]["rsa_spearman_input"] for ck in CKPTS]))
        summ[s] = row
    res["summary_mean_over_4_ckpts"] = summ
    # already-computed scores for context (no refit)
    ctx = {}
    for ck in CKPTS:
        d = json.load(open(STUDY / ck / "linear_summary.json"))["sets"]["dataset0106"]
        ctx[ck] = {"raw_r2": d["raw"]["r2_mean"], **{f: {"r2": d["families"][f]["r2_mean"], "random_r2": d["families"][f]["random_r2"], "d_raw": d["families"][f]["vs_raw"]["delta"]}
                                                       for f in ("layer0/flat", "layer1/flat", "layer2/flat", "layer2")}}
    res["already_computed_scores_0106"] = ctx
    json.dump(res, open(OUT / "stats.json", "w"), indent=1)

    # ---- figures ----
    cols = ["#d62728" if s == "dataset0106" else "#7f7f7f" for s in SETS]
    lab = [s.replace("dataset", "") for s in SETS]
    S_ = res["sets"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for a, (k, t) in zip(ax.ravel(), [("raw_amp_range_median", "median raw amplitude range"), ("frac_spectra_any_position_outside_1_99", "frac spectra with any position outside corpus p1-p99"),
                                      ("hf_power_frac_above_half_nyquist", "HF power frac"), ("fwhm_samples_median", "median FWHM (samples)"),
                                      ("resample_loss_patchmean_withinset", "patch-mean loss, within-set variance"), ("nn_dist_other_sources_median", "NN dist to other-source corpus sample")]):
        a.bar(lab, [S_[s][k] for s in SETS], color=cols); a.set_title(t, fontsize=9); a.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(OUT / "input_stats.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(48)
    ax[0].plot(x, np.array(rep["prior_s2"]["dataset0106"]["input"]["patch_var_profile"]) / np.sum(rep["prior_s2"]["dataset0106"]["input"]["patch_var_profile"]), label="input (within-set var per patch)")
    for mode, c in (("trained", "r"), ("random", "b")):
        p = np.mean([np.array(rep[ck]["dataset0106"][f"{mode}_L2"]["token_profile"]) / np.sum(rep[ck]["dataset0106"][f"{mode}_L2"]["token_profile"]) for ck in CKPTS], 0)
        ax[0].plot(x, p, c, label=f"block2 {mode}")
    ax[0].set_title("0106 per-token variance share"); ax[0].legend(fontsize=7); ax[0].set_xlabel("token")
    for a, key, t in ((ax[1], "cka_input_global_z", "L2 flat CKA with input"), (ax[2], "effective_rank_entropy", "L2 flat effective rank")):
        w = 0.4; xs = np.arange(len(SETS))
        a.bar(xs - w / 2, [summ[s][f"L2_trained_{key}"] for s in SETS], w, label="trained"); a.bar(xs + w / 2, [summ[s][f"L2_random_{key}"] for s in SETS], w, label="random")
        a.set_xticks(xs); a.set_xticklabels(lab, rotation=45); a.set_title(t); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "representation.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, key, t in ((ax[0], "rsa_spearman_input", "L2 flat RSA vs input"), (ax[1], "token_mean_var_frac", "L2 variance fraction in token-mean"), (ax[2], "max_leverage_top5pc_over_uniform", "L2 max sample leverage (x uniform)")):
        xs = np.arange(len(SETS)); w = .4
        a.bar(xs - w / 2, [summ[s][f"L2_trained_{key}"] for s in SETS], w, label="trained"); a.bar(xs + w / 2, [summ[s][f"L2_random_{key}"] for s in SETS], w, label="random")
        a.set_xticks(xs); a.set_xticklabels(lab, rotation=45); a.set_title(t); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "geometry.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in ("dataset0106", "dataset0109"):
        ax[0].plot(raw_all[s][:8].T, lw=.8)
        break
    ax[0].set_title("0106 raw spectra (8)"); ax[1].plot(raw_all["dataset0109"][:8].T, lw=.8); ax[1].set_title("0109 raw spectra (8)")
    fig.tight_layout(); fig.savefig(OUT / "example_spectra.png", dpi=110); plt.close(fig)


def principal_cos(U, V):
    """Cosines of principal angles between column spans of U and V (both [n, k], orthonormal)."""
    return np.linalg.svd(U.T @ V, compute_uv=False)


def loo_ridge_r2(X, T, lam_scale=1e-2):
    """Label-free: leave-one-out kernel-ridge R2 for predicting input PC scores T [n,k] from features X [n,p]."""
    Xc = centered(X); K = Xc @ Xc.T; n = len(K)
    lam = lam_scale * np.trace(K) / n
    H = K @ np.linalg.inv(K + lam * np.eye(n))
    Tc = centered(T)
    resid = (Tc - H @ Tc) / (1 - np.diag(H))[:, None]
    return 1 - (resid ** 2).sum(0) / (Tc ** 2).sum(0)


def extra():
    """Supplementary label-free checks: subspace overlap with the input PCs and token positions."""
    res = json.load(open(OUT / "stats.json"))
    gm, gs = res["global_stats"]["mean"], res["global_stats"]["std"]
    ex = {}
    for s in SETS:
        raw = load_bank("prior_s2", s)["input_raw"][:, 0].astype(np.float64)
        sub = np.sort(np.random.default_rng(1).choice(len(raw), 256, replace=False)) if len(raw) > 256 else slice(None)
        Z = ((raw - gm) / gs)[sub]
        Zc = centered(Z); Ui, Si, _ = np.linalg.svd(Zc, full_matrices=False)
        row = {"input_pc_var_frac_top10": (Si[:10] ** 2 / (Si ** 2).sum()).tolist(), "input_erank": erank(Si ** 2)}
        for mode in ("trained", "random"):
            cs3, cs10, r2s = [], [], []
            for ck in CKPTS:
                X = load_bank(ck, s, "bank__layer2__flat", random=(mode == "random")).reshape(len(raw), -1)[sub].astype(np.float64)
                Uf, Sf, _ = np.linalg.svd(centered(X), full_matrices=False)
                cs3.append(principal_cos(Ui[:, :3], Uf[:, :3]).mean()); cs10.append(principal_cos(Ui[:, :10], Uf[:, :10]).mean())
                T = Ui[:, :10] * Si[:10]
                r2s.append(loo_ridge_r2(X, T))
            r2s = np.mean(r2s, 0)
            row[mode] = {"mean_cos_top3": float(np.mean(cs3)), "mean_cos_top10": float(np.mean(cs10)),
                         "loo_r2_recover_input_pc": r2s.tolist(), "loo_r2_mean_pc1_10": float(r2s.mean())}
        ex[s] = row
    # token positions for 0106: top tokens by within-set variance share
    Z = (load_bank("prior_s2", "dataset0106")["input_raw"][:, 0].astype(np.float64) - gm) / gs
    ip = input_patch_var(Z - Z.mean(0)); ip /= ip.sum()
    tr = np.mean([np.array(res["representation"][ck]["dataset0106"]["trained_L2"]["token_profile"]) / np.sum(res["representation"][ck]["dataset0106"]["trained_L2"]["token_profile"]) for ck in CKPTS], 0)
    rd = np.mean([np.array(res["representation"][ck]["dataset0106"]["random_L2"]["token_profile"]) / np.sum(res["representation"][ck]["dataset0106"]["random_L2"]["token_profile"]) for ck in CKPTS], 0)
    ex["tokens_0106"] = {"input_top8": np.argsort(ip)[::-1][:8].tolist(), "input_top8_share": float(np.sort(ip)[::-1][:8].sum()),
                         "block2_trained_top8": np.argsort(tr)[::-1][:8].tolist(), "block2_trained_top8_share": float(np.sort(tr)[::-1][:8].sum()),
                         "block2_random_top8": np.argsort(rd)[::-1][:8].tolist(), "block2_random_top8_share": float(np.sort(rd)[::-1][:8].sum()),
                         "trained_share_in_input_top8": float(tr[np.argsort(ip)[::-1][:8]].sum()),
                         "trained_tokens_below_1pct_of_mean": int((tr < 0.01 / 48 * 100 / 100 * 1).sum())}
    res["extra"] = ex
    json.dump(res, open(OUT / "stats.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for mode, c in (("trained", "r"), ("random", "b")):
        for s in SETS:
            ax[0].plot(ex[s][mode]["loo_r2_recover_input_pc"], c, alpha=1 if s == "dataset0106" else .2, lw=2 if s == "dataset0106" else .8)
    ax[0].set_title("LOO R2 recovering input PC1-10 from block2 flat\n(red trained, blue random; bold = 0106)"); ax[0].set_xlabel("input PC")
    ax[1].plot(ip, label="input"); ax[1].plot(tr, "r", label="block2 trained"); ax[1].plot(rd, "b", label="block2 random")
    ax[1].set_title("0106 within-set variance share per token"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "subspace_and_tokens.png", dpi=110)


if __name__ == "__main__":
    if "--extra" in sys.argv:
        extra()
    else:
        main()
