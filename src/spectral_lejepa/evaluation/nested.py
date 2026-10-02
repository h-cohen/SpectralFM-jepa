"""Nested cross-validation label probe -- copied from clean-eval (6237feb) nested.py,
nested_ladder.py and canary.py, behaviour preserved.

Every arm (raw input, each encoder block) picks its normalizer+probe recipe by inner K-fold
CV INSIDE each outer training fold; the winner is refit on the whole training fold and only
then predicts the held-out fold. All arms share the outer folds (KFold, shuffle,
random_state=seed+r), so out-of-fold predictions of two arms -- or two backbones on the same
rows -- are paired, and paired_delta resamples the same spectra for both.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .probe import RECIPES, fit_predict, fold_transforms, r2

ENSEMBLE_K = 3      # the block-average readout: the 3 best blocks by inner CV, predictions averaged
MIN_N = 20          # below this, inner folds would fit on a handful of rows; such sets are skipped
LADDER_N = (10, 20, 50, 100, 200, 500, 1000, 2000)


def _inner_scores(X, y, n_inner, seed):
    """Inner-CV R² of every recipe, on these (training) rows only."""
    preds = {rc: np.zeros(len(y)) for rc in RECIPES}
    for itr, ite in KFold(n_inner, shuffle=True, random_state=seed).split(X):
        T = fold_transforms(X[itr], X[ite], seed=seed)
        for norm, probe in RECIPES:
            a, b = T[norm]
            preds[(norm, probe)][ite] = fit_predict(a, y[itr], b, probe, seed)
    return {rc: r2(y, p) for rc, p in preds.items()}


def _outer_fold(arms, families, y, tr, te, n_inner, seed, fixed_arms, ensembles):
    with threadpool_limits(limits=1):
        needed = sorted({a for names in families.values() for a in names}
                        | {a for names, _ in ensembles.values() for a in names})
        inner = {a: _inner_scores(arms[a][tr], y[tr], n_inner, seed) for a in needed}
        cache = {}

        def transforms(arm):
            if arm not in cache:
                cache[arm] = fold_transforms(arms[arm][tr], arms[arm][te], seed=seed)
            return cache[arm]

        chosen = {}
        for fam, names in families.items():
            arm, (norm, probe) = max(((a, rc) for a in names for rc in RECIPES),
                                     key=lambda k: inner[k[0]][k[1]])
            a, b = transforms(arm)[norm]
            chosen[fam] = {"pred": fit_predict(a, y[tr], b, probe, seed), "arm": arm,
                           "norm": norm, "probe": probe, "inner_r2": inner[arm][(norm, probe)]}
        for name, (names, k) in ensembles.items():
            best = {a: max(RECIPES, key=lambda rc: inner[a][rc]) for a in names}
            top = sorted(names, key=lambda a: -inner[a][best[a]])[:k]
            preds = []
            for a_ in top:
                norm, probe = best[a_]
                a, b = transforms(a_)[norm]
                preds.append(fit_predict(a, y[tr], b, probe, seed))
            chosen[name] = {"pred": np.mean(preds, axis=0),
                            "members": [{"arm": a_, "norm": best[a_][0], "probe": best[a_][1],
                                         "inner_r2": inner[a_][best[a_]]} for a_ in top]}
        fixed = {}
        for arm in fixed_arms:
            for norm, probe in RECIPES:
                a, b = transforms(arm)[norm]
                fixed[(arm, norm, probe)] = fit_predict(a, y[tr], b, probe, seed)
    return chosen, fixed


def bootstrap_sd(y, P, n_boot=500, seed=0):
    """SD of the repeat-averaged R² under resampling spectra. P: [R, n]."""
    rng = np.random.default_rng(seed)
    vals = [np.mean([r2(y[i], p[i]) for p in P])
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.std(vals))


def paired_delta(y, PA, PB, n_boot=1000, seed=0):
    """R²(A) - R²(B) on the same folds, with its bootstrap SD (both rescored on the SAME
    resampled spectra, so shared difficulty cancels). PA, PB: [R, n]."""
    def diff(i):
        return float(np.mean([r2(y[i], a[i]) - r2(y[i], b[i]) for a, b in zip(PA, PB)]))
    rng = np.random.default_rng(seed)
    boots = np.array([diff(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    return {"delta": diff(np.arange(len(y))), "sd": float(boots.std()),
            "p_a_better": float((boots > 0).mean())}


def nested_cv(arms, families, y, n_repeats=2, n_folds=5, n_inner=5, seed=42,
              fixed_arms=(), ensembles=None, n_jobs=1):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    ensembles = ensembles or {}
    jobs = []
    for r in range(n_repeats):
        for f, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed + r).split(y)):
            jobs.append((r, f, tr, te))
    args = [(arms, families, y, tr, te, n_inner, seed + 1000 + 10 * r + f, fixed_arms, ensembles)
            for r, f, tr, te in jobs]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(*a) for a in args)
    else:
        outs = [_outer_fold(*a) for a in args]

    res = {fam: {"oof": np.zeros((n_repeats, n)), "chosen": []} for fam in [*families, *ensembles]}
    fixed = {k: np.zeros((n_repeats, n)) for k in outs[0][1]}
    test_folds = [[] for _ in range(n_repeats)]
    for (r, f, tr, te), (chosen, fx) in zip(jobs, outs):
        test_folds[r].append(te.tolist())
        for fam, c in chosen.items():
            res[fam]["oof"][r, te] = c["pred"]
            res[fam]["chosen"].append({"repeat": r, "fold": f, **{k: v for k, v in c.items() if k != "pred"}})
        for k, p in fx.items():
            fixed[k][r, te] = p
    for v in res.values():
        v["r2_per_repeat"] = [r2(y, p) for p in v["oof"]]
        v["r2_mean"] = float(np.mean(v["r2_per_repeat"]))
        v["bootstrap_sd"] = bootstrap_sd(y, v["oof"])
    res["fixed"] = fixed
    res["test_folds"] = test_folds
    return res


def run_nested(bank, input_raw, y, n_repeats=2, n_folds=5, n_inner=5, seed=42, n_jobs=1):
    """Raw input vs the embedding (every block, mean-pooled), each at its nested-selected
    recipe; the top-3 block average; every block alone (depth profile / fixed-block readout);
    raw input at every fixed recipe. Returns (json-able results, {family: oof [R, n]})."""
    arms = {"raw": np.asarray(input_raw, dtype=np.float32), **bank}
    stages = list(bank)
    families = {"raw": ["raw"], "embedding": stages, **{s: [s] for s in stages}}
    ensembles = {f"embedding_top{ENSEMBLE_K}": (stages, ENSEMBLE_K)}
    res = nested_cv(arms, families, y, n_repeats=n_repeats, n_folds=n_folds, n_inner=n_inner,
                    seed=seed, fixed_arms=("raw",), ensembles=ensembles, n_jobs=n_jobs)
    families = {**families, **ensembles}
    y64 = np.asarray(y, dtype=np.float64)
    out = {
        "protocol": {"n": len(y), "n_repeats": n_repeats, "n_folds": n_folds, "n_inner": n_inner,
                     "seed": seed, "n_comp": 1, "recipes": [f"{a}+{b}" for a, b in RECIPES]},
        "families": {fam: {k: v for k, v in res[fam].items() if k != "oof"} for fam in families},
        "raw_fixed_recipes": {f"{norm}+{probe}": {"r2_mean": float(np.mean([r2(y64, p) for p in P])),
                                                  "bootstrap_sd": bootstrap_sd(y64, P, n_boot=200)}
                              for (_, norm, probe), P in res["fixed"].items()},
        "embedding_minus_raw": paired_delta(y64, res["embedding"]["oof"], res["raw"]["oof"]),
        "test_folds": res["test_folds"],
    }
    return out, {fam: res[fam]["oof"] for fam in families}


def best_block(results) -> str:
    """The block with the highest nested-CV score on its own."""
    fam = results["families"]
    blocks = {k: v for k, v in fam.items() if k != "raw" and not k.startswith("embedding")}
    return max(blocks, key=lambda k: blocks[k]["r2_mean"])


# --- label-budget ladder (nested_ladder.py) ---

def n_draws(n, n_full, max_draws=None):
    """More draws where one draw is noisy and cheap; one at the full fold."""
    if n >= n_full:
        return 1
    d = 20 if n <= 100 else 10 if n <= 500 else 5
    return min(d, max_draws) if max_draws else d


def _one_draw(arms, y, sub, te, n_inner, seed):
    out = {}
    with threadpool_limits(limits=1):
        for name, X in arms.items():
            inner = _inner_scores(X[sub], y[sub], min(n_inner, len(sub)), seed)
            norm, probe = max(RECIPES, key=lambda rc: inner[rc])
            a, b = fold_transforms(X[sub], X[te], seed=seed)[norm]
            out[name] = {"r2": r2(y[te], fit_predict(a, y[sub], b, probe, seed)), "recipe": f"{norm}+{probe}"}
    return out


def nested_ladder(arms, y, n_trains=LADDER_N, n_folds=5, n_inner=5, seed=42, n_jobs=1, max_draws=None):
    """Per outer fold and label budget n: random n-row subsets of the training fold; inner CV on
    those n rows picks the recipe; scored on the held-out fold. Arms share subsets (paired)."""
    y = np.asarray(y, dtype=np.float64)
    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(y))
    n_full = min(len(tr) for tr, _ in folds)
    rungs = [n for n in n_trains if n < n_full] + [n_full]
    jobs = []
    for f, (tr, te) in enumerate(folds):
        rng = np.random.default_rng(seed + 100 * f)
        for n in rungs:
            for d in range(n_draws(n, n_full, max_draws)):
                sub = tr if n >= n_full else rng.choice(tr, size=n, replace=False)
                jobs.append((n, (arms, y, np.sort(sub), te, n_inner, seed + 7 * d + 1000 * f)))
    if n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_one_draw)(*a) for _, a in jobs)
    else:
        outs = [_one_draw(*a) for _, a in jobs]

    names = list(arms)
    per = {a: collections.defaultdict(list) for a in names}
    rec = {a: collections.defaultdict(collections.Counter) for a in names}
    gap = {a: collections.defaultdict(list) for a in names[1:]}
    for (n, _), o in zip(jobs, outs):
        for a in names:
            per[a][n].append(o[a]["r2"])
            rec[a][n][o[a]["recipe"]] += 1
        for a in names[1:]:
            gap[a][n].append(o[a]["r2"] - o[names[0]]["r2"])

    def summ(v):
        return {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}
    return {
        "rungs": rungs,
        "draws": {str(n): n_folds * n_draws(n, n_full, max_draws) for n in rungs},
        "arms": {a: {str(n): {**summ(per[a][n]), "recipes": dict(rec[a][n].most_common())} for n in rungs}
                 for a in names},
        "gaps": {a: {str(n): {**summ(gap[a][n]), "frac_positive": float(np.mean(np.array(gap[a][n]) > 0))}
                     for n in rungs} for a in names[1:]},
    }


def ladder_for_set(bank, input_raw, y, block, seed=42, n_jobs=1):
    """Raw input vs ONE block fixed in advance (the fixed-block readout)."""
    arms = {"raw": np.asarray(input_raw, dtype=np.float32), "block": bank[block]}
    return {**nested_ladder(arms, y, seed=seed, n_jobs=n_jobs), "block": block}


# --- shuffled-label canary (canary.py) ---

def _kfold_oof(X, y, seed, n_folds=5):
    pred = np.zeros(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        a, b = fold_transforms(X[tr], X[te], seed=seed)["standardize"]
        pred[te] = fit_predict(a, y[tr], b, "ridgecv", seed)
    return pred


def shuffled_label_canary(X, y, seed=42, threshold=0.02):
    """Permute y and rerun the same probe: an honest pipeline scores R² ~ 0."""
    y = np.asarray(y, dtype=np.float64)
    y_shuffled = np.random.default_rng(seed).permutation(y)
    real = r2(y, _kfold_oof(X, y, seed))
    shuffled = r2(y_shuffled, _kfold_oof(X, y_shuffled, seed))
    return {"real_r2": real, "shuffled_r2": shuffled, "passed": bool(shuffled <= threshold)}


def run_canary(X_block, input_raw, y, seed=42):
    return {"block": shuffled_label_canary(X_block, y, seed), "raw": shuffled_label_canary(input_raw, y, seed)}


# --- comparisons and pairing ---

def compare_with_reference(ours, ref):
    """Largest absolute numeric difference and number of differing recipe choices between two
    nested results (ours vs the parent's nested_results.json)."""
    if list(ours["families"]) != list(ref["families"]):
        raise ValueError(f"family lists differ: {list(ours['families'])} vs {list(ref['families'])}")
    diffs, mismatches = [], 0
    for fam, o in ours["families"].items():
        r = ref["families"][fam]
        diffs += [abs(o["r2_mean"] - r["r2_mean"]), abs(o["bootstrap_sd"] - r["bootstrap_sd"])]
        diffs += [abs(a - b) for a, b in zip(o["r2_per_repeat"], r["r2_per_repeat"])]
        for co, cr in zip(o["chosen"], r["chosen"]):
            pairs = list(zip(co["members"], cr["members"])) if "members" in co else [(co, cr)]
            mismatches += "members" in co and len(co["members"]) != len(cr["members"])
            for mo, mr in pairs:
                mismatches += (mo["arm"], mo["norm"], mo["probe"]) != (mr["arm"], mr["norm"], mr["probe"])
                diffs.append(abs(mo["inner_r2"] - mr["inner_r2"]))
    for k, o in ours["raw_fixed_recipes"].items():
        r = ref["raw_fixed_recipes"][k]
        diffs += [abs(o["r2_mean"] - r["r2_mean"]), abs(o["bootstrap_sd"] - r["bootstrap_sd"])]
    for k in ("delta", "sd", "p_a_better"):
        diffs.append(abs(ours["embedding_minus_raw"][k] - ref["embedding_minus_raw"][k]))
    return {"max_abs_diff": float(max(diffs)), "choice_mismatches": int(mismatches)}


def compare_ladders(ours, ref):
    if ours["rungs"] != ref["rungs"] or ours["block"] != ref["block"]:
        raise ValueError(f"ladders differ in rungs/block: {ours['rungs']}/{ours['block']} vs "
                         f"{ref['rungs']}/{ref['block']}")
    diffs, mismatches = [], 0
    for arm, rungs in ours["arms"].items():
        for n, o in rungs.items():
            r = ref["arms"][arm][n]
            diffs += [abs(o[k] - r[k]) for k in ("median", "p25", "p75")]
            mismatches += o["recipes"] != r["recipes"]
    for arm, rungs in ours["gaps"].items():
        for n, o in rungs.items():
            diffs += [abs(o[k] - ref["gaps"][arm][n][k]) for k in ("median", "p25", "p75", "frac_positive")]
    return {"max_abs_diff": float(max(diffs)), "recipe_mismatches": int(mismatches)}


def pair_with_baseline(y, oof, baseline_dir):
    """Paired R² gaps (ours - baseline) on identical spectra and folds, read from the
    baseline's nested_oof.npz. Refuses if the rows differ: a gap on different rows is meaningless."""
    path = os.path.join(baseline_dir, "nested_oof.npz")
    base = np.load(path)
    if base["y"].shape != np.shape(y) or not np.array_equal(base["y"], y):
        raise ValueError(f"baseline rows in {path} differ from ours (n={len(base['y'])} vs {len(y)}); "
                         "refusing to pair")
    families = ("embedding", f"embedding_top{ENSEMBLE_K}", "raw")
    return {fam: paired_delta(np.asarray(y, dtype=np.float64), oof[fam], base[fam])
            for fam in families if fam in base.files and fam in oof}


def write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
