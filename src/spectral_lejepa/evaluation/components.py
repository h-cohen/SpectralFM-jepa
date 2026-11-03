"""Do the other components of a sample help the parameter_0 probe?

Each measured sample has several component spectra that all carry the sample's label. Arms, per
feature set (raw input, random-weight control, JEPA block-2 flat):
- `comp{c}`: the nested-CV probe on component c alone (comp0 is the usual baseline);
- `pooled_*`: one probe trained on the rows of every component; its held-out predictions are scored
  on component c rows (`pooled_comp{c}`) and as the per-sample mean over components (`pooled_avg`,
  against the sample label).
Every arm uses the same outer folds over samples (KFold, shuffle, random_state=seed+r, as in
nested.nested_cv) and inner folds that keep a sample's rows together, so out-of-fold predictions are
paired per sample and paired_delta applies.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from .nested import _outer_fold, _single_threaded, bootstrap_sd, paired_delta
from .probe import RECIPES, r2


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def paired_rmse_delta(y, PA, PB, n_boot=1000, seed=0):
    """RMSE(A) - RMSE(B) (negative: A is better) with its bootstrap SD, both rescored on the same
    resampled samples. PA, PB: [R, n]."""
    def diff(i):
        return float(np.mean([rmse(y[i], a[i]) - rmse(y[i], b[i]) for a, b in zip(PA, PB)]))
    rng = np.random.default_rng(seed)
    boots = np.array([diff(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    return {"delta": diff(np.arange(len(y))), "sd": float(boots.std())}


def _fold(arms, pooled_arms, y, tr, te, n_inner, seed, backend, recipes, device=None):
    """One outer fold of every arm: {arm_key: held-out predictions}. Single-component arms are fit on
    sample rows tr; pooled arms on the rows of every component of tr, and predict the rows of te for
    every component (stacked component by component)."""
    n = len(y)
    out = {}
    with _single_threaded(backend):
        if arms:
            chosen, _ = _outer_fold(arms, {a: [a] for a in arms}, y, tr, te, n_inner, seed, (), {}, backend, device,
                                    recipes=recipes, groups=np.arange(n))
            out |= {a: c["pred"] for a, c in chosen.items()}
        if pooled_arms:
            n_comp = len(next(iter(pooled_arms.values()))) // n
            y_pool = np.tile(y, n_comp)
            shifts = np.arange(n_comp)[:, None] * n
            chosen, _ = _outer_fold(pooled_arms, {a: [a] for a in pooled_arms}, y_pool,
                                    (tr[None] + shifts).ravel(), (te[None] + shifts).ravel(), n_inner, seed,
                                    (), {}, backend, device, recipes=recipes,
                                    groups=np.tile(np.arange(n), n_comp))
            out |= {a: c["pred"] for a, c in chosen.items()}
    return out


def run_components(arms, y, n_repeats=2, n_folds=5, n_inner=5, seed=42, pooled=True, n_jobs=1,
                   backend="sklearn", device=None, recipes=RECIPES):
    """arms {name: {component: X [n, d]}}, y [n] (the sample labels, rows aligned across components).
    Returns {"scores": {name: {score: {r2_mean, rmse_mean, ..., bootstrap_sd}}},
             "oof": {name: {score: [R, n]}}, "deltas_vs_comp0": {name: {score: {"r2": ..., "rmse": ...}}},
             "test_folds", "protocol"}."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    comps = sorted(next(iter(arms.values())))
    single = {f"{a}|comp{c}": np.asarray(X[c], np.float32) for a, X in arms.items() for c in comps}
    pool = ({f"{a}|pool": np.concatenate([np.asarray(X[c], np.float32) for c in comps]) for a, X in arms.items()}
            if pooled and len(comps) > 1 else {})
    jobs = [(r, f, tr, te) for r in range(n_repeats)
            for f, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed + r).split(y))]
    args = [(single, pool, y, tr, te, n_inner, seed + 1000 + 10 * r + f, backend, tuple(recipes))
            for r, f, tr, te in jobs]
    if backend == "torch":
        from .probe_torch import map_on_streams
        outs = map_on_streams(_fold, args, n_jobs, device)
    elif n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_fold)(*a) for a in args)
    else:
        outs = [_fold(*a) for a in args]

    oof = {a: {} for a in arms}
    for a in arms:
        for c in comps:
            oof[a][f"comp{c}"] = np.zeros((n_repeats, n))
        if pool:
            for c in comps:
                oof[a][f"pooled_comp{c}"] = np.zeros((n_repeats, n))
            oof[a]["pooled_avg"] = np.zeros((n_repeats, n))
    test_folds = [[] for _ in range(n_repeats)]
    for (r, f, tr, te), out in zip(jobs, outs):
        test_folds[r].append(te.tolist())
        for a in arms:
            for c in comps:
                oof[a][f"comp{c}"][r, te] = out[f"{a}|comp{c}"]
            if pool:
                P = out[f"{a}|pool"].reshape(len(comps), len(te))      # row k = component comps[k] of te
                for k, c in enumerate(comps):
                    oof[a][f"pooled_comp{c}"][r, te] = P[k]
                oof[a]["pooled_avg"][r, te] = P.mean(axis=0)

    scores, deltas = {}, {}
    for a, by_score in oof.items():
        scores[a] = {s: {"r2_per_repeat": [r2(y, p) for p in P], "r2_mean": float(np.mean([r2(y, p) for p in P])),
                         "rmse_per_repeat": [rmse(y, p) for p in P], "rmse_mean": float(np.mean([rmse(y, p) for p in P])),
                         "bootstrap_sd": bootstrap_sd(y, P)} for s, P in by_score.items()}
        deltas[a] = {s: {"r2": paired_delta(y, P, by_score["comp0"]), "rmse": paired_rmse_delta(y, P, by_score["comp0"])}
                     for s, P in by_score.items() if s != "comp0"}
    return {"scores": scores, "oof": oof, "deltas_vs_comp0": deltas, "test_folds": test_folds,
            "protocol": {"n": n, "n_repeats": n_repeats, "n_folds": n_folds, "n_inner": n_inner, "seed": seed,
                         "components": comps, "pooled": bool(pool), "recipes": [f"{a}+{b}" for a, b in recipes]}}


def study_json(res, arm_names):
    """Per-arm R², RMSE (label units) and paired deltas vs comp0 (json-able; the out-of-fold arrays go to npz)."""
    return {"protocol": res["protocol"], "test_folds": res["test_folds"],
            "scores": {a: {s: {k: v for k, v in d.items()} for s, d in res["scores"][a].items()} for a in arm_names},
            "deltas_vs_comp0": {a: res["deltas_vs_comp0"][a] for a in arm_names}}


def oof_payload(res, y):
    """npz keys `<arm>__<score>` -> out-of-fold predictions [R, n] per sample, plus the sample labels `y`."""
    return {"y": np.asarray(y, np.float64),
            **{f"{a}__{s}": P for a, by in res["oof"].items() for s, P in by.items()}}


def run_study(model, random_model, ckpt_cfg, name, set_dirs, components, pooled, vcfg, seed, out_dir, probe, model_input):
    """One label set: block-2 flat features of the model and the random-weight control, plus the raw
    spectra, on every requested component; writes results_components.json and components_oof.npz."""
    from pathlib import Path
    from .bank import extract_bank
    from .data import load_labeled_components
    from .nested import write_json
    raw, y, sid, comp = load_labeled_components(set_dirs, components, max_samples=vcfg["max_samples"], seed=seed)
    n = int((comp == components[0]).sum())
    for c in components:
        if not np.array_equal(sid[comp == c], np.arange(n)):
            raise RuntimeError("component rows are not aligned by sample")
    y = y[:n]
    arms = {"raw": {c: raw[comp == c] for c in components}}
    for arm, backbone in (("jepa", model), ("random", random_model)):
        if backbone is None:
            continue
        bank = extract_bank(backbone, model_input(raw, ckpt_cfg), device=vcfg["device"], batch_size=vcfg["batch_size"],
                            readouts=("flat",), flat_blocks=[2])["layer2/flat"]
        arms[arm] = {c: bank[comp == c] for c in components}
    res = run_components(arms, y, seed=seed, pooled=pooled, **probe)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results_components.json", study_json(res, list(arms)))
    np.savez_compressed(out_dir / "components_oof.npz", **oof_payload(res, y))
    return res
