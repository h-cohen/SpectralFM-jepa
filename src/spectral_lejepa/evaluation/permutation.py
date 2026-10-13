"""Frozen-feature label-permutation audit for the seed-101 dataset0120 bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from spectral_lejepa.evaluation import nested
from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.probe import r2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "outputs/final-model/eval/production_p48_s101_20261010-024928_step497990/dataset0120/bank.npz"
DEFAULT_OUTPUT = ROOT / "outputs/production-analysis/permutation_audit.json"
CANARY_REFERENCE = 0.020445631812812892
CANARY_TOLERANCE = 1e-5


def summarize_null(real_score, null_scores, threshold):
    scores = np.asarray(null_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("null_scores must be a non-empty finite vector")
    at_or_above = int(np.sum(scores >= real_score))
    above_threshold = int(np.sum(scores > threshold))
    return {
        "real_r2": float(real_score),
        "null_mean": float(scores.mean()),
        "null_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "null_min": float(scores.min()),
        "null_q05": float(np.quantile(scores, 0.05)),
        "null_median": float(np.median(scores)),
        "null_q95": float(np.quantile(scores, 0.95)),
        "null_max": float(scores.max()),
        "count_at_or_above_real": at_or_above,
        "empirical_p_value": float((1 + at_or_above) / (len(scores) + 1)),
        "threshold": float(threshold),
        "count_above_threshold": above_threshold,
        "fraction_above_threshold": float(above_threshold / len(scores)),
        "permutation_count": int(len(scores)),
    }


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _hash_array(value):
    arr = np.ascontiguousarray(value)
    return _hash_bytes(arr.dtype.str.encode() + repr(arr.shape).encode() + arr.tobytes())


def _data_audit(raw, y):
    _, inverse, counts = np.unique(raw, axis=0, return_inverse=True, return_counts=True)
    duplicate_groups = np.flatnonzero(counts > 1)
    conflicting = sum(
        not np.all(y[inverse == group] == y[np.flatnonzero(inverse == group)[0]])
        for group in duplicate_groups
    )
    return {
        "rows": int(len(y)),
        "unique_raw_spectra": int(len(counts)),
        "duplicate_spectrum_groups": int(len(duplicate_groups)),
        "duplicate_spectrum_excess_rows": int(len(y) - len(counts)),
        "conflicting_duplicate_label_groups": int(conflicting),
    }


def _folds(n, n_folds, seed):
    splits = []
    seen = []
    for index, (train, test) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(n))):
        if np.intersect1d(train, test).size:
            raise AssertionError(f"fold {index} has overlapping train and test rows")
        splits.append({"fold": index, "train_indices": train.tolist(), "test_indices": test.tolist()})
        seen.extend(test.tolist())
    if sorted(seen) != list(range(n)):
        raise AssertionError("outer test folds do not cover each row exactly once")
    return splits


def _score(X, y, seed, n_folds, backend, device):
    pred = nested._kfold_oof(X, y, seed, n_folds=n_folds, backend=backend, device=device)
    return r2(y, pred)


def permutation_audit(block, raw, y, *, permutation_seeds=range(1000, 1100),
                      n_folds=5, fold_seed=42, threshold=0.02,
                      canary_reference=None, canary_tolerance=CANARY_TOLERANCE,
                      backend="sklearn", device=None, source_hashes=None):
    block = np.asarray(block, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    if block.ndim != 2 or raw.ndim != 2 or y.ndim != 1 or len(block) != len(raw) or len(raw) != len(y):
        raise ValueError("block, raw, and y must be aligned 2D, 2D, and 1D arrays")
    if not len(y) or not all(np.isfinite(a).all() for a in (block, raw, y)):
        raise ValueError("audit arrays must be non-empty and finite")
    seeds = [int(s) for s in permutation_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("permutation_seeds must be a non-empty sequence of unique seeds")

    arms = {"block": block, "raw": raw}
    real = {name: _score(X, y, fold_seed, n_folds, backend, device) for name, X in arms.items()}
    canary_y = np.random.default_rng(fold_seed).permutation(y)
    canary = {name: _score(X, canary_y, fold_seed, n_folds, backend, device) for name, X in arms.items()}
    if canary_reference is not None and abs(canary["block"] - canary_reference) > canary_tolerance:
        raise ValueError(f"shuffle-{fold_seed} block canary {canary['block']:.10f} differs from expected "
                         f"{canary_reference:.10f} by more than {canary_tolerance:g}")

    null = {name: [] for name in arms}
    for seed in seeds:
        permuted_y = np.random.default_rng(seed).permutation(y)
        for name, X in arms.items():
            null[name].append(_score(X, permuted_y, fold_seed, n_folds, backend, device))

    return {
        "method": {
            "description": "Frozen-feature single-probe label permutation audit; not nested-CV evaluation or independent deployment validation.",
            "normalizer": "standardize; fit within each outer training fold",
            "probe": "RidgeCV(alphas=logspace(-3,3,20))",
            "outer_folds": "KFold(shuffle=True, random_state=fold_seed)",
            "fold_seed": int(fold_seed), "n_folds": int(n_folds),
            "permutation": "numpy.default_rng(seed).permutation(y)",
            "permutation_seeds": seeds,
            "empirical_p": "(1 + count(null_r2 >= real_r2)) / (n_permutations + 1)",
            "threshold_fraction": "count(null_r2 > threshold) / n_permutations",
            "backend": backend,
        },
        "source_hashes": dict(source_hashes or {
            "block_array_sha256": _hash_array(block), "raw_array_sha256": _hash_array(raw),
            "labels_array_sha256": _hash_array(y),
        }),
        "data_audit": _data_audit(raw, y),
        "folds": _folds(len(y), n_folds, fold_seed),
        "canary": {"shuffle_seed": int(fold_seed), "reference_r2": canary_reference,
                   "tolerance": float(canary_tolerance), "arm_r2": canary,
                   "block_matches_reference": (canary_reference is None or
                                                abs(canary["block"] - canary_reference) <= canary_tolerance)},
        "arms": {name: {**summarize_null(real[name], null[name], threshold),
                        "null_scores": [float(s) for s in null[name]]}
                 for name in arms},
    }


def run_default(bank_path=DEFAULT_BANK, *, permutation_seeds=range(1000, 1100),
                backend="sklearn", device=None):
    bank_path = Path(bank_path)
    bank, raw, y, metadata = load_bank(bank_path, readouts=("flat",))
    if set(bank) != {"layer2/flat"}:
        raise ValueError(f"expected only layer2/flat in the selected bank, found {sorted(bank)}")
    if metadata.get("set") != "dataset0120":
        raise ValueError(f"expected dataset0120 bank, found {metadata.get('set')!r}")
    seed_match = re.search(r"production_p\d+_s(\d+)_", str(metadata.get("checkpoint", "")))
    if not seed_match or int(seed_match.group(1)) != 101:
        raise ValueError(f"expected seed-101 production checkpoint, found {metadata.get('checkpoint')!r}")
    hashes = {
        "bank_file_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "block_array_sha256": _hash_array(bank["layer2/flat"]),
        "raw_array_sha256": _hash_array(raw), "labels_array_sha256": _hash_array(y),
    }
    result = permutation_audit(bank["layer2/flat"], raw, y,
                               permutation_seeds=permutation_seeds, backend=backend,
                               device=device, source_hashes=hashes,
                               canary_reference=CANARY_REFERENCE)
    result["source"] = {"bank_path": str(bank_path), "dataset": metadata["set"],
                        "n": int(metadata["n"]), "training_seed": int(seed_match.group(1)),
                        "evaluation_seed": int(metadata["seed"]),
                        "checkpoint": metadata.get("checkpoint"), "feature": "layer2/flat"}
    return result


def write_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def _publish_wandb(result):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("--wandb requires WANDB_API_KEY in the shell environment")
    import wandb

    with wandb.init(project="spectralfm-lejepa", group="production-analysis",
                    job_type="frozen-feature-permutation-audit",
                    config={"source": result["source"], "method": result["method"],
                            "source_hashes": result["source_hashes"], "data_audit": result["data_audit"]}) as run:
        for arm, stats in result["arms"].items():
            run.summary[f"{arm}/real_r2"] = stats["real_r2"]
            run.summary[f"{arm}/empirical_p_value"] = stats["empirical_p_value"]
            run.summary[f"{arm}/fraction_above_threshold"] = stats["fraction_above_threshold"]
            run.summary[f"{arm}/null_mean"] = stats["null_mean"]
            run.summary[f"{arm}/null_sd"] = stats["null_sd"]
            run.log({f"{arm}/null_r2": wandb.Histogram(stats["null_scores"])})
        run.summary["canary/block_matches_reference"] = result["canary"]["block_matches_reference"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=("sklearn", "torch"), default="sklearn")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true", help="publish summary to W&B group production-analysis")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        result = run_default(args.bank, backend=args.backend, device=args.device)
    result["result_path"] = str(args.output)
    write_result(result, args.output)
    if args.wandb:
        _publish_wandb(result)
    output = {k: v for k, v in result.items() if k != "arms"}
    output["arms"] = {
        arm: {k: v for k, v in stats.items() if k != "null_scores"}
        for arm, stats in result["arms"].items()
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
