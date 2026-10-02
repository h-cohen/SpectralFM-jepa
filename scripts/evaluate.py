"""Evaluate a LeJEPA checkpoint with the clean-eval nested-CV label probe and pair it with the
data2vec baseline (ref_feb25) on identical rows and folds.

  uv run python -m scripts.evaluate --checkpoint outputs/<run>/checkpoint_last.pt
  uv run python -m scripts.evaluate --checkpoint wandb:<entity>/spectralfm-lejepa/lejepa-<run_id>:latest
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from spectral_lejepa.config import load_config
from spectral_lejepa.evaluation.bank import extract_mean_bank, save_bank
from spectral_lejepa.evaluation.data import load_labeled_data, normalize_like_fairseq
from spectral_lejepa.evaluation.nested import (ENSEMBLE_K, best_block, ladder_for_set, pair_with_baseline,
                                               run_canary, run_nested, write_json)
from spectral_lejepa.models.vit_1d import EvalBackbone
from spectral_lejepa.training.checkpoint import load_model
from spectral_lejepa.utils import wandb as wb


def flat_metrics(prefix, d):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flat_metrics(key, v))
        elif isinstance(v, (bool, int, float)):
            out[key] = float(v)
    return out


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def artifact_name(artifact):
    if artifact is None:
        return None
    return getattr(artifact, "qualified_name", None) or f"{artifact.entity}/{artifact.project}/{artifact.name}"


def resolve_checkpoint(ref, run):
    """Return (local .pt path, used W&B artifact or None)."""
    if not ref.startswith("wandb:"):
        return ref, None
    if run is None:
        raise RuntimeError("a wandb: checkpoint reference needs W&B enabled")
    artifact = run.use_artifact(ref[len("wandb:"):], type="model")
    files = list(Path(artifact.download()).glob("*.pt"))
    if len(files) != 1:
        raise RuntimeError(f"expected one .pt file in artifact {ref}, found {files}")
    return str(files[0]), artifact


def summarize(nested, block):
    fam = nested["families"]
    return {"n": nested["protocol"]["n"],
            "embedding_r2": fam["embedding"]["r2_mean"], "embedding_sd": fam["embedding"]["bootstrap_sd"],
            "raw_r2": fam["raw"]["r2_mean"], "raw_sd": fam["raw"]["bootstrap_sd"],
            "top3_r2": fam[f"embedding_top{ENSEMBLE_K}"]["r2_mean"],
            "best_block": block, "best_block_r2": fam[block]["r2_mean"],
            "blocks": {k: v["r2_mean"] for k, v in fam.items() if k.startswith("layer")},
            "embedding_minus_raw": nested["embedding_minus_raw"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="local .pt path or wandb:<artifact ref>")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    ecfg, vcfg, bcfg = cfg["experiment"], cfg["evaluation"], cfg["baseline"]
    seed = ecfg["seed"]

    run = wb.init_run(cfg, job_type=cfg["wandb"]["job_type"])
    ckpt_path, artifact = resolve_checkpoint(a.checkpoint, run)
    model, ckpt = load_model(ckpt_path)
    if artifact is None and run is not None and ckpt.get("wandb_run_id"):
        artifact = run.use_artifact(f"lejepa-{ckpt['wandb_run_id']}:step-{ckpt['step']}")   # W&B lineage edge
    lineage = {"pretraining_run_id": ckpt.get("wandb_run_id"), "pretraining_checkpoint": a.checkpoint,
               "pretraining_checkpoint_sha256": sha256_file(ckpt_path),
               "pretraining_artifact": artifact_name(artifact),
               "pretraining_artifact_digest": getattr(artifact, "digest", None),
               "pretraining_checkpoint_step": ckpt["step"], "pretraining_git_commit": ckpt.get("git_commit"),
               "pretraining_git_dirty": ckpt.get("git_dirty"), "evaluation_git_commit": wb.git_info()["git_commit"],
               "evaluation_config": cfg}
    if run is not None:
        run.config.update({"lineage": lineage})

    backbone = EvalBackbone(model)
    out_root = Path(ecfg["output_dir"]) / f"{Path(ckpt_path).parent.name}_step{ckpt['step']}"
    sets, ladder_block = {}, None
    for name, dirs in cfg["label_sets"].items():
        raw, y = load_labeled_data(dirs, max_samples=vcfg["max_samples"], seed=seed)
        if len(y) < vcfg["min_n"]:
            print(f"[evaluate] skip {name}: n={len(y)} < {vcfg['min_n']}")
            continue
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        bank = extract_mean_bank(backbone, normalize_like_fairseq(raw), device=vcfg["device"],
                                 batch_size=vcfg["batch_size"])
        save_bank(out_dir / "bank.npz", bank, raw, y, {"checkpoint": a.checkpoint, "backbone": "EvalBackbone",
                                                       "set": name, "n": int(len(y)), "seed": seed,
                                                       "stages": tuple(bank)})
        nested, oof = run_nested(bank, raw, y, seed=seed, n_jobs=vcfg["n_jobs"])
        write_json(out_dir / "nested_results.json", {**nested, "meta": lineage})
        np.savez_compressed(out_dir / "nested_oof.npz", y=y, **oof)
        block = best_block(nested)
        summary = summarize(nested, block)
        summary["canary"] = run_canary(bank[block], raw, y, seed)
        if name in bcfg["run_dirs"]:
            summary["vs_baseline"] = pair_with_baseline(y, oof, os.path.join(bcfg["parent_repo"], bcfg["run_dirs"][name]))
        if name == vcfg["ladder_block_from"]:
            ladder_block = block
        if name in vcfg["ladder_sets"]:
            if ladder_block is None:
                raise ValueError("evaluation.ladder_block_from must come before the ladder sets in label_sets")
            ladder = ladder_for_set(bank, raw, y, ladder_block, seed=seed, n_jobs=vcfg["n_jobs"])
            write_json(out_dir / "nested_ladder.json", ladder)
            summary["ladder"] = {"block": ladder_block,
                                 **{f"{arm}_n{n}": v["median"] for arm, rungs in ladder["arms"].items()
                                    for n, v in rungs.items()}}
        write_json(out_dir / "summary.json", summary)
        sets[name] = summary
        wb.log(run, flat_metrics(name, summary))
        print(f"[evaluate] {name}: embedding R2 {summary['embedding_r2']:.4f} vs raw {summary['raw_r2']:.4f}"
              + (f", vs {bcfg['tag']} {summary['vs_baseline']['embedding']['delta']:+.4f}"
                 if "vs_baseline" in summary else ""))

    write_json(out_root / "summary.json", {"lineage": lineage, "sets": sets})
    if run is not None:
        import wandb
        artifact = wandb.Artifact(f"eval-{run.id}", type="evaluation", metadata={"lineage": lineage})
        for path in out_root.rglob("*.json"):
            artifact.add_file(str(path), name=str(path.relative_to(out_root)))
        run.log_artifact(artifact)
    wb.finish(run)
    print("outputs:", out_root)


if __name__ == "__main__":
    main()
