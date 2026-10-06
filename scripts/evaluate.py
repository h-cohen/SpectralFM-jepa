"""Evaluate a LeJEPA checkpoint with the clean-eval nested-CV label probe and pair it with the
data2vec baseline (ref_feb25) on identical rows and folds.

  uv run python -m scripts.evaluate --checkpoint outputs/<run>/checkpoint_last.pt

Readouts and the random-init control come from `evaluation.readouts` and `evaluation.random_control`; W&B gets metrics, a scorecard table and a Δ plot, and never artifacts.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from spectral_lejepa.config import load_config
from spectral_lejepa.evaluation.bank import READOUTS, extract_bank, save_bank
from spectral_lejepa.evaluation.data import load_labeled_data, normalize_like_fairseq
from spectral_lejepa.evaluation.nested import (ENSEMBLE_K, best_block, ladder_for_set, paired_delta,
                                               pair_with_baseline,
                                               run_canary, run_nested, write_json)
from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
from spectral_lejepa.training.checkpoint import load_model, sha256_file
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


def model_input(raw, ckpt_cfg):
    """The model's input spectra, normalized the way the checkpoint was trained."""
    if ckpt_cfg.get("data", {}).get("normalization", "sample_zscore") == "global":
        g = ckpt_cfg["derived"]["global_stats"]
        return ((raw - g["mean"]) / g["std"]).astype(np.float32)
    return normalize_like_fairseq(raw)


def summarize(nested, block):
    fam = nested["families"]
    return {"n": nested["protocol"]["n"],
            "embedding_r2": fam["embedding"]["r2_mean"], "embedding_sd": fam["embedding"]["bootstrap_sd"],
            "raw_r2": fam["raw"]["r2_mean"], "raw_sd": fam["raw"]["bootstrap_sd"],
            "top3_r2": fam[f"embedding_top{ENSEMBLE_K}"]["r2_mean"],
            "best_block": block, "best_block_r2": fam[block]["r2_mean"],
            "blocks": {k: v["r2_mean"] for k, v in fam.items() if k.startswith("layer")},
            "embedding_minus_raw": nested["embedding_minus_raw"]}


WIN_MARGIN = 0.05
# the baselines' raw-input OOFs come from the sklearn backend; the torch backend reproduces them
# up to the float32 rounding inside sklearn's LinearRegression (<= 6e-4 sd(y) seen on real sets),
# while different rows/folds move nearly every prediction by a large fraction of sd(y)
RAW_PAIRING_ATOL = 1e-8
TORCH_RAW_PAIRING_REL = 1e-2


def verdict(raw_r2, emb_r2, margin=WIN_MARGIN):
    """(delta, verdict): 'ceiling' when raw leaves less than `margin` of R² to gain."""
    delta = emb_r2 - raw_r2
    if 1 - raw_r2 < margin:
        return delta, "ceiling"
    return delta, "win" if delta >= margin else "no-win"


def scorecard(sets, key):
    """Counts over per-set summaries; key=None scores the model, key='random_control' the control."""
    rows = [s if key is None else s[key] for s in sets.values()]
    eligible = [r for r in rows if r["verdict"] != "ceiling"]
    return {"wins": sum(r["verdict"] == "win" for r in eligible), "eligible": len(eligible),
            "ceiling": len(rows) - len(eligible),
            "mean_delta": float(np.mean([r["delta_vs_raw"] for r in rows])) if rows else float("nan")}


def scorecard_figure(sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(sets)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(names), 3.5))
    ax.bar(x - 0.2, [sets[n]["delta_vs_raw"] for n in names], 0.4, label="model")
    if all("random_control" in sets[n] for n in names):
        ax.bar(x + 0.2, [sets[n]["random_control"]["delta_vs_raw"] for n in names], 0.4, label="random init")
    ax.axhline(WIN_MARGIN, color="k", ls="--", lw=1, label=f"win margin +{WIN_MARGIN}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("R² − raw R² (nested CV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="local .pt checkpoint path")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    ecfg, vcfg, bcfg = cfg["experiment"], cfg["evaluation"], cfg["baseline"]
    seed = ecfg["seed"]
    if bad := [r for r in vcfg["readouts"] if r not in READOUTS]:
        raise ValueError(f"unknown evaluation.readouts entry {bad[0]!r}; choose from {READOUTS}")
    fb = vcfg.get("flat_blocks")
    if fb is not None and not (isinstance(fb, list) and all(isinstance(b, int) and not isinstance(b, bool) and b >= 0 for b in fb)):
        raise ValueError(f"evaluation.flat_blocks must be null or a list of non-negative ints, got {fb!r}")
    backend = vcfg.get("backend", "sklearn")
    if backend not in ("sklearn", "torch"):
        raise ValueError(f"evaluation.backend must be 'sklearn' or 'torch', got {backend!r}")
    # sklearn: n_jobs worker processes. torch: probe_workers concurrent folds per probe device
    # (probe_device: one device or a list, folds dealt round-robin; ~2 GB per labeled_data-sized fold)
    workers = vcfg.get("probe_workers", 2)
    if not (isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1):
        raise ValueError(f"evaluation.probe_workers must be a positive int, got {workers!r}")
    probe = {"n_jobs": vcfg["n_jobs"] if backend == "sklearn" else workers,
             "backend": backend, "device": vcfg.get("probe_device", "cuda")}

    run = wb.init_run(cfg, job_type=cfg["wandb"]["job_type"])
    ckpt_path = a.checkpoint
    model, ckpt = load_model(ckpt_path)
    git = wb.git_info()
    method = ckpt["config"].get("data", {}).get("normalization", "sample_zscore")
    lineage = {"pretraining_run_id": ckpt.get("wandb_run_id"), "pretraining_checkpoint": a.checkpoint,
               "pretraining_checkpoint_sha256": sha256_file(ckpt_path),
               "pretraining_checkpoint_step": ckpt["step"], "pretraining_git_commit": ckpt.get("git_commit"),
               "pretraining_git_dirty": ckpt.get("git_dirty"), "model_input": method, "evaluation_git_commit": git["git_commit"],
               "evaluation_git_dirty": git["git_dirty"], "evaluation_backend": backend,
               "evaluation_probe_device": probe["device"] if backend == "torch" else None,
               "evaluation_config": cfg}
    if run is not None:
        run.config.update({"lineage": lineage})

    backbone = EvalBackbone(model)
    out_root = Path(ecfg["output_dir"]) / f"{Path(ckpt_path).parent.name}_step{ckpt['step']}"
    random_backbone = None
    if vcfg["random_control"]:
        torch.manual_seed(0)
        random_backbone = EvalBackbone(build_model(ckpt["config"]["model"], ckpt["config"]["data"]["sequence_length"]))
    sets, ladder_block = {}, None
    for name, dirs in cfg["label_sets"].items():
        raw, y = load_labeled_data(dirs, max_samples=vcfg["max_samples"], seed=seed)
        if len(y) < vcfg["min_n"]:
            print(f"[evaluate] skip {name}: n={len(y)} < {vcfg['min_n']}")
            continue
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        bank = extract_bank(backbone, model_input(raw, ckpt["config"]), device=vcfg["device"],
                            batch_size=vcfg["batch_size"], readouts=tuple(vcfg["readouts"]),
                            flat_blocks=vcfg.get("flat_blocks"))
        save_bank(out_dir / "bank.npz", bank, raw, y, {"checkpoint": a.checkpoint, "backbone": "EvalBackbone",
                                                       "set": name, "model_input": method, "n": int(len(y)), "seed": seed,
                                                       "stages": tuple(bank),
                                                       "readouts": tuple(vcfg["readouts"]),
                                                       "flat_blocks": vcfg.get("flat_blocks")})
        nested, oof = run_nested(bank, raw, y, seed=seed, **probe)
        write_json(out_dir / "nested_results.json", {**nested, "meta": lineage})
        np.savez_compressed(out_dir / "nested_oof.npz", y=y, **oof)
        block = best_block(nested)
        summary = summarize(nested, block)
        summary["delta_vs_raw"], summary["verdict"] = verdict(summary["raw_r2"], summary["embedding_r2"])
        summary["canary"] = run_canary(bank[block], raw, y, seed, backend=backend, device=probe["device"])
        if name in bcfg["run_dirs"]:
            summary["vs_ref_mean_only"] = pair_with_baseline(
                y, oof, os.path.join(bcfg["parent_repo"], bcfg["run_dirs"][name]), nested["protocol"],
                raw_atol=RAW_PAIRING_ATOL if backend == "sklearn" else TORCH_RAW_PAIRING_REL * float(np.std(y)))
        if name == vcfg["ladder_block_from"]:
            ladder_block = block
        if name in vcfg["ladder_sets"]:
            if ladder_block is None:
                raise ValueError("evaluation.ladder_block_from must come before the ladder sets in label_sets")
            ladder = ladder_for_set(bank, raw, y, ladder_block, seed=seed, **probe)
            write_json(out_dir / "nested_ladder.json", ladder)
            summary["ladder"] = {"block": ladder_block,
                                 **{f"{arm}_n{n}": v["median"] for arm, rungs in ladder["arms"].items()
                                    for n, v in rungs.items()}}
        if random_backbone is not None:
            rc_dir = out_dir / "random_control"
            rc_dir.mkdir(exist_ok=True)
            rc_bank = extract_bank(random_backbone, model_input(raw, ckpt["config"]), device=vcfg["device"],
                                   batch_size=vcfg["batch_size"], readouts=tuple(vcfg["readouts"]),
                                   flat_blocks=vcfg.get("flat_blocks"))
            rc_nested, rc_oof = run_nested(rc_bank, raw, y, seed=seed, **probe)
            if not np.allclose(rc_oof["raw"], oof["raw"], rtol=0, atol=1e-8):
                raise RuntimeError("random-control raw OOFs differ from the model's: folds/rows are not paired")
            summary["vs_random_control"] = paired_delta(np.asarray(y, dtype=np.float64), oof["embedding"],
                                                        rc_oof["embedding"])
            write_json(rc_dir / "nested_results.json", {**rc_nested, "meta": {**lineage, "model": "random_init_seed0"}})
            np.savez_compressed(rc_dir / "nested_oof.npz", y=y, **rc_oof)
            rc_r2 = rc_nested["families"]["embedding"]["r2_mean"]
            rc_delta, rc_verdict = verdict(rc_nested["families"]["raw"]["r2_mean"], rc_r2)
            summary["random_control"] = {"embedding_r2": rc_r2, "best_block": best_block(rc_nested),
                                         "delta_vs_raw": rc_delta, "verdict": rc_verdict}
        write_json(out_dir / "summary.json", summary)
        sets[name] = summary
        wb.log(run, flat_metrics(name, summary))
        print(f"[evaluate] {name}: embedding R2 {summary['embedding_r2']:.4f} vs raw {summary['raw_r2']:.4f}"
              + f", Δraw {summary['delta_vs_raw']:+.4f}±{summary['embedding_minus_raw']['sd']:.3f} [{summary['verdict']}]"
              + (f", vs {bcfg['tag']} (mean-only ref) {summary['vs_ref_mean_only']['embedding']['delta']:+.4f}"
                 if "vs_ref_mean_only" in summary else ""))

    card = {"model": scorecard(sets, None)}
    if random_backbone is not None:
        card["random_control"] = scorecard(sets, "random_control")
    write_json(out_root / "summary.json", {"lineage": lineage, "scorecard": card, "sets": sets})
    print("[evaluate] scorecard:", card)
    if run is not None:
        import wandb
        cols = ["set", "n", "raw_r2", "model_r2", "model_delta", "model_delta_sd", "model_p_better", "model_verdict",
                "random_r2", "random_delta", "random_verdict", "model_vs_random_delta", "model_vs_random_sd"]
        data = [[n, s["n"], s["raw_r2"], s["embedding_r2"], s["delta_vs_raw"],
                 s["embedding_minus_raw"]["sd"], s["embedding_minus_raw"]["p_a_better"], s["verdict"],
                 s.get("random_control", {}).get("embedding_r2"), s.get("random_control", {}).get("delta_vs_raw"),
                 s.get("random_control", {}).get("verdict"),
                 s.get("vs_random_control", {}).get("delta"), s.get("vs_random_control", {}).get("sd")] for n, s in sets.items()]
        run.log({"scorecard/table": wandb.Table(columns=cols, data=data)})
        wb.log(run, flat_metrics("scorecard", card))
        wb.log_figure(run, "scorecard/delta_vs_raw", scorecard_figure(sets))
    wb.finish(run)
    print("outputs:", out_root)


if __name__ == "__main__":
    main()
