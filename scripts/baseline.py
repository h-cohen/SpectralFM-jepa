"""Record the data2vec baseline (ref_feb25) in W&B by re-scoring the parent's saved banks with
OUR copied evaluation code, and check that it reproduces the parent's own numbers.

  uv run python -m scripts.baseline [--config configs/eval.yaml]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from spectral_lejepa.config import load_config
from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.nested import (best_block, compare_ladders, compare_with_reference,
                                               ladder_for_set, run_nested, write_json)
from spectral_lejepa.utils import wandb as wb
from scripts.evaluate import flat_metrics, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    bcfg, vcfg, seed = cfg["baseline"], cfg["evaluation"], cfg["experiment"]["seed"]
    cfg["wandb"]["tags"] = ["baseline", "data2vec", "spectralfm"]
    parent_git = wb.git_info(bcfg["parent_repo"])
    run = wb.init_run(cfg, job_type="baseline", name=f"baseline_{bcfg['tag']}",
                      extra_config={"baseline_parent_git": parent_git, "baseline_checkpoint": bcfg["checkpoint"]})

    out_root = Path("outputs") / "baseline" / bcfg["tag"]
    ladder_block, worst = None, 0.0
    for name, rel in bcfg["run_dirs"].items():
        src = os.path.join(bcfg["parent_repo"], rel)
        bank, raw, y, meta = load_bank(os.path.join(src, "bank.npz"))
        if len(y) < vcfg["min_n"]:
            continue
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        nested, oof = run_nested(bank, raw, y, seed=seed, n_jobs=vcfg["n_jobs"])
        check = compare_with_reference(nested, json.load(open(os.path.join(src, "nested_results.json"))))
        write_json(out_dir / "nested_results.json", {**nested, "meta": {**meta, "source": src}})
        np.savez_compressed(out_dir / "nested_oof.npz", y=y, **oof)
        block = best_block(nested)
        summary = {**summarize(nested, block), "reproduction": check}
        if name == vcfg["ladder_block_from"]:
            ladder_block = block
        if name in vcfg["ladder_sets"] and os.path.exists(os.path.join(src, "nested_ladder.json")):
            ladder = ladder_for_set(bank, raw, y, ladder_block, seed=seed, n_jobs=vcfg["n_jobs"])
            write_json(out_dir / "nested_ladder.json", ladder)
            summary["ladder_reproduction"] = compare_ladders(ladder, json.load(open(os.path.join(src, "nested_ladder.json"))))
        write_json(out_dir / "summary.json", summary)
        worst = max(worst, check["max_abs_diff"])
        wb.log(run, flat_metrics(name, summary))
        print(f"[baseline] {name}: embedding R2 {summary['embedding_r2']:.4f}, raw {summary['raw_r2']:.4f}, "
              f"max |diff| vs parent {check['max_abs_diff']:.2e}, choice mismatches {check['choice_mismatches']}")
    wb.log(run, {"reproduction/max_abs_diff": worst})
    wb.finish(run)


if __name__ == "__main__":
    main()
