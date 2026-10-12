"""Measure final-checkpoint loss-component gradients without updating parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from spectral_lejepa.data.loader import PackedSpectra, valid_subset
from spectral_lejepa.training.gradient_terms import audit_batch
from spectral_lejepa.training.checkpoint import load_model, sha256_file
from spectral_lejepa.training.loss import LeJEPAObjective

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/final-model/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/production-analysis/gradient_audit.json"
SEED = 4000

def run_audit(checkpoint, device="cuda", seed=SEED):
    checkpoint = Path(checkpoint)
    model, ckpt = load_model(checkpoint, map_location="cpu")
    cfg, train_cfg, data_cfg = ckpt["config"], ckpt["config"]["training"], ckpt["config"]["data"]
    if data_cfg["source"] != "packed" or data_cfg["normalization"] != "global":
        raise ValueError("audit requires the final packed-corpus/global-normalization recipe")
    packed_dir = Path(data_cfg["packed_dir"])
    stats = json.loads((packed_dir / "stats.json").read_text())
    valid_rows = np.load(packed_dir / "valid_rows.npy")
    row_ids = valid_subset(valid_rows, k=1024, seed=seed)
    if len(row_ids) != 1024:
        raise ValueError(f"expected 1024 validation rows, found {len(row_ids)}")
    dataset = PackedSpectra(packed_dir, row_ids, "global", {"mean": stats["mean"], "std": stats["std"]})
    batches = [torch.stack([dataset[i] for i in range(start, start + 256)]) for start in range(0, 1024, 256)]
    inputs = torch.cat(batches).numpy()
    model.to(device).eval()
    objective = LeJEPAObjective(*(train_cfg[k] for k in ("lambda_sigreg", "sigreg_num_slices", "sigreg_knots",
        "sigreg_t_max", "global_weight", "regularizer", "visreg_slices", "visreg_lambda"))).to(device).eval()
    masking_cfg = cfg["masking"]
    global_cfg = {"views": train_cfg["global_views"], "keep": train_cfg["global_keep"]}
    per_batch = [audit_batch(model, objective, x, masking_cfg, global_cfg, seed=seed + i, device=device)
                 for i, x in enumerate(batches)]
    means = {"terms": {k: float(np.mean([b["terms"][k] for b in per_batch])) for k in per_batch[0]["terms"]},
             "total_loss": float(np.mean([b["total_loss"] for b in per_batch])), "blocks": {}}
    for block in per_batch[0]["blocks"]:
        names = per_batch[0]["blocks"][block]["gradient_norms"]
        means["blocks"][block] = {"gradient_norms": {n: float(np.mean([b["blocks"][block]["gradient_norms"][n]
            for b in per_batch])) for n in names},
            "gradient_norm_total": float(np.mean([b["blocks"][block]["gradient_norm_total"] for b in per_batch])),
            "cosine_prediction_vs_other": float(np.mean([b["blocks"][block]["cosine_prediction_vs_other"] for b in per_batch]))}
    row_hash = hashlib.sha256(np.asarray(row_ids, dtype=np.int64).tobytes()).hexdigest()
    input_hash = hashlib.sha256(np.ascontiguousarray(inputs).tobytes()).hexdigest()
    return {
        "source": {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                   "checkpoint_step": int(ckpt["step"]), "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()},
        "config": {"lambda_sigreg": train_cfg["lambda_sigreg"], "global_weight": train_cfg["global_weight"],
                   "regularizer": train_cfg["regularizer"], "sigreg_num_slices": train_cfg["sigreg_num_slices"],
                   "sigreg_knots": train_cfg["sigreg_knots"], "sigreg_t_max": train_cfg["sigreg_t_max"],
                   "global_views": train_cfg["global_views"], "global_keep": train_cfg["global_keep"],
                   "masking": masking_cfg},
        "data": {"packed_dir": str(packed_dir), "normalization": "global", "mean": stats["mean"],
                 "std": stats["std"], "subset_seed": seed, "rows": row_ids.tolist(),
                 "rows_sha256": row_hash, "normalized_inputs_sha256": input_hash,
                 "batch_size": 256, "n_batches": 4},
        "term_weights": {"mse": 1.0, "token_sigreg": train_cfg["lambda_sigreg"],
                         "global_invariance": train_cfg["global_weight"] * (1 - train_cfg["lambda_sigreg"]),
                         "global_sigreg": train_cfg["global_weight"] * train_cfg["lambda_sigreg"]},
        "seed": seed, "per_batch": per_batch, "means": means,
    }
def write_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)
    return path
def _publish_wandb(result):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("--wandb requires WANDB_API_KEY in the shell environment")
    import wandb

    with wandb.init(project="spectralfm-lejepa", group="production-analysis", job_type="frozen-gradient-audit",
                    config={"source": result["source"], "config": result["config"],
                            "data": {k: v for k, v in result["data"].items() if k != "rows"}}) as run:
        for i, batch in enumerate(result["per_batch"]):
            metrics = {f"batch{i}/{k}": v for k, v in batch["terms"].items()}
            for block, vals in batch["blocks"].items():
                metrics.update({f"batch{i}/{block}/grad_{k}": v for k, v in vals["gradient_norms"].items()})
                metrics[f"batch{i}/{block}/cosine_pred_other"] = vals["cosine_prediction_vs_other"]
            run.log(metrics)
        run.summary.update(result["means"])
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = ROOT / json.loads(MANIFEST.read_text())["checkpoint"]
    result = run_audit(checkpoint, args.device)
    result["result_path"] = str(args.output)
    write_result(result, args.output)
    if args.wandb:
        _publish_wandb(result)
    print(json.dumps({"result_path": str(args.output), "source": result["source"], "means": result["means"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
