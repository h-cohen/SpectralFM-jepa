"""Masked-latent pretraining with validation, diagnostics and reproducible checkpoints."""
from __future__ import annotations

import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR

from ..data.loader import (PackedSpectra, SourceWeightedSampler, SpectraDataset, make_loader,
                           manifest_fingerprint, packed_train_rows, read_manifest, source_mass,
                           source_row_probs, subsample, valid_subset)
from ..models.masking import make_mask, random_mask
from ..models.vit_1d import build_model
from ..utils import wandb as wb
from . import diagnostics as diag
from .checkpoint import load_checkpoint, save_checkpoint, sha256_file
from .loss import LeJEPAObjective

RECOMMENDED_MIN_BATCH = 256   # SIGReg's Epps-Pulley statistic is estimated from the batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("training.device is cuda but no GPU is visible; "
                           "set training.device=cpu explicitly for a CPU run")
    return torch.device(name)


def param_groups(model, weight_decay):
    """No weight decay on biases, norms, positional embeddings and the mask token."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or name.endswith("pos_embed") or name.endswith("mask_token"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def schedule_lengths(steps_per_epoch: int, tcfg: dict) -> tuple[int, int]:
    """(total_steps, warmup_steps). Warmup is warmup_epochs, capped at 10% of a short dev run."""
    total = tcfg["max_steps"] or tcfg["epochs"] * steps_per_epoch
    warmup = min(int(tcfg["warmup_epochs"] * steps_per_epoch), total // 10)
    return total, max(1, warmup)


def lr_factor(step, warmup_steps, total_steps, final_ratio):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def draw_views(global_cfg, batch_size, num_patches, generator, device):
    """Random patch subsets (one [B, k] index set per extra view) for the sample-level term; None when off."""
    if not global_cfg:
        return None
    keep, n = global_cfg["keep"], global_cfg["views"]
    return [random_mask(batch_size, num_patches, 1 - keep, generator, device)[1] for _ in range(n)]


@torch.no_grad()
def validation_losses(model, objective, signals, num_patches, masking_cfg, batch_size, seed, device, amp, global_cfg=None):
    """Mean losses over full batches of the fixed valid set; the mask seed is fixed so values
    are comparable across steps."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(batch_size, len(signals))
    totals = {"loss": 0.0, "mse_loss": 0.0, "sigreg_loss": 0.0}
    if global_cfg:
        totals.update(global_inv_loss=0.0, global_sigreg_loss=0.0, global_loss=0.0)
    if objective.regularizer == "visreg":
        totals.update(reg_center=0.0, reg_scale=0.0, reg_shape=0.0)
    n_batches = 0
    for i in range(0, len(signals) - batch_size + 1, batch_size):
        x = signals[i:i + batch_size].to(device)
        masked_idx, visible_idx = make_mask(masking_cfg, len(x), num_patches, generator, device)
        view_idxs = draw_views(global_cfg, len(x), num_patches, generator, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x, masked_idx, visible_idx, view_idxs)
        losses = objective(out["predicted"], out["target_masked"], out["target"], out.get("views"))
        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1
    model.train()
    return {f"valid/{k}": v / n_batches for k, v in totals.items()}


def run_diagnostics(run, model, valid_signals, masking_cfg, step, seed, device, reference_rank):
    """Log representation statistics and retain the initial rank as a collapse reference."""
    model.eval()
    reps = diag.encode(model, valid_signals, device)
    model.train()
    pooled, sv = diag.representation_stats(reps.mean(dim=1))
    token, _ = diag.representation_stats(reps.reshape(-1, reps.shape[-1]))
    metrics = {**{f"representation/{k}": v for k, v in pooled.items()},
               **{f"representation/token_{k}": v for k, v in token.items()}}
    if reference_rank is None:
        reference_rank = pooled["effective_rank"]
    alert = diag.collapse_alert(pooled, reference_rank)
    metrics["representation/collapse_alert"] = int(alert)
    if alert:
        text = (f"step {step}: std={pooled['std']:.3g}, effective_rank={pooled['effective_rank']:.3g} "
                f"(step-0 rank {reference_rank:.3g})")
        print(f"[diagnostics] WARNING possible collapse -- {text}", flush=True)
        wb.alert(run, "Possible representation collapse", text)
    wb.log(run, metrics, step=step)
    if run is not None:
        wb.log_figure(run, "representation/singular_values", diag.singular_value_figure(sv), step)
        masked_idx, _ = make_mask(masking_cfg, 4, len(model.tokenizer.bounds),
                                    torch.Generator().manual_seed(seed + 3 + step))
        wb.log_figure(run, "masking/examples", diag.masking_figure(
            valid_signals[:4].numpy(), model.tokenizer.bounds, masked_idx.numpy()), step)
    return metrics, reference_rank


def load_init_checkpoint(path, cfg) -> dict | None:
    """Load compatible model weights without restoring optimization state."""
    if not path:
        return None
    ckpt = load_checkpoint(path)
    old = ckpt["config"]
    if old["model"] != cfg["model"]:
        raise ValueError(f"training.init_checkpoint model config differs from this run: "
                         f"{old['model']} vs {cfg['model']}")
    if old["data"]["normalization"] != cfg["data"]["normalization"]:
        raise ValueError(f"training.init_checkpoint data.normalization is {old['data']['normalization']!r}, "
                         f"this run uses {cfg['data']['normalization']!r}")
    return {"ckpt": ckpt, "info": {"path": str(path), "sha256": sha256_file(path), "step": ckpt["step"]}}


def load_resume_checkpoint(path, cfg):
    if cfg["training"].get("resume") and cfg["training"].get("init_checkpoint"):
        raise ValueError("training.resume and init_checkpoint are mutually exclusive")
    if not path:
        return None
    result = load_init_checkpoint(path, cfg)
    old = result["ckpt"]["config"]
    ignored = {"resume", "init_checkpoint", "device", "log_every", "val_every", "diag_every", "ckpt_every"}
    for section in ("model", "data", "masking", "training"):
        for key in set(old[section]) | set(cfg[section]):
            if section == "training" and key in ignored:
                continue
            if section == "data" and key == "num_workers":
                continue
            if old[section].get(key) != cfg[section].get(key):
                raise ValueError(f"training.resume incompatible {section}.{key}")
    if old["experiment"]["seed"] != cfg["experiment"]["seed"]:
        raise ValueError("training.resume incompatible experiment.seed")
    for key in ("optimizer", "scaler", "step"):
        if key not in result["ckpt"]:
            raise ValueError(f"training.resume missing {key}")
    return result


def restore_optimization(ckpt, optimizer, scheduler, scaler):
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        scheduler.last_epoch = ckpt["step"]
        scheduler._step_count = ckpt["step"] + 1
        scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]


def train(cfg: dict) -> dict:
    ecfg, dcfg, mcfg, kcfg, tcfg = (cfg[k] for k in ("experiment", "data", "model", "masking", "training"))
    make_mask(kcfg, 1, mcfg["num_patches"])   # validates the masking config before any work
    if tcfg["optimizer"] != "adamw":
        raise ValueError(f"training.optimizer={tcfg['optimizer']!r}: only 'adamw' is implemented")
    global_cfg = None
    if tcfg["global_weight"] < 0:
        raise ValueError(f"training.global_weight={tcfg['global_weight']} must be >= 0")
    if tcfg["global_weight"] > 0:
        if tcfg["global_views"] < 1:
            raise ValueError(f"training.global_views={tcfg['global_views']} must be >= 1")
        if not 0 < tcfg["global_keep"] <= 1 or round(tcfg["global_keep"] * mcfg["num_patches"]) < 1:
            raise ValueError(f"training.global_keep={tcfg['global_keep']} must be in (0, 1] and keep at "
                             f"least one of {mcfg['num_patches']} patches")
        global_cfg = {"views": tcfg["global_views"], "keep": tcfg["global_keep"]}
    seed = ecfg["seed"]
    set_seed(seed)
    device = resolve_device(tcfg["device"])
    amp = bool(tcfg["amp"]) and device.type == "cuda"

    resume = load_resume_checkpoint(tcfg.get("resume"), cfg)
    init = resume or load_init_checkpoint(tcfg.get("init_checkpoint"), cfg)

    norm = dcfg["normalization"]
    weights, sampler = dcfg.get("source_weights"), None
    if weights and dcfg["source"] != "packed":
        raise ValueError("data.source_weights needs data.source=packed")
    if dcfg["source"] == "packed":
        packed_dir = dcfg["packed_dir"]
        stats = json.loads((Path(packed_dir) / "stats.json").read_text())
        global_stats = {"mean": stats["mean"], "std": stats["std"]}
        valid_rows = np.load(Path(packed_dir) / "valid_rows.npy")
        train_rows = subsample(packed_train_rows(packed_dir), dcfg["max_train_samples"], seed)
        train_ds = PackedSpectra(packed_dir, train_rows, norm, global_stats)
        valid_ds = PackedSpectra(packed_dir, valid_subset(valid_rows), norm, global_stats)
        data_derived = {"packed_dir": str(packed_dir), "global_stats": global_stats,
                        "n_rows": stats["n_rows"], "n_dropped": stats["n_dropped"]}
        if weights:
            sampler = SourceWeightedSampler(source_row_probs(packed_dir, train_rows, weights), seed)
            data_derived["source_mass"] = source_mass(packed_dir, train_rows, weights)
    elif dcfg["source"] == "manifests":
        if norm == "global":
            raise ValueError("global normalization needs data.source=packed")
        train_manifest = os.path.join(dcfg["manifest_dir"], "train.tsv")
        valid_manifest = os.path.join(dcfg["manifest_dir"], "valid.tsv")
        train_ds = SpectraDataset(subsample(read_manifest(train_manifest), dcfg["max_train_samples"], seed), norm)
        valid_ds = SpectraDataset(read_manifest(valid_manifest), norm)
        data_derived = {"train_manifest": manifest_fingerprint(train_manifest),
                        "valid_manifest": manifest_fingerprint(valid_manifest)}
    else:
        raise ValueError(f"data.source={dcfg['source']!r}: use 'manifests' or 'packed'")
    valid_signals = torch.stack([valid_ds[i] for i in range(len(valid_ds))])
    batch_size = tcfg["batch_size"]
    loader = make_loader(train_ds, batch_size, shuffle=True, seed=seed,
                         num_workers=dcfg["num_workers"], drop_last=True, sampler=sampler)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError(f"{len(train_ds)} training samples is fewer than one batch ({batch_size})")
    total_steps, warmup_steps = schedule_lengths(steps_per_epoch, tcfg)
    if resume:
        previous = resume["ckpt"]["config"].get("derived", {})
        if previous.get("total_steps", total_steps) != total_steps or previous.get("steps_per_epoch", steps_per_epoch) != steps_per_epoch:
            raise ValueError("training.resume incompatible schedule/data length")
        if not 0 <= resume["ckpt"]["step"] < total_steps:
            raise ValueError("training.resume step must precede total_steps")
    below = batch_size < RECOMMENDED_MIN_BATCH
    if below:
        warnings.warn(f"batch_size={batch_size} < {RECOMMENDED_MIN_BATCH}: SIGReg's batch statistic is "
                      "estimated from fewer samples (logged as derived.batch_size_below_256=true)")

    run_name = f"{ecfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(ecfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    derived = {"derived": {
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps,
        "train_samples": len(train_ds), "valid_samples": len(valid_ds),
        "batch_size_below_256": below, "output_dir": str(out_dir), **data_derived}}
    if init:
        derived["derived"]["resume" if resume else "init_checkpoint"] = init["info"]
    resolved = {**cfg, **derived}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    run = wb.init_run(cfg, job_type=cfg["wandb"]["job_type"], extra_config=derived, name=run_name)
    metadata = {**wb.git_info(), "wandb_run_id": run.id if run is not None else None}

    model = build_model(mcfg, dcfg["sequence_length"])
    if init:
        model.load_state_dict(init["ckpt"]["model"])
    model = model.to(device)
    objective = LeJEPAObjective(tcfg["lambda_sigreg"], tcfg["sigreg_num_slices"],
                                tcfg["sigreg_knots"], tcfg["sigreg_t_max"], tcfg["global_weight"],
                                tcfg["regularizer"], tcfg["visreg_slices"], tcfg["visreg_lambda"]).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, tcfg["weight_decay"]), lr=tcfg["learning_rate"])
    scheduler = LambdaLR(optimizer, lambda s: lr_factor(s, warmup_steps, total_steps, tcfg["final_lr_ratio"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    mask_generator = torch.Generator().manual_seed(seed + 1)
    if resume:
        restore_optimization(resume["ckpt"], optimizer, scheduler, scaler)
    num_patches = mcfg["num_patches"]
    max_norm = tcfg["grad_clip"] or float("inf")

    start_step = resume["ckpt"]["step"] if resume else 0
    latest, reference_rank = {}, None
    metrics, reference_rank = run_diagnostics(run, model, valid_signals, kcfg, start_step, seed, device, reference_rank)
    latest.update(metrics)

    def checkpoint(step, epoch):
        meta = {"config": resolved, "epoch": epoch, "global_step": step, "metrics": latest, **metadata}
        path = out_dir / f"checkpoint_step{step}.pt"
        for target in (path, out_dir / "checkpoint_last.pt"):
            save_checkpoint(target, model=model, optimizer=optimizer, scaler=scaler, step=step,
                            epoch=epoch, config=resolved, metadata=metadata, metrics=dict(latest), scheduler=scheduler,
                            rng_state={"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                                       "python": random.getstate(), "mask": mask_generator.get_state(),
                                       "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})
        if run is not None and cfg["wandb"].get("log_checkpoints", False):
            wb.log_checkpoint(run, path, f"lejepa-{run.id}", ["latest", f"step-{step}"], meta)

    step, samples_seen, history, epoch_pass = start_step, start_step * batch_size, [], start_step // steps_per_epoch
    if resume:
        rng = resume["ckpt"].get("rng_state")
        if rng:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            mask_generator.set_state(rng["mask"])
            if device.type == "cuda" and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            warnings.warn("Legacy resume checkpoint has no RNG state; stochastic sequence restarts")
        print(f"Resuming step {step}/{total_steps}, lr={optimizer.param_groups[0]['lr']:.8g}", flush=True)
    model.train()
    while step < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch_pass)
        epoch_pass += 1
        skip_batches = step % steps_per_epoch
        for batch_index, x in enumerate(loader):
            if batch_index < skip_batches:
                continue
            x = x.to(device, non_blocking=True)
            masked_idx, visible_idx = make_mask(kcfg, len(x), num_patches, mask_generator, device)
            view_idxs = draw_views(global_cfg, len(x), num_patches, mask_generator, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(x, masked_idx, visible_idx, view_idxs)
            losses = objective(out["predicted"], out["target_masked"], out["target"], out.get("views"))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            samples_seen += len(x)
            epoch = step / steps_per_epoch
            last = step == total_steps

            if step % tcfg["log_every"] == 0 or last:
                record = {f"train/{k}": v.item() for k, v in losses.items()}
                if not math.isfinite(record["train/loss"]):
                    raise FloatingPointError(f"non-finite loss at step {step}: {record}")
                record.update({"train/learning_rate": scheduler.get_last_lr()[0], "train/epoch": epoch,
                               "train/global_step": step, "train/samples_seen": samples_seen,
                               "train/grad_norm": float(grad_norm)})
                if resume:
                    print(f"step={step} loss={record['train/loss']:.6g} lr={record['train/learning_rate']:.8g}", flush=True)
                history.append(record)
                wb.log(run, record, step=step)
            if step % tcfg["val_every"] == 0 or last:
                valid = validation_losses(model, objective, valid_signals, num_patches, kcfg,
                                          batch_size, seed + 2, device, amp, global_cfg)
                latest.update(valid)
                wb.log(run, valid, step=step)
            if step % tcfg["diag_every"] == 0 or last:
                metrics, reference_rank = run_diagnostics(run, model, valid_signals, kcfg, step,
                                                          seed, device, reference_rank)
                latest.update(metrics)
            if step % tcfg["ckpt_every"] == 0 or last:
                checkpoint(step, epoch)
            if last:
                break

    wb.finish(run)
    return {"output_dir": str(out_dir), "final_checkpoint": str(out_dir / "checkpoint_last.pt"),
            "history": history, "wandb_run_id": metadata["wandb_run_id"]}
