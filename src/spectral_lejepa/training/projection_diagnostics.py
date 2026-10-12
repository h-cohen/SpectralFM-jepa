"""Validation diagnostics for the auxiliary SIGReg projection space."""
from __future__ import annotations

import torch

from .gradient_terms import audit_batch
from spectral_lejepa.training import diagnostics as diag
from spectral_lejepa.utils import wandb as wb


def run_projection_diagnostics(run, model, objective, valid_signals, masking_cfg, global_cfg,
                               step, seed, device) -> dict:
    """Log projection-space statistics and block gradient terms without changing training state."""
    projector = getattr(model, "sigreg_projector", None)
    if projector is None:
        return {}

    device = torch.device(device)
    signals = torch.as_tensor(valid_signals).detach()
    if signals.ndim != 2 or len(signals) == 0:
        raise ValueError("valid_signals must be a non-empty [N, 245] tensor")

    # Use the same validation cadence/batch size as the existing gradient audit.
    signals = signals[:4 * 256]
    was_training = [(module, module.training) for module in model.modules()]
    rng_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                   if device.type == "cuda" else [])
    model.eval()
    try:
        # Keep dropout, masking helpers, and any audit implementation from advancing
        # the caller's generators. audit_batch uses its own deterministic fork too.
        with torch.random.fork_rng(devices=rng_devices):
            reps = diag.encode(model, signals, device, batch_size=256)
            base_tokens = reps.to(device=device, dtype=torch.float32)
            base_pooled = base_tokens.mean(dim=1)
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                projected_tokens = projector(base_tokens.float()).float().cpu()
                projected_pooled = projector(base_pooled.float()).float().cpu()

            token_stats, token_sv = diag.representation_stats(
                projected_tokens.reshape(-1, projected_tokens.shape[-1]))
            pooled_stats, pooled_sv = diag.representation_stats(projected_pooled)
            metrics = {
                **{f"projection/token/{key}": float(value) for key, value in token_stats.items()},
                **{f"projection/pooled/{key}": float(value) for key, value in pooled_stats.items()},
            }

            per_batch = []
            for start in range(0, len(signals), 256):
                batch = signals[start:start + 256]
                per_batch.append((len(batch), audit_batch(
                    model, objective, batch, masking_cfg, global_cfg,
                    seed=seed + len(per_batch), device=device,
                )))
            for block in ("block2", "block6"):
                names = per_batch[0][1]["blocks"][block]["gradient_norms"]
                for term in names:
                    metrics[f"gradient_terms/{block}/{term}"] = float(
                        sum(size * result["blocks"][block]["gradient_norms"][term]
                            for size, result in per_batch) / len(signals))
    finally:
        for module, training in was_training:
            module.training = training

    if run is not None:
        token_fig = diag.singular_value_figure(token_sv)
        token_fig.axes[0].set_title("SIGReg projection: token space")
        pooled_fig = diag.singular_value_figure(pooled_sv)
        pooled_fig.axes[0].set_title("SIGReg projection: pooled space")
        wb.log_figure(run, "projection/token_singular_values", token_fig, step)
        wb.log_figure(run, "projection/pooled_singular_values", pooled_fig, step)
    return metrics
