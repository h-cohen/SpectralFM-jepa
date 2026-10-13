"""Masked MSE plus token and pooled-view regularization. SIGReg runs in fp32 over the batch axis."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import SIGReg


class LeJEPAObjective(nn.Module):
    def __init__(self, lambda_sigreg: float, num_slices: int = 1024, knots: int = 17, t_max: float = 3.0,
                 global_weight: float = 0.0, robust_weight: float = 0.0):
        super().__init__()
        self.robust_weight = robust_weight
        self.lambda_sigreg = lambda_sigreg
        self.global_weight = global_weight
        self.sigreg = SIGReg(knots=knots, t_max=t_max, num_vectors=num_slices)

    def forward(self, predicted, target_masked, target, views=None, robust=None) -> dict:
        with torch.autocast(device_type=target.device.type, enabled=False):
            mse = F.mse_loss(predicted.float(), target_masked.float())
            lam = self.lambda_sigreg
            reg = self.sigreg(target.float().transpose(0, 1))   # [T, B, D]
            loss = mse + lam * reg
            out = {"mse_loss": mse, "sigreg_loss": reg}
            if self.global_weight > 0 and views is not None:
                views = views.float()
                inv = ((views - views.mean(0, keepdim=True)) ** 2).mean()
                greg = self.sigreg(views)   # [V+1, B, D]
                global_loss = (1 - lam) * inv + lam * greg
                loss = loss + self.global_weight * global_loss
                out.update(global_inv_loss=inv, global_sigreg_loss=greg, global_loss=global_loss)
            if robust is not None:   # clean vs perturbed block-k tokens, scale-free via detached batch variance
                clean, perturbed = (h.float() for h in robust)
                rob = (perturbed - clean).pow(2).mean() / clean.detach().var(0).mean().clamp_min(1e-8)
                loss = loss + self.robust_weight * rob
                out["robust_loss"] = rob
        return {"loss": loss, **out}
