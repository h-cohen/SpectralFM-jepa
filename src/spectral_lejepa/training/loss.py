"""Masked MSE plus token and pooled-view regularization. SIGReg runs in fp32 over the batch axis."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import SIGReg

from spectral_lejepa.training.visreg import VISReg


class LeJEPAObjective(nn.Module):
    def __init__(self, lambda_sigreg: float, num_slices: int = 1024, knots: int = 17, t_max: float = 3.0,
                 global_weight: float = 0.0, regularizer: str = "sigreg", visreg_slices: int = 4096,
                 visreg_lambda: float = 0.6):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.global_weight = global_weight
        if regularizer not in ("sigreg", "visreg"):
            raise ValueError(f"regularizer={regularizer!r}: expected 'sigreg' or 'visreg'")
        self.regularizer = regularizer
        self.visreg_lambda = visreg_lambda
        if regularizer == "sigreg":
            self.sigreg = SIGReg(knots=knots, t_max=t_max, num_vectors=num_slices)
        else:
            self.visreg = VISReg(num_slices=visreg_slices)

    def forward(self, predicted, target_masked, target, views=None) -> dict:
        with torch.autocast(device_type=target.device.type, enabled=False):
            mse = F.mse_loss(predicted.float(), target_masked.float())
            if self.regularizer == "sigreg":
                lam = self.lambda_sigreg
                reg = self.sigreg(target.float().transpose(0, 1))   # [T, B, D]
                loss = mse + lam * reg
                out = {"mse_loss": mse, "sigreg_loss": reg}
            else:
                lam = self.visreg_lambda
                parts = self.visreg(target.float().transpose(0, 1))
                reg = parts["reg"]
                loss = (1 - lam) * mse + lam * reg
                out = {"mse_loss": mse, "sigreg_loss": reg, "reg_center": parts["center"],
                       "reg_scale": parts["scale"], "reg_shape": parts["shape"]}
            if self.global_weight > 0 and views is not None:
                views = views.float()
                inv = ((views - views.mean(0, keepdim=True)) ** 2).mean()
                greg = self.sigreg(views) if self.regularizer == "sigreg" else self.visreg(views)["reg"]  # [V+1, B, D]
                global_loss = (1 - lam) * inv + lam * greg
                loss = loss + self.global_weight * global_loss
                out.update(global_inv_loss=inv, global_sigreg_loss=greg, global_loss=global_loss)
        return {"loss": loss, **out}
