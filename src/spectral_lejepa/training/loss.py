"""LeJEPA objective for masked latent prediction:

    loss = MSE(predicted, target_masked) + lambda_sigreg * SIGReg(target)

MSE compares predictions only with the target embeddings at the SAME masked positions.
SIGReg (lightly's implementation of the LeJEPA regularizer) tests whether embeddings look
like an isotropic Gaussian. It is applied per token position: target [B, 24, D] is
transposed to [24, B, D], so each position is tested across the B independent samples of
the batch (N = B), and the 24 statistics are averaged. SIGReg sums cos/sin over the batch
and scales by N, so it runs in fp32 with autocast off.

Optional sample-level term (global_weight > 0), given pooled embeddings views [V+1, B, D] of the
full spectrum and V random patch subsets: loss += global_weight * ((1-lambda) * mean((views -
views.mean(0))^2) + lambda * SIGReg(views)); SIGReg runs per view across the batch, then averages.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import SIGReg


class LeJEPAObjective(nn.Module):
    def __init__(self, lambda_sigreg: float, num_slices: int = 1024, knots: int = 17, t_max: float = 3.0,
                 global_weight: float = 0.0):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.global_weight = global_weight
        self.sigreg = SIGReg(knots=knots, t_max=t_max, num_vectors=num_slices)

    def forward(self, predicted, target_masked, target, views=None) -> dict:
        with torch.autocast(device_type=target.device.type, enabled=False):
            mse = F.mse_loss(predicted.float(), target_masked.float())
            sigreg = self.sigreg(target.float().transpose(0, 1))   # [24, B, D]
            loss = mse + self.lambda_sigreg * sigreg
            out = {"mse_loss": mse, "sigreg_loss": sigreg}
            if self.global_weight > 0 and views is not None:
                views = views.float()
                inv = ((views - views.mean(0, keepdim=True)) ** 2).mean()
                gsig = self.sigreg(views)                           # [V+1, B, D]
                global_loss = (1 - self.lambda_sigreg) * inv + self.lambda_sigreg * gsig
                loss = loss + self.global_weight * global_loss
                out.update(global_inv_loss=inv, global_sigreg_loss=gsig, global_loss=global_loss)
        return {"loss": loss, **out}
