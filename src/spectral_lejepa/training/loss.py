"""LeJEPA objective for masked latent prediction:

    loss = MSE(predicted, target_masked) + lambda_sigreg * SIGReg(target)

MSE compares predictions only with the target embeddings at the SAME masked positions.
SIGReg (lightly's implementation of the LeJEPA regularizer) tests whether embeddings look
like an isotropic Gaussian. It is applied per token position: target [B, 24, D] is
transposed to [24, B, D], so each position is tested across the B independent samples of
the batch (N = B), and the 24 statistics are averaged. SIGReg sums cos/sin over the batch
and scales by N, so it runs in fp32 with autocast off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import SIGReg


class LeJEPAObjective(nn.Module):
    def __init__(self, lambda_sigreg: float, num_slices: int = 1024, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.sigreg = SIGReg(knots=knots, t_max=t_max, num_vectors=num_slices)

    def forward(self, predicted, target_masked, target) -> dict:
        with torch.autocast(device_type=target.device.type, enabled=False):
            mse = F.mse_loss(predicted.float(), target_masked.float())
            sigreg = self.sigreg(target.float().transpose(0, 1))   # [24, B, D]
        return {"loss": mse + self.lambda_sigreg * sigreg, "mse_loss": mse, "sigreg_loss": sigreg}
