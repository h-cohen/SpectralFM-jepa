"""Previous-project cosine similarity and noise robustness on block-2 readouts (spec 2026-10-14)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

READOUTS = ('block2_mean', 'block2_flat')


@torch.no_grad()
def block2_readouts(model, model_inputs, device, batch_size=64):
    model.to(device).eval()
    flat, mean = [], []
    for i in range(0, len(model_inputs), batch_size):
        x = torch.as_tensor(model_inputs[i:i + batch_size], dtype=torch.float32, device=device)
        h2 = model.encoder.prefix(model.tokenizer(x), 2).float()           # [B, 48, D], before final norm
        flat.append(h2.reshape(len(x), -1).cpu().numpy())
        mean.append(h2.mean(1).cpu().numpy())
    return {'block2_mean': np.concatenate(mean), 'block2_flat': np.concatenate(flat)}


class _Precomputed(torch.nn.Module):
    """Stands in for a backbone: input_values are row indices; returns those rows' embeddings."""

    def __init__(self, embeddings):
        super().__init__()
        self.e = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)

    def forward(self, input_values):
        return SimpleNamespace(last_hidden_state=self.e[input_values[:, 0].long()][:, None, :])


def stack_similarity(legacy_sim, inputs_z, stack_idx, embeddings, k=5):
    """The previous `embedding_similarity.run` on precomputed embeddings; drops its large arrays."""
    df = pd.DataFrame({'data': list(np.asarray(inputs_z, np.float32)), 'stack_idx': np.asarray(stack_idx)})
    idx = torch.arange(len(df), dtype=torch.float32)[:, None]
    loader = [{'data': idx[i:i + 64]} for i in range(0, len(df), 64)]
    out = legacy_sim.run(df, _Precomputed(embeddings), loader, k=k, device='cpu')
    d = out['similarity_distributions']
    return {'embedding_stack_match_rate': float(out['embedding_stack_match_rate']),
            'input_stack_match_rate': float(out['input_stack_match_rate']),
            'match_score_avg': float(out['match_score_avg']),
            'distributions': {key: d[key] for key in d if not key.startswith('sim_matrix')}}
