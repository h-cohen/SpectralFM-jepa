"""Export raw-spectrum → fixed block2-flat TorchScript encoder, with provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spectral_lejepa.training.checkpoint import load_model


class ProductionEncoder(nn.Module):
    """FP32 [B,245] → [B,12288]; normalization and first two blocks included."""
    def __init__(self, model, mean: float, std: float):
        super().__init__()
        if len(model.encoder.blocks) < 2 or len(model.tokenizer.bounds) != 48:
            raise ValueError('Production interface requires48patches and at least2blocks')
        if model.tokenizer.pos_embed.shape[1] != 256 or std <= 0:
            raise ValueError('Production interface requires d256 and positive normalization std')
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        self.register_buffer('index', model.tokenizer.index.detach().clone())
        self.register_buffer('position', model.tokenizer.pos_embed.detach().clone())
        self.projection = model.tokenizer.proj
        self.blocks = nn.ModuleList(list(model.encoder.blocks)[:2])

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.dim() != 2 or raw.size(1) != 245:
            raise ValueError('Expected raw spectra [B,245]')
        if raw.dtype != torch.float32:
            raise ValueError('Expected float32 raw spectra')
        x = (raw - self.mean) / self.std
        x = self.projection(F.pad(x, (0, 1))[:, self.index]) + self.position
        for block in self.blocks:
            x = block(x)
        # This is hidden_states[2], not the sixth block's final LayerNorm output.
        return x.flatten(1)


def export(checkpoint, output):
    model, ckpt = load_model(str(checkpoint))
    if ckpt['config']['data']['normalization'] != 'global':
        raise ValueError('Production export requires global input normalization')
    stats = ckpt['config']['derived']['global_stats']
    encoder = ProductionEncoder(model.eval(), stats['mean'], stats['std']).eval()
    compiled = torch.jit.script(encoder)
    for batch in (1, 3):
        raw = torch.randn(batch, 245)
        with torch.no_grad():
            normalized = (raw - stats['mean']) / stats['std']
            token = model.tokenizer(normalized)
            expected = model.encoder.blocks[1](model.encoder.blocks[0](token)).flatten(1)
            torch.testing.assert_close(compiled(raw), expected, atol=1e-5, rtol=1e-5)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    compiled.save(str(output / 'encoder.pt'))
    digest = hashlib.sha256()
    with open(checkpoint, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest.hexdigest(),
        'step': ckpt['step'], 'run_id': ckpt.get('wandb_run_id'),
        'training_seed': ckpt['config']['experiment']['seed'], 'readout': 'layer2/flat',
        'input': {'shape': ['B', 245], 'dtype': 'float32', 'normalization': 'included global affine'},
        'output': {'shape': ['B', 12288], 'dtype': 'float32'}, 'global_stats': stats,
        'benchmark_status': 'pending fixed-readout evaluation; deployment validation still required'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Exported: {output / "encoder.pt"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/final-model/production'))
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
