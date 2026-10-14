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


def script_encoder(checkpoint):
    """TorchScript raw [B,245] -> block2-flat [B,12288] encoder, verified against the in-memory model."""
    model, ckpt = load_model(str(checkpoint))
    if ckpt['config']['data']['normalization'] != 'global':
        raise ValueError('Production export requires global input normalization')
    stats = ckpt['config']['derived']['global_stats']
    compiled = torch.jit.script(ProductionEncoder(model.eval(), stats['mean'], stats['std']).eval())
    for batch in (1, 3):
        raw = torch.randn(batch, 245)
        with torch.no_grad():
            token = model.tokenizer((raw - stats['mean']) / stats['std'])
            expected = model.encoder.blocks[1](model.encoder.blocks[0](token)).flatten(1)
            torch.testing.assert_close(compiled(raw), expected, atol=1e-5, rtol=1e-5)
    return compiled, ckpt


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export(checkpoint, output):
    compiled, ckpt = script_encoder(checkpoint)
    stats = ckpt['config']['derived']['global_stats']
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    compiled.save(str(output / 'encoder.pt'))
    metadata = {'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256(checkpoint),
        'step': ckpt['step'], 'run_id': ckpt.get('wandb_run_id'),
        'training_seed': ckpt['config']['experiment']['seed'], 'readout': 'layer2/flat',
        'input': {'shape': ['B', 245], 'dtype': 'float32', 'normalization': 'included global affine'},
        'output': {'shape': ['B', 12288], 'dtype': 'float32'}, 'global_stats': stats,
        'benchmark_status': 'pending fixed-readout evaluation; deployment validation still required'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Exported: {output / "encoder.pt"}', flush=True)


def export_ensemble(members, output, force=False):
    """Inference bundle: member_<name>.pt TorchScript encoders + manifest.json (see the inference spec)."""
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f'{output} is not empty; pass --force to overwrite')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, checkpoint in members.items():
        compiled, ckpt = script_encoder(checkpoint)
        file = f'member_{name}.pt'
        compiled.save(str(output / file))
        entries.append({'name': name, 'file': file, 'sha256': _sha256(output / file),
                        'source_checkpoint': str(checkpoint), 'source_sha256': _sha256(checkpoint),
                        'step': ckpt['step'], 'global_stats': ckpt['config']['derived']['global_stats']})
    manifest = {'format_version': 1, 'readout': 'layer2/flat',
                'input': {'length': 245, 'dtype': 'float32', 'normalization': 'inside each encoder'},
                'output': {'shape': ['N', len(entries), 12288], 'dtype': 'float32'}, 'members': entries}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Bundle: {output} ({len(entries)} members)', flush=True)
    return manifest


def main_ensemble(argv=None):
    import yaml
    p = argparse.ArgumentParser(description='Build the inference bundle from the members in a config.')
    p.add_argument('--config', type=Path, default=Path('configs/inference.yaml'))
    p.add_argument('--output', type=Path)
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    export_ensemble(cfg['members'], a.output or Path(cfg['bundle']), force=a.force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/final-model/production'))
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
