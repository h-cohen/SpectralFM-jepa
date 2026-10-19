"""Inference cost per spectrum: raw ridge vs one JEPA member vs the 4-member ensemble vs data2vec.

FLOPs are counted with torch.utils.flop_counter (matmuls and convolutions; elementwise ops, norms and
softmax are not counted) on one [1, 245] spectrum. The linear readout adds 2·d FLOPs on d features.

    python ops/inference_flops.py [--out outputs/inference_flops.json]
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from spectral_lejepa.evaluation.data2vec import load_data2vec
from spectral_lejepa.models.export import ProductionEncoder
from spectral_lejepa.training.checkpoint import load_model

D2V = '/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged/outputs/d2v_alldata/checkpoints/checkpoint_last.pt'
MANIFEST = Path('outputs/inference/robust-ensemble/manifest.json')


def flops(fn, x):
    # nn.TransformerEncoderLayer's eval fast path is one fused kernel the counter cannot see; same maths unfused
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            fn(x)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    return int(counter.get_total_flops())


def params(module):
    return int(sum(p.numel() for p in module.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/inference_flops.json')
    a = ap.parse_args()
    x = torch.randn(1, 245)
    m = json.loads(MANIFEST.read_text())['members'][0]   # members share one architecture
    model, _ = load_model(m['source_checkpoint'])
    # the exported encoder in eager mode: the FLOP counter cannot see inside TorchScript
    member = ProductionEncoder(model.eval(), m['global_stats']['mean'], m['global_stats']['std']).eval()
    d2v, _ = load_data2vec(D2V)
    d2v.eval()
    enc = flops(member, x)
    d2v_full = flops(lambda t: d2v(input_values=t, output_hidden_states=True), x)
    rows = {
        'raw ridge': {'encoder': 0, 'readout': 2 * 245, 'params': 245},
        'JEPA member (blocks 0-1, block-2 flat)': {'encoder': enc, 'readout': 2 * 12288,
                                                   'params': params(member) + 12288},
        'JEPA ensemble (4 members)': {'encoder': 4 * enc, 'readout': 4 * 2 * 12288,
                                      'params': 4 * (params(member) + 12288)},
        'data2vec recent (all 12 layers, mean readout)': {'encoder': d2v_full, 'readout': 2 * 768,
                                                           'params': params(d2v) + 768},
    }
    for r in rows.values():
        r['total'] = r['encoder'] + r['readout']
    Path(a.out).write_text(json.dumps({'member': m['name'], 'rows': rows}, indent=1))
    for k, r in rows.items():
        print(f"{k:48s} {r['total'] / 1e6:10.3f} MFLOPs  {r['params'] / 1e6:8.2f} M params")


if __name__ == '__main__':
    main()
