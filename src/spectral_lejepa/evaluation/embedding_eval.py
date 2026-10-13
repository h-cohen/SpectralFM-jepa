"""Previous-project cosine similarity and noise robustness on block-2 readouts (spec 2026-10-14)."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from spectral_lejepa.evaluation.data import load_labeled_data
from spectral_lejepa.evaluation.legacy import LEGACY_COMMIT, legacy_module
from spectral_lejepa.evaluation.noise import layer_norm_rows, noisy_raw, row_cosine
from spectral_lejepa.evaluation.stacks import load_stack_subset

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


def _cosine(m):
    u = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return (u @ u.T).astype(np.float32)


def _centered_cosine(m):
    """Cosine after removing the mean vector (the previous panel's anisotropy correction)."""
    return _cosine(m - m.mean(0, keepdims=True))


def load_data(cfg, legacy):
    d, seed = cfg['data'], cfg['experiment']['seed']
    stacks = {a: load_stack_subset(f"{d['nova_data_dir']}/{ds}", split, n, seed=seed)
              for a, (ds, split, n) in d['stack_sets'].items()}
    noise_sets = {a: s['raw'] for a, s in stacks.items()}
    noise_sets.update({name: load_labeled_data(dirs)[0] for name, dirs in d['label_sets'].items()})
    panel_raw, groups = legacy['panel'].build_panel(d['nova_data_dir'], seed=seed)
    return {'stack_sets': {a: stacks[a] for a in d['similarity_sets']}, 'noise_sets': noise_sets,
            'panel': {'raw': panel_raw, 'groups': groups}}


def evaluate_model(model, ckpt_cfg, data, cfg, legacy):
    from spectral_lejepa.evaluation.run import model_input
    e = cfg['evaluation']
    embed = lambda raw: block2_readouts(model, model_input(raw, ckpt_cfg), e['device'], e['batch_size'])
    out = {'similarity': {}, 'noise': {}, 'panel': {}, 'arrays': {}}
    for alias, s in data['stack_sets'].items():
        z = layer_norm_rows(s['raw'])[0]   # the previous input space for the input-neighbour baseline
        feats = embed(s['raw'])
        out['similarity'][alias] = {}
        for r in READOUTS:
            res = stack_similarity(legacy['sim'], z, s['stack_idx'], feats[r], k=e['k'])
            for key, v in res.pop('distributions').items():
                out['arrays'][f'dist/{alias}/{r}/{key}'] = np.asarray(v)
            out['similarity'][alias][r] = res
    configs = legacy['noise'].NOISE_CONFIGS
    for alias, raw in data['noise_sets'].items():
        clean = embed(raw)
        out['noise'][alias] = {r: {} for r in READOUTS}
        for t, kind in enumerate(configs):
            noisy = embed(noisy_raw(raw, kind, e['noise_seed'] + t, configs))
            for r in READOUTS:
                cos = row_cosine(clean[r], noisy[r])
                out['arrays'][f'noise/{alias}/{r}/{kind}'] = cos
                out['noise'][alias][r][kind] = float(cos.mean())
    if data['panel'] is not None:
        feats = embed(data['panel']['raw'])
        out['arrays']['panel/input'] = _cosine(layer_norm_rows(data['panel']['raw'])[0])
        for r in READOUTS:
            out['arrays'][f'panel/{r}'] = _cosine(feats[r])
            out['arrays'][f'panel/{r}_centered'] = _centered_cosine(feats[r])
        out['panel'] = {'groups': list(data['panel']['groups'])}
    return out


def _rows(model, res):
    for alias, by_r in res['similarity'].items():
        for r, m in by_r.items():
            for key in ('embedding_stack_match_rate', 'input_stack_match_rate', 'match_score_avg'):
                yield model, alias, r, key, m[key]
    for alias, by_r in res['noise'].items():
        for r, kinds in by_r.items():
            for kind, v in kinds.items():
                yield model, alias, r, f'noise_cosine/{kind}', v


def ensemble_rows(rows, members):
    """Mean of the members' values per (set, readout, metric); refuses an incomplete member set."""
    by = {}
    for model, alias, r, key, v in rows:
        by.setdefault((alias, r, key), {})[model] = v
    out = []
    for (alias, r, key), vals in sorted(by.items()):
        if sorted(vals) != sorted(members):
            raise ValueError(f'{alias}/{r}/{key}: results for {sorted(vals)}, expected {sorted(members)}')
        out.append(('ensemble', alias, r, key, float(np.mean(list(vals.values())))))
    return out


def run_suite(cfg):
    from spectral_lejepa.training.checkpoint import load_model
    out = Path(cfg['experiment']['output_dir'])
    legacy = {'sim': legacy_module('code/eval/evaluations/embedding_similarity.py'),
              'noise': legacy_module('code/eval/evaluations/noise_robustness.py'),
              'panel': legacy_module('code/eval/evaluations/structured_similarity.py',
                                     {'_normalize_like_fairseq': lambda a: a})}   # raw panel spectra
    data = load_data(cfg, legacy)
    rows = []
    for name, path in cfg['models'].items():
        model, ck = load_model(path, map_location='cpu')
        res = evaluate_model(model, ck['config'], data, cfg, legacy)
        (out / name).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / name / 'arrays.npz', **res.pop('arrays'))
        (out / name / 'results.json').write_text(json.dumps({'checkpoint': str(path), **res}, indent=2))
        rows += list(_rows(name, res))
        print(f'[embedding] {name} done', flush=True)
    rows += ensemble_rows(rows, list(cfg['models']))
    with open(out / 'summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows([('model', 'set', 'readout', 'metric', 'value'), *rows])
    (out / 'run_info.json').write_text(json.dumps({
        'legacy_commit': LEGACY_COMMIT, 'config': cfg,
        'deviation': 'noise draws seeded with np.random.seed(noise_seed + type index); previous code was unseeded'},
        indent=2, default=str))
    return out
