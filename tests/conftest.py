import os

os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("LIGHTLY_DID_VERSION_CHECK", "True")

import pytest


@pytest.fixture
def make_checkpoint(tmp_path):
    """Factory: a minimal stage-2-like checkpoint (48 patches, d256, global normalization) on disk."""
    import torch
    import yaml
    from spectral_lejepa.models.vit_1d import build_model

    def make(name, seed=0):
        torch.manual_seed(seed)
        cfg = yaml.safe_load(open('configs/final_model.yaml'))
        model = build_model(cfg['model'], 245)
        ckpt = {'model': model.state_dict(), 'step': 7, 'wandb_run_id': None,
                'config': {'model': cfg['model'], 'data': {'sequence_length': 245, 'normalization': 'global'},
                           'derived': {'global_stats': {'mean': .3784, 'std': .5126}},
                           'experiment': {'seed': seed}}}
        path = tmp_path / f'{name}.pt'
        torch.save(ckpt, path)
        return path
    return make
