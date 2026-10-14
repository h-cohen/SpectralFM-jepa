import glob

import numpy as np
import pytest
import torch

from spectral_lejepa.inference import SpectralEmbedder

pytestmark = pytest.mark.slow
EVAL = {'robust_s401': 'outputs/robustness-continuation/robust_s401/eval/*_step49799',
        'robust_p0': 'outputs/robust-ensemble/robust_p0/eval/*_step49799',
        'robust_p1': 'outputs/robust-ensemble/robust_p1/eval/*_step49799',
        'robust_s101': 'outputs/robust-ensemble/robust_s101/eval/*_step49799'}


@pytest.mark.parametrize('device, atol', [('cuda', 1e-5), ('cpu', 5e-3)])
def test_real_bundle_reproduces_each_members_evaluation_features(device, atol):
    """Evaluation banks were extracted on GPU: CUDA matches them exactly; CPU transformer kernels differ by ~1e-3."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no GPU')
    emb = SpectralEmbedder(device=device)
    for j, name in enumerate(emb.members):
        bank = np.load(glob.glob(EVAL[name])[0] + '/dataset0055/bank.npz', allow_pickle=True)
        got = emb.embed(bank['input_raw'][:, 0].astype(np.float32))[:, j]
        assert np.allclose(got, bank['bank__layer2__flat'][:, 0, 0], atol=atol), name
