import numpy as np
import pytest
import torch

from spectral_lejepa.evaluation.bank import extract_bank
from spectral_lejepa.evaluation.run import tta_config

ROBUST = {'sigma_max': 0.2, 'offset_max': 0.05, 'slope_max': 0.05, 'scale_max': 0.05, 'shift_max': 1, 'prob': 0.0}


class Toy(torch.nn.Module):
    """Two 'blocks': tokens = input reshaped [B, 5, 49], then a tanh of it."""
    def forward(self, input_values, output_hidden_states=True):
        h0 = input_values.reshape(len(input_values), 5, 49)
        return type('O', (), {'hidden_states': (h0, torch.tanh(h0))})()


def x():
    return np.random.default_rng(0).normal(size=(10, 245)).astype(np.float32)


def tta(prob, k=1, seed=0):
    return {'k': k, 'seed': seed, 'robust': {**ROBUST, 'prob': prob}, 'stats': {'mean': 1.0, 'std': 2.0}}


def test_tta_off_matches_plain():
    a = extract_bank(Toy(), x(), device='cpu', readouts=('mean', 'flat'))
    b = extract_bank(Toy(), x(), device='cpu', readouts=('mean', 'flat'), tta=tta(0.0))
    for k in a:      # perturb round-trips through raw units, so only float round-off may differ
        np.testing.assert_allclose(a[k], b[k], atol=1e-5)


def test_tta_on_changes_and_is_deterministic():
    a = extract_bank(Toy(), x(), device='cpu', readouts=('mean',))
    b = extract_bank(Toy(), x(), device='cpu', readouts=('mean',), tta=tta(1.0, k=4))
    c = extract_bank(Toy(), x(), device='cpu', readouts=('mean',), tta=tta(1.0, k=4))
    assert not np.allclose(a['layer1'], b['layer1'])
    for k in b:
        np.testing.assert_array_equal(b[k], c[k])


def test_tta_config_requires_global_normalization():
    vcfg = {'tta': {'k': 8, 'seed': 0, 'robust': ROBUST}}
    with pytest.raises(ValueError, match='global'):
        tta_config(vcfg, {'data': {'normalization': 'sample_zscore'}})
    got = tta_config(vcfg, {'data': {'normalization': 'global'}, 'derived': {'global_stats': {'mean': 3., 'std': 4.}}})
    assert got['stats'] == {'mean': 3., 'std': 4.} and got['k'] == 8
    assert tta_config({}, {}) is None
