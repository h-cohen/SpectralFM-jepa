import numpy as np
import pytest
import torch
import torch.nn.functional as F

from spectral_lejepa.evaluation.noise import denormalize, layer_norm_rows, noisy_raw, row_cosine


def sample():
    x = np.random.default_rng(0).normal(3, 2, (4, 245)).astype(np.float32)
    x[2] = 1.5   # constant spectrum
    return x


def test_layer_norm_matches_torch_and_round_trips():
    x = sample()
    z, mu, sd = layer_norm_rows(x)
    ref = torch.stack([F.layer_norm(torch.from_numpy(r), r.shape) for r in x]).numpy()
    assert np.allclose(z, ref, atol=1e-5)
    assert np.allclose(denormalize(z, mu, sd), x, atol=1e-5) and np.isfinite(z).all()


def test_noise_is_applied_in_the_normalized_space():
    configs = {'add_one': lambda z: z + 1.0}
    x = sample()
    out = noisy_raw(x, 'add_one', 0, configs)
    _, _, sd = layer_norm_rows(x)
    assert np.allclose(out - x, sd, atol=1e-5)


def test_noise_is_reproducible_for_a_seed():
    configs = {'g': lambda z: z + np.random.normal(0, .01, size=z.shape)}
    assert np.array_equal(noisy_raw(sample(), 'g', 3, configs), noisy_raw(sample(), 'g', 3, configs))


def test_row_cosine_is_one_for_scaled_rows():
    a = np.random.default_rng(1).normal(size=(3, 8))
    assert np.allclose(row_cosine(a, 2 * a), 1.0)


@pytest.mark.slow
def test_legacy_noise_formula_is_used_unchanged():
    from spectral_lejepa.evaluation.legacy import legacy_module
    configs = legacy_module('code/eval/evaluations/noise_robustness.py').NOISE_CONFIGS
    x = sample()
    z, mu, sd = layer_norm_rows(x)
    np.random.seed(7)
    expected = denormalize(np.stack([configs['shot_low'](r.astype(float)) for r in z]), mu, sd)
    assert np.allclose(noisy_raw(x, 'shot_low', 7, configs), expected)
