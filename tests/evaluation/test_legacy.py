import pytest

from spectral_lejepa.evaluation.legacy import LEGACY_COMMIT, legacy_module

pytestmark = pytest.mark.slow


def test_noise_configs_come_from_the_pinned_commit():
    m = legacy_module('code/eval/evaluations/noise_robustness.py')
    assert list(m.NOISE_CONFIGS) == ['gaussian_std', 'gaussian_mean', 'shot_low', 'shot_high', 'gain_low', 'gain_high']
    assert m.__file__.endswith(f'@{LEGACY_COMMIT[:7]}:code/eval/evaluations/noise_robustness.py')


def test_overrides_replace_module_globals():
    m = legacy_module('code/eval/evaluations/structured_similarity.py', {'_normalize_like_fairseq': lambda a: a})
    assert m._normalize_like_fairseq(5) == 5
