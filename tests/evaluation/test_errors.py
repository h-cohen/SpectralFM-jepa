import numpy as np
import pytest

from spectral_lejepa.evaluation.errors import error_metrics


def _data():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    good = np.stack([y + .1 * rng.normal(size=200) for _ in range(2)])
    bad = np.stack([y + 1.0 * rng.normal(size=200) for _ in range(2)])
    return y, {'raw': bad, 'model': good}


def test_rmse_family_matches_definitions():
    y, P = _data()
    m = error_metrics(y, P)['model']
    e = P['model'] - y
    assert m['rmse'] == pytest.approx(np.mean(np.sqrt((e ** 2).mean(1))))
    assert m['nrmse'] == pytest.approx(m['rmse'] / y.std())
    assert m['mae'] == pytest.approx(np.abs(e).mean())
    assert m['median_ae'] == pytest.approx(np.mean(np.median(np.abs(e), 1)))
    assert m['p90_ae'] == pytest.approx(np.mean(np.quantile(np.abs(e), .9, axis=1)))
    assert m['bias'] == pytest.approx(e.mean())


def test_paired_and_per_spectrum_comparisons_to_raw():
    y, P = _data()
    out = error_metrics(y, P)
    assert 'vs_raw' not in out['raw']
    d = out['model']['vs_raw']
    assert d['delta_rmse'] < 0 and d['sd'] > 0 and d['p_lower'] > .99
    assert out['model']['win_rate'] > .8


def test_quintile_rmse_and_residual_slope():
    y = np.linspace(-2, 2, 500)
    shrunk = np.stack([.5 * y, .5 * y])          # ridge-like shrinkage toward the mean
    m = error_metrics(y, {'raw': shrunk, 'model': shrunk})['model']
    assert len(m['rmse_by_quintile']) == 5
    assert m['rmse_by_quintile'][0] > m['rmse_by_quintile'][2] < m['rmse_by_quintile'][4]
    assert m['residual_slope'] == pytest.approx(-.5)
