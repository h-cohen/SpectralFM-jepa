import pytest
from spectral_lejepa.utils import wandb as wb


def test_global_terms_logged_with_actual_weights():
    cfg=dict(lambda_sigreg=.05,global_weight=1.)
    values={'train/mse_loss':2.,'train/sigreg_loss':10.,'train/global_inv_loss':.1,
            'train/global_sigreg_loss':20.,'train/global_loss':1.095}
    r=wb.training_metrics(values,cfg,100)
    assert r['train/global_invariance_contribution']==pytest.approx(.095)
    assert r['train/global_sigreg_contribution']==pytest.approx(1.)
    assert sum(r[k] for k in ('train/prediction_contribution','train/regularizer_contribution',
                              'train/global_invariance_contribution','train/global_sigreg_contribution'))==pytest.approx(3.595)


def test_weighted_terms_explain_total():
    expected = (2., .5)
    cfg = dict(lambda_sigreg=.05, global_weight=1.)
    metrics = {'train/mse_loss': 2., 'train/sigreg_loss': 10., 'train/global_loss': 3.,
               'train/global_step': 200000}
    result = wb.training_metrics(metrics, cfg, 500000)
    assert result['train/prediction_contribution'] == pytest.approx(expected[0])
    assert result['train/regularizer_contribution'] == pytest.approx(expected[1])
    assert result['train/global_contribution'] == 3.
    assert result['progress/percent_complete'] == 40.
    assert result['progress/steps_remaining'] == 300000
    assert 'train/loss' not in result  # no invented measurements


def test_validation_metrics_do_not_invent_training_loss():
    result = wb.training_metrics({'valid/loss': .2}, {}, 100)
    assert result == {'valid/loss': .2}
