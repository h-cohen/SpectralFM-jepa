import pytest
from spectral_lejepa.utils import wandb as wb


@pytest.mark.parametrize('regularizer,expected', [('sigreg', (2., .5)), ('visreg', (.8, 6.))])
def test_weighted_terms_explain_total(regularizer, expected):
    cfg = dict(regularizer=regularizer, lambda_sigreg=.05, visreg_lambda=.6, global_weight=1.)
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


def test_presentation_excludes_partial_or_merged_scorecards(tmp_path):
    import json
    from scripts.training_progress import SETS, evaluation_rows, evaluation_table
    partial = {s: {'raw_r2':.3, 'embedding_r2':.4} for s in SETS[:-1]}
    (tmp_path/'partial').mkdir()
    (tmp_path/'partial'/'summary.json').write_text(json.dumps({'sets':partial}))
    complete = {s: {'n':25, 'raw_r2':.3, 'embedding_r2':.4} for s in SETS}
    complete['merged'] = {'raw_r2':0, 'embedding_r2':1}
    (tmp_path/'complete').mkdir()
    (tmp_path/'complete'/'summary.json').write_text(json.dumps({'sets':complete}))
    rows = evaluation_rows([tmp_path])
    assert len(rows) == 1 and rows[0]['wins'] == 8
    cols, values = evaluation_table(rows)
    assert len(values) == 8
    assert all(row[cols.index('random_r2')] is None for row in values)


def presentation_row(step=200000):
    from scripts.training_progress import SETS
    return dict(name=f'long-1/lr50_s0_run_step{step}', step=step, wins=5, strong_wins=2, mean_delta=.1,
                sets={s: {'n':25,'raw_r2':.3,'embedding_r2':.4,'random_control':{'embedding_r2':.35},
                          'vs_random_control':{'sd':.02,'p_a_better':.8}} for s in SETS})


def test_latest_figure_advances_with_completed_evaluation():
    import matplotlib.pyplot as plt
    from scripts.training_progress import seed_comparison_figure
    fig = seed_comparison_figure([presentation_row(), presentation_row(497990)])
    assert '497990' in fig.axes[0].get_title()
    plt.close(fig)


def test_summary_includes_raw_model_and_random_scores():
    from scripts.training_progress import narrative, evaluation_table
    report = narrative([presentation_row()])
    assert '<th>Mean raw R²</th>' in report
    assert '<th>Mean model R²</th>' in report
    assert '<th>Mean random R²</th>' in report
    columns, values = evaluation_table([presentation_row()])
    assert values[0][columns.index('paired_delta_vs_random_sd')] == .02
