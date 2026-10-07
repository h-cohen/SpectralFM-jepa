import matplotlib.pyplot as plt

from scripts.scorecard_figures import dataset_comparison_figure, paired_gains_figure


def test_dataset_comparison_figure_shows_r2_context_and_missing_random_control():
    records = [{
        'label': 'run seed0 step15000',
        'sets': {
            'dataset0055': {'n': 25, 'raw_r2': .2, 'model_r2': .3, 'random_r2': None},
        },
    }]

    fig = dataset_comparison_figure(records, ['dataset0055'])

    assert 'n=25' in fig.axes[0].get_title()
    assert fig.axes[0].get_xlabel() == 'Nested-CV R²'
    assert len(fig.axes[0].collections) == 2  # raw and model only; no fabricated random score
    plt.close(fig)


def test_paired_gains_figure_shows_thresholds_uncertainty_and_missing_values():
    records = [{
        'label': 'run seed0 step15000',
        'sets': {
            'dataset0109': {'n': 96, 'delta_vs_raw': -.02, 'sd_vs_raw': .03,
                            'delta_vs_random': -.04, 'sd_vs_random': .02},
            'dataset0120': {'n': 125, 'delta_vs_raw': .06, 'sd_vs_raw': None,
                            'delta_vs_random': None, 'sd_vs_random': None},
        },
    }]

    fig = paired_gains_figure(records, ['dataset0109', 'dataset0120'])

    for ax in fig.axes:
        assert any(list(line.get_xdata()) == [0, 0] for line in ax.lines)
        assert any(list(line.get_xdata()) == [.05, .05] for line in ax.lines)
        assert 'paired bootstrap SD' in ax.get_xlabel()
    assert len(fig.axes[0].containers) >= 2  # both measured paired comparisons
    assert len(fig.axes[1].containers) == 1  # absent random-control result is omitted
    plt.close(fig)
