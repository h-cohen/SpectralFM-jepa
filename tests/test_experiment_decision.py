import json
import sys
from types import SimpleNamespace

import pytest

from scripts.assess_screen8 import ARMS, assess
from scripts.training_progress import SETS


def summary(model_shift=0.1, raw=0.2):
    return {'sets': {
        name: {
            'raw_r2': raw,
            'embedding_r2': raw + model_shift,
            'random_control': {'embedding_r2': raw - 0.1},
            'embedding_minus_raw': {'delta': model_shift, 'sd': 0.02},
            'vs_random_control': {'delta': model_shift + 0.1, 'sd': 0.03},
        }
        for name in SETS
    }}


def test_assess_applies_candidate_gates_and_keeps_per_set_controls():
    cards = {arm: summary(0.05 if arm.startswith('control_') else 0.1) for arm in ARMS}
    result = assess(cards)

    assert result['missing_arms'] == []
    assert result['decisions']['random50']['candidate'] is True
    for seed in ('0', '1'):
        assert result['arms'][f'random50_s{seed}']['wins'] == 8
        assert result['arms'][f'random50_s{seed}']['hard_set_mean_gain'] == pytest.approx(0.05)
        point = result['arms'][f'random50_s{seed}']['sets']['dataset0106']
        assert point['raw_r2'] == pytest.approx(0.2)
        assert point['model_r2'] == pytest.approx(0.3)
        assert point['random_r2'] == pytest.approx(0.1)
        assert point['control_model_r2'] == pytest.approx(0.25)
        assert point['paired_uncertainty']['model_minus_raw']['sd'] == 0.02


def test_assess_marks_missing_or_partial_arm_pending_and_preserves_none():
    cards = {arm: summary() for arm in ARMS if arm != 'control_s1'}
    cards['block75_s1'] = {'sets': {name: summary()['sets'][name] for name in SETS[:-1]}}
    result = assess(cards)

    assert result['missing_arms'] == ['control_s1', 'block75_s1']
    assert result['decisions']['random50']['candidate'] is None
    assert result['arms']['random50_s1']['sets']['labeled_data']['control_model_r2'] is None
    assert result['decisions']['block75']['status'] == 'pending'


def test_assess_rejects_variant_and_control_raw_mismatch():
    cards = {arm: summary() for arm in ARMS}
    cards['random50_s0']['sets']['dataset0055']['raw_r2'] += 0.001

    with pytest.raises(ValueError, match='raw R² mismatch'):
        assess(cards)


def test_assess_uses_strict_positive_delta_for_wins():
    cards = {arm: summary() for arm in ARMS}
    for seed in ('0', '1'):
        for name in SETS[:4]:
            cards[f'random50_s{seed}']['sets'][name]['embedding_r2'] = cards[f'random50_s{seed}']['sets'][name]['raw_r2']

    result = assess(cards)

    assert result['arms']['random50_s0']['wins'] == 4
    assert result['decisions']['random50']['gates']['wins_each_seed'] is False


def test_cli_reads_step15000_summaries_writes_reports_and_logs_table(tmp_path, monkeypatch):
    from scripts.assess_screen8 import main
    calls = {'logs': []}

    class Run:
        def log(self, data):
            calls['logs'].append(data)

        def finish(self):
            calls['finished'] = True

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=lambda **kwargs: calls.setdefault('init', kwargs) or Run(),
        Table=lambda **kwargs: kwargs,
        Image=lambda figure: {'figure': figure},
    ))
    # setdefault returns the config dictionary, so provide a run explicitly after recording init.
    wandb = sys.modules['wandb']
    wandb.init = lambda **kwargs: (calls.update(init=kwargs) or Run())
    for arm in ARMS:
        directory = tmp_path / 'eval' / f'{arm}_20261007-100000_step15000'
        directory.mkdir(parents=True)
        (directory / 'summary.json').write_text(json.dumps(summary(0.05 if arm.startswith('control_') else 0.1)))
    ignored = tmp_path / 'eval' / 'random50_s0_20261007-100000_step14999'
    ignored.mkdir()
    (ignored / 'summary.json').write_text(json.dumps(summary()))

    main(['--root', str(tmp_path)])

    result = json.loads((tmp_path / 'decision.json').read_text())
    assert result['missing_arms'] == []
    assert result['decisions']['random50']['candidate'] is True
    assert result['decisions']['random50']['mean_raw_r2'] == pytest.approx(0.2)
    assert result['decisions']['random50']['mean_model_r2'] == pytest.approx(0.3)
    assert result['decisions']['random50']['mean_random_r2'] == pytest.approx(0.1)
    markdown = (tmp_path / 'decision.md').read_text()
    assert 'Mean raw R²' in markdown and 'Mean model R²' in markdown and 'Mean random R²' in markdown
    assert '| random50 | candidate | 0.200 | 0.300 | 0.100 |' in markdown
    assert calls['init']['project'] == 'spectralfm-lejepa'
    assert calls['init']['group'] == 'screen-8' and calls['init']['job_type'] == 'decision'
    assert any('decision/table' in item for item in calls['logs'])
    scorecard = next(item['decision/per_dataset'] for item in calls['logs'] if 'decision/per_dataset' in item)
    assert len(scorecard['data']) == 48
    columns, data = scorecard['columns'], scorecard['data']
    point = next(row for row in data if row[columns.index('arm')] == 'random50_s0'
                 and row[columns.index('dataset')] == 'dataset0106')
    assert point[columns.index('raw_r2')] == pytest.approx(0.2)
    assert point[columns.index('model_r2')] == pytest.approx(0.3)
    assert point[columns.index('random_r2')] == pytest.approx(0.1)
    assert point[columns.index('delta_vs_control')] == pytest.approx(0.05)
    figure_logs = [item for item in calls['logs'] if 'decision/per_dataset_r2' in item or 'decision/paired_gains' in item]
    assert len(figure_logs) == 2
    assert (tmp_path / 'per_dataset_r2.png').is_file()
    assert (tmp_path / 'paired_gains.png').is_file()
    assert calls['finished'] is True


def test_screen_figure_records_keep_random_and_matched_control_gains_separate():
    from scripts.assess_screen8 import figure_records
    cards = {arm: summary(.05 if arm.startswith('control_') else .1) for arm in ARMS}
    result = assess(cards)

    candidate = next(row for row in figure_records(result) if row['label'] == 'random50 seed0')
    control = next(row for row in figure_records(result) if row['label'] == 'control seed0')
    point = candidate['sets']['dataset0106']
    assert point['delta_vs_random'] == pytest.approx(.2)
    assert point['delta_vs_control'] == pytest.approx(.05)
    assert point['sd_vs_random'] == pytest.approx(.03)
    assert control['sets']['dataset0106']['delta_vs_control'] is None
