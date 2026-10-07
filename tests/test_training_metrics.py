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
    from scripts.training_progress import seed_comparison_figure, seed_paired_gains_figure
    fig = seed_comparison_figure([presentation_row(), presentation_row(497990)])
    assert '497990' in fig._suptitle.get_text()
    assert len(fig.axes) == 8
    plt.close(fig)
    gains = seed_paired_gains_figure([presentation_row(497990)])
    assert '497990' in gains._suptitle.get_text()
    assert all('paired bootstrap SD' in ax.get_xlabel() for ax in gains.axes)
    plt.close(gains)


def test_refresh_evaluations_reuses_presentation_run_without_replaying_training_history(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import scripts.training_progress as progress
    monkeypatch.setattr(progress, 'OUTPUT', tmp_path)
    monkeypatch.setattr(progress, 'ROOT', tmp_path)
    monkeypatch.setattr(progress, 'evaluation_rows', lambda roots: [presentation_row(497990)])
    (tmp_path / 'run.json').write_text(json.dumps({'id': 'existing123'}))
    calls = {'logs': []}

    class Run:
        summary = {}
        url = 'https://wandb.invalid/runs/existing123'

        def log(self, data):
            calls['logs'].append(data)

        def finish(self):
            calls['finished'] = True

    monkeypatch.setitem(__import__('sys').modules, 'wandb', SimpleNamespace(
        init=lambda **kwargs: (calls.update(init=kwargs) or Run()),
        Table=lambda **kwargs: kwargs,
        Html=lambda html: html,
        Image=lambda figure: figure,
    ))

    progress.main(['--refresh-evaluations'])

    assert calls['init']['id'] == 'existing123'
    assert calls['init']['resume'] == 'allow'
    assert not any('seed0/optimizer_step' in item for item in calls['logs'])
    assert any('evaluation/latest_per_dataset_r2' in item for item in calls['logs'])
    assert any('evaluation/latest_paired_gains' in item for item in calls['logs'])
    assert (tmp_path / 'latest_per_dataset_r2.png').is_file()
    assert calls['finished'] is True


def test_presentation_run_startup_resumes_id_recorded_in_run_manifest(tmp_path, monkeypatch):
    import json
    import scripts.training_progress as progress
    monkeypatch.setattr(progress, 'OUTPUT', tmp_path)
    (tmp_path / 'run.json').write_text(json.dumps({'id': 'existing123', 'url': 'old-url', 'sources': {}}))
    calls = {}

    class Run:
        id = 'existing123'
        url = 'https://wandb.invalid/runs/existing123'

    class Wandb:
        def init(self, **kwargs):
            calls.update(kwargs)
            return Run()

    progress.init_presentation_run(Wandb(), {0: ['source0'], 1: ['source1']}, {'training': {}}, 100)

    assert calls['id'] == 'existing123'
    assert calls['resume'] == 'allow'
    assert json.loads((tmp_path / 'run.json').read_text())['id'] == 'existing123'


def test_resumed_watcher_starts_after_each_source_run_cursor():
    from scripts.training_progress import history_cursors

    class Source:
        def __init__(self, row):
            self.summary = {'_step': row}

    class Api:
        def run(self, path):
            return Source({'old0': 50, 'old1': 60, 'resume0': 500, 'resume1': 600}[path.rsplit('/', 1)[-1]])

    sources = {0: ['old0', 'resume0'], 1: ['old1', 'resume1']}
    last_status = {'0': {'source_run': 'resume0', 'step': 480, 'history_step': 470},
                   '1': {'source_run': 'resume1', 'step': 590, 'history_step': 580}}

    assert history_cursors(Api(), sources, last_status) == {
        'old0': 50, 'resume0': 470, 'old1': 60, 'resume1': 580,
    }


def test_resumed_watcher_prefers_persisted_processed_cursors():
    from scripts.training_progress import history_cursors

    class Api:
        def run(self, path):
            raise AssertionError(f'persisted cursor should avoid querying {path}')

    sources = {0: ['old0', 'resume0']}
    persisted = {'old0': 45, 'resume0': 480}

    assert history_cursors(Api(), sources, {}, persisted) == persisted


def test_resumed_watcher_recovers_history_cursor_when_legacy_status_has_only_optimizer_step():
    from scripts.training_progress import history_cursors

    class Source:
        summary = {'_step': 500, 'train/global_step': 500}

        def scan_history(self, **kwargs):
            return iter([{'_step': 498, 'train/global_step': 499},
                         {'_step': 499, 'train/global_step': 500}])

    class Api:
        def run(self, path):
            return Source()

    cursors = history_cursors(Api(), {0: ['resume0']},
                              {'0': {'source_run': 'resume0', 'step': 500}})

    assert cursors['resume0'] == 499


def test_summary_includes_raw_model_and_random_scores():
    from scripts.training_progress import narrative, evaluation_table
    report = narrative([presentation_row()])
    assert '<th>Mean raw R²</th>' in report
    assert '<th>Mean model R²</th>' in report
    assert '<th>Mean random R²</th>' in report
    columns, values = evaluation_table([presentation_row()])
    assert values[0][columns.index('paired_delta_vs_random_sd')] == .02


def test_load_sources_reads_manifest_into_integer_seed_keys(tmp_path):
    import json
    from scripts.training_progress import load_sources
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps({'0': ['old0', 'resume0'], '1': ['old1', 'resume1']}))

    assert load_sources(manifest) == {0: ['old0', 'resume0'], 1: ['old1', 'resume1']}


@pytest.mark.parametrize('manifest_data', [
    {'0': ['old0'], '1': []},
    {'0': ['old0']},
    {'0': ['old0', 12], '1': ['resume1']},
])
def test_load_sources_rejects_missing_seeds_empty_lists_and_nonstring_ids(tmp_path, manifest_data):
    import json
    from scripts.training_progress import load_sources
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps(manifest_data))

    with pytest.raises(ValueError):
        load_sources(manifest)


def test_evaluation_rows_skips_partially_written_json(tmp_path):
    from scripts.training_progress import SETS, evaluation_rows
    (tmp_path / 'partial').mkdir()
    (tmp_path / 'partial' / 'summary.json').write_text('{"sets":')
    complete = {s: {'raw_r2': .3, 'embedding_r2': .4} for s in SETS}
    (tmp_path / 'complete').mkdir()
    (tmp_path / 'complete' / 'summary.json').write_text(__import__('json').dumps({'sets': complete}))

    rows = evaluation_rows([tmp_path])

    assert len(rows) == 1
    assert rows[0]['wins'] == 8
