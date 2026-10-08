import json

import pytest

from scripts.build_results_report import collect_report_data, render_html


DATASETS = [
    'labeled_data', 'dataset0055', 'dataset0106', 'dataset0109',
    'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120',
]


def _write_summary(path, *, step, model, raw, random, wins):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'lineage': {'pretraining_checkpoint_step': step},
        'scorecard': {'model': {'wins': wins, 'strong_wins': 1, 'sets': 8, 'mean_delta': model - raw}},
        'sets': {
            dataset: {
                'n': 25,
                'embedding_r2': model,
                'raw_r2': raw,
                'random_control': {'embedding_r2': random},
                'embedding_minus_raw': {'delta': model - raw, 'sd': .04},
                'vs_random_control': {'delta': model - random, 'sd': .05},
            } for dataset in DATASETS
        },
    }
    path.write_text(json.dumps(payload))


def test_report_data_normalizes_long_runs_and_exploratory_48_patch(tmp_path):
    _write_summary(
        tmp_path / 'outputs/long-1/eval/lr50_s0_resume_step497990/summary.json',
        step=497990, model=.72, raw=.60, random=.66, wins=5,
    )
    _write_summary(
        tmp_path / 'outputs/screen-7/eval/p48_lr50_step149397/summary.json',
        step=149397, model=.71, raw=.60, random=.65, wins=5,
    )
    _write_summary(
        tmp_path / 'outputs/paired-long/eval/p24_s2_run_step497990/summary.json',
        step=497990, model=.73, raw=.60, random=.66, wins=6,
    )

    data = collect_report_data(tmp_path)

    final = next(row for row in data['records'] if row['family'] == '24-patch' and row['step'] == 497990)
    assert final['sets']['dataset0055']['model_r2'] == .72
    assert final['sets']['dataset0055']['delta_vs_random'] == pytest.approx(.06)
    p48 = next(row for row in data['records'] if row['family'] == '48-patch')
    assert p48['status'] == 'exploratory single seed'
    seed2 = next(row for row in data['records'] if row['family'] == '24-patch' and row['seed'] == 2)
    assert seed2['step'] == 497990


def test_rendered_report_is_self_contained_and_labels_win_rule(tmp_path):
    _write_summary(
        tmp_path / 'outputs/long-1/eval/lr50_s0_resume_step497990/summary.json',
        step=497990, model=.72, raw=.60, random=.66, wins=5,
    )
    html = render_html(collect_report_data(tmp_path))

    assert '<!doctype html>' in html.lower()
    assert 'ΔR² &gt; 0' in html or 'ΔR² > 0' in html
    assert '<script src=' not in html.lower()
    assert '<link href="http' not in html.lower()
    assert 'dataset0055' in html
    assert 'application/json' in html


def test_report_includes_snapshot_of_active_paired_runs(tmp_path):
    _write_summary(
        tmp_path / 'outputs/long-1/eval/lr50_s0_resume_step497990/summary.json',
        step=497990, model=.72, raw=.60, random=.66, wins=5,
    )
    manifest = {
        'snapshot_at': '2026-10-08 16:18 IDT',
        'target_steps': 497990,
        'runs': [{'patches': 48, 'seed': 2, 'step': 700, 'run_id': 'abc123',
                  'state': 'running', 'service': 'spectralfm-paired-p48-s2'}],
    }
    (tmp_path / 'outputs/paired-long').mkdir(parents=True)
    (tmp_path / 'outputs/paired-long/manifest.json').write_text(json.dumps(manifest))

    html = render_html(collect_report_data(tmp_path))

    assert 'Confirmatory comparison in progress' in html
    assert 'abc123' in html
    assert '"step":700' in html
    assert 'r.patches' in html and 'r.seed' in html and 'r.step' in html
