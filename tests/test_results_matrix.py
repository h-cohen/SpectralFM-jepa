import json
import pytest

from scripts.results_matrix import collect_matrix, render_matrix


def test_matrix_keeps_dataset_columns_and_recovers_random_uncertainty(tmp_path):
    root = tmp_path / 'outputs/long-1/eval/p24_s0_run_step497990'
    root.mkdir(parents=True)
    point = {'n': 25, 'embedding_r2': .7, 'embedding_sd': .03,
             'raw_r2': .6, 'raw_sd': .02, 'random_control': {'embedding_r2': .5}}
    (root / 'summary.json').write_text(json.dumps({'sets': {'dataset0055': point}}))
    rc = root / 'dataset0055/random_control'
    rc.mkdir(parents=True)
    (rc / 'nested_results.json').write_text(json.dumps({'families': {'embedding': {'bootstrap_sd': .04}}}))
    result = collect_matrix(tmp_path)
    rows = result['evaluations'][0]['rows']
    assert rows[0]['cells']['dataset0055'] == {'r2': .7, 'sd': .03, 'n': 25}
    assert rows[1]['cells']['dataset0055']['r2'] == .6
    assert rows[2]['cells']['dataset0055']['sd'] == .04
    assert 'dataset0106' not in rows[0]['cells']
    rendered = render_matrix(result)
    assert '0.7000 ± 0.0300' in rendered
    assert 'Raw spectra' in rendered
    assert 'Bootstrap SD' in rendered
    assert '<small class="sample-count">n=25</small>' in rendered
    assert 'class="win-cell"' in rendered


def test_seed_aggregation_uses_sample_sd_and_does_not_invent_missing_sd(tmp_path):
    from scripts.results_matrix import aggregate_seeds
    rows = [{'seed': i, 'cells': {'dataset0055': {'r2': x, 'sd': None, 'n': 25}}}
            for i, x in enumerate([.5, .6, .7])]
    result = aggregate_seeds(rows, 24)
    assert result['cells']['dataset0055']['r2'] == pytest.approx(.6)
    assert result['cells']['dataset0055']['sd'] == pytest.approx(.1)
    assert 'Seed SD' in result['uncertainty']


def test_six_final_runs_are_complete_with_deduplicated_controls(tmp_path):
    from scripts.build_results_report import DATASETS
    for patches in (24, 48):
        for seed in range(3):
            folder = tmp_path / f'outputs/long-1/eval/p{patches}_s{seed}_run_step497990'
            folder.mkdir(parents=True)
            value = (.7 if patches == 24 else .8) + seed * .01
            point = {'n': 30, 'embedding_r2': value, 'embedding_sd': .03,
                     'raw_r2': .6, 'raw_sd': .02,
                     'random_control': {'embedding_r2': .5, 'embedding_sd': .04}}
            (folder / 'summary.json').write_text(json.dumps({'sets': {d: point for d in DATASETS}}))
    result = collect_matrix(tmp_path)
    assert result['complete']
    assert len(result['final_rows']) == 9
    assert len(result['evaluations']) == 6
    assert result['conclusion']['prefer_48']
    assert result['seed_means'][0]['cells']['dataset0055']['sd'] == pytest.approx(.01)
    assert result['final_rows'][0]['cells']['dataset0055']['sd'] == .02
