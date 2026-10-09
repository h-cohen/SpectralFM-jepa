import json

from scripts.readout_variants import collect_variants, render_variants


def test_fixed_readouts_use_nested_family_uncertainty_and_leave_missing_layers_absent(tmp_path):
    folder = tmp_path / 'outputs/long-1/eval/p48_s0_run_step497990'
    (folder / 'dataset0055').mkdir(parents=True)
    (folder / 'summary.json').write_text(json.dumps({'sets': {'dataset0055': {'n': 25, 'raw_r2': .6, 'raw_sd': .02}}}))
    (folder / 'dataset0055/nested_results.json').write_text(json.dumps({'families': {
        'layer2/flat': {'r2_mean': .7, 'bootstrap_sd': .04},
        'layer6': {'r2_mean': .4, 'bootstrap_sd': .08},
        'embedding': {'r2_mean': .9, 'bootstrap_sd': .01}}}))
    result = collect_variants(tmp_path)
    rows = result['rows']
    assert {r['arm'] for r in rows} == {'layer2/flat', 'layer6'}
    flat = next(r for r in rows if r['arm'] == 'layer2/flat')
    assert flat['cells']['dataset0055']['r2'] == .7
    assert flat['cells']['dataset0055']['sd'] == .04
    assert flat['dimension'] == 12288
    assert 'layer6/flat' not in {r['arm'] for r in rows}
    output = render_variants(result)
    assert '0.7000 ± 0.0400' in output
    assert 'win-cell' in output
    assert 'n=25' in output
