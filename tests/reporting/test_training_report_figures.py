from spectral_lejepa.reporting import training_report


def test_candidates_figure_has_no_threshold_annotation(tmp_path, monkeypatch):
    monkeypatch.setattr(training_report, 'OUT', tmp_path)
    training_report.candidates()
    svg = (tmp_path / 'fig4_candidates.svg').read_text()
    assert 'contract' not in svg and '5/8' not in svg


def test_depth_profile_figure_from_member_banks(tmp_path, monkeypatch):
    import json
    import numpy as np
    sets = training_report.DATASETS
    run = tmp_path / 'm1' / 'eval' / 'r_step1'
    rng = np.random.default_rng(0)
    for ds in sets:
        for sub, gain in (('', .1), ('random_control', 0.)):
            d = run / ds / sub
            d.mkdir(parents=True, exist_ok=True)
            fam = {'raw': {'r2_mean': .5}, **{f'layer{k}/flat': {'r2_mean': .5 + gain * (k == 2)} for k in range(7)}}
            (d / 'nested_results.json').write_text(json.dumps({'families': fam}))
            if ds == 'labeled_data':
                np.savez(d / 'bank.npz', **{f'bank__layer{k}': rng.normal(size=(30, 1, 1, 8)) for k in range(7)})
    monkeypatch.setattr(training_report, 'OUT', tmp_path / 'out')
    (tmp_path / 'out').mkdir()
    stats = training_report.depth_profile({'m1': str(tmp_path / 'm1' / 'eval' / 'r_step*')})
    assert (tmp_path / 'out' / 'fig8_depth_isotropy.svg').exists()
    assert stats['trained']['m1']['delta'][2] > stats['trained']['m1']['delta'][6]
    assert len(stats['trained']['m1']['effective_rank']) == 7
