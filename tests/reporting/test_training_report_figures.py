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


def _ablation_job(root, job, design, gain2, erank_top):
    import json
    depth = design[0]
    run = root / job / 'eval' / f'{job}_step9'
    for ds in training_report.DATASETS:
        for sub in ('', 'random_control'):
            d = run / ds / sub
            d.mkdir(parents=True, exist_ok=True)
            fam = {'raw': {'r2_mean': .5},
                   **{f'layer{k}/flat': {'r2_mean': .5 + (gain2 if k == 2 and not sub else 0.)} for k in range(depth + 1)}}
            (d / 'nested_results.json').write_text(json.dumps({'families': fam}))
    geo = lambda r: {'effective_rank': r, 'top_pc_share': .1, 'mean_cosine': .1, 'sigreg': 1.}
    blocks = {str(k): {'trained': geo(erank_top * (k + 1) / (depth + 1)), 'random': geo(1.),
                       'trained_tokens': {'effective_rank': 2., 'top_pc_share': .1}} for k in range(depth + 1)}
    (root / job / 'isotropy.json').write_text(json.dumps({'depth': depth, 'blocks': blocks}))
    (root / job / 'state.json').write_text(json.dumps({'job': job, 'design': list(design)}))


def test_ablation_figures_and_criteria(tmp_path, monkeypatch):
    root = tmp_path / 'abl'
    for job, design, gain, rank in [('lam0_d6_s0', (6, 0., 0), -.01, 5.), ('lam001_d6_s0', (6, .01, 0), .08, 40.),
                                    ('lam005_d6_s0', (6, .05, 0), .08, 90.), ('lam02_d6_s0', (6, .2, 0), .04, 110.),
                                    ('lam005_d6_s1', (6, .05, 1), .07, 90.), ('lam005_d4_s0', (4, .05, 0), .02, 80.),
                                    ('lam005_d4_s1', (4, .05, 1), .06, 80.), ('lam005_d2_s0', (2, .05, 0), -.05, 70.),
                                    ('lam005_d2_s1', (2, .05, 1), -.2, 70.)]:
        _ablation_job(root, job, design, gain, rank)
    monkeypatch.setattr(training_report, 'OUT', tmp_path / 'out')
    (tmp_path / 'out').mkdir()
    res = training_report.ablations(root)
    assert (tmp_path / 'out' / 'fig9_sigreg_ablation.svg').exists()
    assert (tmp_path / 'out' / 'fig10_depth_ablation.svg').exists()
    assert res['criteria'] == {'S1': True, 'S2': True, 'D1': True}
    assert abs(res['jobs']['lam005_d2_s1']['flat'][2] + .2) < 1e-9
    assert len(res['jobs']['lam005_d4_s0']['flat']) == 5


def test_true_vs_predicted_grid(tmp_path, monkeypatch):
    import numpy as np
    rng = np.random.default_rng(0)
    preds = {}
    for ds in ('labeled_data', 'dataset0055'):
        y = rng.normal(5, 2, 40)
        preds[ds] = {'y': y, 'raw spectra': np.stack([y + rng.normal(0, 2, 40)] * 2),
                     'robust ensemble': np.stack([y + rng.normal(0, .5, 40)] * 2)}
    monkeypatch.setattr(training_report, 'OUT', tmp_path)
    best = training_report.true_vs_predicted(preds, 'fig_tvp.svg')
    svg = (tmp_path / 'fig_tvp.svg').read_text()
    assert 'R² =' in svg and 'z-scored' not in svg
    assert best == {'labeled_data': 'robust ensemble', 'dataset0055': 'robust ensemble'}


def test_input_figure_shows_patches_and_mask(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(training_report, 'OUT', tmp_path)
    x = np.random.default_rng(0).normal(size=(3, 245)).astype('float32')
    info = training_report.input_figure(x, ['a', 'b', 'c'])
    assert (tmp_path / 'fig0_input_patches_mask.svg').exists()
    assert info['widths'] == [6] * 5 + [5] * 43
    assert all(len(m) == 36 for m in info['masked'])     # 75% of 48 patches
    assert len({tuple(m) for m in info['masked']}) == 3   # each example gets its own mask


def test_benchmark_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(training_report, 'OUT', tmp_path)
    arms = ('jepa', 'minirocket_s0', 'rocket', 'multirocket', 'pred_avg', 'concat', 'data2vec', 'data2vec_feb25')
    res = {'sets': {ds: {'r2': {'raw': .5, **{a: .5 + .01 * i for i, a in enumerate(arms)}},
                         'vs_raw': {a: {'delta': .01 * i, 'sd': .01} for i, a in enumerate(arms)}}
                    for ds in ('labeled_data', 'dataset0055')}}
    training_report.benchmark_figure(res)
    svg = (tmp_path / 'fig_benchmark.svg').read_text()
    assert all(t in svg for t in ('Raw spectrum', 'data2vec (Feb 25)', 'MiniRocket', 'labeled_data'))
    assert 'ROCKET' not in svg.replace('MiniRocket', '')


def test_error_figures(tmp_path, monkeypatch):
    import numpy as np
    from spectral_lejepa.evaluation.errors import error_metrics
    monkeypatch.setattr(training_report, 'OUT', tmp_path)
    rng = np.random.default_rng(0)
    sets = ['labeled_data', 'dataset0055', 'dataset0106', 'dataset0109',
            'dataset0112', 'dataset0113', 'dataset0114', 'dataset0120']
    oofs = {}
    for s in sets:
        y = rng.normal(size=60)
        oofs[s] = (y, {a: np.stack([y + rng.normal(size=60) * k for _ in range(2)])
                       for a, k in (('raw', .5), ('jepa', .3), ('minirocket_s0', .4))})
    res = {'sets': {s: {'errors': error_metrics(*oofs[s], n_boot=20)} for s in sets}}
    training_report.error_figures(res, oofs)
    for name in ('fig_errors_nrmse', 'fig_errors_cdf', 'fig_errors_quintile', 'fig_errors_residuals'):
        assert 'JEPA ensemble' in (tmp_path / f'{name}.svg').read_text()


def test_correlation_figure(tmp_path, monkeypatch):
    import numpy as np
    from spectral_lejepa.evaluation.bank import save_bank
    monkeypatch.setattr(training_report, 'OUT', tmp_path)
    rng = np.random.default_rng(0)
    raw, y = rng.random((150, 245)) + 1, rng.normal(size=150)
    save_bank(tmp_path / 'iso.npz', {'layer2': rng.normal(size=(150, 16))}, raw, y, {})
    z = rng.normal(size=(150, 1))
    save_bank(tmp_path / 'col.npz', {'layer2': z @ rng.normal(size=(1, 16)) + .01 * rng.normal(size=(150, 16))}, raw, y, {})
    out = training_report.correlation_figure((('raw', tmp_path / 'iso.npz', None), ('iso', tmp_path / 'iso.npz', 'layer2'),
                                              ('collapsed', tmp_path / 'col.npz', 'layer2'),
                                              ('missing', tmp_path / 'nope.npz', 'layer2')), n_spectra=20)
    assert set(out) == {'raw', 'iso', 'collapsed'}
    assert out['iso']['mean_abs_offdiag'] < .2 < .9 < out['collapsed']['mean_abs_offdiag']
    assert (tmp_path / 'fig_correlation.svg').exists()
