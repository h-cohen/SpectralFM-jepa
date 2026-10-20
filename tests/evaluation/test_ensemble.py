import json

import numpy as np

import pytest

from spectral_lejepa.evaluation.bank import save_bank
from spectral_lejepa.evaluation.ensemble import ensemble_predictions, fuse, fuse_set, member_oof


def make_member(root, name, value):
    d = root / name
    bank = d / 'eval' / 'run_x_step10' / 'dataset0055'
    bank.mkdir(parents=True)
    (d / 'state.json').write_text(json.dumps({'checkpoint': str(root / 'training' / 'run_x' / 'c.pt')}))
    np.savez(bank / 'nested_oof.npz', y=np.arange(3.), raw=np.zeros((2, 3)), **{'layer2/flat': np.full((2, 3), value)})
    return d


def test_member_oof_reads_the_bank_of_the_trained_checkpoint(tmp_path):
    d = make_member(tmp_path, 'm1', 1.0)
    assert member_oof(d, 'dataset0055')['layer2/flat'].shape == (2, 3)


def test_ensemble_is_the_equal_weight_mean(tmp_path):
    oofs = [member_oof(make_member(tmp_path, f'm{v}', v), 'dataset0055') for v in (1.0, 3.0)]
    assert np.allclose(ensemble_predictions(oofs), 2.0)


def make_bank(root, name, value, rand):
    """An evaluate run dir: <run>/<dataset>/nested_oof.npz and random_control/nested_oof.npz."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    d = root / name / 'dataset0055'
    (d / 'random_control').mkdir(parents=True)
    raw = np.stack([y + rng.normal(scale=1.0, size=40)] * 2)
    np.savez(d / 'nested_oof.npz', y=y, raw=raw, **{'layer2/flat': np.stack([y + value * rng.normal(size=40)] * 2)})
    np.savez(d / 'random_control' / 'nested_oof.npz', y=y, raw=raw,
             **{'layer2/flat': np.stack([y + rand * rng.normal(size=40)] * 2)})
    return root / name


def test_score_ensemble_beats_raw_when_members_are_better(tmp_path):
    from spectral_lejepa.evaluation.ensemble import score_ensemble
    banks = [make_bank(tmp_path, f'm{i}', .3, 2.0) for i in range(3)]
    out = score_ensemble(banks)
    s = out['sets']['dataset0055']
    assert s['vs_raw']['delta'] > 0 and s['vs_random']['delta'] > 0
    assert out['summary']['raw_wins'] == 1 and out['summary']['n_sets'] == 1


def test_score_ensemble_refuses_mismatched_labels(tmp_path):
    import pytest
    from spectral_lejepa.evaluation.ensemble import score_ensemble
    a = make_bank(tmp_path, 'a', .3, 2.0)
    b = make_bank(tmp_path, 'b', .3, 2.0)
    z = dict(np.load(b / 'dataset0055' / 'nested_oof.npz'))
    z['y'] = z['y'] + 1
    np.savez(b / 'dataset0055' / 'nested_oof.npz', **z)
    with pytest.raises(ValueError, match='labels'):
        score_ensemble([a, b])


def make_bank_npz(path, y, signal, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    bank = {'layer2/flat': (y[:, None] * signal + rng.normal(size=(n, 8))).astype(np.float32),
            'layer2': rng.normal(size=(n, 4)).astype(np.float32),
            'layer6': (y[:, None] * signal / 2 + rng.normal(size=(n, 4))).astype(np.float32)}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_bank(path, bank, rng.normal(size=(n, 245)).astype(np.float32), y, {})


def make_run(root, name, y, seed):
    run = root / name
    make_bank_npz(run / 'dataset0055' / 'bank.npz', y, 2.0, seed)
    make_bank_npz(run / 'dataset0055' / 'random_control' / 'bank.npz', y, 0.0, seed + 100)
    return run


def test_fuse_set_returns_paired_arms_and_weights(tmp_path):
    y = np.random.default_rng(0).normal(size=60)
    runs = [make_run(tmp_path, f'm{i}', y, i) for i in range(2)]
    out = fuse_set([r / 'dataset0055' / 'bank.npz' for r in runs])
    assert set(out['oof']) == {'raw', 'avg', 'stack'} and out['oof']['stack'].shape == (2, 60)
    w = out['weights'][0]
    assert {d['arm'] for d in w} == {f'm{i}:{h}' for i in range(2) for h in ('layer2/flat', 'layer2', 'layer6')}
    assert sum(d['weight'] for d in w) == pytest.approx(1.0)
    assert out['r2']['stack'] > out['r2']['raw']


def test_fuse_writes_json_with_random_rows_and_weights(tmp_path):
    y = np.random.default_rng(0).normal(size=60)
    runs = [make_run(tmp_path, f'm{i}', y, i) for i in range(2)]
    res = fuse(runs, tmp_path / 'out')
    s = res['sets']['dataset0055']
    assert {'stack_vs_avg', 'avg_vs_raw', 'stack_vs_raw', 'avg_vs_random', 'stack_vs_random', 'mean_weights'} <= set(s)
    assert (tmp_path / 'out' / 'fusion.json').exists()
    saved = json.loads((tmp_path / 'out' / 'fusion.json').read_text())['sets']['dataset0055']
    assert len(saved['fold_weights']) == 10 and len(saved['fold_weights'][0]) == 6
    assert (tmp_path / 'out' / 'dataset0055' / 'fusion_oof.npz').exists()


def test_fuse_set_missing_head_names_member_and_arm(tmp_path):
    y = np.random.default_rng(0).normal(size=40)
    f = tmp_path / 'm0' / 'bank.npz'
    f.parent.mkdir(parents=True)
    save_bank(f, {'layer2': np.zeros((40, 4), np.float32)}, np.zeros((40, 245), np.float32), y, {})
    with pytest.raises(ValueError, match=r"layer2/flat.*m0"):
        fuse_set([f])


def test_fuse_set_refuses_mismatched_labels(tmp_path):
    rng = np.random.default_rng(0)
    a, b = tmp_path / 'a' / 'bank.npz', tmp_path / 'b' / 'bank.npz'
    make_bank_npz(a, rng.normal(size=40), 1.0, 0)
    make_bank_npz(b, rng.normal(size=40), 1.0, 1)
    with pytest.raises(ValueError, match='labels'):
        fuse_set([a, b])


def make_select_bank(path, y, seed, with_layer1=True):
    rng = np.random.default_rng(seed)
    n = len(y)
    bank = {'layer2/flat': (y[:, None] + rng.normal(size=(n, 8))).astype(np.float32)}
    if with_layer1:
        bank['layer1/flat'] = (2 * y[:, None] + rng.normal(size=(n, 8))).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_bank(path, bank, rng.normal(size=(n, 245)).astype(np.float32), y, {})


def test_fuse_set_select_picks_a_candidate_readout_per_member(tmp_path):
    y = np.random.default_rng(0).normal(size=60)
    files = [tmp_path / f'm{i}' / 'bank.npz' for i in range(2)]
    for i, f in enumerate(files):
        make_select_bank(f, y, i)
    cands = ('layer2/flat', 'layer1/flat', 'layer1+2/flat')
    out = fuse_set(files, heads=(), select=cands)
    assert out['oof']['select'].shape == (2, 60)
    picks = {c['arm'].split(':', 1)[1] for fold in out['select_chosen'] for c in fold}
    assert picks <= set(cands)
    assert 'stack' not in out['oof']


def test_fuse_set_select_missing_readout_raises(tmp_path):
    y = np.random.default_rng(0).normal(size=40)
    f = tmp_path / 'm0' / 'bank.npz'
    make_select_bank(f, y, 0, with_layer1=False)
    with pytest.raises(ValueError, match='layer1/flat'):
        fuse_set([f], heads=(), select=('layer2/flat', 'layer1+2/flat'))


def test_fuse_with_select_reports_select_rows(tmp_path):
    y = np.random.default_rng(0).normal(size=50)
    runs = []
    for i in range(2):
        make_select_bank(tmp_path / f'm{i}' / 'dataset0055' / 'bank.npz', y, i)
        make_select_bank(tmp_path / f'm{i}' / 'dataset0055' / 'random_control' / 'bank.npz', y, i + 50)
        runs.append(tmp_path / f'm{i}')
    s = fuse(runs, tmp_path / 'out', heads=(), select=('layer2/flat', 'layer1/flat'))['sets']['dataset0055']
    assert {'select_vs_avg', 'select_vs_raw', 'select_vs_random', 'avg_vs_raw'} <= set(s)
    assert 'stack_vs_avg' not in s
