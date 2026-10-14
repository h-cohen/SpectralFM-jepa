import json

import numpy as np

from spectral_lejepa.evaluation.ensemble import ensemble_predictions, member_oof


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
