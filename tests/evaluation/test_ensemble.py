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
