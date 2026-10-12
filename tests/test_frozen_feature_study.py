import numpy as np
import torch

from scripts.frozen_feature_study import extract_features, verify_bank
from spectral_lejepa.models.vit_1d import LeJEPA, EvalBackbone
from spectral_lejepa.evaluation.bank import extract_bank


def tiny_model():
    return LeJEPA(245, 8, 16, 6, 4, 32, 0., 16, 1, 4, 32).eval()


def test_pre_norm_is_the_input_to_final_norm_and_other_readouts_are_unchanged():
    m = tiny_model()
    x = np.random.default_rng(4).normal(size=(5,245)).astype('float32')
    got = extract_features(m,x,device='cpu',batch_size=2)
    old = extract_bank(EvalBackbone(m),x,device='cpu',batch_size=2,readouts=('mean','seg4','flat'))
    assert len(got) == 24
    for key in old: np.testing.assert_allclose(got[key],old[key],atol=1e-6)
    pre = torch.from_numpy(got['layer6_pre_norm/flat']).reshape(5,8,16)
    post = m.encoder.norm(pre).detach().numpy().reshape(5,-1)
    np.testing.assert_allclose(post,got['layer6/flat'],atol=1e-6)
    assert not np.allclose(got['layer6_pre_norm/flat'],got['layer6/flat'])


def test_cached_features_cannot_be_reused_after_row_reordering():
    old={'layer2/flat':np.arange(6,dtype='float32').reshape(3,2)}
    verify_bank(old,old)
    import pytest
    with pytest.raises(ValueError,match='layer2/flat'):
        verify_bank(old,{'layer2/flat':old['layer2/flat'][::-1]})
