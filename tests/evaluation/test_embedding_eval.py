import numpy as np
import pytest

from spectral_lejepa.evaluation.embedding_eval import block2_readouts, stack_similarity
from spectral_lejepa.models.vit_1d import build_model

MODEL = {'num_patches': 48, 'dim': 16, 'depth': 6, 'heads': 4, 'mlp_dim': 32, 'dropout': 0.0,
         'predictor_dim': 16, 'predictor_depth': 1, 'predictor_heads': 4, 'predictor_mlp_dim': 32}


def test_block2_readouts_are_flat_and_mean_of_the_same_states():
    m = build_model(MODEL, 245)
    x = np.random.default_rng(0).normal(size=(5, 245)).astype(np.float32)
    r = block2_readouts(m, x, 'cpu', batch_size=2)
    assert r['block2_flat'].shape == (5, 48 * 16) and r['block2_mean'].shape == (5, 16)
    assert np.allclose(r['block2_flat'].reshape(5, 48, 16).mean(1), r['block2_mean'], atol=1e-6)


@pytest.mark.slow
def test_stack_similarity_feeds_the_given_embeddings_in_row_order():
    from spectral_lejepa.evaluation.legacy import legacy_module
    sim = legacy_module('code/eval/evaluations/embedding_similarity.py')
    rng = np.random.default_rng(0)
    stack = np.repeat(np.arange(4), [10, 10, 10, 7])     # a short last stack
    z = rng.normal(size=(len(stack), 245))
    emb = np.eye(4)[stack] + .01 * rng.normal(size=(len(stack), 4))   # perfectly stack-separated
    out = stack_similarity(sim, z, stack, emb, k=5)
    assert out['embedding_stack_match_rate'] == pytest.approx(1.0)
    assert 0 <= out['input_stack_match_rate'] < .5
