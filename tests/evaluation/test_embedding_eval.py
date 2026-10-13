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


def test_evaluate_suite_flag_dispatches_without_a_checkpoint(monkeypatch):
    from spectral_lejepa.evaluation import embedding_eval, run
    seen = {}
    monkeypatch.setattr(embedding_eval, 'run_suite', lambda cfg: seen.setdefault('name', cfg['experiment']['name']))
    run.main(['--suite', 'embedding', '--config', 'configs/eval_embedding.yaml'])
    assert seen['name'] == 'embedding_eval'


def test_probe_suite_still_requires_a_checkpoint():
    from spectral_lejepa.evaluation import run
    with pytest.raises(SystemExit):
        run.main(['--config', 'configs/eval.yaml'])


def test_evaluate_model_on_synthetic_data():
    from spectral_lejepa.evaluation.embedding_eval import evaluate_model

    class Legacy:   # minimal stand-in with the legacy attribute
        NOISE_CONFIGS = {'gaussian_std': lambda z: z + np.random.normal(0, .01, size=z.shape)}

    rng = np.random.default_rng(0)
    data = {'noise_sets': {'toy': rng.normal(2, 1, (6, 245)).astype(np.float32)}, 'stack_sets': {}, 'panel': None}
    cfg = {'evaluation': {'device': 'cpu', 'batch_size': 4, 'k': 5, 'noise_seed': 0}}
    ck = {'data': {'normalization': 'global'}, 'derived': {'global_stats': {'mean': 2.0, 'std': 1.0}}}
    out = evaluate_model(build_model(MODEL, 245), ck, data, cfg, {'noise': Legacy, 'sim': None})
    for r in ('block2_mean', 'block2_flat'):
        assert 0.9 < out['noise']['toy'][r]['gaussian_std'] <= 1.0
