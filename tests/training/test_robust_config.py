from spectral_lejepa.config import load_config


def test_robust_continuation_config_is_the_registered_recipe():
    c = load_config('configs/robust_continuation.yaml')
    t = c['training']
    assert (t['max_steps'], t['learning_rate'], t['final_lr_ratio'], t['lambda_sigreg']) == (49799, 5e-5, .01, .05)
    assert t['robust'] == {'weight': .1, 'block': 2, 'prob': .5, 'sigma_max': 0.20181098959496105,
                           'offset_max': .05, 'slope_max': .05, 'scale_max': .05, 'shift_max': 1}
    assert t['init_checkpoint'] is None   # set on the command line: training.init_checkpoint=<stage-1 checkpoint>
    assert 'projector' not in c['model'] and 'visreg_slices' not in t


def test_robust_continuation_matches_pretraining_except_the_continuation_keys():
    a, b = load_config('configs/final_model.yaml'), load_config('configs/robust_continuation.yaml')
    assert a['model'] == b['model'] and a['data'] == b['data'] and a['masking'] == b['masking']
