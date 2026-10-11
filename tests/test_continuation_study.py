from scripts.continuation_study import training_config, evaluation_config
from spectral_lejepa.training.trainer import schedule_lengths
from scripts.continuation_report import assess_study


def test_continuations_share_schedule_and_stream_but_not_loss_weight():
    a, b = training_config(202, 'control'), training_config(202, 'low_sigreg')
    assert a['training']['init_checkpoint'] and a['training']['resume'] is None
    assert schedule_lengths(49799, a['training']) == (99598, 1000)
    assert a['training']['learning_rate'] == 5e-5
    assert a['training']['final_lr_ratio'] == .01
    assert a['training']['lambda_sigreg'] == .05
    assert b['training']['lambda_sigreg'] == .01
    for cfg in (a, b):
        cfg['experiment'].pop('name'); cfg['training'].pop('lambda_sigreg')
        cfg['wandb'].pop('tags')
    assert a == b
    assert training_config(203, 'control')['experiment']['seed'] == 203


def test_evaluation_retains_required_menu_and_final_flat_controls():
    c = evaluation_config('control_s202')
    assert c['evaluation']['flat_blocks'] == [2, 6]
    assert c['evaluation']['random_control'] is True
    assert c['evaluation']['probe_workers'] == 1
    assert c['wandb']['group'] == 'continuation-study'


def test_research_preference_requires_both_seeds_and_preserved_fixed_embedding():
    def scores(gain, wins):
        return {'scorecard': {p: {'mean_delta': gain, 'wins': wins} for p in ('menu', 'fixed_block2')}}
    baseline = scores(.08, 5)
    jobs = {f'{arm}_s{seed}': {'arm': arm, 'seed': seed, 'results': scores(gain, 7)}
            for seed in (202, 203) for arm, gain in [('control', .09), ('low_sigreg', .10)]}
    assert assess_study({'jobs': jobs}, baseline)['menu'] == 'lower_sigreg'
    jobs['low_sigreg_s203']['results']['scorecard']['fixed_block2']['mean_delta'] = .07
    assert assess_study({'jobs': jobs}, baseline)['menu'] == 'duration_only'
    jobs['low_sigreg_s203'].pop('results')
    assert assess_study({'jobs': jobs}, baseline)['menu'] == 'pending'
