from spectral_lejepa.reporting.sections.projection_pilot import assess_pilot
from spectral_lejepa.reporting.results import DATASETS
import json
import pytest
from spectral_lejepa.reporting.sections.projection_pilot import collect_pilot


def test_pilot_cannot_advance_from_one_good_seed_or_incomplete_results():
    def result(gain):
        return {'scorecard':{'fixed_block2':{'mean_delta':gain,'wins':7}},
                'sets':{d:{'policies':{'fixed_block2':{'r2':.3}}} for d in DATASETS}}
    jobs={f'{a}_s{s}':{'results':result(.09 if a=='control' else .10)} for s in (301,302) for a in ('control','projected')}
    assert assess_pilot({'jobs':jobs})['status']=='research-preference-supported'
    jobs['projected_s302']['results']['scorecard']['fixed_block2']['mean_delta']=.091
    assert assess_pilot({'jobs':jobs})['status']=='stop-or-inconclusive'
    jobs['projected_s302'].pop('results')
    assert assess_pilot({'jobs':jobs})['status']=='pending'


def test_repeated_dataset_regression_blocks_good_macro_gain():
    jobs={f'{a}_s{s}':{'results':{'scorecard':{'fixed_block2':{'mean_delta':.09 if a=='control' else .10,'wins':7}},
          'sets':{d:{'policies':{'fixed_block2':{'r2':.3}}} for d in DATASETS}}}
          for s in (301,302) for a in ('control','projected')}
    for seed in (301,302):
        jobs[f'projected_s{seed}']['results']['sets']['dataset0113']['policies']['fixed_block2']['r2']=.27
    result=assess_pilot({'jobs':jobs})
    assert result['status']=='stop-or-inconclusive'
    assert result['repeated_regressions_vs_control']==['dataset0113']


def test_pilot_depth_cannot_show_another_checkpoint(tmp_path):
    base=tmp_path/'outputs/projection-pilot';job=base/'control_s301';p=job/'eval/run/dataset0055'
    p.mkdir(parents=True)
    (base/'manifest.json').write_text(json.dumps({'jobs':{'control_s301':{} }}))
    (job/'state.json').write_text(json.dumps({'checkpoint_sha256':'expected'}))
    (p/'nested_results.json').write_text(json.dumps({'meta':{'pretraining_checkpoint_sha256':'wrong'}}))
    with pytest.raises(ValueError,match='checkpoint differs'):collect_pilot(tmp_path)
