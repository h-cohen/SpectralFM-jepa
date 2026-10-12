from scripts.projection_pilot_report import assess_pilot
from scripts.build_results_report import DATASETS


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
