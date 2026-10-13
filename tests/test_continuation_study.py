from scripts.continuation_report import assess_study


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
