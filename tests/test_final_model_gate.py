from scripts.final_model_gate import assess
from scripts.build_results_report import DATASETS


def test_gate_requires_all_eight_sets_and_does_not_claim_deployment_release():
    sets = {d: {'embedding_r2': .7, 'raw_r2': .6,
                'random_control': {'embedding_r2': .65},
                'canary': {'block': {'passed': True}, 'raw': {'passed': True}}} for d in DATASETS}
    result = assess({'sets': sets})
    assert result['benchmark_qualified']
    assert result['wins'] == 8
    assert not result['deployment_release_ready']
    del sets['dataset0055']
    assert not assess({'sets': sets})['benchmark_qualified']


def test_raw_wins_alone_cannot_pass_if_random_control_is_better():
    sets = {d: {'embedding_r2': .7, 'raw_r2': .6,
                'random_control': {'embedding_r2': .75},
                'canary': {'block': {'passed': True}, 'raw': {'passed': True}}} for d in DATASETS}
    assert not assess({'sets': sets})['benchmark_qualified']
