import json
import pytest

from scripts.production_analysis_report import collect_analysis


def test_analysis_does_not_mix_another_checkpoint_with_the_registered_one(tmp_path):
    base = tmp_path / 'outputs/production-analysis'
    base.mkdir(parents=True)
    (base / 'manifest.json').write_text(json.dumps({'checkpoint_sha256': 'expected'}))
    ds = base / 'all-blocks/eval/other_step497990/dataset0055'
    ds.mkdir(parents=True)
    (ds / 'nested_results.json').write_text(json.dumps({'families': {}, 'meta': {'pretraining_checkpoint_sha256': 'other'}}))
    (ds / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='checkpoint'):
        collect_analysis(tmp_path)
