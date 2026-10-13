import json
import pytest
from spectral_lejepa.reporting.sections.frozen_features import collect_frozen,render_frozen


def test_pending_study_does_not_claim_results_and_rejects_mixed_head_checkpoint(tmp_path):
    base=tmp_path/'outputs/frozen-feature-study';base.mkdir(parents=True)
    (base/'manifest.json').write_text(json.dumps({'jobs':{'model':{'checkpoint_sha256':'expected'}}}))
    data=collect_frozen(tmp_path)
    result=render_frozen(data)
    assert 'queued' in result and '0/8' in result
    p=base/'model/dataset0055';p.mkdir(parents=True)
    (p/'heads.json').write_text(json.dumps({'source':{'checkpoint_sha256':'wrong'}}))
    with pytest.raises(ValueError,match='checkpoint'):collect_frozen(tmp_path)
