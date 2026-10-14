import json

import pytest
import torch

from spectral_lejepa.models.export import export_ensemble, script_encoder
from spectral_lejepa.models.vit_1d import EvalBackbone
from spectral_lejepa.training.checkpoint import load_model, sha256_file


def test_bundle_has_one_scripted_encoder_per_member_and_a_manifest(make_checkpoint, tmp_path):
    members = {'a': make_checkpoint('a', 0), 'b': make_checkpoint('b', 1)}
    m = export_ensemble(members, tmp_path / 'bundle')
    assert m['format_version'] == 1 and m['readout'] == 'layer2/flat'
    assert [x['name'] for x in m['members']] == ['a', 'b']
    assert m['output']['shape'] == ['N', 2, 12288]
    for x in m['members']:
        f = tmp_path / 'bundle' / x['file']
        assert x['sha256'] == sha256_file(f) and x['source_sha256'] == sha256_file(members[x['name']])
    assert json.loads((tmp_path / 'bundle' / 'manifest.json').read_text()) == m


def test_scripted_member_equals_block2_flat_of_the_checkpoint(make_checkpoint, tmp_path):
    ck = make_checkpoint('a')
    export_ensemble({'a': ck}, tmp_path / 'bundle')
    enc = torch.jit.load(str(tmp_path / 'bundle' / 'member_a.pt'))
    model, _ = load_model(ck)
    raw = torch.randn(3, 245)
    with torch.no_grad():
        expected = EvalBackbone(model)((raw - .3784) / .5126).hidden_states[2].flatten(1)
        torch.testing.assert_close(enc(raw), expected, atol=1e-5, rtol=1e-5)


def test_refuses_to_overwrite_a_bundle_without_force(make_checkpoint, tmp_path):
    export_ensemble({'a': make_checkpoint('a')}, tmp_path / 'bundle')
    with pytest.raises(FileExistsError):
        export_ensemble({'a': make_checkpoint('a')}, tmp_path / 'bundle')
    export_ensemble({'a': make_checkpoint('a')}, tmp_path / 'bundle', force=True)


def test_script_encoder_returns_the_checkpoint(make_checkpoint):
    compiled, ckpt = script_encoder(make_checkpoint('a'))
    assert ckpt['step'] == 7 and compiled(torch.randn(1, 245)).shape == (1, 12288)
