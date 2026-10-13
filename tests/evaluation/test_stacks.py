import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch

from spectral_lejepa.evaluation.stacks import load_stack_subset


def write_manifest(tmp_path, n_rows=60, short_at=7):
    wav = tmp_path / 'wav'
    wav.mkdir()
    names = []
    for i in range(n_rows):
        x = np.linspace(0, 1, 240 if i == short_at else 245, dtype=np.float32) * (i + 1)
        sf.write(wav / f'f{i}.wav', x, 16000, subtype='FLOAT')
        names.append(f'f{i}.wav')
    (tmp_path / 'valid.tsv').write_text(str(wav) + '\n' + ''.join(f'{n}\t245\n' for n in names))
    return tmp_path


def test_short_wavs_are_zero_padded(tmp_path):
    out = load_stack_subset(str(write_manifest(tmp_path)), 'valid', 60)
    assert out['raw'].shape == (60, 245)
    assert out['raw'][7, 240:].tolist() == [0.0] * 5


@pytest.mark.slow
def test_selection_matches_the_legacy_loader(tmp_path, monkeypatch):
    shim = types.ModuleType('torchaudio')
    shim.load = lambda p: (torch.from_numpy(sf.read(p, dtype='float32')[0])[None], 16000)
    monkeypatch.setitem(sys.modules, 'torchaudio', shim)
    from spectral_lejepa.evaluation.legacy import legacy_module
    old = legacy_module('code/eval/data_loader.py').load_manifest_subset(str(write_manifest(tmp_path)), 'valid', 30)
    new = load_stack_subset(str(tmp_path), 'valid', 30)
    assert new['filename'] == old['filename'].tolist()
    assert new['stack_idx'].tolist() == old['stack_idx'].tolist()
    full = [r for r in old['data'] if len(r) == 245]
    assert np.allclose(new['raw'][[len(r) == 245 for r in old['data']]], np.array(full))
