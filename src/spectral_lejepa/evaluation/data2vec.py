"""The previous method (fairseq data2vec-audio) as an evaluation backbone.

The checkpoint is loaded with the previous project's own loader (`code/eval/checkpoint_loader.py`),
read from its pinned commit into `.tmp/legacy_eval`; it returns a HuggingFace Data2VecAudioModel,
which already has the interface `extract_bank` uses. Needs the `data2vec` extra (transformers).
"""
import copy
import importlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

from .legacy import LEGACY_COMMIT, LEGACY_REPO

FILES = ('code/eval/__init__.py', 'code/eval/model.py', 'code/eval/checkpoint_loader.py')


def is_data2vec(path) -> bool:
    """fairseq checkpoints carry 'cfg' and 'model'; ours carry 'config'."""
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    return 'cfg' in state and state['cfg']['model'].get('_name') == 'data2vec_audio'


def load_data2vec(path, root='.tmp/legacy_eval'):
    """(model, ckpt-like dict) for `evaluation.run`: config is empty (per-sample z-score input,
    as fairseq trained it with normalize=True) and step is the fairseq update count."""
    root = Path(root)
    if not (root / FILES[-1]).exists():
        tar = subprocess.check_output(['git', '-C', str(LEGACY_REPO), 'archive', LEGACY_COMMIT, *FILES])
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(root)
    sys.path.insert(0, str(root / 'code'))
    try:
        model = importlib.import_module('eval.checkpoint_loader').CheckpointLoader.from_file(str(path))
    finally:
        sys.path.remove(str(root / 'code'))
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    return model, {'config': {}, 'step': int(state['optimizer_history'][-1]['num_updates'])}


def random_like(model, seed=0):
    """Same architecture, every parameterised module re-initialised from `seed`."""
    torch.manual_seed(seed)
    rnd = copy.deepcopy(model)
    for m in rnd.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    return rnd
