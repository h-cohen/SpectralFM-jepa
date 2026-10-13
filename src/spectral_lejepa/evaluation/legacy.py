"""The previous project's evaluation code, executed from its pinned commit (never its working tree)."""
import subprocess
import types
from pathlib import Path

LEGACY_REPO = Path('/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged')
LEGACY_COMMIT = '6237feb5335af2f5336be28c68a17802050f2abd'


def legacy_module(relpath, overrides=None):
    source = subprocess.check_output(['git', '-C', str(LEGACY_REPO), 'show', f'{LEGACY_COMMIT}:{relpath}'], text=True)
    module = types.ModuleType(f'legacy_{Path(relpath).stem}')
    module.__file__ = f'{LEGACY_REPO}@{LEGACY_COMMIT[:7]}:{relpath}'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    module.__dict__.update(overrides or {})
    return module
