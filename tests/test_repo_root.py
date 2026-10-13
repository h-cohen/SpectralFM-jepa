import importlib

import pytest

from spectral_lejepa.utils.io import REPO_ROOT


def test_repo_root_is_the_checkout():
    assert (REPO_ROOT / 'pyproject.toml').is_file() and (REPO_ROOT / 'configs').is_dir()


@pytest.mark.parametrize('module, attr', [('spectral_lejepa.evaluation.permutation', 'ROOT'),
                                          ('spectral_lejepa.training.audit', 'ROOT'),
                                          ('spectral_lejepa.utils.wandb', 'PROJECT_ROOT')])
def test_module_roots_point_at_the_checkout(module, attr):
    assert getattr(importlib.import_module(module), attr) == REPO_ROOT


def test_results_report_defaults_to_the_checkout(monkeypatch, tmp_path):
    from spectral_lejepa.reporting import results

    class Stop(Exception):
        pass

    def capture(root):
        raise Stop(root)

    monkeypatch.setattr(results, 'collect_report_data', capture)
    with pytest.raises(Stop) as e:
        results.main(['--output', str(tmp_path / 'r.html')])
    assert e.value.args[0] == REPO_ROOT
