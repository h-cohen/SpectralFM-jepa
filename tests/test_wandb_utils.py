import pytest

from spectral_lejepa.utils import wandb as wb

CFG = {"wandb": {"enabled": True, "entity": None, "project": "p", "group": None,
                 "job_type": "pretrain", "tags": ["t"]}}


def test_git_info_in_this_repo():
    info = wb.git_info()
    assert len(info["git_commit"]) == 40
    assert isinstance(info["git_dirty"], bool)
    assert info["git_branch"]


def test_git_info_outside_a_repo(tmp_path):
    assert wb.git_info(tmp_path)["git_commit"] == "unknown"


def test_disabled_by_env_returns_none():
    # conftest sets WANDB_MODE=disabled
    assert wb.init_run(CFG, job_type="pretrain") is None
    wb.log(None, {"a": 1.0}, step=0)
    wb.finish(None)


def test_disabled_by_config(monkeypatch):
    monkeypatch.delenv("WANDB_MODE")
    assert wb.init_run({"wandb": {**CFG["wandb"], "enabled": False}}, job_type="pretrain") is None


def test_missing_credentials_fail_fast(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_MODE")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        wb.init_run(CFG, job_type="pretrain")


def test_software_versions():
    v = wb.software_versions()
    assert v["torch"].startswith("2.8.0") and v["lightly"] == "1.5.26"
