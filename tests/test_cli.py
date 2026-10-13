import importlib
import subprocess
import sys

import pytest

from spectral_lejepa.__main__ import COMMANDS


@pytest.mark.parametrize('name', sorted(COMMANDS))
def test_every_command_module_has_main(name):
    assert callable(importlib.import_module(COMMANDS[name]).main)


@pytest.mark.parametrize('name', ['pretrain', 'evaluate', 'export', 'gate', 'report', 'pack'])
def test_help_works_from_the_repo_root(name):
    r = subprocess.run([sys.executable, '-m', 'spectral_lejepa', name, '--help'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 'usage' in r.stdout.lower()


def test_unknown_command_lists_commands():
    r = subprocess.run([sys.executable, '-m', 'spectral_lejepa', 'nope'], capture_output=True, text=True)
    assert r.returncode == 2 and 'evaluate' in r.stderr


def test_training_report_fetch_subcommand_reaches_fetch(monkeypatch):
    tr = importlib.import_module('spectral_lejepa.reporting.training_report')
    called = []
    monkeypatch.setattr(tr, 'fetch', lambda: called.append('fetch'))
    monkeypatch.setattr(tr, 'figures', lambda: called.append('figures'), raising=False)
    monkeypatch.setattr(sys, 'argv', ['spectral_lejepa training-report', 'fetch'])
    tr.main()
    assert called == ['fetch']
