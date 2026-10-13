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


def test_top_level_help_exits_zero_and_lists_commands():
    r = subprocess.run([sys.executable, '-m', 'spectral_lejepa', '--help'], capture_output=True, text=True)
    assert r.returncode == 0 and 'evaluate' in r.stdout


@pytest.mark.parametrize('name', ['training-report', 'training-report-html', 'noise-calibration', 'perturbation'])
def test_help_never_runs_a_job_for_commands_without_options(name):
    env = {**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': ''}
    r = subprocess.run([sys.executable, '-m', 'spectral_lejepa', name, '--help'],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    assert importlib.import_module(COMMANDS[name]).__doc__.strip().splitlines()[0] in r.stdout


def test_commands_without_options_reject_unknown_arguments():
    r = subprocess.run([sys.executable, '-m', 'spectral_lejepa', 'training-report-html', '--bogus'],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
