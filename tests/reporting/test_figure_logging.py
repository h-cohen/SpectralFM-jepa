import errno
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from spectral_lejepa.utils.wandb import log_figure


def test_full_media_disk_warns_and_closes_figure(monkeypatch):
    def image(fig):
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Image=image))
    fig = plt.figure()
    with pytest.warns(UserWarning, match='Skipping.*disk'):
        log_figure(SimpleNamespace(log=lambda *a, **k: None), 'representation/spectrum', fig, 75000)
    assert not plt.fignum_exists(fig.number)


def test_other_logging_errors_are_not_hidden(monkeypatch):
    def image(fig):
        raise ValueError('invalid figure')
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Image=image))
    fig = plt.figure()
    with pytest.raises(ValueError, match='invalid figure'):
        log_figure(SimpleNamespace(log=lambda *a, **k: None), 'figure', fig)
    assert not plt.fignum_exists(fig.number)
