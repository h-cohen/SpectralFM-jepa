from spectral_lejepa.reporting import training_report


def test_candidates_figure_has_no_threshold_annotation(tmp_path, monkeypatch):
    monkeypatch.setattr(training_report, 'OUT', tmp_path)
    training_report.candidates()
    svg = (tmp_path / 'fig4_candidates.svg').read_text()
    assert 'contract' not in svg and '5/8' not in svg
