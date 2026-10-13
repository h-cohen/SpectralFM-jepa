from spectral_lejepa.reporting.training_report_html import convert


def test_bottom_line_paragraphs_become_callouts():
    _, _, body = convert('# T\n\n## 1. A\n\n**Bottom line.** It works.\n\nPlain text.\n')
    assert '<p class="bottom"><strong>Bottom line.</strong> It works.</p>' in body
    assert '<p>Plain text.</p>' in body
