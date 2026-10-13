from spectral_lejepa.reporting.training_report_html import convert


def test_bottom_line_paragraphs_become_callouts():
    _, _, body = convert('# T\n\n## 1. A\n\n**Bottom line.** It works.\n\nPlain text.\n')
    assert '<p class="bottom"><strong>Bottom line.</strong> It works.</p>' in body
    assert '<p>Plain text.</p>' in body


def test_cells_where_the_model_beats_raw_are_marked():
    md = ('# T\n\n| Set | Raw R² | Robust ensemble R² | Δ vs raw (± SD) |\n|---|---:|---:|---:|\n'
          '| a | .5 | .6 | +.100 ± .01 |\n| b | .5 | .4 | −.100 ± .01 |\n')
    _, _, body = convert(md)
    win, loss = body.split('<tr>')[2], body.split('<tr>')[3]
    assert win.count('win') == 2 and 'win' not in loss
