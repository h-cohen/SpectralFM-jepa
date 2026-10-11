import json
import pytest

from scripts.production_analysis_report import collect_analysis
from scripts.production_analysis_report import depth_figure


def test_depth_plot_keeps_unmeasured_flat_blocks_as_gaps():
    import numpy as np
    import matplotlib.pyplot as plt
    families = {f'layer{k}/flat': {'r2_mean': .5, 'bootstrap_sd': .01} for k in (2, 6)}
    fig = depth_figure({'dataset0055': {'families': families, 'summary': {'raw_r2': .4}}})
    ax = next(a for a in fig.axes if a.get_title() == '0055')
    flat = next(line for line in ax.lines if line.get_label().startswith('flat'))
    assert list(flat.get_xdata()) == list(range(7))
    values = flat.get_ydata()
    assert np.isfinite(values[[2, 6]]).all()
    assert np.isnan(values[[0, 1, 3, 4, 5]]).all()
    plt.close(fig)


def test_analysis_does_not_mix_another_checkpoint_with_the_registered_one(tmp_path):
    base = tmp_path / 'outputs/production-analysis'
    base.mkdir(parents=True)
    (base / 'manifest.json').write_text(json.dumps({'checkpoint_sha256': 'expected'}))
    ds = base / 'all-blocks/eval/other_step497990/dataset0055'
    ds.mkdir(parents=True)
    (ds / 'nested_results.json').write_text(json.dumps({'families': {}, 'meta': {'pretraining_checkpoint_sha256': 'other'}}))
    (ds / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='checkpoint'):
        collect_analysis(tmp_path)
