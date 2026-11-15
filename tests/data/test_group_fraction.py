import math

import numpy as np
import pytest

from spectral_lejepa.data.loader import exclude_source_rows, group_fraction, sample_group_keys


def _index(tmp_path, entries):
    (tmp_path / "index.tsv").write_text("".join(f"{i}\t{s}\t/x/{n}\n" for i, (s, n) in enumerate(entries)))
    return tmp_path


def _sets(spec):
    """spec: {source: n_groups}; three components per group; non-measured sources get unparsable names."""
    entries = []
    for source, n in spec.items():
        if source.startswith("labeled_regression"):
            ds = source.split("/")[-1]
            entries += [(source, f"{ds}_comp{c}_spec_{g}.wav") for g in range(n) for c in (0, 3, 7)]
        else:
            entries += [(source, f"spectra0000_batch1_spec_{g}.wav") for g in range(n)]
    return entries


SPEC = {"labeled_regression/dataset0055": 200, "labeled_regression/dataset0106": 40, "single_channel_all": 30}
N = 3 * 240 + 30


def test_fraction_one_keeps_all_rows(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    rows, _ = group_fraction(d, np.arange(N), 1.0)
    assert np.array_equal(rows, np.arange(N))


def test_fractions_are_nested(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    g = [set(group_fraction(d, np.arange(N), f)[1].tolist()) for f in (0.015, 0.0625, 0.25)]
    assert g[0] < g[1] < g[2]


def test_fraction_stratified_per_source(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    _, groups = group_fraction(d, np.arange(N), 0.25)
    assert sum(k.startswith("dataset0055/") for k in groups) == math.ceil(0.25 * 200)
    assert sum(k.startswith("dataset0106/") for k in groups) == 10


def test_tiny_fraction_keeps_one_group_per_source(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    _, groups = group_fraction(d, np.arange(N), 0.001)
    assert sorted(k.split("/")[0] for k in groups) == ["dataset0055", "dataset0106"]


def test_components_of_a_group_stay_together(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    rows, groups = group_fraction(d, np.arange(N), 0.0625)
    measured = rows[rows < 720]
    keys = sample_group_keys(d, measured)
    assert set(keys.tolist()) == set(groups.tolist())
    assert len(measured) == 3 * len(groups)


def test_non_prefix_rows_untouched(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    rows, _ = group_fraction(d, np.arange(N), 0.015)
    assert set(range(720, N)) <= set(rows.tolist())


def test_seed_changes_selection(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    a = group_fraction(d, np.arange(N), 0.25, seed=0)[1]
    assert not np.array_equal(a, group_fraction(d, np.arange(N), 0.25, seed=1)[1])


def test_exclude_source_rows_drops_prefix(tmp_path):
    d = _index(tmp_path, _sets(SPEC))
    out = exclude_source_rows(d, np.arange(N), ["labeled_regression/dataset0106"])
    assert np.array_equal(out, np.concatenate([np.arange(600), np.arange(720, N)]))
    with pytest.raises(ValueError, match="matches no source"):
        exclude_source_rows(d, np.arange(N), ["labeled_data"])
