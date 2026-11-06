import numpy as np
import pytest

from spectral_lejepa.data.loader import group_holdout, sample_group_keys


def _index(tmp_path, entries):
    """entries: (source, wav basename) per row."""
    (tmp_path / "index.tsv").write_text("".join(f"{i}\t{s}\t/x/{n}\n" for i, (s, n) in enumerate(entries)))
    return tmp_path


def _sets(spec):
    """spec: {source: n_groups}; three components per group."""
    entries = []
    for source, n_groups in spec.items():
        ds = source.split("/")[-1]
        entries += [(source, f"{ds}_comp{c}_spec_{g}.wav") for g in range(n_groups) for c in (0, 3, 7)]
    return entries


def test_group_keys_parse_dataset_and_index(tmp_path):
    d = _index(tmp_path, [("labeled_regression/dataset0055", "dataset0055_comp0_spec_7.wav"),
                          ("labeled_regression/dataset0055", "dataset0055_comp3_spec_7.wav"),
                          ("labeled_regression/dataset0106", "dataset0106_comp0_spec_7.wav")])
    assert sample_group_keys(d, np.array([0, 1, 2])).tolist() == ["dataset0055/7", "dataset0055/7", "dataset0106/7"]


def test_group_keys_reject_unparsable_name(tmp_path):
    d = _index(tmp_path, [("labeled_regression/dataset0055", "dataset0055_comp0_spec_7.wav"),
                          ("single_channel_all", "spectra0000_batch1_spec_5.wav")])
    with pytest.raises(ValueError, match="row 1"):
        sample_group_keys(d, np.array([0, 1]))


def test_holdout_is_group_disjoint_and_stratified(tmp_path):
    d = _index(tmp_path, _sets({"labeled_regression/dataset0055": 40, "labeled_regression/dataset0106": 40}))
    rows = np.arange(240)
    train, valid = group_holdout(d, rows, 0.05, seed=0)
    assert np.intersect1d(train, valid).size == 0
    assert np.array_equal(np.union1d(train, valid), rows)
    keys = sample_group_keys(d, rows)
    assert not set(keys[train]) & set(keys[valid])
    held = set(keys[valid])
    assert sum(k.startswith("dataset0055/") for k in held) == 2
    assert sum(k.startswith("dataset0106/") for k in held) == 2
    assert len(valid) == 4 * 3


def test_holdout_is_seeded(tmp_path):
    d = _index(tmp_path, _sets({"labeled_regression/dataset0055": 40}))
    rows = np.arange(120)
    a, b = group_holdout(d, rows, 0.05, seed=0), group_holdout(d, rows, 0.05, seed=0)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], group_holdout(d, rows, 0.05, seed=1)[1])


def test_single_group_source_stays_in_train(tmp_path):
    d = _index(tmp_path, _sets({"labeled_regression/dataset0055": 40, "labeled_regression/dataset0111": 1}))
    train, valid = group_holdout(d, np.arange(123), 0.05, seed=0)
    assert set(range(120, 123)) <= set(train.tolist())
    assert not set(range(120, 123)) & set(valid.tolist())


def test_min_one_group_per_source(tmp_path):
    d = _index(tmp_path, _sets({"labeled_regression/dataset0118": 4}))
    _, valid = group_holdout(d, np.arange(12), 0.05, seed=0)
    assert len(valid) == 3
