import numpy as np
import pytest

from spectral_lejepa.data.loader import source_rows


def _index(tmp_path, sources):
    (tmp_path / "index.tsv").write_text("".join(f"{i}\t{s}\tx{i}.wav\n" for i, s in enumerate(sources)))
    return tmp_path


def test_selects_exact_prefix_rows(tmp_path):
    d = _index(tmp_path, ["a", "labeled_regression/dataset0112", "b", "labeled_regression/dataset0112", "labeled_data"])
    rows = np.array([0, 1, 2, 3, 4])
    assert source_rows(d, rows, ["labeled_regression/dataset0112"]).tolist() == [1, 3]
    assert source_rows(d, np.array([0, 3, 4]), ["labeled_regression/dataset0112", "labeled_data"]).tolist() == [3, 4]


def test_unknown_prefix_fails(tmp_path):
    d = _index(tmp_path, ["a", "b"])
    with pytest.raises(ValueError, match="matches no source"):
        source_rows(d, np.array([0, 1]), ["labeled_regression/dataset9999"])


def test_empty_result_fails(tmp_path):
    d = _index(tmp_path, ["a", "b"])
    with pytest.raises(ValueError, match="leaves no training rows"):
        source_rows(d, np.array([0]), ["b"])
