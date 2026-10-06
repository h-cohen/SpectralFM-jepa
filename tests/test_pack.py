import json
import os
import subprocess

import numpy as np
import pytest
import soundfile as sf

from scripts import pack_spectra

PARENT_PY = "/mnt5/home/hadar/nova/.venv-labelprobe/bin/python"


def _make_source(root, name, rows):
    """rows: list of arrays; writes wavs + train.tsv (all but last row) + valid.tsv (last row, plus one repeat)."""
    d = root / name
    wavs = d / "wavs"
    wavs.mkdir(parents=True)
    names = []
    for i, r in enumerate(rows):
        n = f"{i}.wav"
        sf.write(wavs / n, np.asarray(r, dtype=np.float32), 16000, subtype="FLOAT")
        names.append(n)
    (d / "train.tsv").write_text(str(wavs) + "\n" + "".join(f"{n}\t245\n" for n in names[:-1]))
    (d / "valid.tsv").write_text(str(wavs) + "\n" + f"{names[-1]}\t245\n{names[0]}\t245\n")  # names[0] duplicate
    return d


@pytest.fixture
def packed(tmp_path):
    rng = np.random.default_rng(1)
    a = [rng.normal(size=245) for _ in range(5)] + [np.full(245, 3.0)]          # row 5 constant
    b = [rng.normal(size=245) for _ in range(4)] + [np.full(245, np.nan)]       # row 4 NaN
    sa, sb = _make_source(tmp_path, "a", a), _make_source(tmp_path, "b", b)
    pk = tmp_path / "pk"
    pk.mkdir()
    prows = rng.normal(size=(3, 245)).astype(np.float32)
    np.save(pk / "pickle_rows.npy", prows)
    (pk / "pickle_names.tsv").write_text("".join(f"pickle:x\tx_comp0_row{i}\n" for i in range(3)))
    out = tmp_path / "out"
    pack_spectra.main(["--out", str(out), "--sources", f"a={sa}", f"b={sb}", "--workers", "2",
                       "--pickle-dir", str(pk)])
    return out, prows


def test_pack_outputs(packed):
    out, prows = packed
    x = np.load(out / "spectra.npy", mmap_mode="r")
    assert x.shape == (6 + 5 + 3, 245) and x.dtype == np.float32
    idx = [l.rstrip("\n").split("\t") for l in open(out / "index.tsv")]
    assert [int(r[0]) for r in idx] == list(range(14))
    assert [r[1] for r in idx] == ["a"] * 6 + ["b"] * 5 + ["pickle:x"] * 3
    np.testing.assert_array_equal(x[11:], prows)  # pickle rows last
    for r in np.random.default_rng(0).choice(11, 3, replace=False):
        w, _ = sf.read(idx[r][2], dtype="float32")
        np.testing.assert_array_equal(x[r], w)
    drop = np.load(out / "drop_rows.npy")
    assert sorted(np.flatnonzero(drop)) == [5, 10]
    kept = np.asarray(x[~drop], dtype=np.float64)
    st = json.load(open(out / "stats.json"))
    np.testing.assert_allclose(st["mean"], kept.mean(), rtol=1e-6)
    np.testing.assert_allclose(st["std"], kept.std(), rtol=1e-6)
    assert st["n_rows"] == 14 and st["n_dropped"] == 2
    assert st["counts"] == {"a": 6, "b": 5, "pickle:x": 3}
    v = np.load(out / "valid_rows.npy")
    assert len(v) == 12 and (np.diff(v) > 0).all() and not drop[v].any()
    expected = np.sort(np.random.default_rng(0).choice(np.flatnonzero(~drop), 12, replace=False))
    np.testing.assert_array_equal(v, expected)


def test_wrong_length_wav_raises(tmp_path):
    d = _make_source(tmp_path, "a", [np.zeros(245), np.ones(245)])
    bad = d / "wavs" / "0.wav"
    sf.write(bad, np.arange(200, dtype=np.float32), 16000, subtype="FLOAT")
    with pytest.raises(Exception, match="0.wav"):
        pack_spectra.main(["--out", str(tmp_path / "o"), "--sources", f"a={d}", "--workers", "1"])


needs_parent = pytest.mark.skipif(not os.path.exists(PARENT_PY), reason="parent python absent")


def _pack_pickles(tmp_path, n_comp, n_cols):
    """Make a tiny DataFrame pickle and run pack_pickles on it, both with the parent python (it has pandas)."""
    pkl = tmp_path / "t.pkl"
    make = ("import numpy as np, pandas as pd; "
            f"cols=[f'component_{{c}}:{{i}}' for c in range({n_comp}) for i in range({n_cols})]; "
            f"pd.DataFrame(np.random.default_rng(0).normal(size=(2, len(cols))), columns=cols).to_pickle(r'{pkl}')")
    subprocess.run([PARENT_PY, "-c", make], check=True)
    return subprocess.run([PARENT_PY, "scripts/pack_pickles.py", "--out", str(tmp_path / "o"), str(pkl)],
                          capture_output=True, text=True)


@needs_parent
def test_pack_pickles_rejects_244_columns(tmp_path):
    r = _pack_pickles(tmp_path, 1, 244)
    assert r.returncode != 0 and "244" in r.stderr


@needs_parent
def test_pack_pickles_ok(tmp_path):
    r = _pack_pickles(tmp_path, 2, 245)
    assert r.returncode == 0, r.stderr
    assert np.load(tmp_path / "o" / "pickle_rows.npy").shape == (4, 245)
    assert open(tmp_path / "o" / "pickle_names.tsv").readline().rstrip("\n") == "pickle:t\tt_comp0_row0"
