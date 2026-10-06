"""Flatten unconverted feature pickles into float32 rows (runs in the PARENT environment, which has pandas).

  /mnt5/home/hadar/nova/.venv-labelprobe/bin/python scripts/pack_pickles.py --out DIR a.pkl b.pkl ...

Components are found with the parent converter's parse_components; every component of every row becomes
one 245-point spectrum. Writes DIR/pickle_rows.npy [M, 245] and DIR/pickle_names.tsv (`pickle:<stem>\\t<stem>_comp<C>_row<i>`),
which scripts/pack_spectra.py (project environment) appends to the packed array.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PARENT_SCRIPTS = "/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged/fairseq/scripts"
SEQUENCE_LENGTH = 245


def pickle_rows(pkl, parse_components):
    stem = Path(pkl).stem
    with open(pkl, "rb") as f:
        df = pickle.load(f)
    rows, names = [], []
    for comp, cols in parse_components(df).items():
        if len(cols) != SEQUENCE_LENGTH:
            raise ValueError(f"{pkl}: component {comp} has {len(cols)} columns, expected {SEQUENCE_LENGTH}")
        x = df[cols].to_numpy(dtype=np.float32)
        rows.append(x)
        names += [f"{stem}_comp{comp}_row{i}" for i in range(len(x))]
    return rows, [(f"pickle:{stem}", n) for n in names]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("pkls", nargs="+")
    args = ap.parse_args(argv)
    sys.path.insert(0, PARENT_SCRIPTS)
    from convert_features_to_wav_per_component import parse_components

    rows, names = [], []
    for pkl in args.pkls:
        r, n = pickle_rows(pkl, parse_components)
        rows += r
        names += n
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pickle_rows.npy", np.concatenate(rows).astype(np.float32))
    (out / "pickle_names.tsv").write_text("".join(f"{s}\t{n}\n" for s, n in names))
    print(f"{len(names)} rows from {len(args.pkls)} pickles")


if __name__ == "__main__":
    main()
