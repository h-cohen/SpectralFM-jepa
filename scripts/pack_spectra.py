"""Pack every spectrum into one float32 memmap [N, 245] with a row index, drop mask, global stats and hold-out.

  uv run python -m scripts.pack_spectra --out /mnt5/home/hadar/nova/data/spectra_all [--sources NAME=MANIFEST_DIR ...] [--workers 16]

Two environments: unconverted pickles are first flattened by scripts/pack_pickles.py in the parent environment
(pandas), giving DIR/pickle_rows.npy + pickle_names.tsv; this script (project environment) packs the WAVs of each
source (deduplicated union of train.tsv and valid.tsv, train first), then appends the pickle rows last.

Outputs in DIR: spectra.npy, index.tsv (row, source, name), drop_rows.npy (constant or non-finite rows; kept in the
array, excluded from training), stats.json (global mean/std over kept rows, counts), valid_rows.npy (10,000 kept
rows, rng(0), sorted). Reads manifests only; never writes under /mnt5/noy.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from spectral_lejepa.data.loader import SEQUENCE_LENGTH, read_manifest

DATA_ROOT = "/mnt5/noy/SpectralFM/fairseq/data/nova_data"
CHUNK = 20_000
N_VALID = 10_000


def default_sources():
    names = ["single_channel_all", "multi_channel", "sampled_data", "labeled_data"]
    dirs = {n: f"{DATA_ROOT}/{n}" for n in names}
    for d in sorted(glob.glob(f"{DATA_ROOT}/labeled_regression/dataset*")):
        if os.path.exists(f"{d}/train.tsv"):
            dirs[f"labeled_regression/{os.path.basename(d)}"] = d
    return dirs


def source_paths(manifest_dir):
    """Deduplicated train + valid wav paths, manifest order, train first."""
    paths = []
    for split in ("train", "valid"):
        f = os.path.join(manifest_dir, f"{split}.tsv")
        if os.path.exists(f):
            paths += read_manifest(f)
    return list(dict.fromkeys(paths))


def read_chunk(args):
    start, paths = args
    out = np.empty((len(paths), SEQUENCE_LENGTH), np.float32)
    for i, p in enumerate(paths):
        x, _ = sf.read(p, dtype="float32")
        if x.shape != (SEQUENCE_LENGTH,):
            raise ValueError(f"{p}: expected {SEQUENCE_LENGTH} samples, got shape {x.shape}")
        out[i] = x
    return start, out


def stream_stats(x, keep):
    """Global mean/std over kept rows via float64 sums, chunk by chunk."""
    n = s = ss = 0.0
    for a in range(0, len(x), CHUNK):
        c = np.asarray(x[a:a + CHUNK][keep[a:a + CHUNK]], dtype=np.float64)
        n += c.size
        s += c.sum()
        ss += (c * c).sum()
    mean = s / n
    return mean, float(np.sqrt(ss / n - mean * mean))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="*", default=None, help="NAME=MANIFEST_DIR (default: all nova_data sources)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pickle-dir", default=None, help="dir holding pickle_rows.npy/pickle_names.tsv (default: --out)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = dict(s.split("=", 1) for s in args.sources) if args.sources else default_sources()

    index, counts = [], {}
    for name, d in sources.items():
        paths = source_paths(d)
        counts[name] = len(paths)
        index += [(name, p) for p in paths]
    n_wav = len(index)
    pk_dir = Path(args.pickle_dir or out)
    pk_rows = None
    if (pk_dir / "pickle_rows.npy").exists():
        pk_rows = np.load(pk_dir / "pickle_rows.npy")
        pk_names = [l.rstrip("\n").split("\t") for l in open(pk_dir / "pickle_names.tsv")]
        assert len(pk_names) == len(pk_rows)
        for s, n in pk_names:
            counts[s] = counts.get(s, 0) + 1
        index += [tuple(r) for r in pk_names]

    x = np.lib.format.open_memmap(out / "spectra.npy", mode="w+", dtype=np.float32,
                                  shape=(len(index), SEQUENCE_LENGTH))
    wav_paths = [p for _, p in index[:n_wav]]
    jobs = [(a, wav_paths[a:a + CHUNK]) for a in range(0, n_wav, CHUNK)]
    with Pool(args.workers) as pool:
        for start, arr in pool.imap_unordered(read_chunk, jobs):
            x[start:start + len(arr)] = arr
    if pk_rows is not None:
        x[n_wav:] = pk_rows
    x.flush()
    with open(out / "index.tsv", "w") as f:
        f.writelines(f"{i}\t{s}\t{n}\n" for i, (s, n) in enumerate(index))

    drop = np.zeros(len(x), bool)
    for a in range(0, len(x), CHUNK):
        c = np.asarray(x[a:a + CHUNK])
        drop[a:a + CHUNK] = (c.std(axis=1) < 1e-6) | ~np.isfinite(c).all(axis=1)
    np.save(out / "drop_rows.npy", drop)
    kept = np.flatnonzero(~drop)
    valid = np.sort(np.random.default_rng(0).choice(kept, min(N_VALID, len(kept)), replace=False))
    np.save(out / "valid_rows.npy", valid)
    mean, std = stream_stats(x, ~drop)
    json.dump({"mean": mean, "std": std, "counts": counts, "n_rows": len(x), "n_dropped": int(drop.sum())},
              open(out / "stats.json", "w"), indent=1)
    print(f"{len(x)} rows, {int(drop.sum())} dropped, mean {mean:.4f} std {std:.4f}")


if __name__ == "__main__":
    main()
