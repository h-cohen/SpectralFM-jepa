"""Parity of the torch probe backend with stored sklearn nested-CV results.

  CUDA_DEVICE_ORDER=PCI_BUS_ID uv run python -m scripts.parity_gpu_eval <eval_dir> <set> [<set> ...] \
      [--device cuda:0 [cuda:1 ...]] [--workers 2]

For each set: load <eval_dir>/<set>/bank.npz, rerun run_nested(backend="torch") and compare with
<set>/nested_results.json -- every family's r2_mean, and every (fold, family) recipe choice. A flipped
choice is listed with both inner R² values (a near-tie when they agree to ~1e-6).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.nested import run_nested, write_json

R2_TOL = 1e-4


def _picks(c):
    return [(m["arm"], m["norm"], m["probe"], m["inner_r2"]) for m in c.get("members", [c])]


def compare(ours, ref):
    rows, flips = [], []
    for fam, r in ref["families"].items():
        o = ours["families"][fam]
        rows.append((fam, o["r2_mean"], r["r2_mean"], abs(o["r2_mean"] - r["r2_mean"])))
        for co, cr in zip(o["chosen"], r["chosen"]):
            po, pr = _picks(co), _picks(cr)
            if [p[:3] for p in po] != [p[:3] for p in pr]:
                flips.append({"family": fam, "repeat": co["repeat"], "fold": co["fold"],
                              "ours": po, "ref": pr,
                              "inner_r2_gap": max(abs(a[3] - b[3]) for a, b in zip(po, pr))})
    return rows, flips


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("eval_dir")
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--device", nargs="+", default=["cuda"], help="one or more devices (folds round-robin)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2, help="concurrent folds per device (threads, one CUDA stream each)")
    ap.add_argument("--out", help="directory to write our nested results (<set>.json) to")
    a = ap.parse_args(argv)
    ok = True
    for name in a.sets:
        d = Path(a.eval_dir) / name
        bank, raw, y, _ = load_bank(d / "bank.npz", readouts=("mean", "seg4", "flat"))
        ref = json.loads((d / "nested_results.json").read_text())
        t0 = time.time()
        ours, oof = run_nested(bank, raw, y, seed=a.seed, backend="torch", device=a.device, n_jobs=a.workers)
        wall = time.time() - t0
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            write_json(Path(a.out) / f"{name}.json", ours)
        peak = (f" peak GPU mem (first device)={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                if torch.cuda.is_available() and a.device[0].startswith("cuda") else "")
        rows, flips = compare(ours, ref)
        oof_line = ""
        if (d / "nested_oof.npz").exists():
            ref_oof = np.load(d / "nested_oof.npz")
            gaps = {fam: float(np.max(np.abs(oof[fam] - ref_oof[fam])) / np.std(y)) for fam in oof if fam in ref_oof.files}
            oof_line = (f"   max |Δ oof| / sd(y): raw {gaps.get('raw', float('nan')):.1e}, "
                        f"worst {max(gaps, key=gaps.get)} {max(gaps.values()):.1e}")
        worst = max(r[3] for r in rows)
        ok &= worst < R2_TOL
        n_choices = sum(len(f["chosen"]) for f in ref["families"].values())
        print(f"== {name}: n={len(y)} arms={len(bank) + 1} wall={wall:.1f}s  max |Δ r2_mean|={worst:.2e} "
              f"choice flips={len(flips)}/{n_choices}{peak}", flush=True)
        if oof_line:
            print(oof_line)
        for fam, o, r, diff in rows:
            print(f"   {fam:24s} torch {o:+.6f}  sklearn {r:+.6f}  |Δ| {diff:.1e}")
        for f in flips:
            print(f"   flip {f['family']} r{f['repeat']}f{f['fold']}: inner-R² gap {f['inner_r2_gap']:.1e}\n"
                  f"       torch   {f['ours']}\n       sklearn {f['ref']}")
    print("PASS" if ok else "FAIL", f"(every family r2_mean within {R2_TOL})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
