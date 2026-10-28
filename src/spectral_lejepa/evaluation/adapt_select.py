"""Study 2: choose each set's adaptation step by development (inner-CV) score and assemble adapted members.

    python -m spectral_lejepa adapt-select --runs-json R.json --sets dataset0112 ... --out DIR

R.json is {member: {step: evaluate dir}}; writes DIR/choice.json and DIR/<member>/<set> symlinks that `ensemble --banks` reads.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def dev_score(results_json, family="layer2/flat"):
    chosen = json.loads(Path(results_json).read_text())["families"][family]["chosen"]
    return float(np.mean([c["inner_r2"] for c in chosen]))


def select_steps(runs, sets, family="layer2/flat"):
    """runs {member: {step: eval dir}}: per set, the step with the highest member-mean dev score (ties: smaller)."""
    steps = sorted(set.intersection(*(set(v) for v in runs.values())))
    out = {}
    for s in sets:
        score = {st: np.mean([dev_score(Path(runs[m][st]) / s / "nested_results.json", family) for m in runs])
                 for st in steps}
        out[s] = max(steps, key=lambda st: (score[st], -st))
    return out


def composite(runs, choice, out_dir):
    """Symlink out_dir/<member>/<set> to the chosen step's set dir (bank.npz and random_control come along)."""
    for m, by_step in runs.items():
        for s, st in choice.items():
            dst = Path(out_dir) / m / s
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink(Path(by_step[st]).resolve() / s, dst)


def main(argv=None):
    p = argparse.ArgumentParser(prog="adapt-select", description=__doc__.splitlines()[0])
    p.add_argument("--runs-json", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--family", default="layer2/flat")
    a = p.parse_args(argv)
    raw = json.loads(Path(a.runs_json).read_text())
    runs = {m: {int(st): d for st, d in by.items()} for m, by in raw.items()}
    choice = select_steps(runs, a.sets, a.family)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "choice.json").write_text(json.dumps(choice, indent=1))
    composite(runs, choice, out)
    print(json.dumps(choice))


if __name__ == "__main__":
    main()
