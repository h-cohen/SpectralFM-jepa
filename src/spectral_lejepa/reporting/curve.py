"""Study 3a: stage-1 checkpoint curve (dev R2 vs step) and the longer-schedule trigger.

    python -m spectral_lejepa stage1-curve --eval-dir outputs/stage1-curve/eval --final <member final eval dirs...> --out DIR

Writes DIR/curve.json (table and trigger verdict) and DIR/fig_stage1_curve.svg.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from spectral_lejepa.evaluation.adapt_select import dev_score

SETS = ["labeled_data", "dataset0055", "dataset0106", "dataset0109",
        "dataset0112", "dataset0113", "dataset0114", "dataset0120"]


def curve_table(evals, sets, family="layer2/flat"):
    """evals {member: {step: eval dir}} -> {set: {step: mean-over-members dev score}}."""
    steps = sorted(set.intersection(*(set(v) for v in evals.values())))
    return {s: {st: float(np.mean([dev_score(Path(evals[m][st]) / s / "nested_results.json", family)
                                   for m in evals])) for st in steps} for s in sets}


def trigger(table, ref=450000, need=5):
    """True when the final (max) step beats `ref` strictly on at least `need` sets."""
    return sum(t[max(t)] > t[ref] for t in table.values()) >= need


def curve_figure(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, t in table.items():
        ax.plot(list(t), list(t.values()), marker="o", label=s)
    ax.set_xlabel("stage-1 step")
    ax.set_ylabel("dev R2 (inner CV, layer2/flat)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _collect(eval_dir, finals):
    """Curve runs are <member>_<stamp>_step<N> under eval_dir; finals are member eval dirs of the same form."""
    evals = {}
    for d in [*sorted(Path(eval_dir).iterdir()), *map(Path, finals)]:
        m = re.match(r"(.+?)_\d{8}-\d{6}_step(\d+)$", d.name)
        if m:
            evals.setdefault(m[1], {})[int(m[2])] = d
    return evals


def main(argv=None):
    p = argparse.ArgumentParser(prog="stage1-curve", description=__doc__.splitlines()[0])
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--final", nargs="+", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--sets", nargs="+", default=SETS)
    a = p.parse_args(argv)
    table = curve_table(_collect(a.eval_dir, a.final), a.sets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    verdict = trigger(table)
    (out / "curve.json").write_text(json.dumps({"table": table, "trigger": verdict}, indent=1))
    curve_figure(table, out / "fig_stage1_curve.svg")
    print(json.dumps({"trigger": verdict}))


if __name__ == "__main__":
    main()
