"""Screen: one-factor LeJEPA arms (short pretrain + labeled_data eval) run across GPUs,
tabulated against a matched control with a computed win/loss rule, logged to W&B (table and summary metrics; no artifacts).

  uv run python -m scripts.screen --gpus 2,4,5,6 [--config configs/screen.yaml] [--max_evals 2]

--gpus are CUDA device indices, one arm at a time per device. On Geoffrey GPU 3 is broken
and CUDA skips it, so CUDA index k >= 3 is nvidia-smi index k+1.
Decision rule: floor = max(|R2(control_s0) - R2(control_s1)|, 2 * sd(paired delta vs
control_s0)); win if delta > floor, loss if delta < -floor, else neutral.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import yaml

from spectral_lejepa.evaluation.nested import paired_delta
from spectral_lejepa.training.checkpoint import load_checkpoint
from spectral_lejepa.utils import wandb as wb

REPO = Path(__file__).resolve().parents[1]
EVAL_SET = "labeled_data"
COLUMNS = ["arm", "status", "verdict", "embedding_r2", "embedding_sd", "delta_vs_control", "delta_sd",
           "floor", "delta_vs_ref_mean_only", "best_block", "valid_mse_loss", "valid_sigreg_loss",
           "effective_rank", "stage", "pretrain_run_id"]
FOOTNOTE = ("Wins are candidates: the floor uses one control-seed spread and a row bootstrap, not "
            "pretraining-seed variance. valid_* losses depend on each arm's own masking/token count and "
            "effective_rank on its width, so those columns are not comparable across arms.")


def find_checkpoint(root: Path, name: str) -> Path:
    run_dir = re.compile(rf"^{re.escape(name)}_\d{{8}}-\d{{6}}$")
    found = sorted(p for p in root.glob(f"{name}_*/checkpoint_last.pt") if run_dir.match(p.parent.name))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name}_* run dir under {root}, found {len(found)} checkpoint dirs")
    return found[0]


def run_arm(name, overrides, scfg, gpu, eval_slot):
    """Pretrain then evaluate one arm in subprocesses; returns a status dict (never raises)."""
    root = Path(scfg["output_dir"])
    log = root / f"{name}.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    pretrain = [sys.executable, "scripts/pretrain.py", *scfg["shared_overrides"],
                f"experiment.name={name}", *overrides]
    with open(log, "w") as f:
        if subprocess.run(pretrain, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode:
            return {"status": "failed", "stage": "pretrain", "log": str(log)}
        try:
            ckpt = find_checkpoint(root, name)
        except RuntimeError as e:
            return {"status": "failed", "stage": "locate", "error": str(e), "log": str(log)}
        evaluate = [sys.executable, "-m", "scripts.evaluate", "--checkpoint", str(ckpt),
                    "--config", scfg["eval_config"]]
        with eval_slot:
            if subprocess.run(evaluate, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode:
                return {"status": "failed", "stage": "eval", "log": str(log)}
    return {"status": "ok", "checkpoint": str(ckpt), "log": str(log)}


def arm_row(name, status, eval_root):
    row = {"arm": name, **status}
    if status["status"] != "ok":
        return row
    ckpt_path = Path(status["checkpoint"])
    ckpt = load_checkpoint(ckpt_path)
    set_dir = eval_root / f"{ckpt_path.parent.name}_step{ckpt['step']}" / EVAL_SET
    s = json.loads((set_dir / "summary.json").read_text())
    m = ckpt["metrics"]
    row.update(n=s["n"], embedding_r2=s["embedding_r2"], embedding_sd=s["embedding_sd"],
               best_block=s["best_block"], oof=str(set_dir / "nested_oof.npz"),
               delta_vs_ref_mean_only=s.get("vs_ref_mean_only", {}).get("embedding", {}).get("delta"),
               valid_mse_loss=m.get("valid/mse_loss"), valid_sigreg_loss=m.get("valid/sigreg_loss"),
               effective_rank=m.get("representation/effective_rank"), pretrain_run_id=ckpt.get("wandb_run_id"))
    return row


def safe_row(name, status, eval_root):
    try:
        return arm_row(name, status, eval_root)
    except Exception as e:
        return {"arm": name, **status, "status": "failed", "stage": "summary", "error": str(e)}


def status_from_disk(name, scfg, eval_root):
    root = Path(scfg["output_dir"])
    try:
        ckpt = find_checkpoint(root, name)
    except RuntimeError as e:
        return {"status": "failed", "stage": "missing", "error": str(e)}
    if not list(eval_root.glob(f"{ckpt.parent.name}_step*/{EVAL_SET}/summary.json")):
        return {"status": "failed", "stage": "missing", "error": f"no {EVAL_SET} eval summary for {ckpt.parent.name}"}
    return {"status": "ok", "checkpoint": str(ckpt), "log": str(root / f"{name}.log")}


def decide(rows, control="control_s0", repeat="control_s1"):
    ok = {r["arm"]: r for r in rows if r["status"] == "ok"}
    have_controls = control in ok and repeat in ok
    if have_controls:
        base = np.load(ok[control]["oof"])
        y, P0 = base["y"], base["embedding"]
        spread = abs(ok[control]["embedding_r2"] - ok[repeat]["embedding_r2"])
    for r in rows:
        if r["status"] != "ok":
            r["verdict"] = "failed"
        elif r["arm"] == control:
            r["verdict"] = "control"
        elif not have_controls:
            r["verdict"] = "no-control"
        else:
            arm = np.load(r["oof"])
            if not np.array_equal(arm["y"], y):
                raise ValueError(f"arm {r['arm']} was evaluated on different rows than {control}")
            d = paired_delta(y, arm["embedding"], P0)
            floor = max(spread, 2 * d["sd"])
            r.update(delta_vs_control=d["delta"], delta_sd=d["sd"])
            if r["arm"] == repeat:
                r["verdict"] = "control"
                continue
            r.update(floor=floor, verdict="win" if d["delta"] > floor else "loss" if d["delta"] < -floor else "neutral")
    return rows


def to_markdown(rows):
    def cell(v, c):
        if isinstance(v, float):
            return f"{v:+.3f}" if c in ("delta_vs_control", "delta_vs_ref_mean_only") else f"{v:.3f}"
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(cell(r.get(c), c) for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines) + "\n\n" + FOOTNOTE + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/screen.yaml")
    ap.add_argument("--gpus", help="comma-separated CUDA indices (one worker each)")
    ap.add_argument("--max_evals", type=int, default=2, help="concurrent evaluations (CPU-bound)")
    ap.add_argument("--summarize_only", action="store_true", help="rebuild results from disk, run no arms")
    a = ap.parse_args(argv)
    if not a.gpus and not a.summarize_only:
        ap.error("--gpus is required unless --summarize_only")
    scfg = yaml.safe_load(open(a.config))
    root = Path(scfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    eval_root = Path(yaml.safe_load(open(scfg["eval_config"]))["experiment"]["output_dir"])
    run = wb.init_run({"wandb": scfg["wandb"], "screen": scfg}, job_type="screen", name=scfg["name"] + ("-summary" if a.summarize_only else ""))

    if a.summarize_only:
        status = {name: status_from_disk(name, scfg, eval_root) for name in scfg["arms"]}
    else:
        jobs = queue.Queue()
        for item in scfg["arms"].items():
            jobs.put(item)
        status, eval_slot = {}, threading.Semaphore(a.max_evals)

        def worker(gpu):
            while True:
                try:
                    name, overrides = jobs.get_nowait()
                except queue.Empty:
                    return
                print(f"[screen] {name} on CUDA {gpu}", flush=True)
                try:
                    status[name] = run_arm(name, overrides, scfg, gpu, eval_slot)
                except Exception as e:
                    status[name] = {"status": "failed", "stage": "worker", "error": str(e)}
                print(f"[screen] {name}: {status[name]['status']}", flush=True)

        threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    missing = {"status": "failed", "stage": "worker", "error": "no status recorded"}
    rows = decide([safe_row(name, status.get(name, missing), eval_root) for name in scfg["arms"]])
    (root / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    (root / "results.md").write_text(to_markdown(rows))
    print(to_markdown(rows))
    if run is not None:
        import wandb
        run.log({"screen/results": wandb.Table(columns=COLUMNS, data=[[r.get(c) for c in COLUMNS] for r in rows])})
        for r in rows:
            for key in ("embedding_r2", "delta_vs_control"):
                if r.get(key) is not None:
                    run.summary[f"{r['arm']}/{key}"] = r[key]
    wb.finish(run)


if __name__ == "__main__":
    main()
