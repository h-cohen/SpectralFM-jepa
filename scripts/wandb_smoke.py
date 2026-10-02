"""W&B smoke test: authenticate from the environment, log git info, a config and a fake loss curve.

  WANDB_API_KEY must already be exported in your shell.
  uv run python scripts/wandb_smoke.py [--entity ENTITY]
"""
import argparse
import math
import sys

from spectral_lejepa.utils import wandb as wb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entity", default=None)
    ap.add_argument("--project", default="spectralfm-lejepa")
    a = ap.parse_args()
    cfg = {"wandb": {"enabled": True, "entity": a.entity, "project": a.project, "group": None,
                     "job_type": "diagnostic", "tags": ["smoke"]},
           "smoke": {"steps": 20}}
    run = wb.init_run(cfg, job_type="diagnostic", name="wandb-smoke")
    if run is None:
        sys.exit("W&B is disabled (WANDB_MODE=disabled?) -- nothing to test")
    for step in range(cfg["smoke"]["steps"]):
        wb.log(run, {"train/loss": math.exp(-step / 5) + 0.05}, step=step)
    print("run url:   ", run.url)
    print("git_commit:", run.config["git_commit"], "dirty:", run.config["git_dirty"])
    wb.finish(run)


if __name__ == "__main__":
    main()
