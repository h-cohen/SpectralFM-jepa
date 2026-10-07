"""W&B integration: every run records its resolved config, code version and environment.

The API key is never read from or written to the repository. W&B finds it in the
WANDB_API_KEY environment variable or ~/.netrc. WANDB_MODE=disabled (set by the tests)
or `wandb.enabled: false` turns every helper here into a no-op.
"""
from __future__ import annotations

import os
import platform
import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def git_info(repo_dir=None) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo_dir or PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "HEAD"),
                "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": True}


def software_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "numpy", "scikit-learn", "scipy", "lightly", "wandb"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def wandb_enabled(cfg) -> bool:
    return bool(cfg["wandb"]["enabled"]) and os.environ.get("WANDB_MODE", "").lower() != "disabled"


def _check_credentials():
    """Fail before any work starts instead of hanging on an interactive login prompt."""
    if os.environ.get("WANDB_MODE", "").lower() == "offline" or os.environ.get("WANDB_API_KEY"):
        return
    netrc = Path.home() / ".netrc"
    if netrc.is_file() and "api.wandb.ai" in netrc.read_text():
        return
    raise RuntimeError("W&B is enabled but no credentials were found: export WANDB_API_KEY in "
                       "your shell (never put it in the repo), or set wandb.enabled=false.")


def init_run(cfg, job_type, extra_config=None, name=None):
    if not wandb_enabled(cfg):
        return None
    _check_credentials()
    import wandb

    w = cfg["wandb"]
    config = {**cfg, **git_info(), "software": software_versions(), "hostname": socket.gethostname(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              **(extra_config or {})}
    run = wandb.init(project=w["project"], entity=w["entity"], group=w["group"], job_type=job_type,
                      tags=list(w["tags"]), name=name, config=config)
    if job_type == "pretrain":
        run.define_metric("optimizer_step")
        for prefix in ("train/*", "valid/*", "representation/*", "progress/*"):
            run.define_metric(prefix, step_metric="optimizer_step")
        run.define_metric("valid/loss", summary="min")
        run.define_metric("progress/percent_complete", summary="last")
    return run


def training_metrics(metrics, tcfg, total_steps):
    """Measured loss terms plus their actual objective weights and absolute progress."""
    out = dict(metrics)
    for prefix in ("train", "valid"):
        mse, reg, glob = (f"{prefix}/{key}" for key in ("mse_loss", "sigreg_loss", "global_loss"))
        if mse in out and reg in out:
            visreg = tcfg.get("regularizer") == "visreg"
            lam = tcfg["visreg_lambda"] if visreg else tcfg["lambda_sigreg"]
            out[f"{prefix}/prediction_contribution"] = out[mse] * (1 - lam if visreg else 1)
            out[f"{prefix}/regularizer_contribution"] = out[reg] * lam
            out[f"{prefix}/global_contribution"] = out.get(glob, 0.) * tcfg.get("global_weight", 0.)
    step = out.get("train/global_step")
    if step is not None and total_steps:
        out["progress/percent_complete"] = 100 * step / total_steps
        out["progress/steps_remaining"] = max(0, total_steps - step)
    return out


def log(run, metrics, step=None):
    if run is not None:
        if getattr(run, "job_type", None) == "pretrain":
            metrics = training_metrics(metrics, run.config["training"], run.config["derived"]["total_steps"])
            if step is not None:
                metrics["optimizer_step"] = step
        run.log(metrics, step=step)


def log_figure(run, key, fig, step=None):
    if run is not None:
        import wandb
        run.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_checkpoint(run, path, artifact_name, aliases, metadata):
    if run is None:
        return
    import wandb
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=list(aliases))


def alert(run, title, text):
    if run is not None:
        run.alert(title=title, text=text)


def finish(run):
    if run is not None:
        run.finish()
