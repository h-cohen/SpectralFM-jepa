# Projection-space pilot implementation plan

> Use superpowers:executing-plans inline, with a focused lighter-model diagnostics builder and reviewer.

**Goal:** Launch the approved four-run matched projector pilot with verified loss isolation and automatic evaluation.
**Spec:** `docs/superpowers/specs/2026-10-12-projection-space-pilot.md`.
**Architecture:** Optional regularization-only MLP in the model, explicit SIGReg overrides in the loss, narrowly compatible initialization, paired durable job orchestration and report integration. Existing default behavior/export/gate stay unchanged.

## Constraints

- Source checkpoint SHA256 ec4c041dc4f83aa04e9ec568a335cdf7e27dbc7aa7fac965f2fd1f686b566ff0.
- Modes none/control/mlp; control instantiates the same unused 256→512→128 MLP as projected mode. No BatchNorm/dropout; FP32 forward; no detached gradients.
- Masked MSE and pooled invariance use original backbone tensors. Only token/pooled SIGReg sees projected tensors.
- Paired seeds301/302,49,799steps,1000warmup,LR5e-5→5e-7,freshAdamW,unchanged source/mask/data/batch recipe.
- Legacy configs/checkpoints remain strict-compatible; warm start may omit only the new regularization-head parameters. Resume still requires matching config and restores all states.
- Fixed2 primary,flat6 secondary;allblocks evaluated;no checkpoint selection/export replacement.

## Tasks

- [x] TDD: isolated loss inputs,FP32 projected shapes,gradient flow,control parity and exact warm-started base weights; implement core/config/warmstart changes.
- [x] Add deterministic projector geometry and per-term block-gradient diagnostics without altering RNG/weights/gradients (lighter builder).
- [x] Register paired jobs/config hashes and automatic final all-block/raw/random evaluation; add offline pilot chapter.
- [x] Focused checks,real-checkpoint CPU/GPU smoke and lighter review;commit before launch.
- [ ] Launch four healthy UUID-pinned services;verify finite metrics and actual progress;record run IDs/ETA.

## Ledger

- User explicitly requested launch after reviewing the pilot design. Execute research training; independent release validation remains absent. No further permission is needed for this authorized launch.

- Verification: 64 focused/model/loss/trainer/resume/report tests passed; final report/config subset 8 passed. Real source-checkpoint control/projected GPU smoke completed. Projected AMP calibration completed 50 steps with finite losses/gradients; one early scaler reduction, no persistent overflow. Lighter independent review found no pairing, warm-start, loss-isolation, evaluation or gate blocker.
- Registration: `docs/superpowers/registrations/2026-10-12-projection-pilot.json`; source and per-job config hashes recorded before launch. Shared gradient-audit implementation moved into the training library so online diagnostics do not depend on scripts.
