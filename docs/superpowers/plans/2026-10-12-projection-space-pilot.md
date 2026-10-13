# Projection-space pilot implementation plan

> Paths before 2026-10-14 refer to `scripts/`; see `docs/superpowers/specs/2026-10-14-repo-restructure-design.md`.

Status (2026-10-13): completed plan. Historical record; current state is in CLAUDE.md.

> Use superpowers:executing-plans inline, with a focused lighter-model diagnostics builder and reviewer.

**Goal:** Launch the approved four-run matched projector pilot with verified loss isolation and automatic evaluation.
**Spec:** `docs/superpowers/specs/2026-10-12-projection-space-pilot.md`.
**Architecture:** Optional regularization-only MLP in the model, explicit SIGReg overrides in the loss, narrowly compatible initialization, paired durable job orchestration and report integration. Existing default behavior/export/gate stay unchanged.

## Constraints

- Source checkpoint SHA256 ec4c041dc4f83aa04e9ec568a335cdf7e27dbc7aa7fac965f2fd1f686b566ff0.
- Modes none/control/mlp; control instantiates the same unused 256→512→128 MLP as projected mode. No BatchNorm/dropout; FP32 forward; no detached gradients.
- Masked MSE and pooled invariance use original backbone tensors. Only token/pooled SIGReg sees projected tensors.
- Paired seeds 301/302, 49,799 steps, 1000 warmup, LR 5e-5→5e-7, fresh AdamW, unchanged source/mask/data/batch recipe.
- Legacy configs/checkpoints remain strict-compatible; warm start may omit only the new regularization-head parameters. Resume still requires matching config and restores all states.
- Fixed2 primary, flat6 secondary; all blocks evaluated; no checkpoint selection/export replacement.

## Tasks

- [x] TDD: isolated loss inputs, FP32 projected shapes, gradient flow, control parity and exact warm-started base weights; implement core/config/warmstart changes.
- [x] Add deterministic projector geometry and per-term block-gradient diagnostics without altering RNG/weights/gradients (lighter builder).
- [x] Register paired jobs/config hashes and automatic final all-block/raw/random evaluation; add offline pilot chapter.
- [x] Focused checks, real-checkpoint CPU/GPU smoke and lighter review; commit before launch.
- [x] Launch four healthy UUID-pinned services; verify finite metrics and actual progress; record run IDs/ETA.

## Ledger

- User explicitly requested launch after reviewing the pilot design. Execute research training; independent release validation remains absent. No further permission is needed for this authorized launch.

- Verification: 64 focused/model/loss/trainer/resume/report tests passed; final report/config subset 8 passed. Real source-checkpoint control/projected GPU smoke completed. Projected AMP calibration completed 50 steps with finite losses/gradients; one early scaler reduction, no persistent overflow. Lighter independent review found no pairing, warm-start, loss-isolation, evaluation or gate blocker.
- Registration: `docs/superpowers/registrations/2026-10-12-projection-pilot.json`; source and per-job config hashes recorded before launch. Shared gradient-audit implementation moved into the training library so online diagnostics do not depend on scripts.

- Launched four systemd services on the registered UUIDs after commit `b38accf`. W&B control301 `3be6orej`, projected301 `pft83led`, control302 `15o7a28q`, projected302 `49r2flsr`. All active with 88–94% GPU utilization and ~8.9 GB memory; W&B verified 650–700 steps with finite loss/gradient norms. Early AMP skips 3–5 are logged, not hidden; scales adjusted to 2048–8192. Estimated training 2–3 hours from initial throughput, final evaluation follows separately.
- Resolved configs verify 12,748,642 training rows, 49,799 steps, 1000 warmup, source mass .5/.5 and exact registered source hash for all four jobs.

## Completion

- [x] Verify four final checkpoint/config hashes, all 8 dataset outputs/run and all 21 readouts; final training step 49,799.
- [x] Apply preregistered primary criterion and analyze secondary all-block outcomes; paired row-bootstrap uncertainty from matched OOFs.
- [x] Compile report and recommendation: stop this recipe. Projected fixed2 gains vs control −.005030/−.002178, all 7/8 raw wins; starting prior2 mean Δraw +.089366 remains best fixed shortlist. Secondary flat6 recovers to 3/8 both projected but does not replace fixed2.
- [x] Preserve existing export/gate; no new training/promotions. Independent group-labelled validation required before shipping.

Final paired summary and six measured figures published to W&B: https://wandb.ai/hcohen/spectralfm-lejepa/runs/e72aiwt3. Specific payload/destination approved by user after automatic review initially rejected the upload.
