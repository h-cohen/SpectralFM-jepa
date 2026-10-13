# Matched Continuation Implementation Plan

Status (2026-10-13): completed plan. Historical record; current state is in CLAUDE.md.

> **For agentic workers:** Use superpowers:executing-plans inline; focused lighter-model review. Steps use checkbox syntax.

**Goal:** Launch four paired continuation jobs and automatically evaluate/log final outputs.
**Architecture:** Reuse trainer, frozen evaluator, policy comparison and gradient audit. One orchestrator owns immutable registration and per-run state; report reads results.
**Tech Stack:** Existing PyTorch/YAML/W&B/systemd/matplotlib.
**Spec:** `docs/superpowers/specs/2026-10-11-continuation-study.md`.

## Constraints and review focus

- Original checkpoint/export/gate never overwritten. New outputs under outputs/continuation-study.
- Verify declared checkpoint/config SHA before spending GPU time; reject duplicate registration/rerun collisions.
- Configs differ only seed, name, lambda; fixed 99,598 step/1000 warmup schedule, init not resume, fresh optimizer.
- Labels only in inner-selected probes; final results exploratory, seed SD distinct from bootstrap SD.
- Failed training/evaluation is an explicit per-run failure, not partial success. Single-run files avoid shared manifest races; report rebuild serialized.

## Tasks

- [x] Test paired config invariants and exact warmup; implement registration/orchestration with existing primitives.
- [x] Add live/completed continuation chapter to offline report; verify missing results remain pending.
- [x] Lighter-model review and focused checks; commit registration before launch.
- [x] Launch four systemd jobs; verify GPU allocation, W&B run IDs, finite progress; record status/ETA.

## Execution ledger

- Authorization: latest user explicitly approved new experiments. Proceed despite absent independent labels as exploratory research; no release promotion. Two paired seeds added to check repeatability. No extra permission required for authorized launch.
- Real-checkpoint 2 step GPU6 smoke passed; all checkpoint metrics finite. Paired configs resolve 99,598 steps/1000 warmup, fresh init not resume. Config invariant and preference tests observed RED→GREEN.
- Independent lighter review confirmed pairing, init/evaluator pipeline and original gate isolation. CUDA 3 allocation verified UUID matches physical 4; launcher now uses registered GPU UUIDs and verifies runtime UUID to avoid NVML/ordinal ambiguity. Frozen baseline/preference computation and per seed comparisons added to report.
- Registration: `docs/superpowers/registrations/2026-10-11-continuation-study.json`. Config and checkpoint hashes verified before each run; no duplicate starts. Research preference computed only when all four final policy outputs available; deployed model never auto-replaced.
- 14 focused tests passed before launch; 9 report/config/preference tests passed after run-link rendering. Preregistration/code commit 82aa681 precedes all starts. Four services verified active, physical GPUs 0/1/2/4 ~6.5 GB and 92–93% utilization.
- W&B control202 `otcgb58r`, low202 `8604lchm`, control203 `rl1j6ygp`, low203 `pqh525md`. All four at step 550: losses .13689/.03540/.13032/.03084; gradient norms 4.978/1.811/2.368/1.348 finite. Lower weighted loss alone is not improvement. Initial throughput ~.15 s/step suggests ~4 hours remaining training; final nested CV/policies follow.
- Background research results not complete. Per-run state and checkpoints durable; each successful pipeline automatically rebuilds report under lock. No original gate/export mutation and no release promotion.

## Final results (2026-10-12)

- All four states completed, final policy outputs/gradient audits/report written. Fixed mean raw gains: control202 +.081188, control203 +.080210, low202 +.079042, low203 +.078723; baseline +.080077. Menu: .084609/.088908/.093238/.089280; baseline +.093253. Fixed wins 6/5/6/6; menu 6/6/6/7.
- Preregistered preference fixed=`duration_only`, menu=`inconclusive`. Control fixed average improvement only +.000622; low fixed regresses both seeds. No automatic promotion. Practical recommendation retain original 101 and lambda .05, stop duration-only sweeps, independently validate the frozen four readout menu.
- Final block6 flat wins control 1/8 each, low 1/8 and 0/8. Lower lambda improves validation prediction MSE but not consistent downstream performance. No all 8 winner. Full recommendation: `docs/PRODUCTION_RECOMMENDATION.md`; original gate/export unchanged.
- Final compilation verified all four checkpoint/config hashes and eight datasets per policy output. Nine focused report/config/preference tests passed. Independent lighter review confirmed aggregate practical recommendation; original gate False preserved.
