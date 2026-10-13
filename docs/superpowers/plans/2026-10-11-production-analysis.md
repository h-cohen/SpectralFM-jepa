# Production Analysis Implementation Plan

Status (2026-10-13): completed plan. Historical record; current state is in CLAUDE.md.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify useful production features and diagnose the qualification blocker before spending on further training.

**Architecture:** Reuse the existing all-block evaluator and nested-CV primitives. Add focused scripts for restricted policies, permutation auditing and gradient measurements; orchestrate them as durable jobs and compile their outputs into the existing offline report.

**Tech Stack:** PyTorch, NumPy, scikit-learn, matplotlib, W&B, systemd; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-11-production-analysis.md`.

## Global constraints

- Seed101 checkpoint 497,990 is frozen; existing gate/assessment/export stay unchanged.
- Nested 2×5 outer, 5 inner folds, seed42; raw/random controls and source SHA256s.
- All 7 states ×3 readouts; no extrapolation of missing scores.
- Menu declared before this analysis: layer2/flat, layer2, layer3, layer3/seg4.
- 100 permutations, seeds 1000–1099; diagnostics do not override the 0.020 canary gate.
- Local `.tmp`, single GPU fold worker, 4 CPU threads; broken physical GPU3 excluded.
- Independent deployment/development data is not yet supplied. Reused benchmark analysis must be labelled exploratory.

## Review focus

- Outer-test labels cannot choose recipes or readouts.
- Raw/random/fixed candidates share rows/folds; missing control data stays missing.
- Multiscale concatenation dimensions and stage numbering match encoder states.
- Permutation statistics use finite-sample plus-one correction and fixed splits; no seed shopping.
- Gradient measurements do not update weights, nor confuse scalar contributions with gradient magnitudes.

## Tasks

- [x] Test full-layer config and restricted-policy membership/leakage boundaries; implement configs and policy script using existing nested CV.
- [x] Test deterministic permutation audit and fold/row checks; implement frozen-feature audit separately.
- [x] Test per-term gradient extraction preserves parameters; implement checkpoint diagnostic.
- [x] Launch durable all-block, permutation and gradient jobs; record service/run IDs and verify live progress.
- [x] Compile completed outputs and plots into report; compare interface coverage and uncertainty.
- [x] Resolve the training decision: independent development criterion absent; document the recommended controlled experiment and exact data requirement without launching it.

## Execution ledger

- All-block/policy GPU0 service `spectralfm-production-readouts`; W&B all-block run `m507ncw7`. Seven small datasets completed; 4716-row labeled_data currently running. All 21 arm caches include matched random banks. Fixed/checkpoint gate files untouched.
- Permutation GPU1 service completed; W&B `y4cftohx`. 100 seeds 1000–1099, threshold .020, shuffle42 exactly reproduced. Block null 0/100 above limit, raw 3/100; both real-score empirical p 1/101. No exact raw duplicates/conflicting labels; folds disjoint. This does not clear leakage or replace the original failed gate. A test caught initial implementation incorrectly using observed canary score as limit; corrected to the already declared .020 before accepting results.
- Gradient GPU2 service completed; W&B `dwelzk1b`. 4B256 FP32 passes, seed 4000, unchanged weights/parameter gradients. Block2 norms prediction .10799/token SIG .10844/pool SIG .10264; block6 .00733/.12878/.132995. Generic validation data, not downstream-label gradient alignment or actual Adam update magnitudes.
- Report now includes partial all-block depth with pending labels marked, measured gradient/permutation figures, and will add model/random policy score tables and paired gain plots automatically after the queue. Reused benchmark remains explicitly exploratory; prior 3 seed plots separated from new 101 bootstrap plots.
- Independent lighter-model review found no leakage/gate mutation blockers. Added a provenance check rejecting wrong-checkpoint analysis data. 34 related tests passed, followed by 14 focused tests after provenance/report changes; 1 expected NVML warning. Distinct train/test-label selection regression verifies first-fold menu choice unchanged when outer-test labels are perturbed.
- Optional independent labelled development/deployment data path is still pending. Training continuation is not automatically launched from these diagnostic ratios or reused benchmark scores.
- Latest verification: 32 focused policy, permutation, gradient, provenance, report, evaluator and model tests passed in 53.66 s; background readout service remains active with exit status 0. Final report completion awaits the queued all-block and controlled-policy evaluations.

## Completed comparison and decision

- All-block service exited 0. All 8 sets ×21 readouts and four controlled policies completed; policy W&B `hcwixghf`. Report has complete depth curves and model/random policy tables, paired gains and measured audits.
- Fixed2 flat: 5/8, +.080077 mean raw gain. Four-arm menu: 7/8, +.093253 raw, +.140900 matched random. Menu+raw: 7/8, +.086825 raw. Fixed multiscale: 5/8, +.080146 raw. None beats all 8. Menu's 0106 gain +.003094 ±.028176 and 0114 +.001823 ±.001545 are paired bootstrap SD, not guaranteed advantages.
- Menu vs fixed paired gains: 0055 +.086210 ±.037454; 0106 +.093932 ±.035476; 0112 −.076712 ±.037483; 0114 +.001983 ±.000933; other 4 identical. On 0112 one of 10 outer folds chooses seg4 rather than flat. Adding choices can hurt held-out performance despite correct nesting.
- Final6 flat beats raw 2/8, mean −.094886 delta. Largest set: raw .830116, block2 flat .958847, block6 flat .67643. This depth pattern plus gradient imbalance motivates an objective-balance test; it does not establish a causal SIGReg effect or benefits of extra duration.
- Recommended next research experiment: from the same final checkpoint, compare no continuation, matched 2 epoch continuation with lambda .05, and matched 2 epoch continuation with lambda .01. Both continuations use the same fresh schedule (peak 5e-5, 1000 warmup steps, cosine to 5e-7), 99,598 steps, fresh AdamW state, predeclared paired seed 202, sampling/RNG, batch 256, mask 75%, normalization and pooled weight 1. Full objective coefficient effects must be recorded; lower lambda also changes pooled-view invariance weighting (.95→.99), while masked prediction weight stays 1. Keep readout menu fixed. This is a proposed sensitivity experiment, not a validated recipe or launched job.
- Development requirement: separate raw 245 spectra and parameter_0 labels, source/simulation/batch IDs for group-disjoint splits, and a reserved release holdout. Predeclare development metric and acceptable regressions before comparing continuations; release labels cannot select duration/readouts. Existing 8 benchmark labels have already informed design and remain exploratory.
- Original assessment/export untouched; 0120 shuffle42 failure remains. No promotion and no new training launched. Paired policy differences saved locally in `outputs/production-analysis/decision_analysis.json`.
- Completed-report verification: 11 focused report/provenance/matrix/readout tests passed. Independent lighter-model evidence review confirmed counts and uncertainty; recommends retaining current fixed export until menu replication. No inference/export changes made.
