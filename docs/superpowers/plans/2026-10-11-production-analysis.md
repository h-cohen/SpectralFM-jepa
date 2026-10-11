# Production Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify useful production features and diagnose the qualification blocker before spending on further training.

**Architecture:** Reuse the existing all-block evaluator and nested-CV primitives. Add focused scripts for restricted policies, permutation auditing and gradient measurements; orchestrate them as durable jobs and compile their outputs into the existing offline report.

**Tech Stack:** PyTorch,NumPy,scikit-learn,matplotlib,W&B,systemd; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-11-production-analysis.md`.

## Global constraints

- Seed101 checkpoint497,990 is frozen; existing gate/assessment/export stay unchanged.
- Nested2×5outer,5inner folds,seed42; raw/random controls and source SHA256s.
- All7states×3readouts; no extrapolation of missing scores.
- Menu declared before this analysis: layer2/flat,layer2,layer3,layer3/seg4.
-100permutations,seeds1000–1099; diagnostics do not override the0.020canary gate.
- Local`.tmp`,single GPUfold worker,4CPUthreads; broken physicalGPU3excluded.
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
- [x] Launch durable all-block,permutation and gradient jobs; record service/run IDs and verify live progress.
- [ ] Compile completed outputs and plots into report; compare interface coverage and uncertainty.
- [ ] Choose one targeted training experiment only after evidence and an independent development criterion are available. Otherwise report the exact remaining data requirement.

## Execution ledger

- All-block/policy GPU0service `spectralfm-production-readouts`; W&B all-block run `m507ncw7`. Seven small datasets completed;4716-row labeled_data currently running. All21arm caches include matched random banks. Fixed/checkpoint gate files untouched.
- Permutation GPU1service completed; W&B `y4cftohx`.100seeds1000–1099,threshold.020,shuffle42exactlyreproduced. Block null0/100above limit,raw3/100;both real-score empiricalp1/101. No exact raw duplicates/conflicting labels;foldsdisjoint. This does not clear leakage or replace the original failed gate. A test caught initial implementation incorrectly using observedcanaryscore aslimit; corrected to the alreadydeclared.020 before accepting results.
- Gradient GPU2service completed; W&B `dwelzk1b`.4B256FP32passes,seed4000,unchanged weights/parameter gradients. Block2norms prediction.10799/tokenSIG.10844/poolSIG.10264;block6.00733/.12878/.132995. Generic validation data,not downstream-label gradient alignment or actual Adam update magnitudes.
- Report now includes partial all-blockdepth with pending labels marked, measuredgradient/permutation figures,and will add model/random policy scoretables andpairedgainplots automatically after the queue. Reused benchmark remains explicitly exploratory; prior3seed plots separated from new101bootstrap plots.
- Independent lighter-model review found no leakage/gate mutation blockers. Added a provenance check rejecting wrong-checkpoint analysis data.34relatedtests passed,followed by14focused tests after provenance/report changes;1expectedNVMLwarning. Distinct train/test-label selection regression verifies first-fold menu choice unchanged when outer-test labels are perturbed.
- Optional independent labelled development/deployment data path is still pending. Training continuation is not automatically launched from these diagnostic ratios or reused benchmark scores.
- Latest verification:32 focused policy,permutation,gradient,provenance,report,evaluator and model tests passed in53.66s; background readout service remains active with exit status0. Final report completion awaits the queued all-block and controlled-policy evaluations.
