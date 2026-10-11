# Matched Continuation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; focused lighter-model review. Steps use checkbox syntax.

**Goal:** Launch four paired continuation jobs and automatically evaluate/log final outputs.
**Architecture:** Reuse trainer, frozen evaluator, policy comparison and gradient audit. One orchestrator owns immutable registration and per-run state; report reads results.
**Tech Stack:** Existing PyTorch/YAML/W&B/systemd/matplotlib.
**Spec:** `docs/superpowers/specs/2026-10-11-continuation-study.md`.

## Constraints and review focus

- Original checkpoint/export/gate never overwritten. New outputs underoutputs/continuation-study.
- Verify declared checkpoint/config SHA before spending GPUtime;reject duplicate registration/rerun collisions.
- Configs differ onlyseed,name,lambda;fixed99,598step/1000warmup schedule,initnotresume,freshoptimizer.
- Labels onlyininner-selected probes;finalresults exploratory,seedSDdistinctfrombootstrapSD.
- Failedtraining/evaluation is an explicit per-run failure,not partialsuccess. Single-run filesavoidsharedmanifest races;report rebuildserialized.

## Tasks

- [x] Test paired config invariants and exact warmup;implement registration/orchestration with existing primitives.
- [x] Add live/completed continuation chapter to offline report;verify missing results remainpending.
- [x] Lighter-model review and focused checks;commit registration beforelaunch.
- [x] Launch four systemdjobs;verifyGPUallocation,W&BrunIDs,finite progress;recordstatus/ETA.

## Execution ledger

- Authorization:latest user explicitly approvednewexperiments. Proceed despite absentindependentlabels as exploratoryresearch;no releasepromotion. Two pairedseedsadded tocheckrepeatability. No extra permission requiredforauthorizedlaunch.
- Real-checkpoint2stepGPU6smoke passed;allcheckpointmetricsfinite. Pairedconfigsresolve99,598steps/1000warmup,freshinitnotresume. Configinvariant andpreference tests observedRED→GREEN.
- Independent lighterreview confirmed pairing,init/evaluatorpipeline andoriginalgateisolation. CUDA3allocation verified UUIDmatchesphysical4;launcher nowusesregisteredGPUUUIDs andverifiesruntimeUUID toavoid NVML/ordinalambiguity. Frozenbaseline/preferencecomputation andperseedcomparisons added toreport.
- Registration:`docs/superpowers/registrations/2026-10-11-continuation-study.json`. Configandcheckpoint hashesverifiedbeforeeachrun;no duplicate starts. Researchpreferencecomputedonlywhenallfourfinalpolicyoutputsavailable;deployedmodelneverauto-replaced.
-14focused tests passed before launch;9report/config/preference tests passed after run-link rendering. Preregistration/codecommit82aa681 precedes allstarts. Four servicesverifiedactive,physicalGPUs0/1/2/4~6.5GBand92–93%utilization.
- W&Bcontrol202`otcgb58r`,low202`8604lchm`,control203`rl1j6ygp`,low203`pqh525md`. Allfour atstep550:losses.13689/.03540/.13032/.03084;gradientnorms4.978/1.811/2.368/1.348 finite. Lowerweightedlossaloneisnotimprovement. Initialthroughput~.15s/step suggests~4hoursremainingtraining;finalnestedCV/policiesfollow.
- Backgroundresearchresultsnotcomplete. Per-runstate andcheckpointsdurable;eachsuccessfulpipelineautomaticallyrebuildsreportunderlock. Nooriginalgate/exportmutationandnoreleasepromotion.
