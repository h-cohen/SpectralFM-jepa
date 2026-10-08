# Screen 8 and Final Training Direction Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Focused subagents use gpt-6-luna and Cavecrew.

**Goal:** Continue experiments toward a single frozen model beating raw spectra on a majority, ideally all eight datasets, while deciding the final full training recipe from repeated measured effects.
**Architecture:** Preserve two original 10-epoch lr50 runs. Add a matched two-seed, three-arm masking continuation screen using the existing trainer and screen queue. Run jobs as detached user services, sample long-run checkpoints at 250k and 350k, and keep final automatic evaluation. Publish all completed eight-set scorecards and explicit promotion gates in W&B.
**Tech Stack:** Existing PyTorch/W&B/YAML tools, systemd user services, pytest.
**Spec:** CLAUDE.md; plans/SUMMARY.md §§1,6; user approval to continue and requested lighter-model Caveman workflow.

## Global Constraints
- Global affine normalization; all 12,758,642 packed spectra eligible (existing 10,000 unlabeled validation holdout retained); labels never used in pretraining.
- Fixed eight-set nested CV: 2×5 outer, five inner, seed 42. Any ΔR²>0 wins; ≥.05 is strong. Raw and same-architecture random-init controls accompany every evaluation.
- Keep model 24 patches/d256/depth6, SIGReg .05, global term 1, source mix .5/.5, batch256. Do not retry 80% mix or VISReg.
- No writes to read-only data/parent repositories, no model/checkpoint artifacts or stored credentials, commits as Hadar on main. W&B-managed table data is allowed for the requested scorecards.
- Logging changes must not interrupt training. Reduce data workers to four per job for shared CPU capacity.

## Evidence and hypotheses
- At 200k both seeds win 5/8; mean Δ +.069/+ .016. Their 0106 gains are only +.001–.003, 0109 loses .022–.028 while random wins .026, and 0120 loses .003–.020.
- 48 patches helps 0120 and margin over random, but still loses 0106 and does not solve 0109. It remains the next architectural candidate, not an evidence-backed replacement yet.
- Mask changes isolate whether more context or structured prediction retains useful fine spectral information. No reconstruction head, new tokenizer, or teacher change until this screen is measured.
- Completion of the cosine tail tests a separate hypothesis about labeled_data recovery. It is not replaced by the short screen.

## Review Focus
- Prior jobs disappeared at 221750–222250 without errors or saved progress beyond200k. Exact termination cause is unknown. User services run under the independent user manager; verify their PIDs/cgroups and live telemetry.
- Long-run resume restores moments/scaler and original schedule; screen continuations deliberately reset moments/schedule equally in every arm.
- Match seed-specific initial checkpoints and vary only masking within each seed.
- Do not interpret zero-centered noisy small-set differences as proof; report exploratory candidate status and confirm across further seeds.
- Do not conflate continuation-local step15000 with original pretraining step200000. Config and lineage state both.

### Task 1: Durable continuation
- [x] Verify detached user-service support and absence of duplicate jobs.
- [x] Set long-run workers4/checkpoint cadence10000; resume latest available200k checkpoints on GPUs0/1.
- [x] Verify optimizer schedule, finite loss, parent user-manager cgroup and service state.

### Task 2: Controlled screen
- [x] Add config validation test; observe RED; create screen8/eval_small8 configs and verify.
- [x] Arms: control_s0/s1 random75, random50_s0/s1 random50, block75_s0/s1 block75/three blocks.
- [x] All arms initialize from their original200k model weights, fresh AdamW, 15000 total steps, 1500 warmup, same LR5e-4 cosine, validation/diagnostics5000, checkpoints5000.
- [x] Launch four screen workers on CUDA2,3,4,5 (physical2,4,5,6), max two concurrent evaluations. Preserve CUDA_DEVICE_ORDER=PCI_BUS_ID.
- [x] Run focused/full verification and lightweight independent review.

### Task 3: Measured decisions and presentation
- [x] Add source manifest support so presentation follows restarted runs; skip in-progress JSON safely.
- [x] Add assessment with raw/model/random scores, controls, uncertainty and promotion gates; tests RED/GREEN.
- [x] Apply the promotion gates: neither masking candidate passed all gates, so neither was promoted.
- [x] These are recipe-selection gates, not a change to the user-defined win metric and not a statistical confidence claim.
- [x] Assess all six completed scorecards, write decision JSON/Markdown and W&B decision run. All six are complete; no arbitrary automatic heavy training.
- [x] Evaluate resumed250k and350k checkpoints serially on CUDA6 (physical7); both milestone sets completed. Final evaluations completed automatically after both ten-epoch runs.
- [x] Run live presentation as a user service with current source IDs; record service names, logs, links and state in SUMMARY.md.

## Final full training policy
The provisional final recipe is the existing24-patch, mask75, global-term1, 50% mix with10 epochs. Replace masking only after the registered screen gates pass and a longer continuation confirms the effect. If no masking candidate passes, retain that recipe and compare the long cosine tail with a repeated48-patch long run before spending on reconstruction or teacher variants. Final recipe claims require at least three seeds; report per-dataset raw/model/random R², mean±seed SD and Δ. Choose one frozen checkpoint; do not assemble dataset-specific models. All-eight remains aspirational until measured.

## Execution ledger
- User authorizes continuation and direction setting; execute inline, with only focused lighter-model subagents.
- Current durable data stop: checkpoint200000, not logged222k. Earlier logs alone cannot recover optimizer state.
- Detached systemd user-service probe succeeded. Previous terminal session persistence was insufficient.
- Cavecrew evidence investigator independently favors matched masking before a48-patch long run.

- Durable services: long1 seed0/1, screen8 controller, milestones, and presentation ran under independent `/user.slice/.../user@1040.service/app.slice/` cgroups. Screen8, milestone evaluation, and both ten-epoch long runs completed; the presentation watcher remains active.
- Source manifest now uses original histories plus new resume IDs8zrttyj7/3cj18b2q; stale lost trajectories are excluded. Presentation runu2b9gssq; plan runfjbztblr; controlleru4k2kge9; initial pending decision54umpbjc.
- RED/GREEN: config test failed before screen8 config existed, then passed. Builder's source/partial-JSON tests failed then11passed. Assessor tests failed before implementation/reporting fields existed, then5passed. Independent lighter-model review found reporting omissions, corrected with measured raw/model/random and48per-set rows.
- Full suite initially observed the reporting test's RED state because it started while the builder changed that test/module. Confirmed `KeyError: mean_raw_r2`; no unrelated failing tests. After all edits completed, fresh full suite174passed,8deselected, one sandbox NVML warning. `git diff --check` and shell syntax check passed.
- Deferred minor: assessor permits a complete model/raw card with missing random control, displaying nulls. Normal configured evaluation requires random_control=true; candidate status remains exploratory and does not launch training. Final claims require observed controls.
- Ruling: retain exact evaluation protocol and user win rule; use explicit separate promotion gates only for experimental triage.
- Ruling: short null screen does not rule out long training; 48patch remains conditional next architecture before reconstruction/teacher changes.
- W&B table metadata is managed by the SDK as table artifacts; no model/checkpoint files are uploaded. Prior repository table logging already uses this mechanism.
- Screen8 and both long1 seeds completed. Final long1 results are seed0 5/8 wins (mean Δ +0.0442) and seed1 3/8 (mean Δ +0.0105); majority performance did not repeat in both seeds. The current recipe remains provisional.

## Runtime status (2026-10-08 05:08 IDT)

- Both long1 services completed 497,990 steps and their final evaluations. Seed0 wins 5/8 (mean Δ +0.0442); seed1 wins 3/8 (mean Δ +0.0105). These two seeds do not establish repeatable majority performance.
- All six screen8 evaluations completed. Final assessor run: https://wandb.ai/hcohen/spectralfm-lejepa/runs/3hki6g5s . Neither candidate passed every registered gate: random50 missed wins-per-seed (5/8, 4/8); block75 missed positive hard-set gain in each seed (seed1 −0.0424). No masking candidate was promoted and no heavy follow-on training was launched.
- The milestone queue completed both 250k and350k evaluations for each seed. At350k, seed0 won4/8 and seed1 won5/8; later checkpoint scores were not monotonic.
- The presentation service remains active, showing both runs finished at497,990 and39 complete scorecards. The screen service, milestone service, and long1 services are inactive because their jobs completed.
- Current user-manager check: only `spectralfm-presentation` reports `active`. Final recipe choice is open; the screen8 masking variants were not promoted.

## Follow-up paired architecture experiment (pre-registered 2026-10-08)

- Compare fresh 24-patch and48-patch runs on the same seeds0,1,2 (six runs total), each from scratch for10 epochs/497,990 steps. Hold all other settings fixed: packed data, global normalization, 50/50 source weights, mask ratio.75, random masking, d256/depth6, batch256, AdamW5e-4, global weight1, SIGReg.05, and the same training/validation protocol.
- Use four data workers per job,10k-step recovery checkpoints,25k training diagnostics, and detached user services. GPU3 is excluded; CUDA6 (physical GPU7) is reserved for evaluation. Final evaluations share an exclusive file lock so only one eight-dataset nested-CV probe runs at a time.
- Log each run to W&B group `paired-long-confirmatory`; retain optimizer-step coordinates, weighted loss components, learning rate, validation/representation diagnostics, throughput and run configuration. Do not upload checkpoints.
- Evaluate only the pre-registered final checkpoint for the architecture decision, with the unchanged eight-set nested-CV protocol, raw spectrum and same-architecture random-init control. Do not choose a checkpoint by inspecting intermediate downstream evaluations.
- Report per-dataset ΔR², wins/strong wins, paired-fold uncertainty, and across-seed mean±SD. Call a recipe majority-positive only if its mean per-dataset Δ is positive on at least5/8 datasets and at least two of three seeds individually win5/8 or more. Call one architecture better than the other only if the paired-seed mean gain across the eight datasets is positive and the 48-patch arm is better in at least two of three seed pairs. These are exploratory selection rules, not significance claims. All-eight requires positive seed-mean Δ on every dataset and remains aspirational.
- Run only after a CUDA smoke check passes and all six training devices are visible. The accepted 48-patch one-seed/three-epoch result (5/8 wins) remains exploratory and does not count as one of the fresh full-schedule seeds.
- Report artifact: `outputs/training-progress/results_report.html`, generated by `scripts/build_results_report.py`; it embeds all chart code and current scorecards for offline presentation.
