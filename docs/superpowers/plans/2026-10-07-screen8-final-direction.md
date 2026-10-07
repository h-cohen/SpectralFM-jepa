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
- [ ] Promote a masking candidate only if each seed wins≥5/8, hard-set mean gain over matched continuation is positive in both seeds, overall mean gain is positive, seed-mean LD regression≤.01 and other stable-set regressions≤.05.
- [x] These are recipe-selection gates, not a change to the user-defined win metric and not a statistical confidence claim.
- [ ] Assess all six completed scorecards, write decision JSON/Markdown and W&B decision run. Missing results remain pending; no arbitrary automatic heavy training.
- [x] Evaluate resumed250k/350k checkpoints serially on CUDA6 (physical7). Final evaluations remain queued by resume launcher.
- [x] Run live presentation as a user service with current source IDs; record service names, logs, links and state in SUMMARY.md.

## Final full training policy
The provisional final recipe is the existing24-patch, mask75, global-term1, 50% mix with10 epochs. Replace masking only after the registered screen gates pass and a longer continuation confirms the effect. If no masking candidate passes, retain that recipe and compare the long cosine tail with a repeated48-patch long run before spending on reconstruction or teacher variants. Final recipe claims require at least three seeds; report per-dataset raw/model/random R², mean±seed SD and Δ. Choose one frozen checkpoint; do not assemble dataset-specific models. All-eight remains aspirational until measured.

## Execution ledger
- User authorizes continuation and direction setting; execute inline, with only focused lighter-model subagents.
- Current durable data stop: checkpoint200000, not logged222k. Earlier logs alone cannot recover optimizer state.
- Detached systemd user-service probe succeeded. Previous terminal session persistence was insufficient.
- Cavecrew evidence investigator independently favors matched masking before a48-patch long run.

- Durable services: long1 seed0/1, screen8, milestones, presentation all verified active with independent `/user.slice/.../user@1040.service/app.slice/` cgroups. First four screen arms have finite W&B losses; block arms queued. Long runs verified beyond205k with original cosine schedule.
- Source manifest now uses original histories plus new resume IDs8zrttyj7/3cj18b2q; stale lost trajectories are excluded. Presentation runu2b9gssq; plan runfjbztblr; controlleru4k2kge9; initial pending decision54umpbjc.
- RED/GREEN: config test failed before screen8 config existed, then passed. Builder's source/partial-JSON tests failed then11passed. Assessor tests failed before implementation/reporting fields existed, then5passed. Independent lighter-model review found reporting omissions, corrected with measured raw/model/random and48per-set rows.
- Full suite initially observed the reporting test's RED state because it started while the builder changed that test/module. Confirmed `KeyError: mean_raw_r2`; no unrelated failing tests. After all edits completed, fresh full suite174passed,8deselected, one sandbox NVML warning. `git diff --check` and shell syntax check passed.
- Deferred minor: assessor permits a complete model/raw card with missing random control, displaying nulls. Normal configured evaluation requires random_control=true; candidate status remains exploratory and does not launch training. Final claims require observed controls.
- Ruling: retain exact evaluation protocol and user win rule; use explicit separate promotion gates only for experimental triage.
- Ruling: short null screen does not rule out long training; 48patch remains conditional next architecture before reconstruction/teacher changes.
- W&B table metadata is managed by the SDK as table artifacts; no model/checkpoint files are uploaded. Prior repository table logging already uses this mechanism.
- Runtime results and final recipe confirmation remain pending. Services queue complete probes/assessment without depending on an active chat turn. The current provisional recipe is not presented as a newly measured winner.
