# Training Progress Presentation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans.

**Goal:** Make W&B show the training procedure and measured progress toward majority/all-eight frozen-probe wins.
**Architecture:** Add reusable metric enrichment and explicit optimizer-step axes for future jobs. Publish a separate live presentation run, reading original/resumed W&B histories and existing local evaluation JSON; never co-write active training runs or restart optimization.
**Tech Stack:** Existing W&B SDK, matplotlib, Python, pytest.
**Spec:** CLAUDE.md goal/reporting rules and user request on 2026-10-07.

## Global Constraints
- No artifacts or credentials in files; metrics, tables, HTML and figures only.
- Validation cadence and scientific protocol unchanged. Loss reduction is not evidence of improved probe R².
- Latest probe results labeled with evaluated checkpoint step, distinct from current training step.
- Show raw R², model R²/Δ, random-init and uncertainty when available; missing controls remain missing.

## Review Focus
- Weighted contributions must sum to total for SIGReg and VISReg.
- Historical steps retain original optimizer coordinates through resume.
- Filter to the exact eight scored sets; incomplete evaluations cannot claim majority.
- Refresh only new history rows; no overwriting source runs.
- Small-set variability and unsupervised use of evaluation spectra are disclosed.

### Task 1: Metric logging
- [x] Test weighted contribution accounting and progress/axis enrichment; observe RED.
- [x] Implement enrichment helper and explicit axes; verify full fast suite.

### Task 2: Live presentation
- [x] Add script combining seed histories and evaluation scorecards, recipe/protocol notes and per-dataset comparisons.
- [x] Verify partial evaluations and weighted losses with focused tests.
- [x] Publish live view; verify via authenticated API and save local presentation snapshot.
- [x] Record links, processes, limitations and commit as Hadar.

## Execution ledger
- User authorized improvements; execute inline on main as CLAUDE.md requires.

- Root cause confirmed via authenticated read-only API: loss terms already arrive, but histories are split, weighted contributions/progress absent, validation sparse by configured cadence, and probe results live elsewhere.
- RED: missing training_metrics failed three tests; GREEN: SIGReg/VISReg accounting and no fabricated validation measurements verified.
- First full suite: 161 passed, 8 deselected, one sandbox NVML warning.
- Independent review found stale hard-coded latest checkpoint, incomplete aggregate baselines, missing random-control uncertainty. Two regression tests observed RED, then all six focused tests GREEN after fixes.
- Live view published at https://wandb.ai/hcohen/spectralfm-lejepa/runs/hkvku2ea ; session 54577. Remote metrics/media verified and second refresh confirmed.
- Ruling: keep validation cadence and current optimization untouched; presentation mirrors existing measurements, with no claim that lower loss guarantees improved R².
- Ruling: use existing W&B run dashboard/HTML/figures rather than installing report SDK; exports are available locally. Source training runs remain read-only.

- Final full verification: 163 passed, 8 deselected, one sandbox NVML warning; git diff --check clean. Both training runs continue (presentation refresh 206800/206500).

## Offline presentation story (2026-10-08, user-approved design)

- Extend the existing offline report with chapters: training paradigm → SIGReg sensitivity and current diagnostics → embedding geometry → whitening policy comparison → eight-dataset scorecards.
- `scripts/report_evidence.py` computes label-independent geometry and paired nested-CV policy diagnostics from saved legacy final banks (24 patches, seeds 0/1, step 497,990). Fixed final encoder block mean = layer6, 256 dimensions. No model/readout selection using outer test scores.
- Geometry: reproducible sample of at most 2,048 rows; PCA/whitening fitted on first outer training split only. Show covariance spectra, covariance-entropy rank divided by attainable rank, and two-PC before/after point clouds, with held-out rows separate. Whitening equalizes retained training covariance by construction; it does not Gaussianize features or guarantee held-out isotropy.
- Downstream diagnostic: all saved samples in all eight datasets, two seeds, fixed layer6 mean. Identical 2×5 outer folds / 5 inner folds. Compare whitening-only policy (full/8/32/128 PCA + RidgeCV/OLS) against none/standardize policy, choosing recipes within each outer training fold. Result: 3/16 positive policy deltas; mean ΔR² −0.071562. This does not generalize to the main scorecard's adaptive layer/readout search and does not isolate rescaling from PCA truncation.
- SIGReg plot: use actual screen-1 λ=.02 vs λ=.05 controls for validation MSE, singular-value entropy rank and one-dataset downstream R². The earlier recipe differs from the final recipe; the weaker arm has one seed and its paired difference was neutral (−.044 ± .029 bootstrap SD). No SIGReg-off causal claim.
- `scripts/local_wandb.py` reads complete CRC-validated local datastore history records, including fragmented records; ignores incomplete live tails. Ongoing six-run geometry/validation histories are presented without querying or modifying source runs. Logged singular-value entropy rank is explicitly distinguished from covariance-entropy rank.
- `scripts/report_story.py` exports standalone PNG/SVG figures and embeds raster copies in the offline HTML. Dataset/seed selector changes the geometry figure. Existing scorecard controls remain available; the win timeline supports 0–8 wins.
- Refresh diagnostics: `.venv/bin/python -m scripts.report_evidence` (CPU only, one BLAS thread, cached). Refresh HTML/slide figures: `.venv/bin/python -m scripts.build_results_report`.
- Independent lighter-model review found no blocking leakage/provenance/plot bugs; cache eligibility issue corrected so a smaller CPU threshold cannot reuse a now-ineligible probe result. Training configurations/services are untouched.
- Final validation: 14 focused report/evidence/datastore/launcher tests passed; geometry train/test boundary and held-out-label invariance verified. `git diff --check` passed. Generated offline HTML has unique IDs, no remote runtime dependencies, all 16 measured whitening comparisons, all six ongoing run histories, and 19 standalone SVG/PNG figure pairs. Geometry, whitening, SIGReg sensitivity and history images were visually inspected. Snapshot through 2026-10-08 18:43 IDT; 24p at 71,800–72,750 and 48p at 60,100–60,200 observed steps.

### User correction: SIGReg geometry analogy

- Supersedes the whitening chapters above for presentation only. User wants SIGReg illustrated as ellipse-like anisotropy → circle-like isotropy during training, with no separate whitening content.
- Report now contains an explicitly labeled synthetic 2D schematic (identical axes/aspect), early SIGReg weight sensitivity, actual ongoing training diagnostics, and downstream scorecards. The circle represents the desired target; no measured embedding cloud is altered to force this shape, and no SIGReg-off causal claim is made.
- Removed old geometry/policy figures from the current slide directory, retaining them in `outputs/training-progress/archived-diagnostics/`. The presentation HTML excludes the old diagnostic payload entirely. Three current standalone PNG/SVG figure pairs remain.

### Fixed readout and depth analysis (2026-10-09)

- User requested fixed variants and “where the signal lives” before the final model launch. `scripts/readout_variants.py` reads existing nested family scores for17readouts per trained checkpoint: mean/seg4 at0–6, flat at0–2. All6fresh final checkpoints yield102trained and102matched-random rows, with source bootstrap SD for every measured cell. No new scores inferred or evaluations launched.
- Report has an offline fixed-readout selector, default `layer2/flat`; tables include raw, individual seeds, matching random controls and separate means±seed SD. Dataset counts and green raw-win cells are preserved. Full-precision export: `outputs/training-progress/readout_variants.csv`.
- Two eight-panel figures plot R² by block for24/48patches. Lines are three-seed means, bands sample SD, faint dots seed results, dashed lines raw. Block0=tokenizer;block6=final LayerNorm output. Flat is not extrapolated beyond block2. Different pooling dimensions remain explicit.
- Retrospective discovery:48p block2-flat(12,288dims) wins7/8 in every existing seed, failing0106. Final mean/seg4 output is often weaker, but final flattened output was not measured, so do not conclude all final-token information is lost. This fixed interface requires a predeclared follow-up; it does not replace the registered architecture comparison post hoc.
- Verified18focused tests, full uncertainty coverage, correct dimensions, unique HTML IDs, no removed presentation material, offline operation and exported depth plot visually inspected. Final training/README/dataflow update remains pending the user's production interface decision.
