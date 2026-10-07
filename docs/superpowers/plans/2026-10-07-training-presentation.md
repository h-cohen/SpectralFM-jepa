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
