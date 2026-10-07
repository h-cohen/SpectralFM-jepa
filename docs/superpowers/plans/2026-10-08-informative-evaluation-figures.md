# Informative Evaluation Figures Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans.

**Goal:** Make each W&B evaluation and the live progress view explain per-dataset model performance and paired gains clearly.
**Architecture:** Add reusable matplotlib figures to evaluation and screen-8 assessment, then publish equivalent figures from the live progress watcher. Keep numeric tables as the searchable source of detail and export PNG copies locally.
**Tech Stack:** Python, matplotlib, W&B SDK, pytest.
**Spec:** Approved design in the 2026-10-08 conversation; `CLAUDE.md` reporting rules.

## Global Constraints
- Every R² figure includes the raw score and the same-architecture random-init control when measured.
- Dataset identity and sample count are visible; incomplete values remain missing, never zero-filled.
- Gain plots show the zero line and +0.05 strong-win threshold; paired uncertainty is labeled SD, not a confidence interval.
- Do not change training, evaluation protocol, user win rule, or promotion gates.
- W&B receives metrics, tables, HTML and figures only; no model artifacts or credentials.

## Review Focus
- Missing random controls or paired uncertainty must render as missing without hiding available dataset results.
- Screen arms must remain distinguishable by variant and seed.
- A result's checkpoint step must remain explicit in the live progress view.

### Task 1: Evaluation and screen-8 figures

**Files:** `scripts/evaluate.py`, `scripts/assess_screen8.py`, focused tests.

- [x] Add tests for dataset comparison and paired-gain plots, including absent controls/uncertainty.
- [x] Run those tests and observe expected failures.
- [x] Implement the figures; log them alongside existing tables and export assessment PNGs.
- [x] Run focused tests.

### Task 2: Live progress figures

**Files:** `scripts/training_progress.py`, focused tests.

- [x] Add tests ensuring all-dataset gains and score comparisons include measured context and tolerate missing values.
- [x] Run those tests and observe expected failures.
- [x] Implement a refresh path for the existing live presentation run and local PNG exports.
- [x] Run focused tests; refresh and remotely verify live presentation and screen-8 figures.

## Execution ledger
- User approved the two-figure design: raw/model/random per-dataset comparison and paired gains with uncertainty, zero and +0.05 reference lines.
- Execute on `main` as directed by `CLAUDE.md`; training and evaluation protocol remain unchanged.
- Focused verification: 19 tests passed before adding the existing-run refresh case; that new test also passed.
- Cavecrew review found a cursor-coordinate risk on watcher restart; persist raw W&B `_step` separately and recover the exact row for legacy status. Regression test observed RED then GREEN.
- Full fast verification: `OMP_NUM_THREADS=4 JOBLIB_TEMP_FOLDER=/dev/shm PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q` → 182 passed, 8 deselected, one sandbox NVML warning.
- W&B refresh succeeded on presentation run `u2b9gssq`; API confirmed both per-dataset and paired-gain media files.
- Screen-8 assessment run `sy5sup9s` shows two complete arms and four pending. API confirmed both screen figure files. This is exploratory/pending, not a promotion result.
- Active presentation watcher was restarted on the same run and now maintains durable source-row cursors.
