# Long-1 Resume Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task.

**Goal:** Continue the approved 50% mix seeds from 200k to 497,990 steps, aiming for a frozen FM beating raw on a majority, ideally all eight sets.
**Architecture:** Restore model, AdamW, scaler and original cosine position. Reject incompatible configurations. Preserve available RNG and scheduler state in new checkpoints. Legacy checkpoints lack RNG state; document this limitation. Use fresh output directories and linked W&B runs.
**Tech Stack:** PyTorch, YAML, pytest, W&B.
**Spec:** CLAUDE.md and plans/SUMMARY.md sections 1, 2, 6; user authorized continuation on 2026-10-07.

## Global Constraints
- All packed spectra are eligible; labels never enter pretraining.
- Keep global normalization, mask .75, global weight 1, 50% source mix, batch 256, ten epochs, original seed and schedule.
- Preserve nested CV and random-init controls. No writes to parent repositories. Work directly on main as CLAUDE.md directs.

## Review Focus
- Reject model, data, loss or schedule mismatches before expensive setup.
- Resume must preserve AdamW moments and absolute steps.
- Legacy checkpoints reconstruct scheduler position, with explicit RNG limitation.
- New checkpoints save RNG/scheduler state for later continuation.
- Preserve init_checkpoint semantics and fresh training behavior.

### Task 1: True resume
- [ ] Add resume tests: absolute step/LR, optimizer counts, incompatible config, mutual exclusion, scheduler reconstruction.
- [ ] Run tests and observe failure.
- [ ] Implement training.resume and state restoration, scheduler/RNG persistence.
- [ ] Run focused tests and full fast suite; review diff; commit as Hadar.

### Task 2: Launch and evaluate
- [ ] Generate resolved configs from each original checkpoint, changing only resume and bookkeeping.
- [ ] Check healthy GPU availability; launch two detached runs with OMP_NUM_THREADS=4, JOBLIB_TEMP_FOLDER=/dev/shm via bash -ic.
- [ ] Verify progress beyond 200k and finite loss; queue final evaluation using eval_long1.yaml and random control.
- [ ] Update plans/SUMMARY.md and execution ledger with processes, logs and next evaluation milestones.

## Execution ledger
- Approval: user explicitly instructed continuation of the proposed resume plan; execute inline without repeating authorization.
- Main branch: mandated by CLAUDE.md; no worktree.

- Focused verification: 17 training tests passed. Full fast suite running.
- Review: legacy weighted resume is sound; launcher revised to evaluate the exact emitted checkpoint.
- Limitation: RNG restoration saves global/mask generators, but unweighted DataLoader shuffle-generator state is not saved. Current long-1 uses deterministic epoch-seeded weighted sampling. Exact stochastic replay is not claimed.
