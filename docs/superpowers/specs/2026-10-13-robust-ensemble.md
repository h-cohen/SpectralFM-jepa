# Robust four-seed ensemble — registered 2026-10-13

User direction: the goal is winning on a majority of the eight sets, not winning by large margins. Under that contract the seed2 robust continuations won 7/8 and 8/8 and beat the random control on 8/8 (see the contract re-read in `2026-10-13-robustness-continuation.md`). This study gives the other 48p seeds the same treatment so the seed ensemble can be rebuilt from robust members. The user approved launching it.

## Jobs

Same recipe as `robust_s401`: block-2 token invariance, weight 0.1, the registered noise calibration, 49,799 steps, fresh AdamW, LR 5e-5→5e-7 with 1000 warmup. Code: `scripts/robust_ensemble.py`, launcher `scripts/launch_robust_ensemble.sh`.

| Job | Start | Training seed | Physical GPU |
|---|---|---|---|
| robust_p0 | prior 48p seed0 | 410 | 0 |
| robust_p1 | prior 48p seed1 | 411 | 1 |
| robust_s101 | final seed101 | 412 | 2 |

Seed2's member is `robust_s401`, fixed now as the lower of the two existing seed2 robust seeds. There are no new control arms: earlier continuations showed that plain continuation leaves the start scores nearly unchanged (control401/402 vs prior seed2).

Pipeline per job: train → gradient audit → all-block nested evaluation (raw/random) → perturbation evaluation of the job and of its starting checkpoint. W&B group `robust-ensemble`; outputs `outputs/robust-ensemble/`.

## Registered criteria (block2-flat, equal-weight average of the four members' outer-fold predictions)

1. Contract: ensemble beats raw on ≥ 5/8 sets, mean Δraw > 0, mean Δrandom > 0 (seed2 random-weight control).
2. Preferred over the original ensemble as a holdout candidate: raw wins ≥ 7 and random wins ≥ 7. The original ensemble scored 7 and 6.
3. Robustness: every new job's median 1%-noise drift is ≤ 0.5× that of its own starting checkpoint.

Reported but not used as criteria: per-job wins, noisy and held-out perturbation R², shuffled-label canaries.

A pass makes the robust ensemble a holdout candidate next to seed101, seed2 and the original ensemble. It does not change the export or the gate. Only the independent labelled holdout can qualify it.
