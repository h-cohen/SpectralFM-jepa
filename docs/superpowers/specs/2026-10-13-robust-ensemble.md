# Robust four-seed ensemble — registered 2026-10-13

> Paths before 2026-10-14 refer to `scripts/`; see `docs/superpowers/specs/2026-10-14-repo-restructure-design.md`.

Status (2026-10-13): completed; all three registered criteria pass (8/8 vs raw and random, mean Δraw +.0556). See Results.

Status (2026-10-13): registered and launched; the three jobs are running.

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

## Results (2026-10-13)

All three jobs completed 49,799 steps, the gradient audit, all-block evaluation and both perturbation evaluations. `python -m scripts.robust_ensemble analyze` wrote `outputs/robust-ensemble/results_analysis.json`. **All three registered criteria pass.**

| Criterion | Result | Pass |
|---|---|---|
| Contract: raw wins ≥ 5, mean Δraw > 0, mean Δrandom > 0 | 8/8, +.0556, +.0478 | yes |
| Preferred over original ensemble: raw wins ≥ 7, random wins ≥ 7 | 8/8, 8/8 (original 7, 6) | yes |
| Each job's 1%-noise drift ≤ 0.5× its start | p0 .0146 vs .0926; p1 .0136 vs .0913; s101 .0145 vs .0901 | yes |

Ensemble block2-flat, mean outer-test R² over the two repeats (random = seed2 random-weight control):

| Set | Raw R² | Random R² | Robust ensemble R² | Δ vs raw (± bootstrap SD) | Δ vs random |
|---|---|---|---|---|---|
| labeled_data | .830 | .949 | .959 | +.129 ± .004 | +.010 |
| 0055 | .591 | .541 | .640 | +.049 ± .127 | +.099 |
| 0106 | .697 | .702 | .714 | +.017 ± .022 | +.011 |
| 0109 | .918 | .933 | .943 | +.026 ± .011 | +.011 |
| 0112 | .377 | .364 | .421 | +.044 ± .117 | +.057 |
| 0113 | .548 | .561 | .702 | +.154 ± .060 | +.141 |
| 0114 | .983 | .982 | .986 | +.002 ± .001 | +.003 |
| 0120 | .380 | .354 | .405 | +.025 ± .054 | +.051 |

Per-member raw wins (reported, not a criterion): robust_s401 7/8, robust_p0 6/8, robust_p1 8/8, robust_s101 7/8. Ensembling recovers each member's losses: 0112 (s101 −.052, s401 −.001), 0055 and 0120 (p0). Noisy small-set R² at 1% noise rises from .574–.587 to .644–.665 for the new jobs, while clean small-set R² falls from .726–.739 to .686–.690.

Interpretation: under the majority contract this is the strongest benchmark candidate so far. It is the only model that wins 8/8 against both raw and random features, and it is about 6× less sensitive to input noise. Its mean margin (+.056) is smaller than that of the original ensemble (+.086). Several wins are small relative to their bootstrap SD (0106, 0114, 0120, 0055, 0112). The candidate remains retrospective. Export and gate are unchanged, single-shuffle canaries were not rerun, and only the independent labelled holdout can qualify it.
