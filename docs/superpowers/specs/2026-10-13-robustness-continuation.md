# Robustness continuation — design for approval, 2026-10-13

Status: proposal. Not approved, not launched. The seed101 export, its gate and the seed-ensemble holdout candidate stay unchanged.

## Motivation

From `docs/superpowers/specs/2026-10-13-robust-production-roadmap.md` (A results):

- Block2-flat relative drift under 1% gaussian noise is .11 for trained features, against .005 for raw and .0015 for random. At 3% noise the trained features fall below raw on 0055, 0106, 0109 and 0114, even with a fixed ridge probe.
- Drift follows source type (`single_channel_all` .223, `labeled_regression` .042), not oversampling. Memorization is not supported.
- Current invariance acts only on block6 mean-pooled outputs of patch-subset views. Production reads block2 tokens. Nothing in the objective asks block2 tokens to ignore measurement perturbations.

Question: does a block2 token invariance to realistic spectral perturbations cut drift and noisy-input R² loss without losing clean fixed2 accuracy?

## Risk to address first

Labels may depend on fine spectral detail. Raw ridge on labeled_data collapses under 1% noise, and 0112/0120 raw scores drop under noise too. Invariance to unrealistic noise would erase real signal. So perturbation magnitudes come from data, not from A6's fixed test values.

## Step 0 — label-free noise calibration (before registration)

- Per source group, estimate measurement noise as the residual SD after Savitzky–Golay smoothing (window 11, order 2), relative to spectrum SD, on ~2,000 sampled corpus rows each.
- Training noise σ is drawn up to the corpus p90 of that estimate. Baseline offset, slope and scale ranges stay at the A6 magnitudes (5%, 5%, ±5%) unless the corpus shows smaller between-replicate variation where replicates exist.
- Output `outputs/robustness-continuation/noise_calibration.json`. Registration happens after this step and before any training.

## Arms

Four jobs from the exact prior48pseed2 checkpoint (SHA `ec4c041d…6ff0`). The protocol matches the projection pilot: 49,799 steps (one epoch), fresh AdamW, 1,000 warmup, LR 5e-5 → 5e-7. Data, source weights, masking, batch size, SIGReg λ .05 and pooled invariance are unchanged.

| Arm | Seeds | Change |
|---|---|---|
| control | 401, 402 | none (continuation only) |
| robust | 401, 402 | add block2 token invariance term |

Robust term, per batch:

- x' = perturb(x), sampled independently per spectrum. Each of these is applied with probability 0.5: noise σ ~ U(0, σ_p90), offset, linear slope, scale, and ±1-bin shift. Perturb raw values, then apply the same global normalization.
- h2(·) = block2 token states [B, 48, 256] of the full unmasked sequence. Reuse the clean pass of the target encoder; this adds one encoder forward.
- L_rob = mean over tokens and dimensions of (h2(x') − h2(x))², divided by the batch variance of h2(x), detached (scale-free). Gradients flow through both branches, with no stop-grad, matching the existing invariance term.
- Total = existing loss + w_rob · L_rob. Both arms run the extra forward pass so compute and RNG consumption match; the control logs L_rob with weight 0.

w_rob is fixed by a label-free 1,000-step calibration on seed 400 (not used later). Choose from {0.1, 0.3, 1} the smallest weight whose weighted block2 gradient norm is ≥ 0.5× the prediction term's block2 gradient norm. Calibration uses the existing `training/gradient_terms.py`.

## Evaluation (automatic, as in previous studies)

Final gradient audit, then all-block/raw/random nested evaluation (primary protocol), then the A6 perturbation suite with fixed standardize+RidgeCV and inner-selected probes, then report refresh. Also evaluate a held-out perturbation family that is never used in training: 2-bin shift, 5% noise, and a sinusoidal baseline at 5% of range.

## Registered criteria (both seeds must pass)

Primary:

1. Drift: robust block2-flat median relative drift at 1% noise on the eight evaluation sets is ≤ 50% of the matched control's.
2. Noisy accuracy: mean R² under the A6 noise 1% and 3% suite (fixed ridge probe, labeled_data reported separately because every arm collapses there) beats the matched control.
3. Clean non-inferiority: fixed2 mean Δraw ≥ control − .005, raw wins ≥ control − 1, and no dataset regresses below control by more than −.02 on both seeds.

Secondary, report only: held-out perturbation family, Δrandom, 0106, late-block readouts, per-source corpus drift, and prediction MSE.

Decision: if all primary criteria pass, a full-length robust run (or robust continuation of all four 48p seeds for the ensemble) becomes the next proposal. A pass does not promote or export anything. Otherwise stop this recipe.

## Operations

- Code: `perturb()` in `src/spectral_lejepa/data/`, optional `training.robust_weight` (default 0, legacy behaviour unchanged), block2 tap in the model forward. Warm-start compatibility as in the projection pilot; resume semantics unchanged.
- Orchestration copies `scripts/projection_pilot.py`: registration hash, duplicate-start refusal, systemd services `spectralfm-robust-{control,robust}-s{401,402}`, healthy GPUs 0/1/2/4 by UUID, W&B group `robustness-continuation`. Outputs `outputs/robustness-continuation/`.
- Tests: perturbation determinism and identity at zero magnitude, legacy loss unchanged at `robust_weight=0`, warm-start strictness, GPU smoke.
- Cost: about 1.2× a pilot job because of the extra forward pass; four GPUs at once, similar wall time to the projection pilot.

## Open questions for the user

1. Are realistic instrument noise and baseline levels known from the hardware? If so, they replace the step 0 estimate.
2. Should the robust arm start from all four 48p seeds instead of seed2 only, so a pass feeds the ensemble directly? This costs 8 jobs instead of 4.

## Results (2026-10-13) — registered primary criteria FAIL

All four jobs completed 49,799 steps, the gradient audit, all-block evaluation, policies and perturbation evaluation. Analysis: `scripts/robustness_continuation_analysis.py` → `outputs/robustness-continuation/results_analysis.json`. Robust weight 0.1 (calibration rule; weighted term ≈1.7× prediction at block2, stronger than the 0.5× target).

| Check (both seeds) | s401 control → robust | s402 control → robust | Pass |
|---|---|---|---|
| Drift at 1% noise (median of medians) | .080 → .012 | .080 → .013 | yes (≈0.15×) |
| Small-set mean R², noise 1% | .599 → .675 | .607 → .676 | yes |
| Small-set mean R², noise 3% | −.677 → .447 | −.619 → .461 | yes |
| Clean fixed2 mean Δraw (≥ control − .005) | .0846 → .0436 | .0871 → .0469 | **no** |
| Clean raw wins (≥ control − 1) | 7 → 7 | 7 → 8 | yes |
| Repeated per-dataset regression < −.02 | 0055, 0112, 0113, 0120 | | **no** |

Per-dataset clean Δraw, control → robust (s401; s402 within .02): labeled_data .131 → .127, 0055 .051 → .030, 0106 −.104 → +.016, 0109 .007 → .031, 0112 .362 → −.001, 0113 .158 → .132, 0114 .001 → .003, 0120 .070 → .011. Paired robust − control: 0106 +.120±.033 and 0109 +.022±.008 improve; 0112 −.36±.10 is the dominant loss.

Secondary and held-out (fixed ridge, small-set mean R², control → robust): noise 5% −3.8 → +.05; shift 2 .31 → .29; offset 5% −.73 → −1.69; slope 5% .62 → .48; scale 1.05 .47 → .33; sine baseline 5% .13 → −.46. Feature drift falls for every perturbation, but accuracy under baseline-type perturbations (offset, slope, scale, sine) gets worse. labeled_data still collapses under noise for every arm (R² ≈ −1e5), so that probe failure is not about the encoder.

Interpretation:

- The term does what it targets. Block-2 features become about 6× less sensitive to noise, and noisy accuracy rises on both seeds.
- It also removes fine-detail signal that some sets depend on. 0112's +.36 gain, the largest in the benchmark, disappears. This matches the pre-launch concern that σ_max = .20 of row SD counts real narrow features as noise.
- The 0106 loss reverses (−.10 → +.02), and s402 is the first 8/8 raw-win encoder. This is a hint about 0106, not a candidate: its mean gain is half the control's.
- Invariance to baseline shifts did not transfer. Smaller drift with lower R² means the remaining feature change lines up with directions the probe uses.

Decision under the registration: stop this recipe. Nothing is promoted or exported, and the 8-job ensemble extension is not proposed. A follow-up would need a new registration. Candidates are noise-only augmentation at a much smaller σ_max (≤ .03, the range evaluated), with no baseline terms, and a weight below 0.1. The calibration grid should go below 0.1.
