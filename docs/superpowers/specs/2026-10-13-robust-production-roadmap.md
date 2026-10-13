# Robust production roadmap — proposal, 2026-10-13

> Paths before 2026-10-14 refer to `scripts/`; see `docs/superpowers/specs/2026-10-14-repo-restructure-design.md`.

Status (2026-10-13): section A done (results below). Section C ran as the robustness continuation (completed) and the robust four-seed ensemble (completed; 8/8, all criteria pass); see their specs. Section B, independent labelled data, is still missing.

## Evidence review

Source: `outputs/frozen-feature-study/*/linear_summary.json`, block2-flat, primary linear protocol. ΔR² ± paired bootstrap SD.

| Dataset | n | Raw R² | Random R² | s0 Δraw | s1 Δraw | s2 Δraw | s101 Δraw | s2 Δrandom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0055 | 25 | .591 | .541 | +.044±.095 | +.005±.087 | +.071±.085 | +.023±.085 | +.121±.077 |
| 0106 | 64 | .697 | .702 | −.097±.034 | −.111±.037 | −.108±.039 | −.091±.036 | −.113±.037 |
| 0109 | 96 | .918 | .933 | +.005±.014 | +.008±.013 | +.010±.013 | −.006±.013 | −.005±.010 |
| 0112 | 68 | .377 | .364 | +.277±.114 | +.299±.120 | +.367±.120 | +.350±.119 | +.380±.100 |
| 0113 | 70 | .548 | .561 | +.172±.054 | +.185±.055 | +.172±.055 | +.208±.055 | +.159±.063 |
| 0114 | 230 | .983 | .982 | +.000±.001 | +.000±.002 | +.001±.001 | −.000±.002 | +.002±.001 |
| 0120 | 125 | .380 | .354 | +.027±.058 | +.054±.059 | +.072±.054 | +.028±.059 | +.098±.061 |
| labeled_data | 4716 | .830 | .949 | +.130±.004 | +.130±.004 | +.130±.004 | +.129±.005 | +.011±.003 |

Findings:

1. **Win counts are mostly noise.** Seven of eight sets have n ≤ 230. On 0055, 0109 and 0114 the effect is inside one SD. Seed2's 7/8 versus seed101's 5/8 comes entirely from 0109 (+.010 vs −.006, SD .013) and 0114 (+.001 vs −.000, SD .002). Fixed mean Δraw across the four 48p seeds is .070/.071/.089/.080 (mean ≈ .078). Seed2 is the top draw of four; expect regression toward ≈ .078 on fresh data. Seed2 is not a meaningfully better model than seed101.
2. **Robust effects are few.** Consistent across all seeds: gains on 0112 (~+.3) and 0113 (~+.18), a gain on labeled_data, and a loss on 0106 (~−.10, ~3 SD). Everything else is a coin flip.
3. **labeled_data gain is mostly architecture, not pretraining.** Random-weight features reach .949 vs raw .830. Trained adds only +.011 over random. The linear raw probe underfits this n=4716 set. Δraw there measures probe capacity.
4. **0106 is a systematic, training-induced loss.** Random ≈ raw on 0106, trained flat loses to both. Block2 mean (+.022) beats raw while block2 flat loses. Layer0-flat is +.014. So pretraining removes or scrambles 0106-relevant token detail that pooling partly hides. Cause not yet diagnosed.
5. **Late blocks degrade.** Block6 flat loses on most sets (0106 −.30, labeled −.12). The top of the encoder specializes for masked prediction. Production uses an early block. Exported encoder needs only blocks 1–2.
6. **The benchmark is spent as a selection set.** Seeds, readout, block, continuation, head and projector decisions all used these labels. Further benchmark-driven selection adds forking paths without adding evidence.
7. **The 0120 canary is underpowered.** One shuffle seed against a fixed .020 cap on n=125 is a noisy test. The 100-permutation diagnostic is the stronger test. The original failure stands; any fix must be a prospective v2 gate for future candidates.

## What a better and more robust model needs

### A. No new pretraining (recommended first)

1. **Seed ensemble, preregistered.** Use frozen block2-flat from all four 48p seeds (0/1/2/101). Fit one probe per seed inside the same inner CV and average predictions. Rationale: removes seed lottery without choosing a seed. Cost: 4× a two-block encoder at inference. Register criterion before running: mean Δraw ≥ mean of the four single seeds and no dataset worse than the worst single seed.
2. **Stronger raw references, reported only.** Add PLS, SNV/first-derivative + PLS and kernel ridge on raw spectra under the same nested CV. Keep the contract scoring rule against the original raw linear comparator unchanged. Purpose: show whether the encoder beats standard chemometrics, not just linear raw.
3. **Report Δrandom beside Δraw everywhere.** Pretraining claims use Δrandom. Labeled_data shows why.
4. **Pooled uncertainty.** Add inverse-variance pooled ΔR² across sets and across seeds. Report win counts with a sign-flip probability per set. Do not replace the contract rule.
5. **Diagnose 0106 without labels in the loop.** Compare 0106 spectra with the corpus: range, normalization statistics, peak width relative to 245/48 interpolation, share of corpus rows from its source. Check per-token feature variance on 0106 versus other sets. Output: one cause hypothesis to test.
6. **Robustness checks on frozen features.** Perturb inputs (noise, baseline offset and slope, multiplicative scale, ±1-bin shift) and measure feature drift and probe R² drop for raw, random and trained. This tests deployment robustness, not selection.

### A registration (fixed before running)

All A analyses reuse the frozen-feature-study banks and the primary protocol: seed42, 2 repeats × 5 outer folds, 5 inner folds, identical outer folds for every arm. Outputs go to `outputs/production-validation/`. No checkpoint, export or gate changes.

- A1 seed ensemble: per dataset, average the saved block2-flat outer-fold predictions of the 48p seeds 0, 1, 2 and 101 with equal weight. No weighting or seed subset is tuned. Pass if both hold: (a) ensemble mean Δraw ≥ mean of the four single-seed means (.0776); (b) on every dataset, ensemble Δraw ≥ the worst single seed. Report wins, Δraw, Δrandom and paired SD. A pass makes the ensemble the preferred candidate for the independent holdout; it does not qualify a release.
- A2 raw chemometric references: PLS (components selected in inner CV), SNV + Savitzky–Golay first derivative + PLS, and RBF kernel ridge (alpha and gamma selected in inner CV), on raw spectra. Reported beside the contract raw comparator; the contract scoring rule is unchanged.
- A3/A4 reporting: Δrandom beside Δraw; inverse-variance pooled ΔR² across datasets with its SD.
- A5 0106 diagnosis: label-free statistics only; labels may be used only to report already-computed scores.
- A6 robustness: probes fit on clean training folds with the inner-selected recipe, then scored on perturbed test rows. Perturbations and magnitudes are fixed in the script before scoring. Raw, random and trained arms receive identical perturbations.

### A results (2026-10-13)

Scripts: `scripts/seed_ensemble.py`, `chemometric_baselines.py`, `diagnose_0106.py`, `robustness_perturbations.py`, `memorization_check.py`. Outputs: `outputs/production-validation/`.

- **A1 seed ensemble: passed.** 7/8 raw wins, mean Δraw +.0858 (bar .0776), no set below its worst seed. Δrandom +.0780, 6/8. 0106 still −.093±.035. The ensemble is now the preferred holdout candidate, beside the seed101 export and seed2.
- **A3/A4 pooled.** Inverse-variance pooled Δraw +.017±.001, Δrandom +.0037±.0012. labeled_data and 0114 dominate the weights because their SD is smallest. Pretraining's pooled gain over random features is small.
- **A2 chemometrics.** RBF kernel ridge is the strongest classical arm and beats contract raw on 7/8. Seed2 block2-flat beats the inner-selected chemometric union on labeled_data (+.18), 0112 (+.15), 0113 (+.18) and 0120 (+.08); 0055/0109 within noise; 0114 tie; 0106 −.12. On labeled_data, PLS (.600) and KRR (.782) score below linear raw (.830), so their grids likely underfit n=4716.
- **A5 0106: no cause found.** Ruled out: out-of-distribution inputs, rarity (0106 gets 16.7% of sampling mass, ~43× per row), tokenization or peak width, amplitude, and information loss (CKA .974, input PCs recovered R² .97). Weak lead: 0106 has effective rank 2.8 and n=64, raw and random already reach .70, and flat features spread signal and add probe variance. Each seed's loss is ~1.5 SD alone; the evidence is cross-seed consistency.
- **A6 robustness: trained features amplify noise.** Block2-flat relative drift under 1% noise is .11 (raw .005, random .0015). A fixed standardize+RidgeCV probe, which is also the production recipe on labeled_data, does not remove it. At 3% noise the trained arm falls below raw on 0055, 0106, 0109 and 0114. labeled_data collapses for every arm under noise; likely near-constant columns amplified by standardization, not verified. Offset, slope and scale give no trained advantage over raw.
- **Memorization: not supported.** Drift is lower on the oversampled `labeled_regression` rows (.042) than on `single_channel_all` (.223) and `multi_channel` (.177). Non-oversampled `labeled_data` matches the oversampled rows (.051). Masked-prediction error follows the same source pattern. Fragility tracks source type, not repetition: the encoder generalizes poorly to perturbed and single-channel spectra.

Consequence for C: a robustness objective is now the leading training candidate. Use noise, shift and baseline views in the invariance branch, gated by a matched control and a prespecified drift and noisy-R² criterion. 0106 has no specific fix.

### B. Data (release blocker)

Independent 245-value spectra with `parameter_0` and source/simulation/batch IDs. Lock group-disjoint development splits and a reserved release holdout before any scoring. Evaluate the seed101 export, seed2 and the seed ensemble once on the holdout. Also request ≥ 200 labelled rows for any small set that matters for deployment; n=25 cannot separate models.

### C. One training experiment, only if A5 gives a cause and the user authorizes it

Candidate directions, each with matched control and fixed2 primary interface kept:

- **Spectral-physics views for the invariance branch** (baseline offset/slope, multiplicative scatter, noise, small shift). Targets instrument robustness, which masking alone does not train.
- **0106-targeted fix** chosen from A5, for example finer tokenization if peak width is the cause, or source reweighting if 0106-like spectra are rare.
- **Shallower encoder** (depth 3) as a compute-saving production variant, since only blocks 1–2 are exported. Equivalence test, not an accuracy claim.

Do not run more duration, λ or projector sweeps.

## Gate v2 (prospective, for new candidates only)

- Permutation canary: real R² above ≥ 99 of 100 label permutations per set, instead of one shuffle seed against .020.
- Seed-ensemble or ≥ 3-seed mean as the scored model, so one draw does not decide release.
- Holdout evaluation on independent groups is the release criterion. The benchmark becomes a regression check.
