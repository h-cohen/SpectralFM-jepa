# Embedding similarity and noise robustness — design, 2026-10-14

Status: proposed, awaiting user review. It depends on the repository restructure (`2026-10-14-repo-restructure-design.md`).

## Goal

Add the previous project's cosine-similarity and noise-robustness evaluations (`eval.runner` in `SpectralFM-label-regression-eval-merged`) to the standard evaluation. Run them on the robust ensemble and produce per-dataset plots.

## Decisions (user, 2026-10-14)

- **Signal reconstruction is skipped.** It needs a trained decoder, and none exists for this encoder. The reconstruction-based per-dataset plots on that repo's branch `feature/recon-dataset-plots` are therefore out of scope.
- **Models.** The four robust members are robust_s401, robust_p0, robust_p1 and robust_s101. The ensemble is reported as the mean of the members' scores.
- **Readouts.** Block-2 flat (production, 12,288 values) and block-2 mean (256 values, the old pooled convention applied to the block that ships).

## Why the old evaluator is not driven directly

The old evaluator applies per-spectrum layer norm before the model sees the data, and adds noise in that space. Our encoder needs raw spectra under one global affine normalization, and it cannot recover raw values from per-spectrum-normalized input. The metric definitions are therefore reused, but inputs are prepared here.

## Definitions reused from the previous project

Source: commit `6237feb` (`code/eval/runner.py`, `evaluations/`), read-only. It is imported from `<repo>/code` when importable; otherwise the constants below are copied with a source reference.

- **Data.** Manifests `single_channel_1k/train` (sanity), `single_channel_10k/valid` (in_dist), `multi_channel/valid` (multi_ch) and `sampled_data/valid` (samples). Whole stacks of 10 consecutive rows are drawn with seed 42, at the old set sizes: sanity 100, in_dist 500, multi_ch 1,000, samples 1,000.
- **Noise** (`NOISE_CONFIGS`): gaussian_std N(0, .01), gaussian_mean N(2, .001), shot_low Poisson scale .1, shot_high Poisson scale .05, gain_low ×N(1, .05), gain_high ×N(1, .1). All are applied to the per-spectrum-normalized signal, exactly as before.
- **Similarity.**
  - Top-5 cosine neighbours in input space and in embedding space, scored as the fraction from the same stack.
  - N×N cosine distributions for intra-stack, random inter-stack and preset-position inter-stack pairs.
- **Structured panel.**
  - The fixed 100-spectrum panel, seed 42: single_channel_all 3 stacks × 10, multi_channel 3 components × 10, sampled_data 2 × 10, labeled_data 2 × 10.
  - 100×100 cosine matrices, raw and mean-centred.

## Input path

```
raw x ─ per-spectrum norm (μᵢ, σᵢ) ─ old noise ─ × σᵢ + μᵢ ─ global norm ─ encoder blocks 1–2 ─ readout
```

Clean inputs take the same path without the noise step, so clean and noisy inputs differ only by the noise.

## Datasets

- Similarity: sanity and in_dist, which are single-channel, as in the old runner. Stacks are defined only there.
- Panel: its own fixed 100 spectra, which span the four sources.
- Noise robustness: sanity, in_dist, multi_ch and samples, plus the eight labeled evaluation sets (labeled_data, 0055, 0106, 0109, 0112, 0113, 0114, 0120). The old runner used only sanity and in_dist. Noise robustness needs no stack structure.

## Components

- `evaluation/similarity.py`: `topk_stack_match`, `cosine_distributions`, `panel_matrices`. Pure functions on arrays.
- `evaluation/noise.py`: `NOISE_CONFIGS`, `apply_noise(x_norm, kind, generator)`, `denormalize`, `noise_cosine`.
- `evaluation/embedding_eval.py`: loads the checkpoint, prepares inputs, extracts block-2 readouts, runs both metrics per dataset, and writes results.
- Figures go in `reporting/embedding_figures.py`, one folder per dataset:
  - match rate, input vs block-2 mean vs block-2 flat;
  - cosine distributions, intra vs inter;
  - noise cosine per type and readout;
  - panel heatmaps.
  - A cross-dataset summary figure (noise cosine × dataset).
- CLI: `python -m spectral_lejepa evaluate --suite embedding --checkpoint ... --config configs/eval_embedding.yaml`. The existing `--suite probe` path, the nested CV, is unchanged.

## Outputs

- `outputs/embedding-eval/<model>/<dataset>/results.json`.
- `outputs/embedding-eval/summary.csv`, plus the source commit hash and the configuration.
- Figures go to `outputs/embedding-eval/figures/<dataset>/`, and the summary figures are also copied to `docs/figures/embedding-eval/`.
- A new training-report section, "Embedding similarity and noise robustness", in both Markdown and HTML.

## Compute

- One healthy GPU (physical 4) for frozen inference only. Expected under 30 minutes for the four members.
- No training and no W&B.
- Nothing is written in `SpectralFM-label-regression-eval-merged`.

## Tests

- `denormalize(per-spectrum-normalize(x))` equals x to within 1e-6.
- The noise draw is identical to the old implementation for a fixed generator, compared against the old function when importable.
- On a small fixed subset, input-space top-k match and the cosine distributions equal the old functions' output.
- The flat readout has shape (B, 12288). The mean readout has shape (B, 256).

## Interpretation limits

- These metrics describe embedding geometry. They are not accuracy.
- High noise cosine with poor probe R² would not count as a win under the contract.
- Raw and random-weight references were not requested. The figures show the input-space reference only where the old evaluation defined one (top-k match, panel).
