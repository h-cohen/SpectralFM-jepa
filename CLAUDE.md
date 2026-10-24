# SpectralFM-LeJEPA

## Aim and evaluation rules

- Aim: frozen spectral features with the highest mean nested-CV `parameter_0` R² gain over raw spectra across the eight sets (labeled_data, 0055, 0106, 0109, 0112, 0113, 0114, 0120). Since 2026-10-23 accuracy comes first: a set may fall at most .01 below raw. Robustness is deferred.
- Every results table shows raw R², alongside a random-weight encoder of the same architecture and readout.
- Labels never enter pretraining. All eligible spectra, including evaluation spectra, may be used; evaluation is transductive.
- Probe recipes are chosen inside inner CV. Never pick checkpoints, readouts or seeds from outer-test scores. Write down study criteria before seeing results.

## Current state (2026-10-24)

- **Candidate:** family-mix ensemble of 16 frozen encoders — 8 non-robust (stage 1) and 8 robust (stage 2) members, block-2 flat features; family means mixed with one inner-CV weight. Mean ΔR² +.095 vs raw, 8/8 sets (report section 1, Appendix D). Adopted 2026-10-24 with the 5/8 head-to-head test vs the robust ensemble waived by the user.
  - Members and scores: `outputs/accuracy-ensemble/` (`mix/`, `results_mix.json`, `mix_banks.txt`, `mix_groups.txt`); run with `ensemble --stack --no-stack-heads --mix-groups ... --banks ...`.
  - Previous candidate: robust four-seed ensemble (+.056, 8/8), still the noise-robust option. No inference export exists yet for the family mix (`export-ensemble` builds the robust four).
- **Exported single model:** `outputs/final-model/production` (seed101, 5/8). It is not gate-qualified, because the 0120 single-shuffle canary scored .0204 against a .020 limit.
- **Blocker:** no independent group-labelled holdout exists. No benchmark result is release approval.
- **Where things are documented:**
  - Results and figures: `docs/TRAINING_REPORT.md` (and `.html`, published as artifact `4pDknEu4iwtFTgEsbnMJyy`).
  - History: `docs/PRODUCTION_RECOMMENDATION.md`.
  - Study specs and plans: `docs/superpowers/` (local, untracked).
- **Running jobs:** none.
- **SIGReg/depth ablation (done 2026-10-14):** report section 8.
  - 9 two-epoch runs in `outputs/sigreg-depth-ablation/`, W&B group `sigreg-depth-ablation`.
  - λ = 0 collapses every block and block 2 falls below raw. Block 2 gains more at depth 6 than at depth 2.
  - `training-report` regenerates figures 13–14 from these outputs.
- **ROCKET-family benchmark (done 2026-10-15):** report Appendix A (main report compares JEPA with raw and data2vec only).
- **data2vec comparison (done 2026-10-17):** report section 9. `evaluate` accepts fairseq data2vec checkpoints (`evaluation/data2vec.py`, `data2vec` extra); outputs in `outputs/data2vec/`. Recent checkpoint 0/8 vs raw, mean −.197.
- **Error metrics and feature correlation (done 2026-10-19):** report sections 10–11. `benchmark score` adds RMSE, nRMSE, MAE, tail error, residual slope, RMSE by label quintile and per-spectrum win rate (`evaluation/errors.py`); `training-report` draws figures 18–22, including feature × feature correlation and label R² by variance floor across SIGReg weights (`isotropy.feature_correlation`, `isotropy.label_spectrum`).
  - `benchmark run|score` (sktime, `benchmark` extra); outputs in `outputs/benchmark/`, criteria in `docs/superpowers/specs/2026-10-15-rocket-benchmark.md`.
  - MiniRocket ties the ensemble 4–4 by set, mean −.043; mean-prediction average +.075 vs raw on 8/8. B1–B3 all failed.
  - MultiRocket on labeled_data needs the CPU probe (`--device cpu`); 11 GB GPUs run out of memory.

## Train

- **Stage 1** (one member): `python -m spectral_lejepa pretrain --config configs/final_model.yaml experiment.seed=S experiment.name=N`. 497,990 steps, about 19 h on one RTX 2080 Ti.
- **Stage 2:** `python -m spectral_lejepa pretrain --config configs/robust_continuation.yaml training.init_checkpoint=<stage-1 ckpt> experiment.seed=S experiment.name=N`. 49,799 steps, about 2 h.
  - The init checkpoint is required. Without it, the stage-2 recipe trains from scratch.
- **Long runs:** run them as user systemd services, launched through `bash -ic` (`systemd-run --user --unit=spectralfm-<name> bash -ic 'cd <repo> && ...'`).
  - Resume with `training.resume=<checkpoint>`; optimizer, scheduler and RNG state are restored.
  - `ops/launch_final_model.sh` is the single-model pipeline: train, export, evaluate, gate, report.

## Evaluate

- **Member probes:** `python -m spectral_lejepa evaluate --checkpoint C --config configs/eval_production_all_blocks.yaml experiment.output_dir=<dir>`.
  - Nested CV: 2×5 outer, 5 inner, 12 recipes. Covers every block and readout, plus raw and random-weight controls.
  - Writes `<dir>/<run>_step<N>/<set>/nested_oof.npz`.
- **Ensemble:** `python -m spectral_lejepa ensemble --banks <run dirs...> --output E.json`. Paired ΔR² vs raw and vs random, with bootstrap SD.
- **Perturbation robustness:** `python -m spectral_lejepa robustness --checkpoint C --output P.json`. Block-2 drift and fixed-ridge R² under noise, offset, slope, scale, shift and a sinusoid.
- **Embedding suite:** `python -m spectral_lejepa evaluate --suite embedding --config configs/eval_embedding.yaml`, then `embedding-figures`.
  - Runs the previous project's cosine-similarity and noise metrics from its pinned commit (`evaluation/legacy.py`).
  - Writes to `outputs/embedding-eval/`.
- **Inference:** `export-ensemble` builds `outputs/inference/robust-ensemble/` from `configs/inference.yaml`. `embed INPUT --output OUT.npz` or `spectral_lejepa.inference.SpectralEmbedder` returns (N, 4, 12288); README has a usage example.
- **Other tools:** `permutation-audit` and `gradient-audit`. `export` writes TorchScript: raw `(B, 245)` in, `(B, 12288)` out.
- **Report:**
  - `training-report`; run `training-report fetch` first, via `bash -ic`, to pull W&B histories.
  - `training-report-html`; republish the artifact from `docs/TRAINING_REPORT.html`.
  - `report` builds the older results HTML.

## Code map

- `src/spectral_lejepa/`; run `python -m spectral_lejepa --help` for every command.
  - `data/`: loaders, packing, perturbations, noise calibration.
  - `models/`: tokenizer, masking, encoder, predictor, export.
  - `training/`: loss, trainer, checkpoint, diagnostics, audit.
  - `evaluation/`: banks, nested probes, ensemble, gate, perturbation, embedding suite.
  - `reporting/`: reports and figures.
- `configs/`; `ops/`; `tests/` mirrors the package. Run tests with `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q`, and add `-m slow` for tests that need real data or the parent repo.
- Retired options (`model.projector`, `model.sigreg_projector`, `training.regularizer=visreg`) remain in saved configs. The code accepts only their disabled values.

## Execution rules

- **W&B:** credentials come only from the user shell (`bash -ic`). Never print or store keys. Log metrics and figures, not checkpoints.
- **GPUs:** physical GPU3 is broken. With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, logical CUDA 0–2 are physical 0–2, and 3–6 are physical 4–7. Pin study jobs by GPU UUID.
- **Environment:** set `OMP_NUM_THREADS=4`, `JOBLIB_TEMP_FOLDER=/dev/shm` and a writable `MPLCONFIGDIR`.
- **Running jobs:** never overwrite an active launcher or edit code that a running pipeline imports later. Do not restart training for documentation changes.
- **Write protection:** never write under `/mnt5/noy`, or into the parent repo `SpectralFM-label-regression-eval-merged`. That repo is read only via `git show` at the pinned commit.
- **Git:** commit as `Hadar <hal.nls@gmail.com>`; pushing completed work is authorized. `plans/` is local and untracked.
- **Cleanups:** preserve checkpoint/RNG resume, input normalization, loss weights and W&B axes. Keep shape and numerical comments.
