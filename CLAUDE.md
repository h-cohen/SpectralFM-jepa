# SpectralFM-LeJEPA

## Aim and evaluation rules

- Aim: frozen spectral features whose nested-CV `parameter_0` R² exceeds raw spectra on as many as possible of the eight sets: labeled_data, 0055, 0106, 0109, 0112, 0113, 0114, 0120.
- Every results table shows raw R², alongside a random-weight encoder of the same architecture and readout.
- Labels never enter pretraining. All eligible spectra, including evaluation spectra, may be used; evaluation is transductive.
- Probe recipes are chosen inside inner CV. Never pick checkpoints, readouts or seeds from outer-test scores. Write down study criteria before seeing results.

## Current state (2026-10-14)

- **Candidate:** robust seed ensemble. Four members, each a stage-1 LeJEPA run plus a stage-2 robust continuation. Block-2 flat features (12,288); predictions are averaged.
  - It beats raw on 8/8 sets (+.0556) and random weights on 8/8 (+.0478).
  - Member evaluations live in `outputs/robust-ensemble/*/eval` and `outputs/robustness-continuation/robust_s401/eval`.
- **Exported single model:** `outputs/final-model/production` (seed101, 5/8). It is not gate-qualified, because the 0120 single-shuffle canary scored .0204 against a .020 limit.
- **Blocker:** no independent group-labelled holdout exists. No benchmark result is release approval.
- **Where things are documented:**
  - Results and figures: `docs/TRAINING_REPORT.md` (and `.html`, published as artifact `4pDknEu4iwtFTgEsbnMJyy`).
  - History: `docs/PRODUCTION_RECOMMENDATION.md`.
  - Study specs: `docs/superpowers/specs/`.
- **Running jobs:** none.

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
