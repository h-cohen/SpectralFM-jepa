# SpectralFM-LeJEPA

## Contract

- Train a frozen encoder whose nested-CV `parameter_0` R² beats raw spectra on a majority (≥ 5) of eight datasets: labeled_data, 0055, 0106, 0109, 0112, 0113, 0114, 0120.
- Any positive ΔR² is a win; ≥ +0.05 is a strong win. The goal is a majority of wins, not large margins: matching raw on some sets is acceptable. Do not register criteria that require keeping a control's mean margin.
- Keep raw and architecture/readout-matched random-weight controls. Do not change the scoring rule. Every results table shows raw R².
- Labels never enter pretraining. All eligible spectra may be used, including evaluation spectra; evaluation is transductive.
- Select probe recipes inside inner CV. Do not select a production checkpoint using outer-test scores. Register criteria before results.

## Current state (2026-10-13)

| Model (block2-flat, 12,288 features) | Raw wins | Random wins | Mean ΔR² vs raw | Status |
|---|---|---|---|---|
| seed101 final (`outputs/final-model/production`) | 5/8 | 6/8 | +.0801 | Export. Not gate-qualified: 0120 single-shuffle canary .0204 > .020 |
| prior 48p seed2 | 7/8 | 6/8 | +.0894 | Retrospective holdout candidate |
| Ensemble of 48p seeds 0/1/2/101 | 7/8 | 6/8 | +.0858 | Passed registered criteria; holdout candidate |
| robust_s401 / robust_s402 (seed2 + robustness term) | 7/8, 8/8 | 8/8 | +.044, +.047 | Registered criteria failed; majority-contract candidate |
| Robust ensemble (robust_s401, robust_p0, robust_p1, robust_s101) | 8/8 | 8/8 | +.0556 | Passed all registered criteria; preferred holdout candidate |

- 0106 loses for every non-robust model; the robustness term turns it into a small win but erases 0112's +.36 gain.
- Single-shuffle canaries are noisy at n = 25–125. A 100-permutation audit shows real R² above every null for all arms on 0106 and 0120. The canary limit is unchanged.
- Release blocker: independent labelled spectra with group IDs. Score the candidates once on a locked group-disjoint holdout. No benchmark result is release approval.
- Recommendation and history: `docs/PRODUCTION_RECOMMENDATION.md`. Roadmap: `docs/superpowers/specs/2026-10-13-robust-production-roadmap.md`.

## Active job

- None. The robust four-seed ensemble completed on 2026-10-13: spec and results in `docs/superpowers/specs/2026-10-13-robust-ensemble.md`, analysis `outputs/robust-ensemble/results_analysis.json`.

## Completed studies

Each study has a spec or plan under `docs/superpowers/`, outputs under `outputs/<study>/`, and a W&B group of the same name. Orchestration code for studies before 2026-10-13 was removed in the cleanup; git history keeps it.

| Date | Study | Outcome |
|---|---|---|
| 10-02 to 10-08 | Screens 1–8, long-1, paired-long (24/48 patches, masking, VISReg, projector) | 48 patches, random 75% masking and SIGReg kept; VISReg gave no gain; the prediction projector was never adopted |
| 10-09 | Final model seed101 | 5/8, +.0801; gate fails on the 0120 canary |
| 10-11 | Production analysis; continuation study (seeds 202/203 × λ .05/.01) | Duration-only or lower-SIGReg continuations do not help |
| 10-12 | Frozen-feature study; projection pilot (seeds 301/302) | Block2-flat is the best fixed family; the SIGReg projector does not help |
| 10-13 | Production validation (seed ensemble, chemometric baselines, 0106 diagnosis, perturbation drift) | Ensemble passes; trained features amplify input noise about 20×; 0106 cause unknown |
| 10-13 | Robustness continuation (seed2, 401/402 × control/robust) | See current state above |
| 10-13 | Robust ensemble (seeds 0/1/101 robust continuations, training seeds 410/411/412; W&B p0 `8nfxzsde`, p1 `ya6s9o5p`, s101 `ok3up62j`) | 8/8 vs raw and random, +.0556; all criteria pass |

## Repository map

- `src/spectral_lejepa/models/`: tokenizer, masking, shared encoder, predictor.
- `src/spectral_lejepa/training/`: objective (`loss.py`), loop (`trainer.py`), checkpointing, diagnostics, gradient terms.
- `src/spectral_lejepa/data/`: corpus loaders; `perturb.py` for the optional robustness term.
- `src/spectral_lejepa/evaluation/`: feature banks, nested probes, paired bootstrap.
- `scripts/pretrain.py`, `scripts/evaluate.py`: CLI entry points.
- Final pipeline: `scripts/launch_final_model.sh`, `export_production.py`, `final_model_gate.py`.
- Audits: `gradient_audit.py`, `permutation_audit.py`.
- 2026-10-13 studies: `seed_ensemble.py`, `chemometric_baselines.py`, `diagnose_0106.py`, `robustness_perturbations.py`, `memorization_check.py`, `noise_calibration.py`, `robustness_continuation*.py`, `robustness_eval.py`, `robust_ensemble.py`.
- Report: `scripts/build_results_report.py` and its `*_report.py`, `results_matrix.py`, `readout_variants.py`, `report_story.py` section modules. These build offline HTML, CSV and SVG from saved outputs.
- `scripts/training_progress.py`: W&B presentation watcher (service `spectralfm-presentation`, stopped 2026-10-13).
- Training report: `docs/TRAINING_REPORT.md`; figures in `docs/figures/training-report/` from `python -m scripts.training_report` (`fetch` first, via `bash -ic`).
- Data packing: `pack_pickles.py`, `pack_spectra.py`. Corpus details: `docs/DATASETS.md`.
- Retired options (`model.projector`, `model.sigreg_projector`, `training.regularizer=visreg`) remain in saved configs; the code accepts only their disabled values.

## Execution rules

- W&B credentials come from the user shell; launch jobs with `bash -ic`. Never print or store keys. Log metrics and figures, not checkpoint artifacts.
- Keep long jobs in user systemd services. Do not overwrite an active launcher, and do not edit code that a running pipeline imports later. Do not restart training for documentation changes.
- Physical GPU3 is broken. With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, logical CUDA 0–2 map directly; 3–6 map to physical 4–7. Pin study jobs by GPU UUID.
- Set `OMP_NUM_THREADS=4`, `JOBLIB_TEMP_FOLDER=/dev/shm`, writable `MPLCONFIGDIR`.
- Never write under `/mnt5/noy` or modify the parent evaluation repository.
- Commit on `main` as `Hadar <hal.nls@gmail.com>`. The user authorized pushing completed work.
- Preserve checkpoint/RNG resume behavior, input normalization, loss weights and W&B axes during cleanup. Keep useful shape and numerical comments; avoid prose that repeats code.
- `plans/SUMMARY.md` is a tracked historical log. `plans/AGENT_HANDOFF.md` is local and ignored.
