# SpectralFM-LeJEPA

## Contract

- Train a frozen encoder whose nested-CV `parameter_0` R² exceeds raw spectra on a majority of eight datasets: labeled_data, 0055, 0106, 0109, 0112, 0113, 0114, 0120.
- Any positive ΔR² is a win; ≥ +0.05 is a strong win. Keep raw and architecture/readout-matched random controls. Do not change the scoring rule.
- Labels never enter pretraining. All eligible spectra may be used, including evaluation spectra; evaluation is transductive.
- Select probe recipes inside inner CV. Distinguish adaptive layer/readout scores from fixed embeddings. Do not select a production checkpoint using outer-test scores.

## Current candidate

- Plan: `docs/superpowers/plans/2026-10-09-final-production-candidate.md`.
- Configs: `configs/final_model.yaml`, `configs/eval_final_model.yaml`.
- 48 patches, d256, 6 blocks; global normalization; 75% random masking; batch 256; SIGReg 0.05; pooled weight 1; expected 50/50 source sampling. AdamW 5e-4, ten epochs / 497,990 steps, seed 101.
- Export candidate: block-2 flat, 12,288 features; all six blocks train. Other blocks remain available in the full checkpoint for analysis.
- Service: `spectralfm-final-model-s101`; W&B run `ibdymf53`, group `final-model-fixed-block2`.
- Output: `outputs/final-model/`. Pipeline: training → TorchScript export → fixed-readout evaluation → benchmark gate → report.
- Gate:eight finite sets, at least five raw wins, positive mean gains versus raw/random, all canaries passing. Passing is benchmark qualification; deployment-domain validation remains separate.
- Existing 48-patch block-2 flat features win 7/8 in each of three seeds; 0106 loses. This readout choice is retrospective and is being replicated, not an architectural requirement.

## Repository map

- `src/spectral_lejepa/models/`: tokenizer, masking, shared encoder, predictor.
- `src/spectral_lejepa/training/`: objective, loop, checkpointing, diagnostics.
- `src/spectral_lejepa/evaluation/`: feature banks and nested probes.
- `scripts/pretrain.py`, `scripts/evaluate.py`: CLI entry points.
- `scripts/launch_final_model.sh`, `scripts/export_production.py`, `scripts/final_model_gate.py`: final pipeline.
- `scripts/build_results_report.py`: offline HTML, CSV and SVG exports.
- `README.md`: architecture, tensor shapes, commands. `docs/DATASETS.md`: corpus details. Historical decisions stay in `docs/superpowers/plans/`; local status in ignored `plans/SUMMARY.md`.

## Execution rules

- W&B credentials come from the user shell; launch jobs with `bash -ic`. Never print or store keys. Log metrics/figures, not checkpoint artifacts.
- Keep long jobs in user systemd services. Do not overwrite an active launcher or restart training for documentation changes.
- Physical GPU3 is broken. With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, logical CUDA 0–2 map directly; 3–6 map to physical 4–7. Final training uses CUDA 0, evaluation CUDA 6.
- Set `OMP_NUM_THREADS=4`, `JOBLIB_TEMP_FOLDER=/dev/shm`, writable `MPLCONFIGDIR`.
- Never write under `/mnt5/noy` or modify the parent evaluation repository.
- Commit on `main` as `Hadar <hal.nls@gmail.com>`. Push when the user requests it.
- Preserve checkpoint/RNG resume behavior, input normalization, loss weights and W&B axes during cleanup. Keep useful shape/numerical comments; avoid prose that repeats code.
