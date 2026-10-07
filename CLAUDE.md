# CLAUDE.md — SpectralFM-LeJEPA

A LeJEPA-pretrained 1-D ViT foundation model for 245-point spectra. Science, architecture and
tensor shapes are in `README.md`; the latest experiment summary is in `plans/SUMMARY.md` (gitignored).

## Goal and scoring (user-defined; do not change without asking)

- **Goal.** One frozen model whose embedding beats the raw spectrum in `parameter_0`
  linear-probe regression on the majority (ideally all) of 8 sets:
  labeled_data, 0055, 0106, 0109, 0112, 0113, 0114, 0120. The merged small-set set is not scored.
- **Wins.**
  - Any nested-CV ΔR² > 0 vs raw is a win; ≥ +0.05 is a strong win.
  - 0114 counts like every other set.
- **Reporting.**
  - Every summary table shows the raw R² next to model R²/Δ.
  - Report the random-init control alongside, and trust only patterns that repeat across runs or seeds.
- **Equal normalization.** The model input is a global affine map (`data.normalization: global`), so amplitude is preserved as it is for raw.
- **Pretraining data.** It may use all spectra, including the evaluation spectra and all components (client-approved). Labels are never used in pretraining.

## Current best recipe

Configs: `configs/screen6.yaml` (arm `long_g1_lr50`) and `configs/long1.yaml`.
- global normalization, mask 0.75, SIGReg λ = 0.05;
- `training.global_weight=1`;
- `data.source_weights={labeled_regression: 0.5, default: 0.5}`.

At 3 epochs it wins 5/8 (3 strong), with mean Δ +0.045.

## Environment rules

- **W&B.** `WANDB_API_KEY` lives only in `~/.bashrc`, so launch W&B jobs via `bash -ic '...'`.
  - Never write or print the key.
  - W&B gets metrics and figures only, no artifacts (`wandb.log_checkpoints: false`).
- **Read-only paths.** Never write under `/mnt5/noy`. Never modify the parent repo `../SpectralFM-label-regression-eval-merged`.
- **Git.** Commit as `Hadar <hal.nls@gmail.com>`, directly on `main`. The user pushes.
- **GPUs.** 8× RTX 2080 Ti (11 GB, no bf16). GPU 3 is broken: with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  CUDA index k ≥ 3 is nvidia-smi k+1. Check free memory with `nvidia-smi -i N`.
- **CPU.** 40 cores, shared and often overloaded. Set `OMP_NUM_THREADS=4` and `JOBLIB_TEMP_FOLDER=/dev/shm`.
- **Data.** The packed array is at `/mnt5/home/hadar/nova/data/spectra_all/` (12,758,642 × 245 float32; `stats.json`, `valid_rows.npy`, `index.tsv`).

## Commands

```bash
uv sync && OMP_NUM_THREADS=4 uv run pytest                       # fast tests; -m slow = parent-equivalence tests
bash -ic 'uv run python scripts/pretrain.py <section.key=value ...>'
bash -ic 'uv run python -m scripts.evaluate --checkpoint <ckpt> --config configs/eval_small6.yaml'
bash -ic 'uv run python -m scripts.screen --config configs/<screen>.yaml --gpus 0,4,5'
```

- **Configs** are strict: an unknown key is an error. Override any value as `section.key=value`.
- **Evaluation** uses `evaluation.backend: torch`, a float64 GPU probe at exact parity with scikit-learn (`sklearn` stays the reference).
  - `eval_small*.yaml` scores the 7 small sets, then labeled_data.
  - `random_control: true` adds the untrained-network control.
- **Outputs** go to `outputs/<screen>/<run>_<timestamp>/` (checkpoints) and `outputs/<screen>/eval/<run>_step<N>/<set>/summary.json`.

## Code map

- `src/spectral_lejepa/data/loader.py`: packed memmap, input normalization, `SourceWeightedSampler`.
- `src/spectral_lejepa/models/`:
  - `tokenizer.py`: 24 contiguous patches, `Linear(11→256)`;
  - `masking.py`;
  - `vit_1d.py`: Encoder, Predictor, LeJEPA, EvalBackbone.
- `src/spectral_lejepa/training/`:
  - `loss.py`: token MSE + SIGReg | VISReg, and the optional global-view term;
  - `visreg.py`;
  - `trainer.py`;
  - `checkpoint.py`.
- `src/spectral_lejepa/evaluation/`:
  - `bank.py`: readouts mean/seg4/flat;
  - `nested.py`: nested CV and pairing;
  - `probe.py` (scikit-learn) and `probe_torch.py` (GPU).
- `scripts/`:
  - `pretrain.py`, `evaluate.py` (scorecard, random control);
  - `screen.py` (multi-GPU arm queue);
  - `parity_gpu_eval.py`;
  - `pack_*.py` (data packing).
