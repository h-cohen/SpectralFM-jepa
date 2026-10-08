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

- At 3 epochs it wins 5/8 (3 strong), with mean Δ +0.045.
- The long-1 10-epoch run was stopped at 4 epochs (step 200k): both seeds win 5/8 (seed 0: mean Δ +0.069).
- Next step: a true resume to 10 epochs (see `plans/SUMMARY.md` §6).

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

## Training-progress presentation

- Live W&B view: https://wandb.ai/hcohen/spectralfm-lejepa/runs/u2b9gssq .
- `scripts/training_progress.py --watch` reads the original/resumed long-1 histories and completed eight-set local scorecards into a separate presentation run; it never writes to source training runs.
- Presentation user service: `spectralfm-presentation`; source manifest `outputs/training-progress/sources.json`; log `outputs/training-progress.log`; local HTML, figures, scorecards and watcher status under `outputs/training-progress/`.
- Panels: `seed0/`, `seed1/` (absolute optimizer steps, weighted losses, LR, progress, validation and representation health); `evaluation/` (raw/FM/random controls, per-dataset R², paired gains with bootstrap SD and the +0.05 threshold, experiment comparisons); `procedure/guide` (recipe, protocol and interpretation).
- Use `scripts/scorecard_figures.py` for evaluation and screen-8 plots. The live presentation can refresh its existing W&B run with `bash -ic '.venv/bin/python -m scripts.training_progress --refresh-evaluations'`; local PNG exports live under `outputs/training-progress/` and `outputs/screen-8/`.
- The live watcher resumes the presentation run recorded in `outputs/training-progress/run.json` and persists per-source W&B `_step` cursors in `outputs/training-progress/cursors.json`; status keeps both optimizer and history steps.
- Probe results remain checkpoint measurements. Falling pretraining loss is not proof of downstream gains. Validation/diagnostics remain every 25k steps.
- New training runs define explicit `optimizer_step` axes and weighted objective contributions through `utils/wandb.py`; the already-running jobs use the separate view to receive these additions without restart.

## Current experiment direction (2026-10-08)

- Plan: `docs/superpowers/plans/2026-10-07-screen8-final-direction.md`. Registered in W&B: https://wandb.ai/hcohen/spectralfm-lejepa/runs/fjbztblr .
- Long1 seeds0/1 run under `spectralfm-long1-s0` / `spectralfm-long1-s1`, with 10k-step checkpoints. Previous terminal-managed runs vanished near222k; their latest saved state was200k, so those unsaved steps were lost.
- Screen8 completed all six arms; its final assessment is https://wandb.ai/hcohen/spectralfm-lejepa/runs/3hki6g5s (controller: https://wandb.ai/hcohen/spectralfm-lejepa/runs/u4k2kge9). Neither random50 nor block75 passed every registered gate, so no masking candidate was promoted and no heavy follow-on training was launched.
- `scripts/assess_screen8.py` writes/logs complete eight-set decisions. Candidate gates are exploratory; they never change the user-defined win metric or automatically launch heavy training.
- The long1 milestone queue completed 250k and350k evaluations serially on CUDA6; both final 497,990-step checkpoints were also evaluated. Final scores are listed below.
- Inspect lifecycle with `systemctl --user status spectralfm-long1-s0 spectralfm-long1-s1 spectralfm-screen8 spectralfm-long1-milestones spectralfm-presentation`. Independent user services replace execution-tool sessions.
- Completed ten-epoch 24-patch runs ended at497,990 steps: seed0 wins5/8 (mean Δ +0.0442), seed1 wins3/8 (mean Δ +0.0105). The final result does not establish repeatable majority performance. Screen8 promoted neither masking change.
- Next is a fresh paired 24-vs48-patch comparison, seeds0/1/2, full10-epoch schedule, same packed data/mix/objective/evaluation, with raw and per-architecture random-init controls. Checkpoints are recovery only; the registered final checkpoint decides the recipe. See the follow-up section in the plan.
- All six confirmatory services are now active under `spectralfm-paired-p{24,48}-s{0,1,2}`. W&B run IDs: 24p s0 `tsf3hftz`, s1 `zdir34g7`, s2 `7ghobw5h`; 48p s0 `kknz92e3`, s1 `m1p2atdu`, s2 `ceanfxq1`. Logs: `outputs/paired-long/p{24,48}_s{0,1,2}.log`. Progress snapshot and links are in the local HTML report.
- Presentation-ready local report: `outputs/training-progress/results_report.html`; regenerate with `.venv/bin/python -m scripts.build_results_report`. W&B presentation `u2b9gssq` follows the previous long1 runs; new runs log individually in W&B group `paired-long-confirmatory`.
- Subagents use lighter `gpt-6-luna` via native Cavecrew delegation. No Caveman runtime cost telemetry is claimed.
