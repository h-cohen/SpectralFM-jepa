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

Config: `configs/paired_long.yaml` with `model.num_patches=48`; matched training seeds0/1/2. Preferred by the completed paired study's pre-launch exploratory rule.
- global normalization, mask 0.75, SIGReg λ = 0.05;
- `training.global_weight=1`;
- `data.source_weights={labeled_regression: 0.5, default: 0.5}`.

- At the registered ten-epoch final checkpoint,48p seed wins are[5,4,7]; its seed-mean profile wins5/8 with mean ΔR²+.067132.24p has[6,5,4],5/8 seed-mean wins,+.046065.
-48p has positive paired architecture gain in2/3 seeds (mean+.021067). Both recipes pass the majority rule; all-eight remains unmet. This is exploratory recipe selection, not statistical significance.

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
- Offline report now tells the objective → geometry → downstream story. `scripts/report_evidence.py` builds CPU-only diagnostics from saved final legacy banks; `scripts/report_story.py` exports slide figures under `outputs/training-progress/figures/`. `scripts/local_wandb.py` reads local histories without modifying training runs.
- Whitening diagnostic (fixed layer6 mean, all eight datasets, both legacy final seeds, nested-selected policies): 3/16 positive deltas, mean ΔR² −0.071562. This is separate from the main adaptive layer/readout scorecard. SIGReg λ=.02/.05 early sensitivity is suggestive only; no matched SIGReg-off result exists. Covariance isotropy alone does not establish predictive benefit.
- User clarified the presentation: show SIGReg's anisotropy→isotropy intuition, not a separate whitening procedure. The HTML now excludes all whitening content and the previous geometry/policy figures; those generated figures are archived under `outputs/training-progress/archived-diagnostics/`. Current slide figures: `sigreg_anisotropy_to_isotropy` (explicit synthetic schematic, same scales), `sigreg_early_sensitivity`, `paired_training_geometry_history`, each PNG/SVG. Actual curves are measured; the ellipse→circle is a target illustration, not manufactured experimental evidence.
- Final paired study completed (2026-10-09): all six final checkpoints and eight-set evaluations exist under `outputs/long-1/eval/p{24,48}_s*_step497990/`. Seed wins:24p[6,5,4],48p[5,4,7]. Both seed-mean profiles beat raw5/8; mean ΔR²24p+.046065,48p+.067132.48p wins2/3 pairs with mean advantage+.021067, passing the pre-launch exploratory preference rule in commit `ae25377`. All-eight remains unmet.
- Single offline report includes final backbone×dataset R²±bootstrap-SD matrix (raw,6trained,2architecture-matched random controls), separate three-seed means±sample-SD, and a complete46-scorecard archive matrix. `scripts/results_matrix.py` extracts source uncertainties, including random-control nested JSONs. Full-precision exports: `outputs/training-progress/{final_results_matrix.csv,all_backbones_results_matrix.csv,results_matrix.json}`. Historical and fresh cohort trajectories are separated.
- Latest presentation corrections: sample counts beneath dataset headers; green cells for strictly positive unrounded R² gain over each row's matching raw control (including aggregate rows). Removed synthetic SIGReg illustration, lifecycle completion section, masking decision section and historical trajectory plot. Only measured SIGReg sensitivity/history figures remain. Historical early-stopping note: legacy seeds both6/8 at250k versus5/8 and3/8 final; retrospective evidence motivates an independently validated stopping rule, not selecting by outer-test scores.17focused tests passed; generated report checked for all removals, sample counts, win highlights, unique IDs and offline dependencies.
- Readout analysis added: `scripts/readout_variants.py` loads17fixed stage/readout families for all6final backbones (102trained+102matched-random rows). The report has a fixed-readout selector, bootstrap/seed-SD tables, and two8-panel “Where the signal lives” plots, by architecture. Plot bands are seed SD; flat only measured atblocks0–2, mean/seg4 at0–6; block6 includes final LayerNorm.
- Production candidate discovery is retrospective:48p fixed `layer2/flat` (12,288dims) wins7/8 in each of seeds0/1/2;0106 consistently loses. Final mean-pooling is much weaker. This does not establish information loss in final tokens (final flatten unmeasured). A fresh fixed-readout replication with predeclared seed101 was proposed but has NOT been launched: user first requested report variants/depth analysis before the final recipe decision. README/dataflow work likewise remains pending this decision.
- User subsequently approved the fixed interface and final run. Launched2026-10-09at11:58:30UTC−4 under durable `spectralfm-final-model-s101`; W&B https://wandb.ai/hcohen/spectralfm-lejepa/runs/ibdymf53 (group`final-model-fixed-block2`). CUDA0trains;CUDA6(physical7)evaluates. Configs:`configs/final_model.yaml`,`configs/eval_final_model.yaml`. Same proven10epochs/497,990steps; seed101; fixed block2-flat; no adaptive layer or checkpoint search.
- Automatic pipeline:`scripts/launch_final_model.sh` trains→exports→evaluates→gates→refreshes report. Export:`outputs/final-model/production/encoder.pt`, rawFP32(B,245)→(B,12288), normalization included. SHA256-linked metadata and predeclared gate checks prevent mismatched export/evaluation. Benchmark qualification requires8finite sets,>=5raw wins,positive mean raw/random gains,all canaries; deployment-domain validation remains separate. No automatic seed retry or extended training.
- Verified two-stepCUDA smoke, actual smoke checkpoint TorchScript export, B1/B3 parity, and24focused tests. GPU0/service/W&B advancement verified (step4,000loss.842grad18.04); current training config and core training files match prelaunch hashes. README is88lines with Mermaid architecture/tensor table and commands; report embeds the same tensor contract via `scripts/architecture_diagram.py` and exports `tensor_dataflow.svg`.
