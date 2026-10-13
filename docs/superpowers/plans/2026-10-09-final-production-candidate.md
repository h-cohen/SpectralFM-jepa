# Final fixed-interface candidate — approved 2026-10-09

Status (2026-10-13): completed plan. Historical record; current state is in CLAUDE.md.

## Recipe and rationale

- From scratch; predeclared seed101. 48 patches, d256, 6 encoder blocks, 8 heads; predictor 192×4.
- Global affine input normalization from packed corpus statistics. All eligible packed training rows; expected source mass 50/50 regression-source/rest. Labels excluded.
- Random 75% masking; batch 256; token SIGReg .05; pooled objective weight 1 with 2 half-subset views. AdamW 5e-4, wd .05; one-epoch warmup, cosine decay to .001× peak; 10 epochs/497,990 steps. No untested schedule extension.
- Fixed production features: block2, flat, 12,288 dimensions. Training still uses all 6 blocks; inference needs tokenizer+first 2 blocks only, without the final block6 LayerNorm.
- Nomination evidence is retrospective: this fixed interface wins 7/8 in each prior 48p seed; 0106 loses. It is not a pre-registered result of the previous layer/readout-selection study. This fresh run tests the now-fixed interface; no later checkpoint/readout/seed selection.

## Execution

- Training config: `configs/final_model.yaml`; fixed evaluation: `configs/eval_final_model.yaml` (`readouts=[flat]`, `flat_blocks=[2]`).
- Durable user service: `spectralfm-final-model-s101`; CUDA 0 trains, CUDA 6 (physical 7) evaluates. Physical GPU3 excluded. Output: `outputs/final-model/`.
- Pipeline: `scripts/launch_final_model.sh`: train → extract final checkpoint → export → serialized eight-set evaluation with raw/random controls → benchmark gate → offline report refresh.
- Recovery checkpoints every 10k; validation/representation diagnostics every 25k; W&B group `final-model-fixed-block2`. No checkpoint artifacts uploaded.
- Portable export: `outputs/final-model/production/encoder.pt`; raw FP32 (B,245) → FP32 (B,12288). Includes normalization. Metadata records source checkpoint SHA256, seed, step, stats, readout and gate outcome.
- Before launch: two-step CUDA smoke succeeds; scripted export matches the reference block2 readout at B1 and B3; config equality proves only the approved model patch count/experiment identity differs from the tested recipe.

## Predeclared gate

- Exactly 8 complete finite evaluations, >=5 positive raw wins, positive mean gain versus raw, positive mean gain versus same-architecture/same-readout random initialization, all raw/model canaries pass.
- Export and evaluation must share checkpoint SHA256, seed101, step 497,990 and fixed block2-flat interface.
- Passing qualifies the benchmark candidate; independent deployment-domain validation remains required. Existing spectra were eligible for unlabeled pretraining, so the evaluation is transductive.
- A failure is reported, not followed by an automatic seed retry, checkpoint search or schedule extension.

## Documentation

- README shortened around architecture, algorithm, tensor shapes and execution.
- Same tensor flow added to offline HTML report, with all 6 training blocks and the 2 block inference path distinguished.
- Run manifest records prelaunch time and file hashes. Local and W&B logs retain progress and final assessment.

## Launch and verification ledger

- Launched durable service 2026-10-09 at 11:58:30 UTC−4. W&B run `ibdymf53`: https://wandb.ai/hcohen/spectralfm-lejepa/runs/ibdymf53 . Output run directory: `outputs/final-model/production_p48_s101_20261009-185847` (directory timestamp uses host timezone).
- CUDA 0 smoke completed 2 steps; its real checkpoint exported successfully and matched native block2 features. Targeted suite 24 passed, including scripted dynamic B1/B3 parity, unchanged recipe proof, gate controls and diagram/report tensor labels.
- Live job advanced to 4,000 steps with finite loss .8420, gradient norm 18.04, recent throughput .150 s/step. Service active and GPU0 utilization 75%, 6,546 MiB. NVML warning is present because of the known broken GPU; CUDA smoke and actual training both work.
- Resolved training pool 12,748,642 rows; diagnostic validation subset 2,048; global corpus 12,758,642. Source masses default/regression .5/.5. Training config and core training code hashes unchanged since registration.
- Diagram visually inspected after rasterizing SVG; final report uses scoped SVG styles. Postlaunch report/gate logging refinements are recorded separately from prelaunch file hashes; they do not change training or qualification thresholds.

## Recovery (2026-10-10)

- Original run failed at 75,000 during W&B diagnostic PNG creation: `OSError: [Errno 28] No space left on device`. Logged loss .201128 and gradient norm .481664 were finite; failure was media logging, not observed numerical divergence.
- Last intact recovery checkpoint is 70,000 with optimizer, scheduler, scaler and complete Python/NumPy/Torch/CUDA/mask RNG state. Resume validation confirms the original 497,990 step schedule. 5,000 unsaved steps are replayed.
- Launcher now supports `--resume CHECKPOINT`, preserves the previous training log, and uses repository-local `.tmp` on the filesystem with 1.5 TiB free instead of shared `/tmp`.
- `log_figure` closes figures in a finally block and warns on ENOSPC/EDQUOT without aborting training; unrelated logging errors still propagate. Loss/checkpoint failures remain fatal. Regression tests cover both paths.
- Recovery service: `spectralfm-final-model-s101-recovery`; new W&B run `ylymmcnn`: https://wandb.ai/hcohen/spectralfm-lejepa/runs/ylymmcnn . Prior run `ibdymf53` remains in manifest lineage. Same seed, model, objective, sampling, schedule and final fixed-readout gate.

## Completed candidate

- Training reached 497,990; export and all eight fixed block2-flat evaluations completed. Checkpoint SHA256, seed101, readout, finite output shape (1,12288) and native reference parity verified.
- Raw wins 5/8; three strong wins; mean ΔR² +0.080077 vs raw / +0.072296 vs matched random. Losses: 0106 −0.0908, 0109 −0.0056, 0114 −0.0002.
- Gate remains not qualified: 0120 shuffled-label block R² 0.0204456318 exceeds the existing 0.020 threshold. Other checks pass. A single borderline shuffle result is not proof of leakage; no threshold was relaxed or seed rerun to obtain a pass. Assessment: https://wandb.ai/hcohen/spectralfm-lejepa/runs/4qnq98xi .
- Report refresh failed on an undefined `_finite` helper. Added the helper and regression tests for completed assessments and missing deltas; 11 focused tests pass. Report regenerated with candidate scores and explicit qualification blocker. Training/evaluation not rerun.
- Artifacts:`outputs/final-model/production/{encoder.pt,metadata.json}`,`outputs/final-model/assessment.json`. Deployment remains unqualified pending investigation and independent domain validation.
