# Final fixed-interface candidate — approved 2026-10-09

## Recipe and rationale

- From scratch; predeclared seed101.48patches,d256,6encoderblocks,8heads; predictor192×4.
- Global affine input normalization from packed corpus statistics. All eligible packed training rows; expected source mass50/50regression-source/rest. Labels excluded.
- Random75%masking; batch256; tokenSIGReg.05; pooled objective weight1 with2half-subset views. AdamW5e-4,wd.05; one-epoch warmup,cosine decay to.001×peak;10epochs/497,990steps. No untested schedule extension.
- Fixed production features: block2,flat,12,288dimensions. Training still uses all6blocks; inference needs tokenizer+first2blocks only, without the final block6LayerNorm.
- Nomination evidence is retrospective: this fixed interface wins7/8 in each prior48pseed;0106loses. It is not a pre-registered result of the previous layer/readout-selection study. This fresh run tests the now-fixed interface; no later checkpoint/readout/seed selection.

## Execution

- Training config:`configs/final_model.yaml`; fixed evaluation:`configs/eval_final_model.yaml` (`readouts=[flat]`,`flat_blocks=[2]`).
- Durable user service:`spectralfm-final-model-s101`; CUDA0trains,CUDA6(physical7)evaluates. PhysicalGPU3excluded. Output:`outputs/final-model/`.
- Pipeline:`scripts/launch_final_model.sh`: train → extract final checkpoint → export → serialized eight-set evaluation with raw/random controls → benchmark gate → offline report refresh.
- Recovery checkpoints every10k; validation/representation diagnostics every25k; W&B group`final-model-fixed-block2`. No checkpoint artifacts uploaded.
- Portable export:`outputs/final-model/production/encoder.pt`; rawFP32(B,245)→FP32(B,12288). Includes normalization. Metadata records source checkpoint SHA256,seed,step,stats,readout and gate outcome.
- Before launch: two-step CUDA smoke succeeds; scripted export matches the reference block2readout atB1andB3; config equality proves only the approved model patch count/experiment identity differs from the tested recipe.

## Predeclared gate

- Exactly8complete finite evaluations,>=5positive raw wins,positive mean gain versus raw,positive mean gain versus same-architecture/same-readout random initialization,all raw/model canaries pass.
- Export and evaluation must share checkpoint SHA256,seed101,step497,990andfixed block2-flat interface.
- Passing qualifies the benchmark candidate; independent deployment-domain validation remains required. Existing spectra were eligible for unlabeled pretraining, so the evaluation is transductive.
- A failure is reported, not followed by an automatic seed retry, checkpoint search or schedule extension.

## Documentation

- README shortened around architecture,algorithm,tensor shapes and execution.
- Same tensor flow added to offline HTML report, with all6trainingblocks and the2block inference path distinguished.
- Run manifest records prelaunch time and file hashes. Local and W&B logs retain progress and final assessment.

## Launch and verification ledger

- Launched durable service2026-10-09at11:58:30UTC−4. W&B run`ibdymf53`: https://wandb.ai/hcohen/spectralfm-lejepa/runs/ibdymf53 . Output run directory:`outputs/final-model/production_p48_s101_20261009-185847` (directory timestamp uses host timezone).
- CUDA0smoke completed2steps; its real checkpoint exported successfully and matched native block2features. Targeted suite24passed, including scripted dynamic B1/B3 parity, unchanged recipe proof, gate controls and diagram/report tensor labels.
- Live job advanced to4,000steps with finite loss.8420,gradient norm18.04, recent throughput.150s/step. Serviceactive andGPU0utilization75%,6,546MiB. NVML warning is present because of the known broken GPU; CUDA smoke and actual training both work.
- Resolved training pool12,748,642rows; diagnostic validation subset2,048; global corpus12,758,642. Source masses default/regression.5/.5. Training config and core training code hashes unchanged since registration.
- Diagram visually inspected after rasterizing SVG; final report uses scoped SVG styles. Postlaunch report/gate logging refinements are recorded separately from prelaunch file hashes; they do not change training or qualification thresholds.

## Recovery (2026-10-10)

- Original run failed at75,000during W&B diagnostic PNG creation: `OSError: [Errno28] No space left on device`. Logged loss.201128andgradient norm.481664were finite; failure was media logging, not observed numerical divergence.
- Last intact recovery checkpoint is70,000with optimizer,scheduler,scaler and complete Python/NumPy/Torch/CUDA/mask RNG state. Resume validation confirms the original497,990step schedule.5,000unsaved steps are replayed.
- Launcher now supports `--resume CHECKPOINT`, preserves the previous training log, and uses repository-local `.tmp` on the filesystem with1.5TiBfree instead of shared `/tmp`.
- `log_figure` closes figures in a finally block and warns on ENOSPC/EDQUOT without aborting training; unrelated logging errors still propagate. Loss/checkpoint failures remain fatal.Regression tests cover both paths.
- Recovery service:`spectralfm-final-model-s101-recovery`; new W&B run`ylymmcnn`: https://wandb.ai/hcohen/spectralfm-lejepa/runs/ylymmcnn . Prior run`ibdymf53`remains in manifest lineage. Same seed,model,objective,sampling,schedule and final fixed-readout gate.
