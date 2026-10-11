# SpectralFM-LeJEPA — experiment summary (2026-10-07)

Local working document (gitignored). Code is at `main` (`beabdf3` and later); Resume implementation committed on main as da18751; user handles pushing.

## 1. Goal and criterion

- **Goal.** A single frozen foundation model (FM) whose embedding beats the raw 245-point spectrum
  in `parameter_0` linear-probe regression on the majority, ideally all, of the 8 sets.
  - Sets: labeled_data (LD, n = 4,716), 0055 (25), 0106 (64), 0109 (96), 0112 (68), 0113 (70), 0114 (230), 0120 (125).
  - The merged small-set set is not scored.
- **Metric.** Nested CV: 2×5 outer, 5 inner, seed 42.
  - The arm and the recipe are chosen inside each fold; a recipe is a normalizer {none, standardize, whiten, whiten8/32/128} × probe {RidgeCV, OLS}.
  - ΔR² = R²(embedding) − R²(raw), on the same rows, folds and fold-internal normalizers.
  - Any ΔR² > 0 is a **win**; ≥ +0.05 is a **strong win** (user rule).
- **Equal normalization.** The model input is a fixed global affine map, so it keeps amplitude,
  exactly as raw does.
- **Pretraining data.** All 12,758,642 spectra, including the evaluation spectra and every component; labels are never used. The client approved this.
- **Control.** Every model is also scored against a random-init copy (same architecture and input normalization).

## 2. Fixed setup

- **Model.** 1-D ViT, 24 patches (5×11 + 19×10 samples), d = 256, 6 pre-LN blocks, 8 heads.
  - Predictor: d = 192, 4 blocks.
  - 6.63M parameters.
  - One shared encoder for context and target: no EMA, no stop-gradient.
- **Loss.** MSE at masked positions + 0.05 · SIGReg, applied per position over the batch ([24, B, D]); lightly's implementation.
  - Optional sample-level term β·[(1−λ)·view-invariance + λ·SIGReg(pooled views)].
  - Views: the full spectrum plus V random patch subsets.
- **Optimizer.** AdamW 5e-4, wd 0.05, batch 256, warmup 10% of steps, cosine to 1e-3× the peak, fp16.
- **Data.** One packed memmap: `/mnt5/home/hadar/nova/data/spectra_all/` (global μ = 0.3784, σ = 0.5126).
- **Readouts per encoder layer:** mean (256 dims), seg4 (1,024), and flat (6,144; layers 0–2 only).
- **Probe.** A float64 GPU port of the probe, at exact parity with scikit-learn (|ΔR²| ≤ 2e-5, 0/200 recipe flips); LD takes about 20 min.

## 3. Experiments

Every number below is model R², with ✅ where it beats raw. *Δ vs rand* is the mean over the 8 sets of (model Δ − random-init Δ).

| model | recipe | LD | 0055 | 0106 | 0109 | 0112 | 0113 | 0114 | 0120 | wins | ≥.05 | mean Δ | Δ vs rand |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **raw R²** | | **0.830** | **0.591** | **0.697** | **0.918** | **0.377** | **0.548** | **0.983** | **0.380** | | | | |
| s2 zscore | per-sample z-score, mask .5, 1 ep | 0.923✅ | 0.123 | 0.373 | 0.686 | 0.376 | 0.451 | 0.985✅ | 0.378 | 2/8 | 1 | -0.129 | +0.027 |
| s2 global | global norm, mask .5, 1 ep | 0.961✅ | 0.502 | 0.602 | 0.894 | 0.087 | 0.539 | 0.985✅ | 0.373 | 2/8 | 1 | -0.047 | -0.042 |
| s2 mask75 | global norm, mask .75, 1 ep | 0.964✅ | 0.523 | 0.619 | 0.916 | 0.455✅ | 0.551✅ | 0.982 | 0.342 | 3/8 | 2 | +0.003 | +0.009 |
| s2 wide | mask75 recipe, d384×8, 1 ep | 0.964✅ | 0.395 | 0.618 | 0.919✅ | 0.218 | 0.434 | 0.985✅ | 0.329 | 3/8 | 1 | -0.058 | – |
| s3 cont_natural | mask75 +15k steps, natural mix | 0.963✅ | 0.610✅ | 0.625 | 0.909 | 0.275 | 0.631✅ | 0.984✅ | 0.365 | 4/8 | 2 | +0.005 | – |
| s3 cont_lr50 | +15k, 50% labeled-regression mix | 0.965✅ | 0.626✅ | 0.628 | 0.909 | 0.363 | 0.498 | 0.984✅ | 0.389✅ | 4/8 | 1 | +0.005 | – |
| s3 cont_lr80 | +15k, 80% mix | 0.965✅ | 0.575 | 0.598 | 0.903 | 0.506✅ | 0.575✅ | 0.983✅ | 0.363 | 4/8 | 2 | +0.018 | – |
| s4 cont_g025 | +15k, global term β=.25 | 0.964✅ | 0.690✅ | 0.595 | 0.910 | 0.386✅ | 0.553✅ | 0.982 | 0.365 | 4/8 | 2 | +0.015 | – |
| s4 cont_g1 | +15k, global term β=1 | 0.963✅ | 0.605✅ | 0.615 | 0.909 | 0.440✅ | 0.560✅ | 0.983✅ | 0.365 | 5/8 | 2 | +0.015 | +0.021 |
| s4 cont_g1_v4 | +15k, β=1, 4 views keep .25 | 0.966✅ | 0.559 | 0.662 | 0.903 | -0.381 | 0.531 | 0.984✅ | 0.362 | 2/8 | 1 | -0.092 | – |
| s5 visreg λ.6 | mask75 recipe, VISReg λ=.6, 1 ep | 0.962✅ | 0.440 | 0.512 | 0.882 | 0.139 | 0.505 | 0.984✅ | 0.375 | 2/8 | 1 | -0.066 | – |
| s5 visreg λ.9 | mask75 recipe, VISReg λ=.9, 1 ep | 0.967✅ | 0.627✅ | 0.666 | 0.906 | 0.317 | 0.533 | 0.984✅ | 0.372 | 3/8 | 1 | +0.006 | – |
| s5 visreg_cont | +15k with VISReg | 0.964✅ | 0.331 | 0.621 | 0.902 | 0.301 | 0.542 | 0.983✅ | 0.355 | 2/8 | 1 | -0.041 | – |
| s6 long_g1 | β=1, 3 ep, seed 0 | 0.969✅ | 0.543 | -0.174 | 0.920✅ | 0.295 | 0.520 | 0.984✅ | 0.389✅ | 4/8 | 1 | -0.110 | -0.104 |
| s6 long_g1_s1 | β=1, 3 ep, seed 1 | 0.969✅ | 0.507 | 0.616 | 0.930✅ | 0.410✅ | 0.507 | 0.982 | 0.394✅ | 4/8 | 1 | -0.001 | +0.005 |
| s6 long_g025 | β=.25, 3 ep | 0.967✅ | 0.437 | 0.677 | 0.930✅ | 0.144 | 0.530 | 0.983 | 0.407✅ | 3/8 | 1 | -0.031 | -0.025 |
| **s6 long_g1_lr50** | **β=1 + 50% mix, 3 ep** | 0.962✅ | 0.707✅ | 0.593 | 0.892 | 0.686✅ | 0.562✅ | 0.985✅ | 0.301 | 5/8 | 3 | +0.045 | +0.051 |
| s7 p12_lr50 | as s6 long_g1_lr50, 12 patches | 0.952✅ | 0.457 | 0.655 | 0.912 | 0.799✅ | 0.549✅ | 0.985✅ | 0.380 | 4/8 | 2 | +0.046 | +0.055 |
| s7 p48_lr50 | as s6 long_g1_lr50, 48 patches | 0.960✅ | 0.586 | 0.635 | 0.917 | 0.588✅ | 0.617✅ | 0.985✅ | 0.393✅ | 5/8 | 3 | +0.045 | +0.108 |
| long-1 lr50_s0 @200k | β=1 + 50% mix, 4 of 10 ep | 0.929✅ | 0.647✅ | 0.699✅ | 0.889 | 0.684✅ | 0.687✅ | 0.983 | 0.361 | 5/8 | 4 | +0.069 | +0.075 |
| long-1 lr50_s1 @200k | β=1 + 50% mix, 4 of 10 ep | 0.962✅ | 0.539 | 0.698✅ | 0.896 | 0.404✅ | 0.586✅ | 0.985✅ | 0.377 | 5/8 | 1 | +0.016 | +0.021 |
| long-1 lr80_s0 @200k | β=1 + 80% mix, 4 of 10 ep | 0.844✅ | 0.407 | 0.691 | 0.905 | 0.693✅ | 0.707✅ | 0.985✅ | 0.288 | 4/8 | 2 | +0.025 | +0.030 |
| *random init* | same arch, global norm, untrained | 0.945✅ | 0.463 | 0.677 | 0.944✅ | 0.368 | 0.566✅ | 0.983✅ | 0.331 | 4/8 | 1 | -0.006 | 0 |

The screens tested:

| Screen | Changes |
|---|---|
| s1 (not shown) | per-sample z-score, single_channel_one only; every change hurt |
| s2 | input normalization and capacity |
| s3 | source-balanced continued pretraining |
| s4 | sample-level global-view term |
| s5 | VISReg (arXiv 2606.02572) in place of SIGReg |
| s6 | 3-epoch from-scratch runs of the combined levers |
| s7 | tokenizer resolution |

Configs are in `configs/screen{2..7}.yaml` and `configs/long1.yaml`. Some outputs live in the
old worktree `../SpectralFM-jepa-wt/outputs`.

## 4. Findings

1. **Global affine input normalization is required.** Per-sample z-score deletes amplitude, which
   carries most of the small-set signal (0055 raw R² 0.59 → 0.18 under z-score), and it lost
   heavily on every small set (s2 zscore: mean Δ −0.129).
2. **Mask 0.75 > 0.5.** A wider model (d384×8) gives no gain.
3. **The sample-level global term (β = 1, 2 half-subset views) helps.** It is the best lever in
   short continuation (s4 cont_g1: 5/8). More views or a weaker β does not help.
4. **The labeled-regression-balanced mix helps most consistently.** The 14 small families hold
   1.2% of the spectra and get 50% of each batch.
   - 0112 goes from about 0.38 (raw) to 0.69–0.80 in every mixed run: s3 lr80, s6 lr50, s7 p12.
   - s6 long_g1_lr50 has the best mean Δ (+0.045) and the best Δ vs random init (+0.051).
5. **VISReg gives no gain.** λ = 0.6 (the paper's small-data setting) is worse; λ = 0.9 ≈ SIGReg;
   switching to it mid-training hurts.
6. **Tokenizer resolution.** 12 ≈ 24 ≈ 48 patches overall: 48 patches gives 5/8 and mean Δ +0.045,
   wins 0120, and has the largest margin over its own random-init control (+0.108).
7. **The small sets are noise-dominated.**
   - Two seeds of one recipe (s6 long_g1 vs long_g1_s1) differ by 0.79 on 0106 and by 0.12 on 0112.
   - Single-set differences below ~0.1 are not interpretable; only patterns across runs count.
8. **The random-init network is already a strong baseline** (4/8 wins, LD 0.945). It beats raw
   on 0109 (0.944 vs 0.918), where most pretrained models lose. Part of the win count therefore
   comes from the architecture and readout, not from what pretraining learned.
9. **The hard sets:**
   - **0106:** beaten only by long-1 lr50 at 200k, by +0.001 to +0.003.
   - **0109:** lost by every model with the mix (best pretrained 0.930; random init 0.944).
   - **0120:** won only by some 3-epoch runs and by s7 p48, by +0.01 to +0.03.
10. **On the small sets, the mean Δ does not trend with training steps**
    (mask75 checkpoints at 0, 20k, 40k and 50k steps: −0.023, −0.023, −0.022, −0.015).
    LD improves with training: 0.945 (random) → 0.964 → 0.969.

11. **Long run at 200k steps (4 of 10 epochs).**
    - **Both seeds of the 50% mix win 5/8.** Seed 0: 4 strong wins, mean Δ +0.069, the best of any model. Seed 1: mean Δ +0.016.
    - **0106 is beaten for the first time,** by both seeds, but only by +0.001 to +0.003, which is within noise.
    - **The 80% mix starves labeled_data** (0.844, against 0.93–0.96 elsewhere): labeled_data is in the "default" group.
    - **labeled_data dips mid-schedule for seed 0** (0.929): learning rate is still high at 40% of the cosine.

**Status against the goal:**
- **Majority reached, and it repeats across seeds.** 5/8 wins for s6 long_g1_lr50, s7 p48, and both long-1 lr50 seeds at 200k.
- **Consistent wins:** labeled_data, 0112, 0113 and 0114. The fifth win varies by model (0055, 0106 or 0120).
- **"All 8" not reached:** 0109 is lost by every pretrained model with the mix, and random init beats raw there (0.944).

## 5. Run state when stopped (2026-10-07 17:20)

- **long-1** (`configs/long1.yaml`; 10 epochs = 497,990 steps): lr50_s0, lr50_s1 and lr80_s0, stopped at step 200k (4 epochs) on the user's instruction.
  - Checkpoints every 50k: `outputs/long-1/<run>/checkpoint_step{50000..200000}.pt`.
  - The 200k checkpoints are scored, with the random control, in `outputs/long-1/eval/`.
- **screen-7** `lr50_s2` (third seed of the s6 winner): stopped at 25k; not usable.
- **W&B project** `spectralfm-lejepa` holds metrics, tables and figures only (free tier, no artifacts).

## 6. What can be tested next (priority order)

1. **True resume of long-1 lr50 (seeds 0 and 1) to 10 epochs.**
   - Checkpoints already hold the optimizer and scaler state; add `training.resume=<ckpt>` that restores the step, optimizer, scaler and LR schedule.
   - `init_checkpoint` restarts the schedule, so it is not a resume.
   - Check whether the cosine tail lifts labeled_data back to ≥ 0.96 while keeping the small-set gains.
   - Drop lr80: it costs labeled_data.
2. **Seed-averaged claims.** Score 3 seeds of the final recipe and report per-set mean ± sd. Claim a set only if the mean Δ > 0.
3. **Targeting 0106/0109/0120,** where pretraining loses information that random features keep:
   - an auxiliary input-reconstruction head (MAE-style) on masked patches, to keep fine spectral detail;
   - a lower mask ratio, or block masking, combined with the 50% mix;
   - 48 patches together with the long schedule (best margin over random init; wins 0120);
   - a multi-scale tokenizer (parallel 12 + 48 patches);
   - stronger mixing toward the losing families (per-family weights, e.g. upweight 0106/0109/0120).
4. **Readout.** A fixed, pre-registered readout (e.g. the layer chosen on LD) removes selection
   noise on the tiny sets. That changes the protocol, so it needs the user's approval.
5. **Evaluation noise.** More outer repeats (2 → 10) on the small sets would shrink protocol
   variance. That also changes the protocol, so it needs approval.
6. **Other JEPA variants.**
   - EMA-teacher I-JEPA or data2vec-style targets, against the shared encoder;
   - V-JEPA multi-block masking;
   - the projector head (`model.projector=mlp`);
   - a λ sweep (0.02–0.1).
7. **Scale.** Batch 512–1024 (SIGReg statistics improve with N); depth 8 together with the mix.

## 7. Housekeeping

- **Push.** `git push origin main` (user; the agent's push is blocked).
- **Worktrees.** `../SpectralFM-jepa-wt` (branch `screen-3`, merged) and `../SpectralFM-jepa-gpu` (branch `gpu-eval`, merged) can be removed once their outputs are no longer needed. The wt holds the s3–s5 checkpoints and rescore outputs.
- **Drive.** The rclone token for gdrive expired; run `rclone config reconnect gdrive:`.

## 8. Resumed training (2026-10-07)

- User authorized continuing training toward majority/all-eight wins using all available spectra and Superpowers plan files.
- Plan: `docs/superpowers/plans/2026-10-07-resume-long1.md`.
- Implementation commit: `da18751`; strict `training.resume` restores weights, AdamW, scaler, absolute step and cosine schedule. New checkpoints include scheduler and global/mask RNG state.
- Legacy 200k checkpoints lack RNG state: continued optimization is correct, exact uninterrupted stochastic replay is unavailable. Unweighted loader shuffle RNG is also not saved; these runs use deterministic epoch-seeded source-weighted sampling.
- Both 50% mix seeds resumed from 200000 to target 497990 with all packed training rows and unchanged recipe/evaluation protocol. lr80 dropped.
- Seed 0: GPU 0; `outputs/long-1/lr50_s0_resume_20261007-191908/`; log `outputs/long-1/resume_s0.log`; W&B https://wandb.ai/hcohen/spectralfm-lejepa/runs/wxrdfq4n . Verified at step 200750 with finite loss.
- Seed 1: GPU 1; `outputs/long-1/lr50_s1_resume_20261007-191916/`; log `outputs/long-1/resume_s1.log`; W&B https://wandb.ai/hcohen/spectralfm-lejepa/runs/wuw24e91 . Verified at step 200650 with finite loss.
- Persistent execution sessions: seed 0 = 75671, seed 1 = 3722. Initial detached launch did not survive; obsolete `.pid` files belong to failed launch, not current jobs.
- Launcher `scripts/resume_long1.sh` automatically evaluates the exact final emitted checkpoint with `configs/eval_long1.yaml`, all eight sets and random-init control, if training exits successfully.
- Checkpoints every 50k; validation/diagnostics every 25k. No new score claims until evaluation completes.
- Verification: 157 fast tests passed, 8 deselected, one sandbox NVML warning. Independent review found no blocker for weighted long-1 resume.
- Next: inspect 250k/300k progress and diagnostics; compare final scorecards with raw and random control across both seeds; then determine whether a third seed or 48-patch long run is justified.

## 9. W&B presentation (2026-10-07)

- User requested a presentation of the training procedure and progress.
- Live view: https://wandb.ai/hcohen/spectralfm-lejepa/runs/hkvku2ea . Original runs remain read-only and continue training.
- Watcher session 54577, log `outputs/training-progress.log`. Local exports under `outputs/training-progress/`.
- Backfills original/resumed histories for both seeds on absolute optimizer coordinates; logs actual weighted objective contributions and percentage/remaining steps. Copies measured validation and representation health without inventing more frequent measurements.
- 27 completed eight-dataset scorecards published, including raw/model/random R², paired uncertainty versus raw/random where measured, and checkpoint steps. Partial and merged-set scores cannot inflate win counts.
- Includes historical experiment ΔR² heatmap, latest evaluated checkpoint comparison by seed, and a recipe/protocol guide. Latest evaluation figures advance automatically when completed evaluations appear.
- Verified remote metrics/media and a subsequent live refresh (205150/204900 steps). Latest probe claims still refer to 200k checkpoints.
- Future native pretraining runs also define explicit optimizer-step axes, loss contributions and progress metrics. Current training jobs were not restarted.

## 10. Durable continuation and controlled direction screen (2026-10-08)

### Correction to prior run-state claims

The earlier training/presentation execution sessions did not survive. Logs end at222250/221750; no training processes remained and all healthy GPUs were idle. No checkpoint beyond200k existed. Therefore roughly22k logged steps per seed were lost; they were never scored. Exact termination cause is unknown. Training now runs under independent systemd user services, verified by PID/cgroup and subsequent live telemetry. Checkpoints are every10k instead of50k.

### Running experiments

- `spectralfm-long1-s0` / `spectralfm-long1-s1`: original ten-epoch recipe from200k, target497990. Directories `outputs/long-1/lr50_s0_resume_20261007-235742` and `...lr50_s1_resume_20261007-235743`. W&B https://wandb.ai/hcohen/spectralfm-lejepa/runs/8zrttyj7 and https://wandb.ai/hcohen/spectralfm-lejepa/runs/3cj18b2q . Original logs are appended, preserving failed attempts.
- `spectralfm-screen8`: six matched15k continuations (`configs/screen8.yaml`), fresh optimizer/schedule identically in every arm, varying only random75/random50/block75 within each seed. Four workers CUDA2/3/4/5, max two evaluations. Model/data/seed-specific initial checkpoints held fixed. First four arms confirmed via remote loss metrics; block arms queued. Log `outputs/screen-8.log`. Controller https://wandb.ai/hcohen/spectralfm-lejepa/runs/u4k2kge9 .
- `spectralfm-long1-milestones`: serial probes at250k/350k on CUDA6; waits for exact checkpoints and verifies main service remains active. Final evaluations belong to the original resume launchers, preventing duplicate writers. Log `outputs/long-1/milestones.log`.
- `spectralfm-presentation`: live W&B view https://wandb.ai/hcohen/spectralfm-lejepa/runs/u2b9gssq ; source manifest follows new resumed IDs and ignores the lost trajectories. Completed scorecards from screen8 automatically enter the presentation.
- Registered design/gates: https://wandb.ai/hcohen/spectralfm-lejepa/runs/fjbztblr ; plan `docs/superpowers/plans/2026-10-07-screen8-final-direction.md`. All science, configuration, source lineage and measured terms are logged; no model/checkpoint artifacts or credentials are stored. W&B-managed tables use SDK metadata artifacts.

### Why this direction

Both200k seeds lose0109 against raw and their random controls; 0106 gains are only+.001–.003 and0120 remains slightly below raw. More context or structured masking can test detail retention without changing architecture. 48 patches already helps0120 but does not solve0106/0109, so it is reserved rather than promoted from one seed. Complete the cosine tail separately to test labeled_data recovery.

Promotion gates: at least5/8 raw wins in each seed, hard-set mean gain over matched continuation positive in both seeds, positive overall mean gain, seed-mean LD regression no worse than.01 and stable-set regression no worse than.05. These are exploratory recipe-selection guards, not new win thresholds or confidence claims. A null15k screen is inconclusive for longer training.

Provisional final recipe stays24 patches/mask75/global weight1/50% mix/10 epochs. Choose a replacement only after repeated masking gains and longer confirmation; otherwise investigate repeated48-patch long training before reconstruction/teacher changes. Final claims need three seeds and per-set raw/FM/random means and seed SD. One frozen model must meet the goal; dataset-specific model selection is excluded.

### Agent cost

Evidence investigator, focused builder and independent reviewer all use lighter gpt-6-luna with Cavecrew. No external Caveman telemetry is available/claimed. Current experiment decisions rely on source evidence and predefined gates.

- Final code verification for this continuation:174fast tests passed,8deselected, one sandbox NVML warning. Registered runtime results remain pending; no masking arm has been promoted.

## 11. Informative W&B scorecard figures (2026-10-08)

- User approved a per-dataset R² figure (raw/FM/random init) and paired-gain figure (vs raw and random init, paired bootstrap SD, zero and +0.05 threshold).
- Added shared plotting in `scripts/scorecard_figures.py`; evaluation runs log both figures and save PNGs. Screen-8 decision logs per-arm/per-seed/per-set figures, distinguishing random-init and matched-continuation gains. Live presentation logs latest evaluated seed figures and retains the historical Δ heatmap/table.
- Live presentation can update existing run `u2b9gssq` without replaying source training histories: `bash -ic '.venv/bin/python -m scripts.training_progress --refresh-evaluations'`.
- At initial figure rollout, the 29 complete scorecards showed both seeds' latest evaluated step at 200,000; newer live training weights were not represented as evaluated.
- The initial screen-8 assessment `sy5sup9s` was pending and has since been superseded by the completed final assessment recorded in §12.
- The live presentation `u2b9gssq` now reports 35 complete scorecards and refreshes current training telemetry; the latest evaluated long-run probe checkpoint is 250,000, not the live weight step.
- The active presentation watcher was restarted on the same W&B run. It persists per-source W&B history `_step` in `outputs/training-progress/cursors.json` and records optimizer and history coordinates in status to avoid replay or gaps on later restart.
- Local figure exports: `outputs/training-progress/latest_per_dataset_r2.png`, `outputs/training-progress/latest_paired_gains.png`, `outputs/screen-8/per_dataset_r2.png`, `outputs/screen-8/paired_gains.png`.
- Full verification:182passed,8deselected, one expected sandbox NVML warning. Cavecrew review caught and guided a cursor fallback fix; regression passed.

## 12. Current operational check (2026-10-08 05:08 IDT)

- **Active:** `spectralfm-long1-s0`, `spectralfm-long1-s1`, `spectralfm-long1-milestones`, and `spectralfm-presentation` each reported active in the user systemd manager. Both long runs are progressing with finite losses at approximately 345.6k/345.7k of 497,990 steps. Screen8 is absent from the active-service list because its controller completed successfully.
- **Long1 results:** the two 250k scorecards each beat raw on 6/8 datasets. Mean ΔR² is +0.0812 for seed0 and +0.0385 for seed1. These are intermediate checkpoint results; final recipe claims await the completed schedule and further seeds.
- **Screen8 complete:** all six arms are evaluated, `missing_arms=[]`, and the final assessor/W&B presentation is https://wandb.ai/hcohen/spectralfm-lejepa/runs/3hki6g5s. Random50 failed the wins-per-seed gate (5/8 and 4/8). Block75 failed the hard-set-gain-each-seed gate (seed1 −0.0424). Neither is promoted; no follow-on heavy training was launched.
- **Waiting by design:** both 250k milestone evaluations completed. The active serial milestone queue awaits the 350k checkpoints before evaluating each seed. Main launchers handle final evaluations after training.
- **Presentation:** live W&B view https://wandb.ai/hcohen/spectralfm-lejepa/runs/u2b9gssq is active with 35 completed scorecards and telemetry refreshed through roughly 345k. Current training telemetry and evaluated checkpoint scores are shown as separate evidence.
- **Next decisions:** review both 350k results against the 250k trajectory, then review final checkpoint scores when both ten-epoch runs finish. Keep the provisional 24-patch/mask75 recipe unless a repeated candidate clears the registered gates; if neither masking candidate does, compare a repeated 48-patch long run before reconstruction/teacher changes. All-eight and final recipe confirmation remain open goals.

## 13. Paired architecture comparison and presentation report (2026-10-08 16:20 IDT)

- The previous 24-patch ten-epoch runs completed: seed0 5/8 raw wins (mean ΔR² +0.0442), seed1 3/8 (+0.0105). The only prior 48-patch measurement is seed0 at149,397 steps/3 epochs, with5/8 wins; it is exploratory. Screen8 promoted neither masking variant.
- A fresh paired 24-vs48-patch experiment is now running from scratch on seeds0/1/2, 10 epochs each, all12,758,642 packed spectra eligible, same50/50 source mix/global norm/mask/objective/schedule, with matching raw and random-init controls.
- Six durable services started at16:14 IDT under user systemd. CUDA0/1/2 map to24-patch seeds0/1/2; CUDA3/4/5 map to48-patch seeds0/1/2. CUDA6 (physical GPU7) is reserved for final evaluation; an exclusive `flock` serializes nested-CV probes. Physical GPU3 remains excluded.
- W&B runs (group `paired-long-confirmatory`): 24p seeds0/1/2 = `tsf3hftz`, `zdir34g7`, `7ghobw5h`; 48p seeds0/1/2 = `kknz92e3`, `m1p2atdu`, `ceanfxq1`. Each run has separate loss/progress/validation/representation metrics and local log under `outputs/paired-long/`.
- The first W&B audit at16:20 showed steps1,800–2,350; every run was active and had finite loss and gradient norm. The one-time scheduler-before-optimizer warning appeared for the 24p seed starts; subsequent metrics are advancing. No CUDA OOM or non-finite losses observed.
- `outputs/paired-long/manifest.json` stores the report's active-run snapshot. Rebuild the self-contained report with `.venv/bin/python -m scripts.build_results_report`; artifact is `outputs/training-progress/results_report.html`. It presents final historical scorecards, dataset-level raw/FM/random comparisons, checkpoint trajectories, screen8 gates, and clickable W&B run links with the current progress snapshot.
- Code/config/report tests:6passed; full repo suite before the last report snapshot enhancement:187passed,8deselected, one expected sandbox NVML warning. The report-specific and launcher-specific suite after that enhancement:6passed.
- Nothing is waiting on a launch. The six full runs and their final evaluations are in progress; the next scientific decision follows the paired final scorecards. Intermediate downstream probes will not choose checkpoints.

## Completed paired study and compiled report (2026-10-09)

- All six fresh paired jobs have final497,990-step eight-set evaluations under `outputs/long-1/eval/p{24,48}_s*_step497990/`.
-24p seed wins[6,5,4];48p[5,4,7]. Both seed-mean profiles beat raw5/8. Mean ΔR²24p+.046065;48p+.067132. Paired architecture gains[+.027441,−.005270,+.041031], mean+.021067:48p passes the pre-launch exploratory preference rule. All-eight remains unmet.
- Report: `outputs/training-progress/results_report.html`. Final9backbone×8dataset matrix includes raw,6trained,2random-init controls; all72cells have R²±measured bootstrap SD. Separate architecture means show three-seed sample SD. Complete historical matrix covers46evaluation scorecards and125model/raw/random rows; absent historical datasets remain blank.
- Full-precision CSVs: `final_results_matrix.csv` and `all_backbones_results_matrix.csv`; structured provenance/data: `results_matrix.json`, in the same report directory. No separate whitening material is displayed; SIGReg conceptual illustration and measured final histories are retained. Fresh finals are not joined onto legacy continuation trajectories.
- Validation:17focused tests passed; source uncertainty coverage and complete six-seed roster checked; independent lighter-model review confirmed arithmetic and uncertainty extraction. Pre-launch registration independently verified in commit `ae25377`. `git diff --check` clean.

## Approved final fixed-interface run (2026-10-09)

- User approved48patches,predeclaredseed101,unchanged10epoch schedule,andfixedblock2-flat12,288dembedding. Durable service`spectralfm-final-model-s101` launched11:58:30UTC−4; W&B https://wandb.ai/hcohen/spectralfm-lejepa/runs/ibdymf53 . CUDA0training;CUDA6/physical7finalevaluation.
- Pipeline/configs:`scripts/launch_final_model.sh`,`configs/final_model.yaml`,`configs/eval_final_model.yaml`. Automatic final TorchScript export→fixed-readout8set raw/random evaluation→predeclared qualification gate→report refresh. Export includes normalization andfirst2blocks,rawFP32(B,245)→(B,12288). No layer/seed/checkpoint selection or schedule extension.
- Resolved497,990steps,49,799warmupsteps,12,748,642eligibletrainingrows,2,048diagnosticvalidationspectra;source masses50/50. Qualify only with8finite sets,>=5raw wins,positive mean raw/random gains,andallcanaries. Benchmarkqualificationdoesnotreplaceindependentdeploymentvalidation.
- CUDA2step smoke andreal-checkpoint export succeeded. Scripted B1/B3 parity and24focused tests passed. Final serviceactive;GPU0~74%utilization/6,546MiB;localW&Bstep4,000loss.8420grad18.04finite. Recent.150s/step impliesroughly21hoursremainingtraining,then evaluation.
- README now88lines,architecture/algorithm/commands-focused. Inline SVGtensorflow added toHTMLreport andexported as`outputs/training-progress/tensor_dataflow.svg`;B1single-input/outputsizesandtraininglossbranchesincluded. Reportshowsfinalcandidateprogressandwillincludeitsfixed-readoutscoretableafterevaluation. Filehashmanifestandfullplan:`outputs/final-model/manifest.json`,`docs/superpowers/plans/2026-10-09-final-production-candidate.md`.

## 14. Offline story report (2026-10-08 18:43 IDT)

- User approved report chapters: training objective → SIGReg evidence → anisotropy/isotropy → downstream whitening → final scorecards. Offline HTML: `outputs/training-progress/results_report.html`; 19 slide-ready SVG/PNG pairs: `outputs/training-progress/figures/`.
- New CPU diagnostics use saved legacy final seed0/1 banks at step497,990; fixed final-block mean (`layer6`, 256 dims). Geometry shows covariance spectra, normalized covariance-entropy rank and before/after PCA-whitening scatter; whitening fit excludes held-out rows.
- Full eight-dataset whitening-policy diagnostic uses identical nested folds (2×5 outer,5 inner), inner-selected whitening variants/probes versus none/standardize/probes. Only3/16 dataset–seed comparisons improve; mean ΔR²−0.071562. This is distinct from the main adaptive layer/readout scorecard and does not isolate whitening rescaling from truncation.
- SIGReg sensitivity plot uses screen1 λ=.02 versus .05 controls; geometry/downstream association is suggestive, one earlier-regime dataset, no matched SIGReg-off causal evidence. Ongoing six-run validation/rank histories are recovered from local CRC-validated W&B records; source runs are read-only.
- Rebuild evidence: `.venv/bin/python -m scripts.report_evidence`; rebuild HTML/figures: `.venv/bin/python -m scripts.build_results_report`. Evidence is cached and CPU-only, one BLAS thread. No training changes.
- Verified14 focused tests, no leakage/provenance blockers in lighter-model independent review; exported figures visually inspected; HTML has no remote runtime dependencies. Latest observations:24p71,800–72,750;48p60,100–60,200 steps. Service/GPU status inaccessible in this sandbox, so local observations are not mislabeled as verified live lifecycle state.

## Production follow-up analysis (2026-10-11; current status)

- Seed101 training and fixed-block2 evaluation finished:5/8 raw wins,mean delta R²+.08008. The original qualification remains false because the0120 shuffled-label canary exceeded.020; diagnostics never overwrite that gate.
- Frozen checkpoint SHA25630486b3e78c354596d7bb5f09ebf351805309b0e9505c88d7ed698145df7fbbe. Durable service`spectralfm-production-readouts` evaluates all7states×3readouts,then a predeclared small menu,raw-inclusive menu and multiscale policy with matched random controls. Seven small sets completed;4716-row labeled_data is running. W&B`m507ncw7`.
-100-permutation audit completed (`y4cftohx`):both real scores exceed all100 null scores; original failed shuffle42 reproduced; no exact duplicates or conflicting labels. This is diagnostic evidence,not a qualification override or proof against all leakage.
- Gradient audit completed (`dwelzk1b`):prediction and each SIGReg gradient are comparable atblock2;each SIGReg term is roughly18times prediction atblock6. Measurements support investigating late objective balance,not automatically lowering regularization everywhere or extending training.
- Offline report includes measured audit figures and partial depth curves,with pending results explicitly marked. Queue adds final policy tables/paired-gain plots and rebuilds it.32focused tests passed; independent lighter-model review checked selection/provenance and unchanged gate.
- Plan:`docs/superpowers/plans/2026-10-11-production-analysis.md`. Next training decision awaits controlled-policy results and an independent labelled development criterion; none supplied yet. Benchmark follow-up is exploratory.
- Completed follow-up:all21readouts and policies finished with exit0. Four-arm menu7/8meanrawgain+.093253;fixed2flat5/8+.080077;menu+raw7/8+.086825;multiscale5/8+.080146. Small0106/0114wins uncertain;0109loses. Final6flat2/8;largestdatasetblock2.95885→block6.67643. Offline report now includes completed analyses and recommendations. No new training/promotion;proposed paired continuation separates extra duration fromlowerSIGReg,with independent development labels still needed. Original gate unchanged.

## Authorized continuation study (2026-10-11)

- Four paired continuations launched as user systemd services:`spectralfm-continuation-{control,low-sigreg}-s{202,203}`. Same frozen101checkpoint,freshAdamW,99598additionalsteps,1000warmup,LR5e-5→5e-7,lambda.05/.01,alltrainingrows andunchangedsource/masking/batchrecipe. PhysicalGPUs0/1/2/4confirmedidlebeforelaunch,thenactive~6.5GB/92–93%utilization;UUIDpinned.
- W&Bcontrol202`otcgb58r`,low202`8604lchm`,control203`rl1j6ygp`,low203`pqh525md`;groupcontinuation-study.14focusedtests andrealcheckpoint2stepsmoke passedwithfinitevalidationmetrics. Registrationcommittedbeforelaunchin82aa681.
- Finalpipelinegradientaudit→allmean/seg4+flat2/6evaluation→fixed/menu/raw/randompolicycomparison→serializedreportrefresh. Predeclaredresearchcriterionrequiresconsistentbenefitonbothseedsandblock2preservation. Finalscoresremainpending;noautomaticdeploymentpromotionororiginalgatechange.

## Final continuation recommendation (2026-10-12)

- All four pipelines completed; config/checkpoint hashes and all eight policy datasets verified. Nine focused checks passed. Recommendation: retain original seed101, SIGReg .05, existing fixed block2 export; stop duration-only sweeps.
- Fixed mean raw gain: baseline+.080077; control202/203+.081188/+.080210; low202/203+.079042/+.078723. Controls average only+.000622abovebaseline. Menu gains: baseline+.093253; controls+.084609/+.088908; low+.093238/+.089280. No new menu improves both seeds over baseline.
- Predeclared research rule remains fixedduration_only/menuinconclusive; tiny duration gains do not justify automatic promotion. Final6flat winscontrol1/8each,low1/8and0/8. Lowerpredictionloss is not a downstream improvement. Original failed gate/export unchanged; independent group-disjoint labels still needed.
- Compiled recommendation: `docs/PRODUCTION_RECOMMENDATION.md`; offline report updated with the completed-study conclusion. No further experiments or deployment launched.
