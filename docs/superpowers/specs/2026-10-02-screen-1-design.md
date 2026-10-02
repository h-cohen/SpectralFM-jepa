# Screen 1 — which change closes the gap to data2vec on labeled_data?

Date: 2026-10-02
Status: approved in brainstorming, pending written-spec review
Builds on: `2026-10-02-lejepa-spectralfm-design.md` (baseline spec)

## 1. Goal

Find which single change to the LeJEPA baseline most improves nested-CV R² on
**labeled_data** (n=4,716), the target set where the baseline trails data2vec
(`ref_feb25`): 0.741 vs 0.853 (Δ −0.112 ± 0.020; W&B eval run `myh00vk1`).
Method: a one-factor screen of short (5-epoch) runs against a matched
short control, then 1–2 full runs of the winner(s), proposed after the screen
and started only on the user's approval.

Constraints from the user: minimal code; every procedure logged to W&B.

## 2. Evidence the arms are chosen from (baseline run `ciw1ug3r`)

- Prediction task nearly trivial: end MSE 0.006 on unit-variance targets —
  smooth spectra + 50% random masking leave visible neighbours around almost
  every masked 10-sample patch (interpolation).
- Best readout is an early block (layer1 on labeled_data); depth discards
  label information.
- λ·SIGReg ≈ 12× MSE at the end of training.
- Scale mismatch vs data2vec: ~5M vs 93M params, 24 vs 47 tokens.

## 3. Arms

All arms inherit `configs/pretrain.yaml` plus shared overrides
`training.epochs=5`, `wandb.group=screen-1`, `wandb.tags=[lejepa,spectralfm,1d,screen]`,
`experiment.output_dir=outputs/screen-1`, `experiment.name=<arm>`.

| arm | overrides | hypothesis |
|---|---|---|
| control_s0 | `experiment.seed=0` | matched short control |
| control_s1 | `experiment.seed=1` | seed noise |
| block50 | `masking.strategy=block` | contiguous spans remove interpolation |
| random75 | `masking.mask_ratio=0.75` | harder task, same mask structure |
| block75 | `masking.strategy=block masking.mask_ratio=0.75` | both |
| patch49 | `model.num_patches=49` | finer tokens (245 = 49×5 exactly) keep detail |
| lambda02 | `training.lambda_sigreg=0.02` | less isotropy pressure |
| projector | `model.projector=mlp` | encoder escapes the Gaussian constraint |
| wide | `model.dim=384 model.depth=8 model.mlp_dim=1536 model.predictor_dim=256 model.predictor_heads=8 model.predictor_mlp_dim=1024` | capacity |

## 4. Code changes (minimal)

### 4.1 Block masking — `models/masking.py`
- `block_mask(batch_size, num_patches, mask_ratio, num_blocks, generator=None, device=None)`
  → `(masked_idx, visible_idx)`, same contract as `random_mask` (sorted, disjoint,
  complete, exactly `n_mask = num_masked(P, ratio)` masked per sample).
- Per sample: masked run lengths = a uniformly random composition of `n_mask`
  into `num_blocks` positive parts; visible gaps = `num_blocks + 1` parts summing
  to `P − n_mask`, interior gaps ≥ 1, end gaps ≥ 0 (random composition). Laid out
  gap, run, gap, run, …, gap. All randomness from the given CPU generator.
- Rejects `num_blocks < 1`, `num_blocks > n_mask`, or `P − n_mask < num_blocks − 1`
  with a `ValueError`.
- `make_mask(masking_cfg, batch_size, num_patches, generator, device)` dispatches on
  `masking_cfg["strategy"]` (`random` | `block`); unknown strategy → `ValueError`.
- Trainer: the three `random_mask` call sites (training step, validation,
  masking figure) call `make_mask`; the "only random" check is removed.
- `configs/pretrain.yaml`: add `masking.num_blocks: 3` (ignored by `random`).
  Default `strategy: random` — the baseline is unchanged.

### 4.2 Screen configs
- `configs/screen.yaml`: `name: screen-1`, `shared_overrides: [...]`,
  `arms: {arm_name: [override, ...]}` exactly as §3, `eval_config: configs/eval_screen.yaml`.
- `configs/eval_screen.yaml`: copy of `eval.yaml` with `label_sets` = labeled_data
  only, `ladder_sets: []`, `ladder_block_from: labeled_data`, `baseline.run_dirs` =
  labeled_data only, `experiment.output_dir: outputs/screen-1/eval`,
  `wandb.group: screen-1`, `wandb.tags: [lejepa, spectralfm, 1d, screen, eval]`.

### 4.3 Launcher — `scripts/screen.py`
- `uv run python -m scripts.screen --gpus 2,3,4,5 [--config configs/screen.yaml] [--max_evals 2]`.
  `--gpus` are **CUDA** indices (GPU 3 on Geoffrey is broken, so CUDA index ≥3 is
  nvidia-smi index +1; documented in the help text).
- A thread per GPU pulls the next arm from a queue and runs, as subprocesses with
  `CUDA_VISIBLE_DEVICES=<gpu>`: `scripts/pretrain.py <shared> <arm overrides>` then
  `python -m scripts.evaluate --checkpoint <arm dir>/checkpoint_last.pt --config <eval_config>`.
  A semaphore caps concurrent evaluations at `--max_evals` (CPU-bound).
  stdout+stderr → `outputs/screen-1/<arm>.log`. A non-zero exit marks the arm
  `failed` (with the log path) and the screen continues.
- The arm's checkpoint dir is found as the single `outputs/screen-1/<arm>_*/`
  directory; more than one → error for that arm.
- After all arms: `summarize(...)` builds one row per arm from the eval
  `summary.json`, `nested_oof.npz` and the checkpoint metrics:
  labeled_data embedding R² ± bootstrap SD, best block, paired Δ vs control_s0
  (`paired_delta` on the shared folds) ± sd, Δ vs ref_feb25 (context only),
  end-of-training `train/mse_loss`, `train/sigreg_loss`, `representation/effective_rank`
  (from the checkpoint's `metrics`), pretrain + eval W&B run ids, status.
- Decision rule (computed, not eyeballed):
  `floor_arm = max(|R²(control_s0) − R²(control_s1)|, 2 × sd(Δ vs control_s0))`;
  verdict `win` if Δ > floor, `loss` if Δ < −floor, else `neutral`
  (controls get `control`).
- Writes `outputs/screen-1/results.json` and `results.md` (table).
- **W&B:** one run `job_type=screen`, `group=screen-1`, name `screen-1`, config =
  the resolved screen definition + git metadata (via `utils/wandb.init_run`);
  logs the table as `wandb.Table` (`screen/results`), the per-arm R² and Δ as
  summary metrics, and `results.json` + `results.md` as an artifact
  (`screen-1-results`, type `screen`). Arm pretrain/eval runs already log to W&B
  in the same group.

## 5. Tests

- `tests/test_masking.py` (additions): block_mask exact count, sorted/disjoint/
  complete, masked indices form exactly `num_blocks` contiguous runs, seed
  reproducibility, invalid settings rejected; `make_mask` dispatch + unknown strategy.
- `tests/test_screen.py`: `summarize` + decision rule on synthetic eval outputs
  (fake summary.json / nested_oof.npz / checkpoint metrics) — win/loss/neutral/
  failed rows correct; an end-to-end launcher run on 2 tiny arms (CPU,
  `WANDB_MODE=disabled`, tiny model/steps via a test screen config) producing
  `results.json`, with one deliberately failing arm recorded as `failed`.
- Existing suite stays green (baseline masking unchanged).

## 6. Running

`--gpus` = the free CUDA devices at launch (check `nvidia-smi`; GPUs 0/1 are often
used by others). ~9 arms × ~35 min pretrain on 4 GPUs ≈ 3 rounds, evals capped at 2
concurrent → ~3 h total. Results reported to the user with the proposed full runs.

## 7. Out of scope

Overlapping conv stem, multi-block I-JEPA targets, multi-seed full runs,
matched-scale (93M / 9.1M) runs — candidates for screen 2 depending on results.
