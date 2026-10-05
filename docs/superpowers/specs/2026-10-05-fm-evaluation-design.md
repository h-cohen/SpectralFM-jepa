# Foundation-model evaluation: readout families, random-init control, "beats raw" scorecard

Date: 2026-10-05
Status: approved in brainstorming, pending written-spec review
Sub-project A of the foundation-model effort (sub-project B = all-data LeJEPA pretraining, separate spec).

## 1. Goal and success criterion

The project goal (user/client, 2026-10-05) is one LeJEPA-family foundation model whose
frozen embedding beats **raw input** on `parameter_0` label regression on most — preferably
all — labeled sets. This sub-project makes the evaluation measure exactly that:

- **Win** on a set: nested-CV R²(embedding) − R²(raw) **≥ +0.05**.
- **Ceiling** set: 1 − R²(raw) < 0.05 (a +0.05 win is impossible; reported separately).
- **No-win** otherwise.
- Headline: `wins / eligible sets` (eligible = not ceiling) and mean Δ, over the 9 sets
  with n ≥ 20 (labeled_data, labeled_regression_all, dataset0055/0106/0109/0112/0113/0114/0120).
  Today only dataset0114 (raw 0.983) is a ceiling set.

## 2. Evidence (readout spike, 2026-10-05; fixed 5-fold protocol, labeled_data)

| readout | R² |
|---|---:|
| raw | 0.824 |
| LeJEPA-20ep layer1 mean (old headline readout) | 0.762 |
| LeJEPA layer1 seg4 / flat | 0.846 / 0.925 |
| random-init layer1 flat | 0.898 |
| data2vec layer0 mean / seg4 | 0.858 / 0.918 |

Mean pooling over tokens discards position information; position-preserving readouts of
early blocks beat raw. A random-init control already gets most of that gain, so every
model must be reported next to its random-init control.

## 3. Readout families — `evaluation/bank.py`

- `extract_bank(model, signals_z, device, batch_size, readouts)` returns arms
  `{arm_name: [N, d]}` for every hidden-state stage and requested readout:
  - `mean`: mean over tokens → arm name `layer{i}` (unchanged naming);
  - `seg4`: means of 4 equal token segments (parent's `np.linspace(0, T, 5)` edges,
    `max(lo+1, hi)` guard), concatenated → `layer{i}/seg4`;
  - `flat`: all tokens concatenated → `layer{i}/flat`.
- Bank file: arms stored as `bank__<arm_name>` with shape `[N, 1, 1, d]` (component 0,
  one statistic), `/` in names encoded as `__` in the key (e.g. `bank__layer1__seg4`);
  `_meta` lists `readouts`. `load_bank` decodes names back.
- Parent banks (`pool_stats` of 10): `load_bank(path, readouts)` yields `layer{i}` (mean)
  and `layer{i}/seg4` (stats seg0..seg3 concatenated); `flat` unavailable (skipped).
- With `readouts=[mean]` everything is byte-identical to the current behaviour
  (existing equivalence tests guard it).
- Config: `evaluation.readouts: [mean, seg4, flat]` (default in `eval.yaml`;
  `eval_screen.yaml` too).

## 4. Nested CV — unchanged methodology

`run_nested(bank, input_raw, y, ...)` already treats each bank entry as an arm: the
`embedding` family selects (arm = block × readout, recipe) inside each outer fold; per-arm
families give the depth × readout profile; `embedding_top3` averages the 3 best arms.
No change to `evaluation/nested.py` or `evaluation/probe.py` beyond what naming needs
(none expected).

## 5. Random-init control — `scripts/evaluate.py`

`evaluation.random_control: true` → after the checkpoint model, build the same
architecture from the checkpoint's model config with `torch.manual_seed(0)`, untrained,
and run the identical pipeline (bank → nested CV) for every set. Outputs go to
`<out_root>/<set>/random_control/` with the same files.

## 6. Scorecard

- Per set `summary.json`: `verdict` and `delta_vs_raw` for the model, and
  `random_control: {embedding_r2, delta_vs_raw, verdict}`.
- Top-level `summary.json`: `scorecard: {model: {wins, eligible, ceiling, mean_delta},
  random_control: {...}}`.
- Constants: `WIN_MARGIN = 0.05` in `scripts/evaluate.py`.

## 7. W&B — results and plots only

- Evaluation: summary metrics per set (as now), plus a `wandb.Table` `scorecard` (set, n,
  raw R², model R², Δ, verdict, random-control R², Δ, verdict) and one matplotlib figure
  (per-set Δ vs raw for model and random control, horizontal line at +0.05). **No
  evaluation artifact.**
- Lineage without artifacts: drop `use_artifact`/`resolve_checkpoint`'s `wandb:` path;
  `--checkpoint` is a local path. Lineage keys: `pretraining_run_id`,
  `pretraining_checkpoint` (path), `pretraining_checkpoint_sha256`,
  `pretraining_checkpoint_step`, `pretraining_git_commit`, `pretraining_git_dirty`,
  `evaluation_git_commit`, `evaluation_git_dirty`, `evaluation_config`.
- Pretraining: `wandb.log_checkpoints: false` (new key in `pretrain.yaml`, default false);
  when false the trainer saves checkpoints locally only. Metrics and figures unchanged.
- Baseline script and screen launcher: no artifacts either (screen results →
  `wandb.Table` + summary; drop the results artifact).

## 8. Screen launcher fixes (`scripts/screen.py`)

Re-apply the stashed, unreviewed fix-wave (`git stash` "unreviewed screen.py fix-wave"),
completed and reviewed: guarded per-arm summary rows, worker exceptions recorded as failed,
`--summarize_only` (rebuild from disk), `control_s1` paired Δ, sign only on Δ columns,
`stage` column, results.md footnote (wins are candidates; valid losses/rank not comparable
across arms).

## 9. Re-scoring (operational, after code review)

1. 20-epoch baseline (`outputs/lejepa_baseline_20261002-161540/checkpoint_last.pt`) on all
   9 sets with `random_control: true`.
2. The 9 screen-1 checkpoints on labeled_data only (`eval_screen.yaml`, readouts all three)
   via `scripts.screen --summarize_only` after re-running their evals.
3. Short decision memo (results table + recommendation) feeding sub-project B.

## 10. Tests

- Readouts: shapes (`mean` d, `seg4` 4d, `flat` T·d), values equal manual computation on a
  tiny model; bank roundtrip with `/` names; parent-bank loading yields mean + seg4.
- `readouts=[mean]` path unchanged (existing equivalence + row-parity slow tests).
- Verdict/ceiling logic (unit, synthetic numbers).
- `scripts.evaluate` end-to-end (existing test extended): random-control rows present,
  scorecard present, no `use_artifact` call, lineage keys present.
- Trainer: `wandb.log_checkpoints=false` path never calls `log_checkpoint` (W&B disabled
  test via a monkeypatched helper).
- Screen fixes: as in §8.

## 11. Out of scope

`raw ⊕ embedding` rows; data2vec `flat` readout (needs fairseq/HF loader; data2vec is a
reference row with mean/seg4 only); any pretraining change (sub-project B).
