# All-data LeJEPA Foundation Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Pretrain LeJEPA on every available spectrum, about 14M from all sources, with an amplitude-preserving ("equal") input normalization. Then screen normalization and capacity, and run the winner long. The aim is to beat raw input on most label sets.

**Architecture:**
- A one-off packer writes every spectrum into one float32 memmap with an index and global stats.
- The trainer reads it through `data.source: packed`, with `data.normalization: sample_zscore | global`.
- `scripts/evaluate.py` normalizes model inputs exactly as the checkpoint was trained.
- Screen 2 reuses `scripts.screen`.

**Spec:** `docs/superpowers/specs/2026-10-06-all-data-fm-design.md`

## Global Constraints
- Never write under `/mnt5/noy`. The packed data goes to `/mnt5/home/hadar/nova/data/spectra_all/` (outside the repo, not committed).
- `data.source: manifests` and `data.normalization: sample_zscore` stay the defaults, and they must reproduce current behaviour exactly; the existing tests guard this.
- `global` normalization is `(x − mean) / std`, with two float scalars from `stats.json`. The trainer copies them into the resolved config as `derived.global_stats = {"mean": m, "std": s}`. Evaluation reads them from `ckpt["config"]["derived"]["global_stats"]`.
- Constant rows (std < 1e-6) and non-finite rows are excluded from training through `drop_rows.npy`. They are not physically removed.
- The validation hold-out is 10,000 rows, `np.random.default_rng(0).choice` over the kept rows, sorted, stored as `valid_rows.npy`.
- No new dependencies in the project environment. Pickles are read in the parent environment `/mnt5/home/hadar/nova/.venv-labelprobe/bin/python` (it has pandas 2.3.3).
- Tests use small synthetic data in `tmp_path`, with `WANDB_MODE=disabled`. Use `uv run` with `export PATH="$HOME/.local/bin:$PATH"`.
- Commit trailer (second `-m`):
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt
  ```

## Review Focus
1. **Packer row order versus `index.tsv`.** Row r of `spectra.npy` must be exactly the spectrum named in index row r. Test: read back the synthetic WAVs by name and compare.
2. **Evaluating a global-normalized checkpoint** must feed `(raw − mean) / std`, not the per-sample z-score. Test: compare the eval bank for a tiny global-trained checkpoint against a manual computation.
3. **An old checkpoint whose config has no `data.normalization`** must evaluate exactly as before, with the per-sample z-score. Test: a checkpoint config without the key.
4. **Dropped rows and the hold-out never enter training batches.** Test: train rows are disjoint from both.
5. **The pickle step on a single-channel DataFrame:** whatever columns `parse_components` returns must yield exactly 245 values per row, or fail loudly. Test: a synthetic DataFrame through the same function (run in the parent environment).

---

### Task 1: Packer

**Files:**
- Create: `scripts/pack_pickles.py` (runs in the parent environment), `scripts/pack_spectra.py` (runs in the project environment), `tests/test_pack.py`

**Interfaces:**
- `scripts/pack_pickles.py --out DIR PKL...` reads each pickle with `pickle.load`, gets components with the parent converter's `parse_components(df)`. The import is `sys.path.insert(0, "<parent>/fairseq/scripts")` and then `from convert_features_to_wav_per_component import parse_components`.
  - For every component and every row, it writes float32 rows to `DIR/pickle_rows.npy` with shape `[M, 245]`.
  - It writes names to `DIR/pickle_names.tsv`, one `source<TAB>name` per row, with source = `pickle:<pkl stem>` and name = `<pkl stem>_comp<C>_row<i>`.
  - It raises `ValueError` if any component doesn't have exactly 245 columns.
- `scripts/pack_spectra.py --out DIR [--sources NAME=MANIFEST_DIR ...] [--workers 16]`:
  - **Defaults:** single_channel_all, multi_channel, sampled_data, labeled_data, and each `labeled_regression/dataset*` that has a `train.tsv`, all under `/mnt5/noy/SpectralFM/fairseq/data/nova_data`.
  - **Rows per source:** the deduplicated union of `train.tsv` and `valid.tsv` paths (via `spectral_lejepa.data.loader.read_manifest`), in manifest order, train first.
  - **Output array:** preallocate `DIR/spectra.npy` with `np.lib.format.open_memmap(float32, (N_wav + M, 245))`. Fill the WAV rows in parallel (`multiprocessing.Pool`, chunks of 20,000 paths, each worker reading with soundfile and returning `(start, array)`). Then copy `pickle_rows.npy` (if present) after them.
  - **Index:** write `DIR/index.tsv` (`row<TAB>source<TAB>name`).
  - **Dropped rows:** compute `drop = (std < 1e-6) | ~isfinite.all(1)` per row and save `drop_rows.npy`.
  - **Stats:** write `stats.json` with global `mean` and `std` over the kept rows (streamed float64 sums), per-source counts, `n_rows`, `n_dropped`.
  - **Hold-out:** save `valid_rows.npy` as described in the Global Constraints.
  - **Wrong-length files:** a WAV with length ≠ 245 raises with its path.

- [ ] **Step 1: Write the tests** (`tests/test_pack.py`).
  - Build 2 synthetic sources in `tmp_path`, each with manifests and wavs. Include one constant row and one 245-length row of NaN, written as a float32 WAV.
  - Build a fake `pickle_rows.npy`/`pickle_names.tsv` pair.
  - Run `pack_spectra.main([...])`.
  - Assert:
    - the shape;
    - index names map to the right rows, by re-reading 3 random WAVs by name and comparing rows exactly;
    - the pickle rows are appended last;
    - `drop_rows` holds exactly the constant and NaN rows;
    - the `stats.json` mean and std match numpy over the kept rows (rtol 1e-6);
    - `valid_rows` is deterministic, sorted, and disjoint from `drop_rows`, with size min(10000, kept).
  - Add a test that a 200-length WAV raises with its filename.
  - Add a test that `pack_pickles` rejects a component with 244 columns. Mark it `skipif` when the parent Python is absent, and run it as a subprocess with the parent Python on a tiny synthetic DataFrame pickle with columns named the way `parse_components` expects. Read `parse_components` in the parent file to build valid column names.
- [ ] **Step 2: Implement both scripts.** Each is about 100 lines, with a module docstring explaining usage and the two environments.
- [ ] **Step 3: Run tests** with `uv run pytest tests/test_pack.py -v`, then the fast suite. Commit: `Add spectra packer (all sources + unconverted pickles) with global stats`.

### Task 2: Packed dataset and normalization in training

**Files:**
- Modify: `src/spectral_lejepa/data/loader.py`, `src/spectral_lejepa/training/trainer.py`, `configs/pretrain.yaml`, `tests/test_loader.py`, `tests/test_train_smoke.py`

**Interfaces:**
- `normalize(x: Tensor, method: str, stats: dict | None) -> Tensor`. `sample_zscore` = `F.layer_norm(x, x.shape[-1:])`; `global` = `(x − stats["mean"]) / stats["std"]`; anything else raises `ValueError`.
- `SpectraDataset(paths, normalization="sample_zscore", stats=None)`. Default behaviour is unchanged.
- `PackedSpectra(packed_dir, rows, normalization, stats)` opens `spectra.npy` with `mmap_mode="r"` lazily inside `__getitem__`, which makes it worker-safe.
- `make_loader(dataset, batch_size, shuffle, seed, num_workers, drop_last)` now takes a Dataset. Update its single call site in the trainer and the loader tests.
- `configs/pretrain.yaml` `data:` gains:
  - `source: manifests   # manifests | packed`
  - `packed_dir: null`
  - `normalization: sample_zscore   # sample_zscore | global`
- **Trainer.** If `source == "packed"`:
  - read `stats.json`, `drop_rows.npy` and `valid_rows.npy`;
  - train rows = all rows minus dropped minus valid, then `subsample(train_rows, max_train_samples, seed)` (subsample works on any list);
  - valid signals = `PackedSpectra` over the first `min(len(valid_rows), 2048)` valid rows;
  - `derived` records `packed_dir`, `train_samples`, `valid_samples`, `global_stats` (always, when packed) and `stats.json`'s `n_rows`/`n_dropped`.
  - With `manifests`, behaviour is as today, plus normalization pass-through.
  - With `global` and `manifests`, raise `ValueError` ("global normalization needs data.source=packed").
- [ ] **Step 1: Tests.**
  - Loader: `normalize` for both methods and an unknown one.
  - `PackedSpectra` returns `(raw − m) / s` for `global` and the layer-norm for `sample_zscore`.
  - Trainer smoke: pack a tiny synthetic dataset by building a small `spectra.npy`, `stats.json`, `drop_rows.npy` and `valid_rows.npy` directly with numpy in the test (no packer). Train 6 steps with `source=packed`, `normalization=global`. Assert the checkpoint's `config.derived.global_stats` equals stats.json and the history is finite.
  - The train rows exclude dropped and valid rows: expose `packed_train_rows(packed_dir) -> np.ndarray` in `loader.py` and test it directly.
  - The existing smoke and reproducibility tests are unchanged.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run the tests and the fast suite.** Commit: `Train from the packed array with sample_zscore or global input normalization`.

### Task 3: Evaluation normalizes like the checkpoint

**Files:**
- Modify: `scripts/evaluate.py`, `tests/test_evaluate_script.py`

- `model_input(raw, ckpt_cfg)`:
  - `sample_zscore`, or a missing key, gives `normalize_like_fairseq(raw)`, the current behaviour;
  - `global` gives `((raw − g["mean"]) / g["std"]).astype(np.float32)` with `g = ckpt_cfg["derived"]["global_stats"]`.
- Use it for both `extract_bank` calls (model and random control). Record `"model_input": <method>` in the bank meta and in lineage.
- [ ] **Step 1: Tests.**
  - `model_input` for both methods, plus a config without the key.
  - Extend the end-to-end test with a second checkpoint trained with `normalization=global` from a tiny packed dir (reuse the Task 2 test helper). The eval runs, and the saved bank `meta["model_input"] == "global"`.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run the tests and the fast suite.** Commit: `Evaluate with the checkpoint's own input normalization`.

### Task 4: Screen-2 config

**Files:**
- Create: `configs/screen2.yaml`

Use the same structure as `configs/screen.yaml`.
- `name: screen-2`, `output_dir: outputs/screen-2`, `eval_config: configs/eval.yaml`.
- **Shared overrides:**
  - `data.source=packed`
  - `data.packed_dir=/mnt5/home/hadar/nova/data/spectra_all`
  - `data.num_workers=12`
  - `training.epochs=1`
  - `training.val_every=5000`, `training.diag_every=5000`, `training.ckpt_every=20000`
  - `experiment.output_dir=outputs/screen-2`
  - `wandb.group=screen-2`
  - `wandb.tags=[lejepa,spectralfm,1d,screen2,alldata]`
- **Arms:**
  - `zscore: [data.normalization=sample_zscore]`
  - `global: [data.normalization=global]`
  - `global_wide: [data.normalization=global, model.dim=384, model.depth=8, model.mlp_dim=1536, model.predictor_dim=256, model.predictor_heads=8, model.predictor_mlp_dim=1024]`
  - `global_mask75: [data.normalization=global, masking.mask_ratio=0.75]`
- `wandb` block as in `screen.yaml`, group `screen-2`.
- [ ] Add a test in `tests/test_screen.py` that every override key in `configs/screen2.yaml` exists in `configs/pretrain.yaml`, by applying each arm through `load_config`. Commit: `Add screen-2 config (all data; normalization and capacity arms)`.

### Task 5 (controller): Pack the data
1. Run the pickles step: `/mnt5/home/hadar/nova/.venv-labelprobe/bin/python scripts/pack_pickles.py --out /mnt5/home/hadar/nova/data/spectra_all /mnt5/noy/nova_samples/full_chnl/spectra0000_batch9.pkl /mnt5/noy/nova_samples/full_chnl/spectra0000_batch10.pkl`
2. Run `uv run python scripts/pack_spectra.py --out /mnt5/home/hadar/nova/data/spectra_all --workers 16`.
3. Check: `n_rows` ≈ 12.76M + about 1.2M, the drop count, and that the global mean and std look sensible.

### Task 6 (controller): Screen 2
Once the current re-score jobs are done, run `uv run python -m scripts.screen --config configs/screen2.yaml --gpus 0,1,2,3 --max_evals 2`. Report the per-arm scorecards and the comparison against the random control.

### Task 7 (controller): Long run
Run the winning arm for 3–5 epochs, evaluate it with `eval.yaml`, and report against the goal.
