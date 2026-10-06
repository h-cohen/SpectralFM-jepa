# All-data LeJEPA foundation model (sub-project B)

Date: 2026-10-06
Status: proceeding under the user's `/goal`: "win the label regression for the majority (or all) datasets vs the raw data with equal normalization… training allowed on ALL datasets… squeeze out every possible thing… make decisions according to this goal".

## 1. Goal and success criterion
The scorecard from `2026-10-05-fm-evaluation-design.md` (nested-CV R² Δ ≥ +0.05 over raw per set; ceiling sets reported separately). Target: wins on a majority (ideally all) of the 8 eligible sets, and beating the random-init control.

## 2. Evidence driving the design (2026-10-06)
- **Small sets, current model (spectra0000-only pretraining, per-sample z-score input):** 1 win out of 7 eligible sets (labeled_regression_all +0.107). Large losses on 0055, 0106, 0109 and 0113.
- **Amplitude is the missing information.**
  - Raw input after the same per-spectrum z-score loses heavily: 0055 drops 0.591 → 0.176, 0106 0.697 → 0.530, 0109 0.918 → 0.706.
  - Appending each spectrum's [μ, log σ] to the embedding recovers much of it: 0106 goes 0.412 → 0.691, merged 0.493 → 0.517.
  - So the model input must keep amplitude ("equal normalization").
- **The model never saw the small sets' simulation datasets.** It was pretrained only on spectra0000. Client policy allows pretraining on all spectra, including evaluation spectra (labels never used).
- **Data loading is file-bound** at about 2k WAV files/s. Packing removes that limit.

## 3. Data: one packed array of every spectrum
- **Output** (outside the repo): `/mnt5/home/hadar/nova/data/spectra_all/`.
  - `spectra.npy`: float32 `[N, 245]`, readable with `np.load(mmap_mode="r")`.
  - `index.tsv`: `row<TAB>source<TAB>name`.
  - `stats.json`: per-source counts, plus global `mean` and `std` over all values, plus the dropped-row counts.
- **WAV sources**, read from their manifests with the union of train and valid, deduplicated: single_channel_all, multi_channel, sampled_data, labeled_data, and all labeled_regression sets. Every component is included.
- **Pickle sources** (never converted): `/mnt5/noy/nova_samples/full_chnl/spectra0000_batch9.pkl` and `batch10.pkl`.
  - These are read in the parent's evaluation environment (`/mnt5/home/hadar/nova/.venv-labelprobe/bin/python`, which has pandas), using the parent converter's column parsing (`parse_components`).
  - The resulting rows are written to `pickle_rows.npy`, then appended to the packed array.
- **Rows dropped:** constant rows (std < 1e-6, as the converter does) and non-finite rows. The counts go in `stats.json`.
- **Validation hold-out:** 10,000 rows chosen with seed 0, stored in `valid_rows.npy`. All other rows are training rows.
- **Read-only inputs:** nothing under `/mnt5/noy` is modified.

## 4. Input normalization: `data.normalization`
- `sample_zscore`: the current behaviour, `F.layer_norm` per sample.
- `global`: `(x − μ_g) / σ_g`, using the scalar constants from `stats.json`, recorded in the resolved config. This is an affine map, so per-spectrum amplitude and offset survive. The raw arm (absolute spectrum, fold-internal normalizers) sees the same information, which is what "equal normalization" means here.
- **Evaluation:** `scripts/evaluate.py` normalizes the model input exactly as the checkpoint's config says. `sample_zscore` keeps the current `normalize_like_fairseq`; `global` uses the checkpoint's constants. The raw arm and the random control are unchanged; the control uses the same input normalization as its model.

## 5. Training from the packed array: `data.source: packed`
- `PackedSpectra(path, rows, normalization)` returns float32 `[245]`, reading the memmap.
- The DataLoader shuffles with a seeded generator. Validation and diagnostics use the hold-out rows.
- `data.source: manifests` keeps the current path unchanged.

## 6. Screen 2: decide normalization and capacity on all data
Each arm runs about 1 epoch of all data (~14M samples ≈ 55k steps at batch 256), in parallel on 4 GPUs, with W&B group `screen-2` (metrics and figures only):

| arm | settings |
|---|---|
| zscore | `sample_zscore`, baseline architecture |
| global | `global`, baseline architecture |
| global_wide | `global`, dim 384, depth 8 |
| global_mask75 | `global`, `mask_ratio 0.75` |

- **Evaluation:** the full 9-set scorecard per arm (`eval.yaml`; flat readout on layers 0–2; random control on).
- **Decision:** pick the arm with the most wins. Ties go to the higher mean Δ, and the comparison against the random control decides between those.
- **Long run:** the winning arm for 3–5 epochs (decided from the screen's loss and diagnostic trends), then the scorecard. More levers (projector, λ, block masking) only if the long run still loses on a majority of sets.

## 7. Out of scope
Changing the evaluation methodology or the win rule; label use in pretraining; new JEPA variants (that's after the long run, if needed).
