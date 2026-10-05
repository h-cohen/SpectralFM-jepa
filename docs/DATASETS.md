# Pretraining datasets

Data root: `/mnt5/noy/SpectralFM/fairseq/data/nova_data/` (RunAI sees it as `/storage/noy/...`; our loader remaps `/storage/noy` -> `/mnt5/noy`).

## Format
- One sample = one mono float32 WAV of exactly 245 points (nominal 16 kHz; ~1.06 KB per file).
- fairseq manifests `train.tsv` / `valid.tsv`: line 1 = wav root dir (some still say `/storage/noy`, hence the remap), then `filename<TAB>245`.
- Filenames: single-channel `spectra0000_batch{B}_spec_{N}.wav` or `spec_N.wav`; multi-component `dataset{D}_comp{C}_spec_{S}.wav`. Row counts below exclude the header line.

## Datasets
| Dataset | Path (under data root) | Train / valid rows | Components | Labels | Notes |
|---|---|---|---|---|---|
| single_channel_all | `single_channel_all/` (wavs in `wav/`) | 9,099,930 / 10,000 | 1 | no | Full single-component simulation (spectra0000); largest set |
| multi_channel | `multi_channel/` | 3,241,852 / 170,624 | 14 | no | dataset0002 + dataset0005 |
| sampled_data | `sampled_data/` | 21,306 / 1,122 | 28 | no | dataset0024 |
| labeled_data | `labeled_data/` | 62,722 / 66,024 | 14 | parameter_0 | dataset0022; valid.tsv holds ALL 66,024 files (union = 66,024 unique) |
| labeled_regression | `labeled_regression/dataset{0055,0106..0120}/` | 140,390 / 7,394 (147,784 total) | varies | parameter_0 (most) | 14 sub-datasets, each with `wavs/`, 95/5 train/valid split, `labels.tsv` (`filename<TAB>parameter_0`) |
| single_channel_{one,5m,10k,1k,100,100_var}, single_sample | `single_channel_*/`, `single_sample/` | one 999,000/1,000; 5m 4,995,000/5,000; 10k 10,611/500; 1k 950/50; 100 90/10; 100_var 100/10; sample 1/1 | 1 | no | Nested subsets of single_channel_all, NOT additional data |
| base_libri_100 | `base_libri_100/` | 90 / 10 | - | no | LibriSpeech audio, not spectra: NOT usable |

labeled_regression gaps: 0107, 0108, 0119 have no `labels.tsv` (spectra still usable); 0111 `labels.tsv` is empty; 0110 and 0116 have no wavs or manifests.

**Total usable for pretraining: 12,758,642 unique spectra** (single_channel_all 9,109,930 + multi_channel 3,412,476 + sampled_data 22,428 + labeled_data 66,024 + labeled_regression 147,784; nested single_channel subsets and base_libri_100 excluded). About 13 GB on disk.

## Source pickles (`/mnt5/noy/nova_samples/`) and conversion status
| Source | Converted into | Status |
|---|---|---|
| `full_chnl/spectra0000_batch{0..8}.pkl` (≈1M spectra each) | single_channel_all | converted (9,999,953 rows incl. valid) |
| `full_chnl/spectra0000_e4.pkl`, `_e5.pkl` | single_channel_all (`e4`, `e5` filename prefixes) | converted (9,999 + 99,978) |
| `full_chnl/spectra0000_batch9.pkl` (0.74 GB), `batch10.pkl` (1.63 GB) | — | **NOT converted: ≈373k + ≈825k ≈ 1.2M spectra** (estimated from file size at ≈1,976 B/spectrum) |
| `multi_channel/dataset0002`, `dataset0005` (+ identical `.pkl` copies) | multi_channel | converted |
| `sampled_data/dataset0024.pkl` | sampled_data | converted |
| `labeled_data_test/v3/dataset0022` (= `v4/`, identical) | labeled_data | converted |
| `labeled_regression/dataset{0055,0106..0120}/dataset*` (pickles live next to the wavs) | labeled_regression | converted (0110, 0116: 692-byte placeholders, no data) |
| `one_chnl/`, `5m_spectograms/`, `debug_chnl/` | single_channel_one / _5m / debug | copies of `full_chnl` batches; nothing new |

All 16 labeled_regression sets (incl. dataset0114) are already in the manifests: wav counts equal train+valid rows for every set; spectra with NaN `parameter_0` still have wavs (usable for pretraining) but no label.
Converting batch9/batch10 (`fairseq/scripts/convert_features_to_wav_per_component.py`) would add ≈1.2M spectra (≈13.96M total).

## Policy (client-approved)
Pretraining may use ALL spectra, including the labeled/evaluation ones; labels are never used in pretraining. Every component is treated as an independent sample (no grouping by parent spectrum).

## Evaluation label sets
One scalar `parameter_0` per spectrum (all components of a spectrum share it). Sets used for evaluation:
- `labeled_data`: 4,716 spectra (comp0)
- `labeled_regression` sets with n>=20: 0055 (25), 0106 (64), 0109 (96), 0112 (68), 0113 (70), 0114 (230), 0120 (125)
- `labeled_regression_all`: their merge (695)

## Caveats
- labeled_data: valid.tsv contains all files (a superset of train.tsv), so train + valid double counts; use the 66,024 union.
- Unusable or missing: base_libri_100 (speech), labeled_regression 0110 / 0116 (no wavs), 0111 (empty labels), 0107 / 0108 / 0119 (no labels, eval-unusable).
- "16 kHz" is nominal only; the 245 points are spectral channels, not audio samples.
