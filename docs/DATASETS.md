# Pretraining datasets

Root: `/mnt5/noy/SpectralFM/fairseq/data/nova_data/` (manifests may say `/storage/noy/…`; the loader remaps it to `/mnt5/noy/…`).
Sample: one mono float32 WAV, exactly 245 points (the 16 kHz rate is nominal). Manifests: `train.tsv` / `valid.tsv`, line 1 = wav dir, then `filename<TAB>245`.

| Dataset | Path | Spectra (train + valid) | Components | Labels |
|---|---|---:|---:|---|
| single_channel_all | `single_channel_all/` (wavs in `wav/`) | 9,109,930 | 1 | – |
| multi_channel | `multi_channel/` (dataset0002, dataset0005) | 3,412,476 | 14 | – |
| sampled_data | `sampled_data/` (dataset0024) | 22,428 | 28 | – |
| labeled_data | `labeled_data/` (dataset0022) | 66,024 ¹ | 14 | `parameter_0` |
| labeled_regression | `labeled_regression/dataset{0055,0106–0120}/` | 147,784 | varies | `parameter_0` ² |
| **Total** | | **12,758,642** | | |

¹ `valid.tsv` lists all 66,024 files (a superset of `train.tsv`): use the union.
² `labels.tsv` (`filename<TAB>parameter_0`). There are no labels for 0107, 0108 and 0119 (their spectra are still usable). 0111 has an empty labels file. 0110 and 0116 contain no data.

**Missing:** about **1.2M single-channel spectra (≈373k + ≈825k) have no WAV files.** They are in `/mnt5/noy/nova_samples/full_chnl/spectra0000_batch9.pkl` and `batch10.pkl`, but both files are **truncated** (0.74 GB and 1.63 GB against 1.98 GB for a full batch; `pickle.load` fails with "pickle data was truncated"), and no other copy exists. They can't be used unless the original source data is regenerated.

**Policy (client-approved):** pretraining uses all spectra above, including the evaluation spectra; labels are never used. Every component is an independent sample.

**Evaluation sets** (`parameter_0`, component 0): labeled_data (4,716 spectra), labeled_regression 0055 (25), 0106 (64), 0109 (96), 0112 (68), 0113 (70), 0114 (230), 0120 (125), and their merge labeled_regression_all (695).

Excluded: the `single_channel_{one,5m,10k,1k,100,100_var}` and `single_sample` directories, which are subsets of single_channel_all, and `base_libri_100`, which is speech, not spectra.
