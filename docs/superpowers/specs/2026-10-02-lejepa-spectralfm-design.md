# LeJEPA SpectralFM — Design

Date: 2026-10-02
Status: approved in brainstorming, pending written-spec review
Source brief: `spectralfm_lejepa_project_prompt.md` (user-supplied)

## 1. Goal

Replace data2vec with a LeJEPA-style objective for SpectralFM's 245-point 1-D
spectra, in a small new repository, and answer one question:

> Does a LeJEPA-pretrained encoder give better downstream `parameter_0`
> regression than the Feb-25 data2vec baseline (`ref_feb25`) under the
> unchanged clean-eval nested-CV methodology?

Success for this spec = infrastructure done (brief §24) + one reproducible
λ=0.05 baseline pretraining run + its clean-eval comparison against `ref_feb25`.
No sweeps.

## 2. Reconnaissance findings (what the design rests on)

Parent repo: `../SpectralFM-label-regression-eval-merged`, branch
`label-regression/clean-eval`, HEAD `6237feb`. Read-only reference.

### Data
- Each sample: one WAV, IEEE float32, mono, nominal 16 kHz, shape `(245,)`.
  Filenames carry identity (`spectra0000_batch{B}_spec_{N}.wav`,
  `spec_N.wav`, `dataset{D}_comp{C}_spec_{ROW}.wav`).
- Manifests: fairseq `.tsv`, line 1 = root dir, then `fname<TAB>245`.
- Training preprocessing (fairseq `FileAudioDataset.postprocess`):
  per-sample `F.layer_norm(x, x.shape)` (population var, eps 1e-5 inside
  sqrt). No global stats, no augmentation, no crop/pad in practice.
- Pretraining subset chosen: `nova_data/single_channel_one`
  (999,000 train / 1,000 valid; manifest root `/storage/...` → remap to
  `/mnt5/noy/...`).
- Labeled sets: `nova_data/labeled_data` (4,716 spectra, 168 distinct labels),
  `labeled_regression/dataset{0055,0106,0109,0112,0113,0114,0115,0117,0118,0120}`
  and merged `labeled_regression_all`. `labels.tsv` = `filename<TAB>parameter_0`,
  shared by all components of a spectrum. Labeled sets come from different
  simulation datasets than `single_channel_*`, so pretraining cannot see them.

### Evaluation (clean-eval `label_probe`)
- Backbone input: component 0, per-sample z-score `(x-mean)/(std+1e-8)`
  (`features.normalize_like_fairseq`). Raw-input baseline: unnormalized spectrum.
- Extraction: `model(input_values=x, output_hidden_states=True).hidden_states`
  → tuple of `[B,T,D]`, named `layer0..layerN`; headline uses mean over T.
- Recipes: {none, standardize, whiten, whiten8, whiten32, whiten128} ×
  {RidgeCV(α=logspace(-3,3,20)), OLS} = 12.
- Headline `nested_cv`: outer 2 repeats × 5-fold `KFold(shuffle, seed+r)`,
  inner 5-fold (`seed+1000+10r+f`), normalizers fit on training rows only.
  Families: raw, embedding (block × recipe chosen inside the fold), per-block,
  `embedding_top3`, `raw_fixed_recipes`. Metric: OOF R² averaged over repeats;
  `bootstrap_sd` (500) and `paired_delta` (1000, shared folds). Sets with
  n < 20 skipped. Seed 42.
- `nested_ladder`: 5-fold, rungs (10,20,50,100,200,500,1000,2000) + full fold,
  draws 20/10/5/1, inner-CV recipe choice per draw, raw vs one fixed block.
- `ref_feb25` banks and `nested_results.json` / `nested_oof.npz` already exist
  under the parent's `code/eval_outputs/label_probe_regression*_ref_feb25/`.
- Baseline checkpoint: `/mnt5/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt`
  (data2vec, ~93M params, 47 tokens, 13.5k updates × 2048 on
  `single_channel_all`; labeled_data R² 0.44 embedding vs 0.38 raw).

### LeJEPA implementation: `lightly` (PyPI)
- The reference package `lejepa` is **not on PyPI** (GitHub
  `galilai-group/lejepa`, v0.0.1). Instead we use **`lightly` 1.5.26** from
  PyPI, which ships `lightly.loss.SIGReg`, `lightly.loss.LeJEPALoss` and
  `lightly.models.modules.LeJEPAProjectionHead`.
- `SIGReg(knots=17, t_max=3.0, num_vectors=1024)`, input `(..., N, D)`
  (leading dims are batch dims → per-position `[24, B, D]` works directly).
  Same Epps–Pulley/trapezoid math as the reference minimal example;
  statistic multiplied by N, averaged over slices and leading dims.
- Only `SIGReg` (and optionally `LeJEPAProjectionHead`) is used.
  `LeJEPALoss` is the multi-view invariance objective and does not apply to
  masked prediction.
- Caveats: heavy transitive deps (torchvision, pytorch_lightning, hydra-core,
  pydantic, requests, lightly_utils); `import lightly` starts a background
  PyPI version check unless `LIGHTLY_DID_VERSION_CHECK=True`, which our package
  `__init__` sets before importing it; slice directions are drawn from the
  **global** torch RNG (reproducible under our seeding, but changing
  `num_vectors` shifts subsequent random draws — documented); computes in
  input dtype, so we pass fp32.
- Paper differences to record in README: the paper applies SIGReg to a
  projector-MLP output, uses multi-crop views with an invariance loss, and
  weights `(1-λ)·inv + λ·sigreg` with λ≈0.05 (explored 0.01–0.1). This
  project is a masked-latent-prediction (I-JEPA-style) variant with
  `MSE + λ·SIGReg`, λ=0.05, no projector by default.

### Choice of λ (measured)
SIGReg value on synthetic embeddings (D=256, per-position `[24,B,D]`,
1024 slices):

| embedding | B=256 | B=1024 |
|---|---:|---:|
| ideal N(0, I) | 1.05 | 1.04 |
| variance 0.25 | 39 | 155 |
| rank 8 | 8.1 | 31 |
| near-collapsed | 103 | 412 |

SIGReg has a floor ≈1 at a perfect Gaussian and deviations scale with N,
while MSE on unit-variance targets lives in ~0.1–1 (predicting 0 ≈ 1.0).
At λ=1 small non-Gaussianity outweighs prediction by 10–100×. The paper's
`0.95·inv + 0.05·sigreg` on the same statistic corresponds to λ≈0.05 in
our `mse + λ·sigreg` form (paper range 0.01–0.1 ↔ ≈0.01–0.11). **Baseline
λ = 0.05.** The brief's λ=1.0 is kept as a planned comparison point.

### Environment
- Host: 7 usable RTX 2080 Ti (11 GB); GPU 3 (bus 61:00.0) is broken.
  Turing ⇒ no bf16 ⇒ fp16 autocast.
- `uv` not installed; `WANDB_API_KEY` not set; no `~/.netrc`.

## 3. Decisions

| Topic | Decision |
|---|---|
| Pretraining data | `single_channel_one` (1M) |
| Eval reuse | Minimal simplified copy of clean-eval nested CV + ladder + canary, verified numerically against parent outputs |
| Tokenizer | Non-overlapping contiguous split, widths `[11]*5 + [10]*19` |
| SIGReg input | Per position: `Z.transpose(0,1)` `[24,B,D]`, N=B, averaged over positions & slices |
| Target encoder | Same module and parameters as context encoder; no EMA, no detach |
| SIGReg implementation | `lightly.loss.SIGReg` (PyPI), not the GitHub `lejepa` package |
| λ_sigreg | 0.05 (see measured table in §2) |
| Projector | `projector: none` (default) \| `mlp` (lightly `LeJEPAProjectionHead`) |
| Precision | fp16 autocast + GradScaler; SIGReg computed in fp32 |
| Scale vs baseline | ~5M-param encoder vs 93M data2vec; documented, not hidden |

## 4. Architecture

```
x [B,245] ──layer_norm──► tokenizer ──► tokens [B,24,D] (+pos)
                                         │
              random mask (per sample, 12 of 24)
            visible_idx [B,12]      masked_idx [B,12]
                 │                         │
   encoder(tokens[visible]) → context [B,12,D]
   predictor(context, masked_idx) → predicted [B,12,D]
   encoder(tokens all)  → target [B,24,D]   (same encoder object)
   target_masked = gather(target, masked_idx) [B,12,D]

loss = mse(predicted, target_masked) + λ · SIGReg(target.transpose(0,1))
```

### 4.1 Data — `data/loader.py`
- `read_manifest(path, root_override)` → list of absolute wav paths;
  remaps `/storage/noy` → `/mnt5/noy`.
- `SpectraDataset(paths)` → `torch.float32 [245]`, per-sample
  `F.layer_norm(x, x.shape)` (eps 1e-5), asserting length 245.
- Optional `max_samples` (seeded) for dev subsets.
- DataLoader: shuffle with seeded generator, `drop_last=True`.

### 4.2 Tokenizer — `models/tokenizer.py`
- Boundaries from `np.array_split(np.arange(245), 24)` → widths
  `[11]*5 + [10]*19`, starts `0,11,22,33,44,55,65,…,235`.
- Implementation: one gather index `[24,11]` into the signal right-padded by
  one zero (index 245 → 0.0) so 10-wide patches get a trailing zero;
  `Linear(11 → D)`; learned positional embedding `[24, D]`.
- Exposes `patch_bounds` (for masking diagnostics).
- Invariant (tested): every sample index 0..244 appears in exactly one patch.

### 4.3 Masking — `models/masking.py`
- `random_mask(B, num_patches=24, mask_ratio=0.5, generator)` →
  `masked_idx [B,n_mask]`, `visible_idx [B,24-n_mask]`, both sorted ascending.
  `n_mask = round(mask_ratio * 24)`, `1 ≤ n_mask ≤ 23` enforced.
- Uses `argsort(torch.rand(B,24, generator=g))`; same generator ⇒ same masks.
- Fixed count per sample ⇒ no padding masks.

### 4.4 1-D ViT — `models/vit_1d.py`
- `Encoder(dim=256, depth=6, heads=8, mlp_dim=1024, dropout=0.0)`:
  pre-LN `nn.TransformerEncoderLayer(norm_first=True, batch_first=True)` ×
  depth + final LayerNorm. Input: tokens that already include positional
  embedding. Option to return all block outputs.
- `Predictor(in_dim=256, dim=192, depth=4, heads=6, mlp_dim=768)`:
  `Linear(256→192)` on context; `mask_token` (learned, `[192]`) + predictor
  positional embedding at masked positions; context also gets predictor
  positional embedding at visible positions; transformer over the
  concatenation; read out the mask slots; `LayerNorm` + `Linear(192→256)`.
- `LeJEPA(tokenizer, encoder, predictor)` with `forward(x, masked_idx,
  visible_idx)` → `dict(predicted, target, target_masked)`.
- `EvalBackbone(lejepa)`: `forward(input_values, output_hidden_states=True)`
  → `SimpleNamespace(hidden_states=(tokens, block1, …, block6))`, each
  `[B,24,256]` → stages `layer0..layer6`. Must not define attributes named
  `feature_extractor` or `feature_projection`.
- All sizes configurable in `pretrain.yaml`.

### 4.4b Optional projector — `projector: none | mlp`
- `none` (baseline): MSE and SIGReg act on encoder-space tokens.
- `mlp`: lightly `LeJEPAProjectionHead(256, hidden_dim=1024, output_dim=128)`
  applied token-wise. Predicted and target tokens are passed through the
  projector **in one call** (concatenated `[B·12 + B·24, 256]`) so BatchNorm
  sees one set of batch statistics; then MSE(proj(pred), proj(target)[masked])
  and SIGReg(proj(target) as `[24,B,128]`). The eval wrapper ignores the
  projector (encoder blocks only), as in the paper.
- Rationale for `none` as baseline: nested CV picks the best of all encoder
  blocks, so the constrained last block is not the only readout; the
  projector is the second controlled experiment (after λ).

### 4.5 Loss — `training/loss.py`
```python
sigreg = lightly.loss.SIGReg(knots=17, t_max=3.0, num_vectors=1024)
mse = F.mse_loss(predicted, target_masked)
sig = sigreg(target.float().transpose(0, 1))     # [24, B, D]
loss = mse + lambda_sigreg * sig
return loss, mse, sig
```
λ=0.05 (`lambda_sigreg`); no automatic tuning.

## 5. Training — `training/trainer.py`, `scripts/pretrain.py`

- AdamW(lr 5e-4, wd 0.05; no decay on norm/bias/pos-emb/mask_token).
- Linear warmup 1 epoch → cosine to lr/1000.
- Batch 256, no accumulation. Batch < 256 ⇒ printed warning +
  `batch_size_below_256=true` in W&B config.
- fp16 autocast + GradScaler; grad-norm logged.
- Single GPU; 20 epochs ≈ 78k steps ≈ 20M samples (data2vec baseline ≈ 27.6M).
  `max_steps` override for dev.
- Seed sets python/numpy/torch(+cuda), DataLoader generator, mask generator.
- Validation every `val_every` steps on the 1k valid split with a fixed-seed
  mask generator.
- Checkpoints every `ckpt_every` steps + final, in `outputs/<run_name>/`:
  model + optimizer + scaler state, step, epoch, resolved config, git info,
  W&B run id, latest valid/diagnostic metrics.

### Config — `config.py`
YAML → nested dict; CLI `key.sub=value` overrides (YAML-parsed values);
resolved dict saved with every run and checkpoint. No Hydra/OmegaConf.

## 6. W&B and diagnostics

### `utils/wandb.py`
- `init_run(cfg, job_type)`; key only from environment. `wandb.enabled=false`
  or `WANDB_MODE=disabled` ⇒ no-op (tests).
- Config logged: resolved config + `git_commit`, `git_branch`, `git_dirty`,
  python/torch/numpy/lightly/wandb versions, data manifest path +
  sha256 + row count, GPU name.
- Default project `spectralfm-lejepa`, entity null; tags `lejepa, spectralfm, 1d`
  (+ `baseline` on the first run).

### Metrics
- Every `log_every` (50) steps: `train/{loss, mse_loss, sigreg_loss,
  learning_rate, epoch, global_step, samples_seen, grad_norm}`.
- Validation: `valid/{loss, mse_loss, sigreg_loss}`.
- Representation diagnostics, `training/diagnostics.py`, every `diag_every`
  (1000) steps on a fixed 2,048-sample valid batch, unmasked encoder output,
  both per-token `[N·24, D]` and mean-pooled `[N, D]`:
  `representation/{mean_norm, std, effective_rank, covariance_trace,
  covariance_condition, mean_pairwise_cosine}` + singular-value spectrum plot.
  Effective rank = exp(entropy of normalized singular values).
  Condition = λ_max / smallest eigenvalue above 1e-12·λ_max.
- Early-collapse alert (no stop-gradient ⇒ the encoder can also lower MSE by
  making targets trivial): if `representation/std < 0.1` or effective rank
  falls below 25% of its step-0 value, print a warning, log
  `representation/collapse_alert=1` and a W&B alert. Training is not stopped.
- Masking diagnostic: figure of 4 fixed valid signals, patch boundaries,
  visible vs masked patches shaded; at step 0 and every diagnostic step.

### Artifacts and lineage
- Checkpoint artifact `lejepa-<run_id>`, type `model`, aliases `latest`,
  `step-N`; metadata = git commit, run id, config, epoch, step, metrics.
- Eval runs `use_artifact(...)` and log `pretraining_run_id`,
  `pretraining_checkpoint`, `pretraining_git_commit`, `evaluation_git_commit`,
  `evaluation_config`.
- `scripts/wandb_smoke.py`: init → git info → small config → fake loss
  curve → finish (phase 3).

## 7. Evaluation — `evaluation/`, `scripts/evaluate.py`, `scripts/baseline.py`

Copied from clean-eval `6237feb`, simplified, behaviour preserved:

| New file | Source | Kept | Dropped |
|---|---|---|---|
| `evaluation/data.py` | `data_loader.load_labeled_data`, `features.normalize_like_fairseq` | comp-0 loading, filename/labels.tsv conventions, 245 pad/trunc, seeded subsample (max 5000), eps-1e-8 z-score | torchaudio, pandas, manifest/eval-subset loaders |
| `evaluation/bank.py` | `readouts.extract_bank`, `build_bank_cache` | `hidden_states` extraction, mean pooling, `bank.npz` keys (`bank__<stage>`, `input_raw`, `input_z`, `y`, `_meta`) | 9 non-mean pool stats, fe/extract_features taps |
| `evaluation/probe.py` | `normalize.py`, `regressors.make_regressor` | none/standardize/whiten/whitenK (float64, rank ≤ n/2), RidgeCV, OLS, 12 `RECIPES` | l2, pca/hgb/knn/xgb probes |
| `evaluation/nested.py` | `nested.py`, `nested_ladder.py`, `canary.py`, `protocol.r2` | all nested families, bootstrap/paired_delta, ladder, canary, same JSON/NPZ schemas | staged search A–D, panel, old ladder, HTML reports |

- The bank reader accepts both our mean-only banks and the parent's
  10-statistic banks (selects the mean stat).
- Evaluation input normalization stays the parent's eps-1e-8 z-score even
  though training uses `layer_norm` (eps 1e-5): the baseline was evaluated
  that way. (~1% scale difference on low-variance spectra; documented.)
- Default label sets: `labeled_data`, `labeled_regression_all`, each small
  set with n ≥ 20. Components: 1 (component 0).
- `scripts/evaluate.py <checkpoint|artifact>`: builds banks, runs nested CV +
  ladder + canary, writes `outputs/eval/<run>/<set>/`, W&B `job_type=eval`,
  logs per-set nested R² (embedding, raw, top3, best block), depth profile,
  ladder, and paired Δ vs `ref_feb25` using the parent's `nested_oof.npz`
  (same seed ⇒ identical folds; asserted by comparing `y` and fold ids).
- `scripts/baseline.py`: re-scores the parent's `ref_feb25` banks with our
  code; W&B `job_type=baseline`, tags `baseline, data2vec, spectralfm`;
  records parent repo path + commit and bank paths.

### Equivalence requirement
`tests/test_eval_equivalence.py` (marked `slow`, skipped if parent outputs
absent): our `nested_cv` on parent `ref_feb25` banks for `dataset0055` and
`dataset0120` reproduces every R², SD and paired Δ in their
`nested_results.json` within 1e-10; optional slow case for `labeled_data`.
If exact match proves impossible (e.g. BLAS nondeterminism), the tolerance
and cause are documented rather than silently loosened.

## 8. Repository layout

```
pyproject.toml  uv.lock  README.md  .gitignore
configs/pretrain.yaml  configs/eval.yaml
src/spectral_lejepa/
  config.py
  data/loader.py
  models/tokenizer.py  models/masking.py  models/vit_1d.py
  training/loss.py  training/diagnostics.py  training/trainer.py
  evaluation/data.py  evaluation/bank.py  evaluation/probe.py  evaluation/nested.py
  utils/wandb.py
scripts/pretrain.py  scripts/evaluate.py  scripts/baseline.py  scripts/wandb_smoke.py
tests/  test_loader.py test_tokenizer.py test_masking.py test_model.py
        test_loss.py test_train_smoke.py test_eval_equivalence.py
docs/superpowers/specs/  docs/superpowers/plans/
```

`.gitignore` adds: `outputs/`, `wandb/`, `*.pt`, `*.npz`, `.venv/`, `.env*`,
data dirs. No credentials anywhere in the tree.

## 9. Dependencies

uv-managed, Python 3.10. torch (cu128 index), numpy, scikit-learn, scipy,
soundfile, matplotlib, wandb, pyyaml, lightly (1.5.26);
dev: pytest. Not included: torchaudio, pandas, transformers, fairseq, hydra.
`uv` itself installed to `~/.local/bin` (official installer).

## 10. Tests

| Test | Checks |
|---|---|
| loader | batch `[B,245]`, float32, per-sample mean≈0 / std≈1, no NaN, path remap |
| tokenizer | `[B,245]→[B,24,D]`, widths `[11]*5+[10]*19`, exact disjoint coverage of 0..244, zero pad only in 10-wide patches |
| masking | shapes, sorted, disjoint, union = 0..23, count = round(ratio·24), reproducible with same seed, different with different seed |
| model | shapes at every stage; context and target use the same `Encoder` instance (`id`), parameter count has no duplicate encoder; no EMA buffers; gradient reaches encoder via target path (SIGReg-only loss) and via context path (MSE-only with target); `target.requires_grad` |
| loss | synthetic: SIGReg small for N(0,I), large for collapsed/constant; mse exact; components returned separately; gradients flow to both inputs |
| train smoke | tiny synthetic data on CPU, few steps, loss finite and decreasing, checkpoint save/load round-trip, W&B disabled |
| eval equivalence | §7 |

All tests run with `WANDB_MODE=disabled`.

## 11. Build order (brief phases)

1. Repo skeleton, uv env, `.gitignore`, config loader.
2. Data loader + test.
3. W&B smoke test — needs `WANDB_API_KEY` exported by the user. **Stop if it fails.**
4. Tokenizer + test.
5. Masking + test.
6. Encoder / predictor + test.
7. Loss + test.
8. Trainer, diagnostics, checkpoints; tiny overfit/smoke run.
9. Evaluation copy + equivalence test; baseline W&B run.
10. First real pretraining (λ=0.05, batch 256, 20 epochs, no projector). **Stop and report.**
11. Evaluate vs `ref_feb25`. **Stop and report.**

Each step = one small commit.

## 12. Known risks / caveats

- λ=0.05 is derived from the paper's weighting, not tuned on this data.
  Small λ + no stop-gradient risks early target collapse; the collapse alert
  and separate loss terms make this visible. First controlled experiment:
  λ ∈ {0.02, 0.05, 0.2, 1.0}.
- No projector by default (unlike the paper); SIGReg constrains encoder
  outputs directly. `projector: mlp` is the second controlled experiment.
- Encoder ~18× smaller than the data2vec baseline and sees ~25% fewer samples;
  a negative result must be read with that in mind.
- clean-eval CV is not grouped by label (4,716 spectra, 168 label values);
  inherited unchanged so the comparison stays like-for-like.
- `single_channel_one` manifest roots are `/storage/...`; loader remaps.
