# SpectralFM LeJEPA

One experiment: does a LeJEPA-pretrained 1-D ViT give better downstream `parameter_0`
regression than the SpectralFM data2vec baseline (`ref_feb25`), under the unchanged
clean-eval nested-CV methodology?

## The pipeline

| Step | What | Where |
|---|---|---|
| Input | one 245-point float32 spectrum per WAV; per-sample z-score (`F.layer_norm`, eps 1e-5, as fairseq did for data2vec) | `data/loader.py` |
| Tokenize | 24 contiguous, non-overlapping patches: 5 x 11 + 19 x 10 samples (245 = 24 x 10 + 5); 10-wide patches padded with one zero; shared `Linear(11 -> 256)` + learned positions | `models/tokenizer.py` |
| Mask | per sample, 12 of 24 tokens chosen uniformly at random (`mask_ratio 0.5`) | `models/masking.py` |
| Encode | pre-LN transformer, 6 blocks, dim 256, 8 heads; the **same** encoder runs on the visible tokens (context) and on all tokens (target). No EMA, no stop-gradient | `models/vit_1d.py` |
| Predict | 4-block predictor (dim 192) sees the context plus a mask token at each masked position and predicts the encoder's embedding there | `models/vit_1d.py` |
| Loss | `MSE(predicted, target[masked]) + 0.05 * SIGReg(target)`; SIGReg (lightly) is applied per position over the batch (`[24, B, D]`) | `training/loss.py` |
| Train | AdamW 5e-4, wd 0.05, batch 256, 1-epoch warmup + cosine, fp16, 20 epochs on `single_channel_one` (999k spectra) | `training/trainer.py` |
| Evaluate | clean-eval nested CV: every encoder block mean-pooled; 12 recipes (normalizer x RidgeCV/OLS) chosen inside each outer fold; R² vs raw input and vs `ref_feb25` on identical rows/folds | `evaluation/` |

## Running

    export PATH="$HOME/.local/bin:$PATH"     # uv
    export WANDB_API_KEY=...                 # in your shell only; never in the repo
    uv sync
    uv run pytest                            # fast tests (W&B disabled)
    uv run pytest -m slow                    # equivalence with the parent repo's saved outputs
    uv run python scripts/pretrain.py                         # baseline run
    uv run python scripts/pretrain.py training.max_steps=300  # any value can be overridden
    uv run python -m scripts.evaluate --checkpoint outputs/<run>/checkpoint_last.pt
    uv run python -m scripts.baseline                         # data2vec baseline into W&B

## Choices that differ from the LeJEPA paper

- **Masked latent prediction instead of multi-crop invariance.** It is an I-JEPA-style
  predictor with SIGReg added, rather than the paper's view-invariance loss.
- **λ = 0.05 in `mse + λ·sigreg`.** This matches the paper's `0.95·inv + 0.05·sigreg`.
  SIGReg's floor is about 1 for a perfect Gaussian, and its deviations scale with the batch
  size, so λ = 1 would make Gaussianity outweigh prediction by 10–100×. The measured table
  is in the design spec.
- **No projector by default.** `model.projector=mlp` adds lightly's `LeJEPAProjectionHead`;
  predictions and targets are projected in one call so BatchNorm sees one batch. Evaluation
  never uses the projector.
- **fp16 rather than bf16.** The RTX 2080 Ti has no bf16. SIGReg runs in fp32.

## What is inherited from the parent (clean-eval, commit 6237feb), and what differs

- **Data semantics are unchanged:** the manifests, the 245-point WAVs, the per-sample
  z-score, and the labeled-set conventions (`labels.tsv`, `dataset<D>_comp<C>_spec_<S>.wav`,
  component 0, ordering, the seeded subsample to 5,000, and set merging).
- **The nested CV, ladder and canary are copied with their behaviour preserved.**
  `tests/test_eval_equivalence.py` reproduces the parent's saved `ref_feb25` numbers to
  1e-10 with zero differing recipe choices.
- **Only the mean pooling statistic is stored** in our banks (every headline number uses
  it). The parent's 10-statistic banks still load.
- **Evaluation input uses the parent's eps-1e-8 z-score**, not training's eps-1e-5
  `layer_norm`. This keeps the evaluation identical to how the baseline was scored, at
  about a 1% scale difference on low-variance spectra.
- **Inherited caveat: the CV is not grouped by label.** labeled_data has 4,716 spectra but
  only 168 distinct labels. This is left unchanged so the comparison stays like-for-like.
- **Scale differs from the baseline.** The encoder is about 5M parameters versus
  data2vec's 93M, and it sees about 20M samples versus 27.6M.

## Reproducibility

Every run logs to W&B project `spectralfm-lejepa`:
- the full resolved config,
- git commit, branch and dirty flag,
- package versions and the GPU,
- the sha256 and row count of the manifests,
- losses (total, MSE and SIGReg separately),
- representation diagnostics (effective rank, condition number, std, cosine similarity,
  and a collapse alert),
- masking figures,
- checkpoints as `lejepa-<run_id>` artifacts.

Evaluation runs record which artifact or checkpoint they consumed, the pretraining run and
commit, and their own commit.
