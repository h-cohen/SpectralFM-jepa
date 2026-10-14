# SpectralFM-LeJEPA

SpectralFM learns frozen features from unlabeled 245-point spectra with masked joint-embedding prediction (LeJEPA). The features aim to predict `parameter_0` better than the raw spectra do, on as many of eight labeled evaluation sets as possible, using a linear probe. Labels never enter pretraining. Evaluation spectra are part of the unlabeled corpus, so evaluation is transductive in the inputs.

Full results, diagrams and figures: [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md). Corpus: [`docs/DATASETS.md`](docs/DATASETS.md). Status and release blocker: [`docs/PRODUCTION_RECOMMENDATION.md`](docs/PRODUCTION_RECOMMENDATION.md).

## Final model: robust seed ensemble

Several encoders with the same architecture, each trained from a different random seed in two stages, then used frozen:

```mermaid
flowchart LR
    C["12.76M unlabeled spectra"] --> S1["Stage 1: LeJEPA pretraining<br/>10 epochs, per seed"]
    S1 --> S2["Stage 2: robust continuation<br/>1 epoch, block-2 invariance"]
    S2 --> F["Frozen blocks 1–2<br/>flatten → 12,288 features"]
    F --> P["Linear probe per member<br/>(recipe chosen by inner CV)"]
    P --> E["Ensemble: mean of members' predictions"]
```

| Stage | Shape | Operation |
|---|---|---|
| Input | `(B, 245)` | Raw `float32` spectrum |
| Normalized | `(B, 245)` | Fixed global affine map `(x − 0.378) / 0.513` (corpus statistics) |
| Tokens | `(B, 48, 256)` | 48 contiguous patches (5 of width 6, 43 of width 5, padded), `Linear(6 → 256)` + learned positions |
| Encoder | `(B, 48, 256)` | 1-D ViT: 6 pre-LayerNorm transformer blocks, 8 heads, MLP 1024; shared by context and target paths (no EMA, no stop-gradient) |
| Predictor | `(B, 36, 256)` | 4 blocks, width 192; predicts the 36 masked tokens from the 12 visible ones |
| **Feature** | `(B, 12288)` | State after encoder block 2, before any final norm, flattened. 1.59M parameters per member |

![Input, patches and mask](docs/figures/training-report/fig0_input_patches_mask.svg)

Three example inputs (single_channel_all, multi_channel, dataset0055) after normalization: (a) the 245-bin spectrum, (b) its 48 contiguous patches, (c) one pretraining mask, in which 36 of 48 patches (shaded) are hidden and the predictor must recover their embeddings from the 12 visible ones.

### Training objective

```text
stage 1:  L = MSE(predicted, target_masked) + 0.05·SIGReg(tokens)
              + [0.95·view invariance + 0.05·SIGReg(pooled views)]
stage 2:  L + 0.1 · mean‖h₂(x′) − h₂(x)‖² / Var(h₂(x))
```

- **SIGReg** pushes each token position, across the batch, toward an isotropic Gaussian; this prevents collapse.
- **Pooled views** are the full sequence plus two random 24-token subsets, each encoded and mean-pooled.
- **Stage 2.** `x′` is a perturbed copy of each spectrum. Each of these is applied with probability 0.5, in raw units, then renormalized:
  - Gaussian noise, up to the corpus p90 noise level;
  - ±5% offset, slope and scale;
  - a ±1-bin shift.

  The term makes the shipped block-2 features invariant to measurement-type changes.
- **Recipe.** Batch 256, AdamW (weight decay 0.05), 75% random masking, AMP.
  - Stage 1: 497,990 steps, LR 5e-4 with cosine decay.
  - Stage 2: 49,799 steps from the stage-1 checkpoint, fresh AdamW, LR 5e-5 with a 1,000-step warmup.
  - Sampling: half of each batch, in expectation, comes from regression-source spectra.

### Result

The ensemble beats raw spectra on 8 of 8 sets (mean ΔR² +.056) and a random-weight encoder of the same architecture on 8 of 8 (+.048). See the report for per-set R², uncertainty and limitations.

## Robustness tests

| Test | Command | What it measures |
|---|---|---|
| Perturbation suite | `python -m spectral_lejepa robustness --checkpoint CKPT --output OUT.json` | For each set, relative block-2 drift and fixed-ridge R² under noise (1/3/5%), offset, slope, scale, ±1/2-bin shift and a sinusoidal baseline. Probes are fit on clean rows. Shift-2, 5% noise and sinusoid are never seen in training |
| Embedding suite | `python -m spectral_lejepa evaluate --suite embedding --config configs/eval_embedding.yaml` | The previous project's metrics, run from its pinned commit: cosine(clean, noisy) under six noise types, top-5 stack retrieval, cosine distributions and a 100-spectrum panel. Members and the ensemble mean are scored on 12 sets |

The robust continuation cuts 1%-noise drift about 6× and keeps features stable (cosine ≥ .997) under small noise and ±10% gain. Large baseline offsets and heavy shot noise still move the features.

## Inference

The trained encoders stay outside git. Build the bundle once from the member checkpoints listed in `configs/inference.yaml`: `python -m spectral_lejepa export-ensemble` writes `outputs/inference/robust-ensemble/` (four TorchScript encoders and a checksummed manifest).

```python
from spectral_lejepa.inference import SpectralEmbedder, EnsembleRegressor
emb = SpectralEmbedder()                    # bundle: argument, $SPECTRALFM_BUNDLE, or the default path
F = emb.embed('path/to/wavs/')              # (N, 4, 12288): block-2 flat features per member
model = EnsembleRegressor().fit(F, y)       # one ridge per member; predictions are averaged
y_hat = model.predict(emb.embed(new_spectra))
```

- **Command line.** `python -m spectral_lejepa embed INPUT --output OUT.npz [--device cpu|cuda]`.
  - INPUT is a `.npy` of shape (N, 245), one mono 245-sample `.wav`, or a folder of `.wav` files, with ids taken from sorted file names.
  - The `.npz` holds `embeddings`, `ids`, `members` and `bundle_manifest`.
- **Keep the member axis.** The encoders are independent, so fit one model per member and average the predictions; do not average the embeddings.
- **GPU and CPU.** GPU output matches the evaluation features exactly; CPU differs by about 1e-3.
- **Portability.** `src/spectral_lejepa/inference.py` needs only torch, numpy, soundfile and scikit-learn. Copied with a bundle to another machine, it runs as `python inference.py INPUT --output OUT.npz --bundle DIR`.
- **Trust.** Bundles are TorchScript, which executes code when loaded: load only bundles you built or trust.
- **Input files.** Wavs must be stored as float (FLOAT/DOUBLE), not integer PCM.
- **Output size.** Each spectrum takes 196 KB (4 × 12,288 float32), so 15k spectra make about 3 GB. The CLI streams embeddings to disk instead of RAM.

## Commands

Run from the repository root with the project venv (`uv sync`). W&B uses `WANDB_API_KEY` from the shell; keep it out of the repository.

```bash
# Stage 1, once per member (seed via override)
python -m spectral_lejepa pretrain --config configs/final_model.yaml experiment.seed=<seed> experiment.name=<member>

# Stage 2, once per member, from that member's stage-1 checkpoint
python -m spectral_lejepa pretrain --config configs/robust_continuation.yaml \
  training.init_checkpoint=<stage-1 checkpoint_last.pt> experiment.seed=<seed> experiment.name=<member>

# Evaluate each member: nested-CV probes for every block/readout, raw and random-weight controls
python -m spectral_lejepa evaluate --checkpoint <stage-2 checkpoint_last.pt> \
  --config configs/eval_production_all_blocks.yaml experiment.output_dir=outputs/<study>/<member>/eval

# Score the ensemble (block-2 flat) from the members' evaluation run dirs
python -m spectral_lejepa ensemble --banks outputs/<study>/*/eval/*_step* --output outputs/<study>/ensemble.json

# Robustness tests, figures, report
python -m spectral_lejepa robustness --checkpoint <ckpt> --output outputs/<study>/<member>/perturbation.json
python -m spectral_lejepa evaluate --suite embedding --config configs/eval_embedding.yaml
python -m spectral_lejepa embedding-figures
python -m spectral_lejepa training-report && python -m spectral_lejepa training-report-html

# Export one member as TorchScript: raw (B, 245) float32 -> (B, 12288)
python -m spectral_lejepa export --checkpoint <ckpt> --output <dir>
```

`python -m spectral_lejepa --help` lists every command. Long runs belong in user systemd services. `ops/launch_final_model.sh` is the durable single-model pipeline (train, export, evaluate, gate, report), with `--resume CHECKPOINT` for recovery.

## References

- Balestriero & LeCun (2025), [LeJEPA](https://arxiv.org/abs/2511.08544).
- Assran et al. (2023), I-JEPA, CVPR.
