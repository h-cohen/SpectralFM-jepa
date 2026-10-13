# Embedding Similarity and Noise Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python -m spectral_lejepa evaluate --suite embedding --config configs/eval_embedding.yaml` scores the four robust members on the previous project's cosine-similarity and noise-robustness metrics. It writes per-dataset results and figures, plus a training-report section.

**Architecture:** The previous project's metric code is executed from its pinned commit `6237feb` via `git show`, and never from its working tree, which has local edits. It is reused unchanged wherever the code is pure:
- `embedding_similarity.run`, fed precomputed embeddings through a tiny adapter;
- `_compute_similarity_distributions`;
- `structured_similarity.build_panel`, with its normalizer swapped for identity so that it returns raw spectra;
- `NOISE_CONFIGS`.

Our code prepares inputs: raw spectra, the old per-spectrum layer norm, the old noise, back to raw units, our global normalization, then the encoder. It extracts the block-2 readouts and computes the noise cosine.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pandas, scikit-learn, soundfile, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-embedding-similarity-noise-design.md`

## Global Constraints

- Legacy source: repo `/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged`, commit `6237feb5335af2f5336be28c68a17802050f2abd`. Read it via `git show`. Never write to it.
- Never write under `/mnt5/noy`.
- Models: robust_s401, robust_p0, robust_p1 and robust_s101. The ensemble is the mean of the members' scores.
- Readouts: `block2_flat` (12,288 values) and `block2_mean` (256 values).
- Datasets:
  - similarity: sanity (single_channel_1k/train, 100) and in_dist (single_channel_10k/valid, 500);
  - noise: those two plus multi_ch (multi_channel/valid, 1,000), samples (sampled_data/valid, 1,000) and the eight labeled sets;
  - panel: the fixed 100 spectra, seed 42.
- Stacks are 10 rows, seed 42, k = 5. There are six noise types (`NOISE_CONFIGS`).
- Noise draws are seeded with `np.random.seed(noise_seed + type index)`. The old code was unseeded; this deviation is recorded in the results JSON.
- Compute: physical GPU 4 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4`). No training and no W&B.
- Tests: `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q`. Tests that need the legacy repo or real data are marked `@pytest.mark.slow`.
- Commit on `main` as `Hadar <hal.nls@gmail.com>` with the session trailer. Push at the end.

## Review Focus

1. **Short or padded spectra.** Wavs not exactly 245 points must be cut or zero-padded the same way in both input spaces. Task 2's loader test includes a 240-point wav.
2. **Constant spectra.** A flat row has std 0. Layer norm uses eps 1e-5, so denormalization must use `sqrt(var + 1e-5)` and stay finite. Task 3's test includes a constant row.
3. **Stacks shorter than k.** If a manifest drops a wav, a stack can have fewer than 10 rows. The old `run` still works, and our adapter must keep row order (Task 4 test).
4. **Ensemble with a missing member.** If one member's results are missing, the summary must fail loudly rather than average three (Task 6 test).
5. **Re-running.** Re-running must overwrite `outputs/embedding-eval/<model>/...` consistently and never append to the CSV (Task 6 test).

---

### Task 1: Pinned legacy loader

**Files:**
- Create: `src/spectral_lejepa/evaluation/legacy.py`
- Test: `tests/evaluation/test_legacy.py`

**Interfaces:**
- Produces: `LEGACY_REPO: Path`, `LEGACY_COMMIT: str`, and `legacy_module(relpath: str, overrides: dict | None = None) -> types.ModuleType`. `relpath` is relative to the repo root, for example `'code/eval/evaluations/noise_robustness.py'`. `overrides` replaces module globals after exec.

- [ ] **Step 1: Write the failing test.**
  ```python
  import pytest

  from spectral_lejepa.evaluation.legacy import LEGACY_COMMIT, legacy_module

  pytestmark = pytest.mark.slow


  def test_noise_configs_come_from_the_pinned_commit():
      m = legacy_module('code/eval/evaluations/noise_robustness.py')
      assert list(m.NOISE_CONFIGS) == ['gaussian_std', 'gaussian_mean', 'shot_low', 'shot_high', 'gain_low', 'gain_high']
      assert m.__file__.endswith(f'@{LEGACY_COMMIT[:7]}:code/eval/evaluations/noise_robustness.py')


  def test_overrides_replace_module_globals():
      m = legacy_module('code/eval/evaluations/structured_similarity.py', {'_normalize_like_fairseq': lambda a: a})
      assert m._normalize_like_fairseq(5) == 5
  ```
- [ ] **Step 2:** Run `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q -m slow tests/evaluation/test_legacy.py`. Expected: FAIL with ModuleNotFoundError.
- [ ] **Step 3: Implement** `legacy.py`:
  ```python
  """The previous project's evaluation code, executed from its pinned commit (never its working tree)."""
  import subprocess
  import types
  from pathlib import Path

  LEGACY_REPO = Path('/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged')
  LEGACY_COMMIT = '6237feb5335af2f5336be28c68a17802050f2abd'


  def legacy_module(relpath, overrides=None):
      source = subprocess.check_output(['git', '-C', str(LEGACY_REPO), 'show', f'{LEGACY_COMMIT}:{relpath}'], text=True)
      name = f'legacy_{Path(relpath).stem}'
      module = types.ModuleType(name)
      module.__file__ = f'{LEGACY_REPO}@{LEGACY_COMMIT[:7]}:{relpath}'
      exec(compile(source, module.__file__, 'exec'), module.__dict__)
      module.__dict__.update(overrides or {})
      return module
  ```
- [ ] **Step 4:** Run the test. Expected: 2 passed.
- [ ] **Step 5:** Commit: `feat: load the previous project's evaluation code from its pinned commit`.

### Task 2: Stack-subset loader (raw spectra)

**Files:**
- Create: `src/spectral_lejepa/evaluation/stacks.py`
- Test: `tests/evaluation/test_stacks.py`

**Interfaces:**
- Produces: `load_stack_subset(dataset_dir: str, split: str, n: int, seed: int = 42, stack_size: int = 10, length: int = 245) -> dict`, with keys:
  - `raw`: float32 array [N, length];
  - `stack_idx`: int array [N];
  - `filename`: list of str.

  The block selection matches the legacy `load_manifest_subset`. Wavs are read with soundfile, then cut or zero-padded to `length`.

- [ ] **Step 1: Write the failing test.** The legacy function needs `torchaudio`, which is not installed. The test injects a soundfile-backed shim only inside the test.
  ```python
  import sys
  import types

  import numpy as np
  import pytest
  import soundfile as sf
  import torch

  from spectral_lejepa.evaluation.stacks import load_stack_subset


  def write_manifest(tmp_path, n_rows=60, short_at=7):
      wav = tmp_path / 'wav'
      wav.mkdir()
      names = []
      for i in range(n_rows):
          x = np.linspace(0, 1, 240 if i == short_at else 245, dtype=np.float32) * (i + 1)
          sf.write(wav / f'f{i}.wav', x, 16000, subtype='FLOAT')
          names.append(f'f{i}.wav')
      (tmp_path / 'valid.tsv').write_text(str(wav) + '\n' + ''.join(f'{n}\t245\n' for n in names))
      return tmp_path


  def test_short_wavs_are_zero_padded(tmp_path):
      out = load_stack_subset(str(write_manifest(tmp_path)), 'valid', 60)
      assert out['raw'].shape == (60, 245)
      assert out['raw'][7, 240:].tolist() == [0.0] * 5


  @pytest.mark.slow
  def test_selection_matches_the_legacy_loader(tmp_path, monkeypatch):
      shim = types.ModuleType('torchaudio')
      shim.load = lambda p: (torch.from_numpy(sf.read(p, dtype='float32')[0])[None], 16000)
      monkeypatch.setitem(sys.modules, 'torchaudio', shim)
      from spectral_lejepa.evaluation.legacy import legacy_module
      old = legacy_module('code/eval/data_loader.py').load_manifest_subset(str(write_manifest(tmp_path)), 'valid', 30)
      new = load_stack_subset(str(tmp_path), 'valid', 30)
      assert new['filename'] == old['filename'].tolist()
      assert new['stack_idx'].tolist() == old['stack_idx'].tolist()
      full = [r for r in old['data'] if len(r) == 245]
      assert np.allclose(new['raw'][[len(r) == 245 for r in old['data']]], np.array(full))
  ```
- [ ] **Step 2:** Run both tests (`-m "slow or not slow"`). Expected: FAIL with ModuleNotFoundError.
- [ ] **Step 3: Implement** `stacks.py`:
  ```python
  """Whole-stack subsets of a fairseq manifest, selected exactly as the previous project did; raw spectra."""
  import os

  import numpy as np
  import soundfile as sf

  REMAPS = [('/storage/noy/', '/mnt5/noy/'), ('/storage/', '/mnt5/'), ('/mnt5/noy/', '/storage/noy/'), ('/mnt5/', '/storage/')]


  def _root(root):
      if os.path.isdir(root):
          return root
      for src, dst in REMAPS:
          if root.startswith(src) and os.path.isdir(dst + root[len(src):]):
              return dst + root[len(src):]
      return root


  def load_stack_subset(dataset_dir, split, n, seed=42, stack_size=10, length=245):
      with open(os.path.join(dataset_dir, f'{split}.tsv')) as f:
          root = _root(f.readline().strip())
          rows = [ln.strip().split('\t')[0] for ln in f if ln.strip()]
      n_total = len(rows) // stack_size
      chosen = np.sort(np.random.default_rng(seed).choice(n_total, max(1, min(n // stack_size, n_total)), replace=False))
      raw, stack_idx, names = [], [], []
      for s, b in enumerate(chosen):
          for name in rows[b * stack_size:(b + 1) * stack_size]:
              data = np.asarray(sf.read(os.path.join(root, name), dtype='float32')[0]).reshape(-1)
              row = np.zeros(length, np.float32)
              row[:min(len(data), length)] = data[:length]
              raw.append(row)
              stack_idx.append(s)
              names.append(name)
      return {'raw': np.stack(raw), 'stack_idx': np.array(stack_idx), 'filename': names}
  ```
  The legacy loader loads stereo as a mean; NOVA wavs are mono, so `reshape(-1)` is equivalent there. Unlike the legacy loader, a missing wav raises instead of being skipped silently.
- [ ] **Step 4:** Run the tests. Expected: 2 passed.
- [ ] **Step 5:** Commit: `feat: raw whole-stack subsets matching the previous evaluation's selection`.

### Task 3: Noise in the legacy space, scored in ours

**Files:**
- Create: `src/spectral_lejepa/evaluation/noise.py`
- Test: `tests/evaluation/test_noise.py`

**Interfaces:**
- Produces:
  - `layer_norm_rows(raw) -> (z, mu, sd)`, identical to `torch.nn.functional.layer_norm(x, x.shape)` per row, with `sd = sqrt(var + 1e-5)`;
  - `denormalize(z, mu, sd) -> raw`;
  - `noisy_raw(raw, kind, seed, configs) -> raw'`;
  - `row_cosine(a, b) -> [N]`.

- [ ] **Step 1: Write the failing test.**
  ```python
  import numpy as np
  import pytest
  import torch
  import torch.nn.functional as F

  from spectral_lejepa.evaluation.noise import denormalize, layer_norm_rows, noisy_raw, row_cosine


  def sample():
      x = np.random.default_rng(0).normal(3, 2, (4, 245)).astype(np.float32)
      x[2] = 1.5   # constant spectrum
      return x


  def test_layer_norm_matches_torch_and_round_trips():
      x = sample()
      z, mu, sd = layer_norm_rows(x)
      ref = torch.stack([F.layer_norm(torch.from_numpy(r), r.shape) for r in x]).numpy()
      assert np.allclose(z, ref, atol=1e-5)
      assert np.allclose(denormalize(z, mu, sd), x, atol=1e-5) and np.isfinite(z).all()


  def test_noise_is_applied_in_the_normalized_space():
      configs = {'add_one': lambda z: z + 1.0}
      x = sample()
      out = noisy_raw(x, 'add_one', 0, configs)
      _, _, sd = layer_norm_rows(x)
      assert np.allclose(out - x, sd, atol=1e-5)


  def test_noise_is_reproducible_for_a_seed():
      configs = {'g': lambda z: z + np.random.normal(0, .01, size=z.shape)}
      assert np.array_equal(noisy_raw(sample(), 'g', 3, configs), noisy_raw(sample(), 'g', 3, configs))


  def test_row_cosine_is_one_for_scaled_rows():
      a = np.random.default_rng(1).normal(size=(3, 8))
      assert np.allclose(row_cosine(a, 2 * a), 1.0)


  @pytest.mark.slow
  def test_legacy_noise_formula_is_used_unchanged():
      from spectral_lejepa.evaluation.legacy import legacy_module
      configs = legacy_module('code/eval/evaluations/noise_robustness.py').NOISE_CONFIGS
      x = sample()
      z, mu, sd = layer_norm_rows(x)
      np.random.seed(7)
      expected = denormalize(np.stack([configs['shot_low'](r.astype(float)) for r in z]), mu, sd)
      assert np.allclose(noisy_raw(x, 'shot_low', 7, configs), expected)
  ```
- [ ] **Step 2:** Run it. Expected: FAIL with ModuleNotFoundError.
- [ ] **Step 3: Implement** `noise.py`:
  ```python
  """The previous evaluation's noise, applied to per-spectrum layer-normed rows, then mapped back to raw units."""
  import numpy as np

  EPS = 1e-5   # torch.nn.functional.layer_norm default, as the previous dataloader used


  def layer_norm_rows(raw):
      raw = np.asarray(raw, np.float64)
      mu = raw.mean(1, keepdims=True)
      sd = np.sqrt(raw.var(1, keepdims=True) + EPS)
      return (raw - mu) / sd, mu, sd


  def denormalize(z, mu, sd):
      return (z * sd + mu).astype(np.float32)


  def noisy_raw(raw, kind, seed, configs):
      """Seeded per call; the previous code drew from the unseeded global NumPy state."""
      z, mu, sd = layer_norm_rows(raw)
      np.random.seed(seed)
      return denormalize(np.stack([configs[kind](row) for row in z]), mu, sd)


  def row_cosine(a, b):
      a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
      return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
  ```
- [ ] **Step 4:** Run the fast and slow tests in the file. Expected: 5 passed.
  Panel loading risk: legacy `_load_wav` re-reads the whole manifest for every spectrum, and single_channel_all has 9.1M rows. If `build_panel` takes more than 10 minutes in Task 7, pass a cached `_load_wav` through `legacy_module` overrides that reads each manifest once. The selection is unchanged, and the change gets a ledger ruling.
- [ ] **Step 5:** Commit: `feat: legacy noise applied in normalized space and mapped back to raw spectra`.

### Task 4: Block-2 readouts and the similarity adapter

**Files:**
- Create: `src/spectral_lejepa/evaluation/embedding_eval.py` (part 1)
- Test: `tests/evaluation/test_embedding_eval.py`

**Interfaces:**
- Consumes: `evaluation.run.model_input(raw, ckpt_cfg)` and `training.checkpoint.load_model(path, map_location)`.
- Produces:
  - `READOUTS = ('block2_mean', 'block2_flat')`;
  - `block2_readouts(model, model_inputs, device, batch_size=64) -> {'block2_mean': [N, 256], 'block2_flat': [N, 12288]}`;
  - `stack_similarity(legacy_sim, inputs_z, stack_idx, embeddings, k=5) -> dict`, the legacy `run()` output without its large arrays.

- [ ] **Step 1: Write the failing test.**
  ```python
  import numpy as np
  import pytest
  import torch

  from spectral_lejepa.evaluation.embedding_eval import block2_readouts, stack_similarity
  from spectral_lejepa.models.vit_1d import build_model

  MODEL = {'num_patches': 48, 'dim': 16, 'depth': 6, 'heads': 4, 'mlp_dim': 32, 'dropout': 0.0,
           'predictor_dim': 16, 'predictor_depth': 1, 'predictor_heads': 4, 'predictor_mlp_dim': 32}


  def test_block2_readouts_are_flat_and_mean_of_the_same_states():
      m = build_model(MODEL, 245)
      x = np.random.default_rng(0).normal(size=(5, 245)).astype(np.float32)
      r = block2_readouts(m, x, 'cpu', batch_size=2)
      assert r['block2_flat'].shape == (5, 48 * 16) and r['block2_mean'].shape == (5, 16)
      assert np.allclose(r['block2_flat'].reshape(5, 48, 16).mean(1), r['block2_mean'], atol=1e-6)


  @pytest.mark.slow
  def test_stack_similarity_feeds_the_given_embeddings_in_row_order():
      from spectral_lejepa.evaluation.legacy import legacy_module
      sim = legacy_module('code/eval/evaluations/embedding_similarity.py')
      rng = np.random.default_rng(0)
      stack = np.repeat(np.arange(4), [10, 10, 10, 7])     # a short last stack
      z = rng.normal(size=(len(stack), 245))
      emb = np.eye(4)[stack] + .01 * rng.normal(size=(len(stack), 4))   # perfectly stack-separated
      out = stack_similarity(sim, z, stack, emb, k=5)
      assert out['embedding_stack_match_rate'] == pytest.approx(1.0)
      assert 0 <= out['input_stack_match_rate'] < .5
  ```
- [ ] **Step 2:** Run it. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** part 1 of `embedding_eval.py`:
  ```python
  """Previous-project cosine similarity and noise robustness on block-2 readouts (spec 2026-10-14)."""
  from types import SimpleNamespace

  import numpy as np
  import pandas as pd
  import torch

  READOUTS = ('block2_mean', 'block2_flat')


  @torch.no_grad()
  def block2_readouts(model, model_inputs, device, batch_size=64):
      model.to(device).eval()
      flat, mean = [], []
      for i in range(0, len(model_inputs), batch_size):
          x = torch.as_tensor(model_inputs[i:i + batch_size], dtype=torch.float32, device=device)
          h2 = model.encoder.prefix(model.tokenizer(x), 2).float()           # [B, 48, D], before final norm
          flat.append(h2.reshape(len(x), -1).cpu().numpy())
          mean.append(h2.mean(1).cpu().numpy())
      return {'block2_mean': np.concatenate(mean), 'block2_flat': np.concatenate(flat)}


  class _Precomputed(torch.nn.Module):
      """Stands in for a backbone: input_values are row indices; returns those rows' embeddings."""

      def __init__(self, embeddings):
          super().__init__()
          self.e = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)

      def forward(self, input_values):
          return SimpleNamespace(last_hidden_state=self.e[input_values[:, 0].long()][:, None, :])


  def stack_similarity(legacy_sim, inputs_z, stack_idx, embeddings, k=5):
      df = pd.DataFrame({'data': list(np.asarray(inputs_z, np.float32)), 'stack_idx': np.asarray(stack_idx)})
      idx = torch.arange(len(df), dtype=torch.float32)[:, None]
      loader = [{'data': idx[i:i + 64]} for i in range(0, len(df), 64)]
      out = legacy_sim.run(df, _Precomputed(embeddings), loader, k=k, device='cpu')
      d = out['similarity_distributions']
      return {'embedding_stack_match_rate': float(out['embedding_stack_match_rate']),
              'input_stack_match_rate': float(out['input_stack_match_rate']),
              'match_score_avg': float(out['match_score_avg']),
              'distributions': {key: d[key] for key in d if not key.startswith('sim_matrix')}}
  ```
  `Encoder.prefix(tokens, k)` already exists in `models/vit_1d.py`: it returns the state after the first k blocks, before the final norm. This matches `perturbation.block2_flat`, which uses `hidden[1]`. The test checks shapes, and Task 7 checks that the values match `perturbation.block2_flat` on a real checkpoint.
- [ ] **Step 4:** Run the tests. Expected: 2 passed.
- [ ] **Step 5:** Commit: `feat: block-2 readouts and legacy stack-similarity adapter`.

### Task 5: Suite runner and `evaluate --suite embedding`

**Files:**
- Modify: `src/spectral_lejepa/evaluation/embedding_eval.py` (part 2: `evaluate_model`, `run_suite`)
- Modify: `src/spectral_lejepa/evaluation/run.py`: add `--suite`, and make `--checkpoint` required only for `probe`.
- Create: `configs/eval_embedding.yaml`
- Test: `tests/evaluation/test_embedding_eval.py` (append)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `evaluate_model(model, ckpt_cfg, data, cfg, legacy) -> dict`, with keys `similarity[set][readout]`, `noise[set][readout][kind]` (mean cosine) and `panel[readout]` (raw and centred 100×100 matrices). It also writes `noise_per_sample` arrays.
  - `run_suite(cfg) -> Path` writes:
    - `<out>/<model>/results.json`;
    - `<out>/<model>/arrays.npz` (distributions, panel matrices, per-sample noise cosines);
    - `<out>/summary.csv` (model, set, readout, metric, value, plus ensemble rows);
    - `<out>/run_info.json` (legacy commit, config, noise-seed deviation note).

- [ ] **Step 1:** Write `configs/eval_embedding.yaml`:
  ```yaml
  # Previous-project cosine similarity and noise robustness on block-2 readouts (spec 2026-10-14).
  experiment:
    name: embedding_eval
    seed: 42
    output_dir: outputs/embedding-eval
  data:
    nova_data_dir: /mnt5/noy/SpectralFM/fairseq/data/nova_data
    stack_sets:            # alias: [dataset dir, split, n]  (previous runner DATASET_SPECS)
      sanity: [single_channel_1k, train, 100]
      in_dist: [single_channel_10k, valid, 500]
      multi_ch: [multi_channel, valid, 1000]
      samples: [sampled_data, valid, 1000]
    similarity_sets: [sanity, in_dist]
    label_sets:
      labeled_data: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data]
      dataset0055: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0055]
      dataset0106: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0106]
      dataset0109: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0109]
      dataset0112: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0112]
      dataset0113: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0113]
      dataset0114: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0114]
      dataset0120: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0120]
  evaluation:
    device: cuda
    batch_size: 64
    k: 5
    noise_seed: 0
  models:
    robust_s401: outputs/robustness-continuation/training/robust_robust_s401_20261013-020629/checkpoint_last.pt
    robust_p0: outputs/robust-ensemble/training/robust_p0_20261013-172822/checkpoint_last.pt
    robust_p1: outputs/robust-ensemble/training/robust_p1_20261013-172821/checkpoint_last.pt
    robust_s101: outputs/robust-ensemble/training/robust_s101_20261013-172822/checkpoint_last.pt
  ```
- [ ] **Step 2: Write the failing tests** (append):
  ```python
  def test_evaluate_suite_flag_dispatches_without_a_checkpoint(monkeypatch, tmp_path):
      from spectral_lejepa.evaluation import embedding_eval, run
      seen = {}
      monkeypatch.setattr(embedding_eval, 'run_suite', lambda cfg: seen.setdefault('name', cfg['experiment']['name']))
      run.main(['--suite', 'embedding', '--config', 'configs/eval_embedding.yaml'])
      assert seen['name'] == 'embedding_eval'


  def test_probe_suite_still_requires_a_checkpoint():
      from spectral_lejepa.evaluation import run
      with pytest.raises(SystemExit):
          run.main(['--config', 'configs/eval.yaml'])


  def test_evaluate_model_on_synthetic_data(tmp_path):
      from spectral_lejepa.evaluation.embedding_eval import evaluate_model

      class Legacy:   # minimal stand-ins with the legacy call signatures
          NOISE_CONFIGS = {'gaussian_std': lambda z: z + np.random.normal(0, .01, size=z.shape)}

      rng = np.random.default_rng(0)
      data = {'noise_sets': {'toy': rng.normal(2, 1, (6, 245)).astype(np.float32)},
              'stack_sets': {}, 'panel': None}
      cfg = {'evaluation': {'device': 'cpu', 'batch_size': 4, 'k': 5, 'noise_seed': 0}}
      ck = {'data': {'normalization': 'global'}, 'derived': {'global_stats': {'mean': 2.0, 'std': 1.0}}}
      out = evaluate_model(build_model(MODEL, 245), ck, data, cfg, {'noise': Legacy, 'sim': None})
      for r in ('block2_mean', 'block2_flat'):
          assert 0.9 < out['noise']['toy'][r]['gaussian_std'] <= 1.0
  ```
  The checkpoint config keeps global stats under `derived.global_stats`, which is what `model_input` reads.
- [ ] **Step 3:** Run them. Expected: FAIL (`unrecognized arguments: --suite`, or ImportError).
- [ ] **Step 4: Implement part 2.** Append to `embedding_eval.py`:
  ```python
  import csv
  import json
  from pathlib import Path

  from spectral_lejepa.evaluation.data import load_labeled_data
  from spectral_lejepa.evaluation.legacy import LEGACY_COMMIT, legacy_module
  from spectral_lejepa.evaluation.noise import layer_norm_rows, noisy_raw, row_cosine
  from spectral_lejepa.evaluation.stacks import load_stack_subset


  def _centered_cosine(m):
      c = m - m.mean(0, keepdims=True)
      c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
      return (c @ c.T).astype(np.float32)


  def _cosine(m):
      u = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
      return (u @ u.T).astype(np.float32)


  def load_data(cfg, legacy):
      d = cfg['data']
      stacks = {a: load_stack_subset(f"{d['nova_data_dir']}/{ds}", split, n, seed=cfg['experiment']['seed'])
                for a, (ds, split, n) in d['stack_sets'].items()}
      noise_sets = {a: s['raw'] for a, s in stacks.items()}
      noise_sets.update({name: load_labeled_data(dirs)[0] for name, dirs in d['label_sets'].items()})
      panel_raw, groups = legacy['panel'].build_panel(d['nova_data_dir'], seed=cfg['experiment']['seed'])
      return {'stack_sets': {a: stacks[a] for a in d['similarity_sets']}, 'noise_sets': noise_sets,
              'panel': {'raw': panel_raw, 'groups': groups}}


  def evaluate_model(model, ckpt_cfg, data, cfg, legacy):
      from spectral_lejepa.evaluation.run import model_input
      e = cfg['evaluation']
      embed = lambda raw: block2_readouts(model, model_input(raw, ckpt_cfg), e['device'], e['batch_size'])
      out = {'similarity': {}, 'noise': {}, 'panel': {}, 'arrays': {}}
      for alias, s in data['stack_sets'].items():
          z = layer_norm_rows(s['raw'])[0]
          feats = embed(s['raw'])
          out['similarity'][alias] = {}
          for r in READOUTS:
              res = stack_similarity(legacy['sim'], z, s['stack_idx'], feats[r], k=e['k'])
              for key, v in res.pop('distributions').items():
                  out['arrays'][f'dist/{alias}/{r}/{key}'] = np.asarray(v)
              out['similarity'][alias][r] = res
      for alias, raw in data['noise_sets'].items():
          clean = embed(raw)
          out['noise'][alias] = {r: {} for r in READOUTS}
          for t, kind in enumerate(legacy['noise'].NOISE_CONFIGS):
              noisy = embed(noisy_raw(raw, kind, e['noise_seed'] + t, legacy['noise'].NOISE_CONFIGS))
              for r in READOUTS:
                  cos = row_cosine(clean[r], noisy[r])
                  out['arrays'][f'noise/{alias}/{r}/{kind}'] = cos
                  out['noise'][alias][r][kind] = float(cos.mean())
      if data['panel'] is not None:
          feats = embed(data['panel']['raw'])
          out['arrays']['panel/input'] = _cosine(layer_norm_rows(data['panel']['raw'])[0])
          for r in READOUTS:
              out['arrays'][f'panel/{r}'] = _cosine(feats[r])
              out['arrays'][f'panel/{r}_centered'] = _centered_cosine(feats[r])
          out['panel'] = {'groups': list(data['panel']['groups'])}
      return out


  def _rows(model, res):
      for alias, by_r in res['similarity'].items():
          for r, m in by_r.items():
              for key in ('embedding_stack_match_rate', 'input_stack_match_rate', 'match_score_avg'):
                  yield model, alias, r, key, m[key]
      for alias, by_r in res['noise'].items():
          for r, kinds in by_r.items():
              for kind, v in kinds.items():
                  yield model, alias, r, f'noise_cosine/{kind}', v


  def run_suite(cfg):
      from spectral_lejepa.training.checkpoint import load_model
      out = Path(cfg['experiment']['output_dir'])
      legacy = {'sim': legacy_module('code/eval/evaluations/embedding_similarity.py'),
                'noise': legacy_module('code/eval/evaluations/noise_robustness.py'),
                'panel': legacy_module('code/eval/evaluations/structured_similarity.py',
                                       {'_normalize_like_fairseq': lambda a: a})}   # raw panel spectra
      data = load_data(cfg, legacy)
      rows = []
      for name, path in cfg['models'].items():
          model, ck = load_model(path, map_location='cpu')
          res = evaluate_model(model, ck['config'], data, cfg, legacy)
          (out / name).mkdir(parents=True, exist_ok=True)
          np.savez_compressed(out / name / 'arrays.npz', **res.pop('arrays'))
          (out / name / 'results.json').write_text(json.dumps({'checkpoint': str(path), **res}, indent=2))
          rows += list(_rows(name, res))
      rows += ensemble_rows(rows, list(cfg['models']))
      with open(out / 'summary.csv', 'w', newline='') as f:
          csv.writer(f).writerows([('model', 'set', 'readout', 'metric', 'value'), *rows])
      (out / 'run_info.json').write_text(json.dumps({
          'legacy_commit': LEGACY_COMMIT, 'config': cfg,
          'deviation': 'noise draws seeded with np.random.seed(noise_seed + type index); previous code was unseeded'},
          indent=2, default=str))
      return out


  def ensemble_rows(rows, members):
      """Mean of the members' values per (set, readout, metric); refuses an incomplete member set."""
      by = {}
      for model, alias, r, key, v in rows:
          by.setdefault((alias, r, key), {})[model] = v
      out = []
      for (alias, r, key), vals in sorted(by.items()):
          if sorted(vals) != sorted(members):
              raise ValueError(f'{alias}/{r}/{key}: results for {sorted(vals)}, expected {sorted(members)}')
          out.append(('ensemble', alias, r, key, float(np.mean(list(vals.values())))))
      return out
  ```
  Modify `run.py` `main`:
  - Change `ap.add_argument("--checkpoint", required=True, ...)` to `required=False`.
  - Add `ap.add_argument("--suite", choices=("probe", "embedding"), default="probe")`.
  - Right after `a = ap.parse_args(argv)`, insert:
    ```python
    if a.suite == "embedding":
        from spectral_lejepa.evaluation.embedding_eval import run_suite
        return run_suite(load_config(a.config, a.overrides))
    if not a.checkpoint:
        ap.error("--checkpoint is required for --suite probe")
    ```
- [ ] **Step 5:** Run the test file. Expected: all pass.
- [ ] **Step 6:** Commit: `feat: evaluate --suite embedding runs previous-project similarity and noise metrics`.

### Task 6: Ensemble completeness, re-run safety, and figures

**Files:**
- Create: `src/spectral_lejepa/reporting/embedding_figures.py`
- Modify: `src/spectral_lejepa/__main__.py`: add `'embedding-figures': 'spectral_lejepa.reporting.embedding_figures'`.
- Test: `tests/reporting/test_embedding_figures.py`, plus an addition to `tests/evaluation/test_embedding_eval.py`.

**Interfaces:**
- Consumes: the `run_suite` outputs (`summary.csv`, `<model>/arrays.npz`, `<model>/results.json`).
- Produces:
  - `figures(root: Path, docs: Path | None) -> list[Path]` writes, under `root/figures/<set>/`:
    - `noise_cosine.svg` for every noise set;
    - `match_rate.svg` and `cosine_distributions.svg` for similarity sets.

    It also writes `root/figures/panel.svg` (input, block2_mean and block2_flat, raw and centred, for robust_s401) and `root/figures/summary_noise.svg` (set × noise type heatmap of ensemble cosine, one panel per readout). `summary_noise.svg` and `panel.svg` are copied to `docs` when given.
  - `main()` runs `figures(Path('outputs/embedding-eval'), Path('docs/figures/embedding-eval'))`.

- [ ] **Step 1: Write the failing tests.**
  ```python
  # tests/evaluation/test_embedding_eval.py (append)
  def test_ensemble_refuses_a_missing_member():
      from spectral_lejepa.evaluation.embedding_eval import ensemble_rows
      rows = [('a', 's', 'block2_flat', 'm', 1.0), ('b', 's', 'block2_flat', 'm', 3.0)]
      assert ensemble_rows(rows, ['a', 'b']) == [('ensemble', 's', 'block2_flat', 'm', 2.0)]
      with pytest.raises(ValueError, match='expected'):
          ensemble_rows(rows[:1], ['a', 'b'])
  ```
  ```python
  # tests/reporting/test_embedding_figures.py
  import csv
  import json

  import numpy as np

  from spectral_lejepa.reporting.embedding_figures import figures

  KINDS = ['gaussian_std', 'gaussian_mean', 'shot_low', 'shot_high', 'gain_low', 'gain_high']
  READOUTS = ['block2_mean', 'block2_flat']


  def fake_run(root):
      rows = [('model', 'set', 'readout', 'metric', 'value')]
      for m in ['robust_s401', 'robust_p0', 'ensemble']:
          for s in ['sanity', 'dataset0055']:
              for r in READOUTS:
                  rows += [(m, s, r, f'noise_cosine/{k}', .9) for k in KINDS]
          for r in READOUTS:
              rows += [(m, 'sanity', r, key, .5) for key in ('embedding_stack_match_rate', 'input_stack_match_rate', 'match_score_avg')]
          if m == 'ensemble':
              continue
          d = root / m
          d.mkdir(parents=True)
          arrays = {f'panel/{r}{c}': np.eye(100, dtype=np.float32) for r in READOUTS for c in ('', '_centered')}
          arrays['panel/input'] = np.eye(100, dtype=np.float32)
          for r in READOUTS:
              for key in ('intra_stack_emb', 'intra_stack_inp', 'random_inter_emb', 'random_inter_inp', 'preset_inter_emb', 'preset_inter_inp'):
                  arrays[f'dist/sanity/{r}/{key}'] = np.linspace(-1, 1, 50)
          np.savez(d / 'arrays.npz', **arrays)
          (d / 'results.json').write_text(json.dumps({'panel': {'groups': ['single_channel_all'] * 30 + ['multi_channel'] * 30 + ['sampled_data'] * 20 + ['labeled_data'] * 20}}))
      with open(root / 'summary.csv', 'w', newline='') as f:
          csv.writer(f).writerows(rows)


  def test_figures_cover_every_set_and_copy_summaries(tmp_path):
      fake_run(tmp_path / 'run')
      paths = figures(tmp_path / 'run', tmp_path / 'docs')
      names = {p.relative_to(tmp_path / 'run' / 'figures').as_posix() for p in paths}
      assert {'sanity/noise_cosine.svg', 'sanity/match_rate.svg', 'sanity/cosine_distributions.svg',
              'dataset0055/noise_cosine.svg', 'panel.svg', 'summary_noise.svg'} <= names
      assert sorted(p.name for p in (tmp_path / 'docs').iterdir()) == ['panel.svg', 'summary_noise.svg']


  def test_rerun_overwrites_instead_of_accumulating(tmp_path):
      fake_run(tmp_path / 'run')
      a = figures(tmp_path / 'run', None)
      b = figures(tmp_path / 'run', None)
      assert sorted(a) == sorted(b) and all(p.read_bytes() for p in b)
  ```
- [ ] **Step 2:** Run the tests. Expected: FAIL with ImportError, or the ensemble test passing (it was implemented in Task 5). If the ensemble test passes, keep it as a regression pin; it was written against Task 5's code, which already exists.
- [ ] **Step 3: Implement** `embedding_figures.py`:
  ```python
  """Per-dataset figures for the embedding suite (outputs/embedding-eval)."""
  import csv
  import json
  import shutil
  from pathlib import Path

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np

  READOUTS = ('block2_mean', 'block2_flat')
  COLORS = {'input': '#999999', 'block2_mean': '#3a6ea5', 'block2_flat': '#c0504d'}
  PANEL_MODEL = 'robust_s401'
  plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                       'svg.fonttype': 'none', 'svg.hashsalt': 'spectralfm'})


  def _save(fig, path):
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.tight_layout()
      fig.savefig(path, metadata={'Date': None})
      plt.close(fig)
      return path


  def _summary(root):
      with open(root / 'summary.csv') as f:
          return [(m, s, r, k, float(v)) for m, s, r, k, v in list(csv.reader(f))[1:]]


  def noise_figure(rows, s, path):
      kinds = sorted({k.split('/', 1)[1] for _, ss, _, k, _ in rows if ss == s and k.startswith('noise_cosine/')},
                     key=lambda k: k)
      fig, ax = plt.subplots(figsize=(7, 3))
      x = np.arange(len(kinds))
      for i, r in enumerate(READOUTS):
          ens = [next(v for m, ss, rr, k, v in rows if (m, ss, rr, k) == ('ensemble', s, r, f'noise_cosine/{kd}')) for kd in kinds]
          ax.bar(x + (i - .5) * .38, ens, .38, label=f'{r} (ensemble mean)', color=COLORS[r])
          for m in sorted({m for m, ss, *_ in rows if ss == s and m != 'ensemble'}):
              vals = [next(v for mm, ss, rr, k, v in rows if (mm, ss, rr, k) == (m, s, r, f'noise_cosine/{kd}')) for kd in kinds]
              ax.plot(x + (i - .5) * .38, vals, 'k.', ms=3)
      ax.set(xticks=x, xticklabels=kinds, ylim=(min(0, ax.get_ylim()[0]), 1.02), ylabel='cosine(clean, noisy)',
             title=f'{s}: noise robustness (dots = members)')
      ax.legend(frameon=False, fontsize=7, loc='lower left')
      return _save(fig, path)


  def match_figure(rows, s, path):
      fig, ax = plt.subplots(figsize=(5, 3))
      models = sorted({m for m, ss, _, k, _ in rows if ss == s and k == 'embedding_stack_match_rate'})
      x = np.arange(len(models))
      inp = [next(v for m, ss, r, k, v in rows if (m, ss, r, k) == (mm, s, 'block2_flat', 'input_stack_match_rate')) for mm in models]
      ax.bar(x - .27, inp, .27, label='input', color=COLORS['input'])
      for i, r in enumerate(READOUTS):
          ax.bar(x + i * .27, [next(v for m, ss, rr, k, v in rows if (m, ss, rr, k) == (mm, s, r, 'embedding_stack_match_rate')) for mm in models],
                 .27, label=r, color=COLORS[r])
      ax.set(xticks=x, xticklabels=models, ylim=(0, 1), ylabel='top-5 same-stack rate', title=f'{s}: stack retrieval')
      ax.legend(frameon=False, fontsize=7)
      plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
      return _save(fig, path)


  def distribution_figure(root, members, s, path):
      fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
      keys = [('intra_stack', 'same stack'), ('random_inter', 'random pair'), ('preset_inter', 'same position')]
      for ax, space in zip(axes, ('inp',) + READOUTS):
          for key, label in keys:
              suffix, r = ('inp', READOUTS[1]) if space == 'inp' else ('emb', space)
              vals = np.concatenate([np.load(root / m / 'arrays.npz')[f'dist/{s}/{r}/{key}_{suffix}'] for m in members])
              ax.hist(vals, bins=40, range=(-1, 1), histtype='step', density=True, label=label)
          ax.set(title='input' if space == 'inp' else space, xlabel='cosine')
      axes[0].legend(frameon=False, fontsize=7)
      fig.suptitle(f'{s}: pairwise cosine, members pooled', fontsize=9)
      return _save(fig, path)


  def panel_figure(root, path):
      a = np.load(root / PANEL_MODEL / 'arrays.npz')
      names = ['panel/input'] + [f'panel/{r}{c}' for r in READOUTS for c in ('', '_centered')]
      fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3.2))
      for ax, n in zip(axes, names):
          ax.imshow(a[n], vmin=-1, vmax=1, cmap='RdBu_r')
          for b in (30, 60, 80):
              ax.axhline(b - .5, color='k', lw=.5)
              ax.axvline(b - .5, color='k', lw=.5)
          ax.set(title=n.split('/', 1)[1], xticks=[], yticks=[])
      fig.suptitle(f'Structured 100-spectrum panel ({PANEL_MODEL}); blocks: single_channel_all, multi_channel, sampled_data, labeled_data', fontsize=9)
      return _save(fig, path)


  def summary_figure(rows, path):
      sets = sorted({s for m, s, _, k, _ in rows if m == 'ensemble' and k.startswith('noise_cosine/')})
      kinds = sorted({k.split('/', 1)[1] for *_, k, _ in rows if k.startswith('noise_cosine/')})
      fig, axes = plt.subplots(1, 2, figsize=(11, .45 * len(sets) + 1.6))
      for ax, r in zip(axes, READOUTS):
          M = np.array([[next(v for m, ss, rr, k, v in rows if (m, ss, rr, k) == ('ensemble', s, r, f'noise_cosine/{kd}')) for kd in kinds] for s in sets])
          ax.imshow(M, vmin=0, vmax=1, cmap='viridis', aspect='auto')
          for (i, j), v in np.ndenumerate(M):
              ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='w' if v < .6 else 'k')
          ax.set(xticks=range(len(kinds)), xticklabels=kinds, yticks=range(len(sets)), yticklabels=sets, title=f'{r}: ensemble noise cosine')
          plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
      return _save(fig, path)


  def figures(root, docs=None):
      root = Path(root)
      rows = _summary(root)
      members = sorted({m for m, *_ in rows if m != 'ensemble'})
      out = root / 'figures'
      paths = []
      for s in sorted({s for _, s, _, k, _ in rows if k.startswith('noise_cosine/')}):
          paths.append(noise_figure(rows, s, out / s / 'noise_cosine.svg'))
      for s in sorted({s for _, s, _, k, _ in rows if k == 'embedding_stack_match_rate'}):
          paths.append(match_figure(rows, s, out / s / 'match_rate.svg'))
          paths.append(distribution_figure(root, members, s, out / s / 'cosine_distributions.svg'))
      paths.append(panel_figure(root, out / 'panel.svg'))
      paths.append(summary_figure(rows, out / 'summary_noise.svg'))
      if docs is not None:
          Path(docs).mkdir(parents=True, exist_ok=True)
          for p in (out / 'panel.svg', out / 'summary_noise.svg'):
              shutil.copy(p, Path(docs) / p.name)
      return paths


  def main():
      figures(Path('outputs/embedding-eval'), Path('docs/figures/embedding-eval'))


  if __name__ == '__main__':
      main()
  ```
  Add `'embedding-figures': 'spectral_lejepa.reporting.embedding_figures'` to `COMMANDS`. Add `'embedding-figures': ()` to `NO_OPTIONS` (its main parses no options).
- [ ] **Step 4:** Run `tests/reporting/test_embedding_figures.py`, `tests/evaluation/test_embedding_eval.py` and `tests/test_cli.py`. Expected: all pass. The CLI tests now cover 14 commands.
- [ ] **Step 5:** Commit: `feat: per-dataset figures for the embedding suite`.

### Task 7: Real run, report section, docs, push

**Files:**
- Modify: `docs/TRAINING_REPORT.md`: add section "Embedding similarity and noise robustness" before "Limitations", and renumber.
- Regenerate: `docs/TRAINING_REPORT.html`.
- Modify: `CLAUDE.md` (repository map: `evaluate --suite embedding`, `embedding-figures`) and the spec status line.

- [ ] **Step 1:** Run the slow tests from Tasks 1–4.
  Run: `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q -m slow tests/evaluation`. Expected: all pass.
- [ ] **Step 2:** Readout cross-check on a real checkpoint: `block2_readouts(...)['block2_flat']` must equal `perturbation.block2_flat(...)` on 16 dataset0055 spectra.
  ```bash
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/src .venv/bin/python -c "
  import numpy as np
  from spectral_lejepa.training.checkpoint import load_model
  from spectral_lejepa.evaluation.run import model_input
  from spectral_lejepa.evaluation.perturbation import block2_flat
  from spectral_lejepa.evaluation.embedding_eval import block2_readouts
  m, ck = load_model('outputs/robust-ensemble/training/robust_p0_20261013-172822/checkpoint_last.pt', map_location='cpu')
  raw = np.load('outputs/frozen-feature-study/prior_s2/dataset0055/bank.npz')['input_raw'][:16, 0].astype(np.float32)
  x = model_input(raw, ck['config'])
  a, b = block2_readouts(m, x, 'cuda')['block2_flat'], block2_flat(m, x)
  print('max abs diff', float(np.abs(a - b).max())); assert np.allclose(a, b, atol=1e-5)"
  ```
  Expected: `max abs diff` ≤ 1e-5.
- [ ] **Step 3:** Run the suite.
  Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/claude-1040/mpl PYTHONPATH=$PWD/src .venv/bin/python -m spectral_lejepa evaluate --suite embedding --config configs/eval_embedding.yaml 2>&1 | tee outputs/embedding-eval.log`
  Expected: `outputs/embedding-eval/{robust_s401,robust_p0,robust_p1,robust_s101}/results.json`, `summary.csv` with ensemble rows, and `run_info.json`. All noise cosines lie in [−1, 1], and none is NaN. Check:
  `.venv/bin/python -c "import csv;r=list(csv.DictReader(open('outputs/embedding-eval/summary.csv')));v=[float(x['value']) for x in r];print(len(r),min(v),max(v));assert all(-1<=x<=100 for x in v)"`
- [ ] **Step 4:** Make the figures.
  Run: `MPLCONFIGDIR=/tmp/claude-1040/mpl PYTHONPATH=$PWD/src .venv/bin/python -m spectral_lejepa embedding-figures`
  Expected: `outputs/embedding-eval/figures/<set>/...` for 12 noise sets and 2 similarity sets, plus `docs/figures/embedding-eval/{panel,summary_noise}.svg`. Look at `summary_noise.svg` and `panel.svg` rendered to PNG in the scratchpad, as for the training report, before writing the text.
- [ ] **Step 5:** Add the training-report section with measured numbers only:
  - a table of ensemble noise cosine (mean over the six types) per set for block2_mean and block2_flat;
  - the sanity/in_dist top-5 stack match: input vs block2_mean vs block2_flat;
  - the two figures;
  - a note on the seeded-noise deviation and that "geometry metrics are not accuracy".

  Then run `python -m spectral_lejepa training-report-html`. Update CLAUDE.md and the spec status line (`Status: implemented 2026-10-14; results in TRAINING_REPORT section …`).
- [ ] **Step 6:** Run the full fast and slow suites; all must pass. Then commit (`docs: embedding similarity and noise robustness results`), push, and republish the HTML artifact from `docs/TRAINING_REPORT.html`, keeping the same URL.
