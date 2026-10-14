# Inference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python -m spectral_lejepa embed INPUT --output OUT.npz` and `SpectralEmbedder().embed(x)` turn raw spectra into the robust-ensemble embedding `(N, M, 12288)`. The encoders come from a bundle that `export-ensemble` builds once from the training checkpoints.

**Architecture:**
- `models/export.py` gains `script_encoder()`, factored out of `export()` with no behaviour change, and `export_ensemble()`, which writes `member_<name>.pt` files plus `manifest.json`.
- `src/spectral_lejepa/inference.py` is self-contained: it imports only torch, numpy, soundfile and scikit-learn. It loads the bundle, reads spectra, embeds them and offers `EnsembleRegressor`.

**Tech Stack:** Python 3.10, PyTorch (TorchScript), NumPy, soundfile, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-inference-design.md`

## Global Constraints

- Default bundle: `outputs/inference/robust-ensemble/`, which is git-ignored. Resolution order: the `bundle` argument, then `SPECTRALFM_BUNDLE`, then the default.
- Bundle files are `member_<name>.pt` (TorchScript `ProductionEncoder`) and `manifest.json`. The manifest has `format_version: 1`, `readout: "layer2/flat"`, `input`, `output` and `members`; each member has `name`, `file`, `sha256`, `source_checkpoint`, `source_sha256`, `step` and `global_stats`.
- Embedding: float32 `(N, M, 12288)`, with members in manifest order. Members are never averaged or concatenated.
- Inputs: an ndarray or `.npy` of shape `(N, 245)`; a mono 245-sample `.wav`; a folder of `*.wav` sorted by name; or a list of `.wav` paths. Anything else raises `ValueError` before any output is written.
- `inference.py` must not import `spectral_lejepa` or any other project module.
- The CLI `.npz` has the keys `embeddings`, `ids`, `members` and `bundle_manifest`.
- Members, from `configs/inference.yaml`:
  - robust_s401: `outputs/robustness-continuation/training/robust_robust_s401_20261013-020629/checkpoint_last.pt`
  - robust_p0: `outputs/robust-ensemble/training/robust_p0_20261013-172822/checkpoint_last.pt`
  - robust_p1: `outputs/robust-ensemble/training/robust_p1_20261013-172821/checkpoint_last.pt`
  - robust_s101: `outputs/robust-ensemble/training/robust_s101_20261013-172822/checkpoint_last.pt`
- Tests: `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q`; the real-bundle test is `@pytest.mark.slow`. Commit as `Hadar <hal.nls@gmail.com>` with the session trailer; push at the end.

## Review Focus

1. **A single 1-D spectrum `(245,)`.** A user embedding one spectrum expects `(1, M, 12288)`, not an error. Test in Task 2.
2. **float64 input.** NumPy defaults to float64. It must be cast to float32, not rejected, because the encoders demand float32. Test in Task 2.
3. **A CPU-only machine.** A bundle exported on a CUDA box must load with `device='cpu'`, via `map_location`. Test in Task 2.
4. **Batching.** `N` larger than `batch_size` must keep row order. Test in Task 2 with `batch_size=2`.
5. **Folders with other files.** `.txt`/`.json` files beside the wavs are ignored, and ids follow the sorted wav names. Test in Task 2.

---

### Task 1: `export-ensemble` builds the bundle

**Files:**
- Modify: `src/spectral_lejepa/models/export.py`: factor `script_encoder()` out of `export()`, and add `export_ensemble()` and `main_ensemble()`.
- Create: `src/spectral_lejepa/models/export_ensemble.py` (CLI shim) and `configs/inference.yaml`.
- Modify: `src/spectral_lejepa/__main__.py` (add the `export-ensemble` command).
- Test: `tests/models/test_export_ensemble.py`, plus the shared fixture in `tests/conftest.py`.

**Interfaces:**
- Produces:
  - `script_encoder(checkpoint) -> (torch.jit.ScriptModule, ckpt: dict)`.
  - `export_ensemble(members: dict[str, path], output: Path, force: bool = False) -> dict`, which returns the manifest.
  - The `tiny_checkpoint(tmp_path, name, seed) -> Path` fixture factory in `tests/conftest.py`, named `make_checkpoint`.

- [ ] **Step 1: Add the shared fixture to `tests/conftest.py`.** Append it, keeping the existing content:
  ```python
  import pytest


  @pytest.fixture
  def make_checkpoint(tmp_path):
      """Factory: a minimal stage-2-like checkpoint (48 patches, d256, global normalization) on disk."""
      import torch
      import yaml
      from spectral_lejepa.models.vit_1d import build_model

      def make(name, seed=0):
          torch.manual_seed(seed)
          cfg = yaml.safe_load(open('configs/final_model.yaml'))
          model = build_model(cfg['model'], 245)
          ckpt = {'model': model.state_dict(), 'step': 7, 'wandb_run_id': None,
                  'config': {'model': cfg['model'], 'data': {'sequence_length': 245, 'normalization': 'global'},
                             'derived': {'global_stats': {'mean': .3784, 'std': .5126}},
                             'experiment': {'seed': seed}}}
          path = tmp_path / f'{name}.pt'
          torch.save(ckpt, path)
          return path
      return make
  ```
- [ ] **Step 2: Write the failing tests** in `tests/models/test_export_ensemble.py`:
  ```python
  import json

  import pytest
  import torch

  from spectral_lejepa.models.export import export_ensemble, script_encoder
  from spectral_lejepa.models.vit_1d import EvalBackbone
  from spectral_lejepa.training.checkpoint import load_model, sha256_file


  def test_bundle_has_one_scripted_encoder_per_member_and_a_manifest(make_checkpoint, tmp_path):
      members = {'a': make_checkpoint('a', 0), 'b': make_checkpoint('b', 1)}
      m = export_ensemble(members, tmp_path / 'bundle')
      assert m['format_version'] == 1 and m['readout'] == 'layer2/flat'
      assert [x['name'] for x in m['members']] == ['a', 'b']
      assert m['output']['shape'] == ['N', 2, 12288]
      for x in m['members']:
          f = tmp_path / 'bundle' / x['file']
          assert x['sha256'] == sha256_file(f) and x['source_sha256'] == sha256_file(members[x['name']])
      assert json.loads((tmp_path / 'bundle' / 'manifest.json').read_text()) == m


  def test_scripted_member_equals_block2_flat_of_the_checkpoint(make_checkpoint, tmp_path):
      ck = make_checkpoint('a')
      export_ensemble({'a': ck}, tmp_path / 'bundle')
      enc = torch.jit.load(str(tmp_path / 'bundle' / 'member_a.pt'))
      model, _ = load_model(ck)
      raw = torch.randn(3, 245)
      with torch.no_grad():
          expected = EvalBackbone(model)((raw - .3784) / .5126).hidden_states[2].flatten(1)
          torch.testing.assert_close(enc(raw), expected, atol=1e-5, rtol=1e-5)


  def test_refuses_to_overwrite_a_bundle_without_force(make_checkpoint, tmp_path):
      export_ensemble({'a': make_checkpoint('a')}, tmp_path / 'bundle')
      with pytest.raises(FileExistsError):
          export_ensemble({'a': make_checkpoint('a')}, tmp_path / 'bundle')
      export_ensemble({'a': make_checkpoint('a')}, tmp_path / 'bundle', force=True)


  def test_script_encoder_returns_the_checkpoint(make_checkpoint):
      compiled, ckpt = script_encoder(make_checkpoint('a'))
      assert ckpt['step'] == 7 and compiled(torch.randn(1, 245)).shape == (1, 12288)
  ```
- [ ] **Step 3:** Run `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q tests/models/test_export_ensemble.py`. Expected: FAIL with `ImportError` (`export_ensemble`).
- [ ] **Step 4: Implement** in `models/export.py`. Replace `export()` with this pair, keeping its behaviour and metadata unchanged, and add the ensemble export:
  ```python
  def script_encoder(checkpoint):
      """TorchScript raw [B,245] -> block2-flat [B,12288] encoder, verified against the in-memory model."""
      model, ckpt = load_model(str(checkpoint))
      if ckpt['config']['data']['normalization'] != 'global':
          raise ValueError('Production export requires global input normalization')
      stats = ckpt['config']['derived']['global_stats']
      compiled = torch.jit.script(ProductionEncoder(model.eval(), stats['mean'], stats['std']).eval())
      for batch in (1, 3):
          raw = torch.randn(batch, 245)
          with torch.no_grad():
              token = model.tokenizer((raw - stats['mean']) / stats['std'])
              expected = model.encoder.blocks[1](model.encoder.blocks[0](token)).flatten(1)
              torch.testing.assert_close(compiled(raw), expected, atol=1e-5, rtol=1e-5)
      return compiled, ckpt


  def _sha256(path):
      digest = hashlib.sha256()
      with open(path, 'rb') as f:
          for chunk in iter(lambda: f.read(1024 * 1024), b''):
              digest.update(chunk)
      return digest.hexdigest()


  def export(checkpoint, output):
      compiled, ckpt = script_encoder(checkpoint)
      stats = ckpt['config']['derived']['global_stats']
      output = Path(output); output.mkdir(parents=True, exist_ok=True)
      compiled.save(str(output / 'encoder.pt'))
      metadata = {'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256(checkpoint),
          'step': ckpt['step'], 'run_id': ckpt.get('wandb_run_id'),
          'training_seed': ckpt['config']['experiment']['seed'], 'readout': 'layer2/flat',
          'input': {'shape': ['B', 245], 'dtype': 'float32', 'normalization': 'included global affine'},
          'output': {'shape': ['B', 12288], 'dtype': 'float32'}, 'global_stats': stats,
          'benchmark_status': 'pending fixed-readout evaluation; deployment validation still required'}
      (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
      print(f'Exported: {output / "encoder.pt"}', flush=True)


  def export_ensemble(members, output, force=False):
      """Inference bundle: member_<name>.pt TorchScript encoders + manifest.json (see docs inference spec)."""
      output = Path(output)
      if output.exists() and any(output.iterdir()) and not force:
          raise FileExistsError(f'{output} is not empty; pass --force to overwrite')
      output.mkdir(parents=True, exist_ok=True)
      entries = []
      for name, checkpoint in members.items():
          compiled, ckpt = script_encoder(checkpoint)
          file = f'member_{name}.pt'
          compiled.save(str(output / file))
          entries.append({'name': name, 'file': file, 'sha256': _sha256(output / file),
                          'source_checkpoint': str(checkpoint), 'source_sha256': _sha256(checkpoint),
                          'step': ckpt['step'], 'global_stats': ckpt['config']['derived']['global_stats']})
      manifest = {'format_version': 1, 'readout': 'layer2/flat',
                  'input': {'length': 245, 'dtype': 'float32', 'normalization': 'inside each encoder'},
                  'output': {'shape': ['N', len(entries), 12288], 'dtype': 'float32'}, 'members': entries}
      (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
      print(f'Bundle: {output} ({len(entries)} members)', flush=True)
      return manifest


  def main_ensemble(argv=None):
      import yaml
      p = argparse.ArgumentParser(description='Build the inference bundle from the members in a config.')
      p.add_argument('--config', type=Path, default=Path('configs/inference.yaml'))
      p.add_argument('--output', type=Path)
      p.add_argument('--force', action='store_true')
      a = p.parse_args(argv)
      cfg = yaml.safe_load(a.config.read_text())
      export_ensemble(cfg['members'], a.output or Path(cfg['bundle']), force=a.force)
  ```
  Create `src/spectral_lejepa/models/export_ensemble.py`:
  ```python
  """python -m spectral_lejepa export-ensemble [--config configs/inference.yaml] [--output DIR] [--force]"""
  from spectral_lejepa.models.export import main_ensemble as main

  if __name__ == '__main__':
      main()
  ```
  Create `configs/inference.yaml`:
  ```yaml
  # Inference bundle: robust seed-ensemble members (stage-2 checkpoints, kept outside git).
  # Build once: python -m spectral_lejepa export-ensemble
  bundle: outputs/inference/robust-ensemble
  members:
    robust_s401: outputs/robustness-continuation/training/robust_robust_s401_20261013-020629/checkpoint_last.pt
    robust_p0: outputs/robust-ensemble/training/robust_p0_20261013-172822/checkpoint_last.pt
    robust_p1: outputs/robust-ensemble/training/robust_p1_20261013-172821/checkpoint_last.pt
    robust_s101: outputs/robust-ensemble/training/robust_s101_20261013-172822/checkpoint_last.pt
  ```
  In `__main__.py` `COMMANDS`, add `'export-ensemble': 'spectral_lejepa.models.export_ensemble',`.
- [ ] **Step 5:** Run the Task 1 tests, `tests/models/test_final_model.py` and `tests/test_cli.py`. Expected: all pass.
- [ ] **Step 6:** Commit: `feat: export-ensemble builds the inference bundle`.

### Task 2: `inference.py` loads the bundle and embeds spectra

**Files:**
- Create: `src/spectral_lejepa/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: Task 1's bundle layout and the `make_checkpoint` fixture.
- Produces:
  - `DEFAULT_BUNDLE = Path('outputs/inference/robust-ensemble')`, `LENGTH = 245`;
  - `resolve_bundle(bundle=None) -> Path`;
  - `load_spectra(x) -> (np.ndarray (N,245) float32, list[str])`;
  - `SpectralEmbedder(bundle=None, device='auto', batch_size=256)` with `.members: list[str]`, `.manifest: dict`, `.embed(x) -> (N,M,12288)` and `.embed_with_ids(x) -> (array, ids)`;
  - `EnsembleRegressor(alphas=np.logspace(-3, 3, 20))` with `.fit(F, y)` and `.predict(F)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_inference.py`:
  ```python
  import ast
  import json

  import numpy as np
  import pytest
  import soundfile as sf
  import torch

  from spectral_lejepa import inference as inf
  from spectral_lejepa.models.export import export_ensemble
  from spectral_lejepa.models.vit_1d import EvalBackbone
  from spectral_lejepa.training.checkpoint import load_model


  @pytest.fixture
  def bundle(make_checkpoint, tmp_path):
      members = {'a': make_checkpoint('a', 0), 'b': make_checkpoint('b', 1)}
      export_ensemble(members, tmp_path / 'bundle')
      return tmp_path / 'bundle', members


  def spectra(n=5, seed=0):
      return np.random.default_rng(seed).normal(.4, .5, (n, 245)).astype(np.float32)


  def test_module_is_self_contained():
      tree = ast.parse(open(inf.__file__).read())
      names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
      names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
      assert not [m for m in names if m.startswith('spectral_lejepa')]


  def test_embedding_is_each_members_block2_flat_in_manifest_order(bundle):
      path, members = bundle
      x = spectra()
      e = inf.SpectralEmbedder(path, device='cpu', batch_size=2).embed(x)   # batches keep row order
      assert e.shape == (5, 2, 12288) and e.dtype == np.float32
      for j, name in enumerate(['a', 'b']):
          model, _ = load_model(members[name])
          with torch.no_grad():
              ref = EvalBackbone(model)((torch.from_numpy(x) - .3784) / .5126).hidden_states[2].flatten(1).numpy()
          assert np.allclose(e[:, j], ref, atol=1e-5)


  def test_single_spectrum_and_float64_are_accepted(bundle):
      emb = inf.SpectralEmbedder(bundle[0], device='cpu')
      one = spectra(1)[0].astype(np.float64)
      assert emb.embed(one).shape == (1, 2, 12288)


  def test_tampered_member_is_rejected(bundle):
      path, _ = bundle
      with open(path / 'member_a.pt', 'ab') as f:
          f.write(b'x')
      with pytest.raises(ValueError, match='checksum'):
          inf.SpectralEmbedder(path, device='cpu')


  def test_bundle_resolution_order(bundle, monkeypatch, tmp_path):
      path, _ = bundle
      monkeypatch.setenv('SPECTRALFM_BUNDLE', str(path))
      assert inf.resolve_bundle() == path
      assert inf.resolve_bundle(str(path)) == path
      monkeypatch.delenv('SPECTRALFM_BUNDLE')
      monkeypatch.setattr(inf, 'DEFAULT_BUNDLE', tmp_path / 'missing')
      with pytest.raises(FileNotFoundError, match='export-ensemble'):
          inf.resolve_bundle()


  def test_load_spectra_formats_and_ids(tmp_path):
      x = spectra(3)
      np.save(tmp_path / 'x.npy', x)
      assert inf.load_spectra(x)[1] == ['0', '1', '2']
      assert np.array_equal(inf.load_spectra(tmp_path / 'x.npy')[0], x)
      d = tmp_path / 'wavs'
      d.mkdir()
      for name, row in zip(['b.wav', 'a.wav', 'c.wav'], x):
          sf.write(d / name, row, 16000, subtype='FLOAT')
      (d / 'notes.txt').write_text('ignored')
      arr, ids = inf.load_spectra(d)
      assert ids == ['a.wav', 'b.wav', 'c.wav'] and np.allclose(arr[1], x[0])
      arr2, ids2 = inf.load_spectra([d / 'c.wav', d / 'a.wav'])
      assert ids2 == ['c.wav', 'a.wav'] and np.allclose(arr2[0], x[2])


  @pytest.mark.parametrize('bad, message', [
      (np.zeros((2, 240), np.float32), '245'),
      (np.full((1, 245), np.nan, np.float32), 'finite'),
      (np.zeros((0, 245), np.float32), 'empty'),
  ])
  def test_bad_arrays_are_rejected(bad, message):
      with pytest.raises(ValueError, match=message):
          inf.load_spectra(bad)


  def test_bad_wavs_are_rejected(tmp_path):
      sf.write(tmp_path / 'short.wav', np.zeros(240, np.float32), 16000)
      with pytest.raises(ValueError, match='short.wav'):
          inf.load_spectra(tmp_path / 'short.wav')
      sf.write(tmp_path / 'stereo.wav', np.zeros((245, 2), np.float32), 16000)
      with pytest.raises(ValueError, match='mono'):
          inf.load_spectra(tmp_path / 'stereo.wav')
      (tmp_path / 'empty').mkdir()
      with pytest.raises(ValueError, match='no .wav'):
          inf.load_spectra(tmp_path / 'empty')


  def test_ensemble_regressor_averages_member_predictions():
      rng = np.random.default_rng(0)
      F = rng.normal(size=(60, 2, 8))
      y = F[:, 0, 0] + F[:, 1, 1]
      reg = inf.EnsembleRegressor().fit(F, y)
      each = [m.predict(F[:, j]) for j, m in enumerate(reg.models_)]
      assert np.allclose(reg.predict(F), np.mean(each, 0))
  ```
- [ ] **Step 2:** Run `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q tests/test_inference.py`. Expected: FAIL with `ImportError` (no `spectral_lejepa.inference`).
- [ ] **Step 3: Implement** `src/spectral_lejepa/inference.py`:
  ```python
  """SpectralFM inference: raw 245-point spectra -> robust-ensemble embedding (N, M, 12288).

  Self-contained (torch, numpy, soundfile, scikit-learn only): copy this file and a bundle anywhere.
      from spectral_lejepa.inference import SpectralEmbedder, EnsembleRegressor
      F = SpectralEmbedder().embed('spectra/')            # (N, M, 12288): one block-2 flat vector per member
      reg = EnsembleRegressor().fit(F_train, y_train)     # one ridge per member, predictions averaged
      python -m spectral_lejepa embed INPUT --output OUT.npz [--bundle DIR] [--device auto|cpu|cuda]
  Build the bundle once with: python -m spectral_lejepa export-ensemble
  """
  import argparse
  import hashlib
  import json
  import os
  from pathlib import Path

  import numpy as np
  import soundfile as sf
  import torch

  DEFAULT_BUNDLE = Path('outputs/inference/robust-ensemble')
  LENGTH = 245


  def resolve_bundle(bundle=None):
      path = Path(bundle or os.environ.get('SPECTRALFM_BUNDLE') or DEFAULT_BUNDLE)
      if not (path / 'manifest.json').is_file():
          raise FileNotFoundError(f'no inference bundle at {path}; build it with '
                                  '`python -m spectral_lejepa export-ensemble` or set SPECTRALFM_BUNDLE')
      return path


  def _check(arr, where):
      if arr.ndim != 2 or arr.shape[1] != LENGTH:
          raise ValueError(f'{where}: expected spectra of {LENGTH} points, got shape {arr.shape}')
      if len(arr) == 0:
          raise ValueError(f'{where}: empty input')
      if not np.isfinite(arr).all():
          raise ValueError(f'{where}: values must be finite (row {int(np.where(~np.isfinite(arr).all(1))[0][0])})')
      return arr.astype(np.float32, copy=False)


  def _read_wav(path):
      data, _ = sf.read(str(path), dtype='float32', always_2d=True)
      if data.shape[1] != 1:
          raise ValueError(f'{path.name}: expected a mono wav, got {data.shape[1]} channels')
      if data.shape[0] != LENGTH:
          raise ValueError(f'{path.name}: expected {LENGTH} samples, got {data.shape[0]}')
      return data[:, 0]


  def load_spectra(x):
      """(spectra (N, 245) float32, ids). x: array, .npy, .wav, folder of .wav, or list of .wav paths."""
      if isinstance(x, np.ndarray):
          arr = x[None] if x.ndim == 1 else x
          return _check(np.asarray(arr), 'array'), [str(i) for i in range(len(arr))]
      if isinstance(x, (list, tuple)):
          paths = [Path(p) for p in x]
      else:
          p = Path(x)
          if p.suffix == '.npy':
              arr = np.load(p)
              arr = arr[None] if arr.ndim == 1 else arr
              return _check(arr, p.name), [str(i) for i in range(len(arr))]
          paths = sorted(p.glob('*.wav')) if p.is_dir() else [p]
          if p.is_dir() and not paths:
              raise ValueError(f'{p}: no .wav files')
      if not paths:
          raise ValueError('empty input')
      return _check(np.stack([_read_wav(q) for q in paths]), 'wav input'), [q.name for q in paths]


  def _sha256(path):
      h = hashlib.sha256()
      with open(path, 'rb') as f:
          for chunk in iter(lambda: f.read(1 << 20), b''):
              h.update(chunk)
      return h.hexdigest()


  class SpectralEmbedder:
      """Loads the bundle's TorchScript members; embed() returns (N, M, 12288) float32 in manifest order."""

      def __init__(self, bundle=None, device='auto', batch_size=256):
          self.bundle = resolve_bundle(bundle)
          self.manifest = json.loads((self.bundle / 'manifest.json').read_text())
          self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                     else 'cpu' if device == 'auto' else device)
          self.batch_size = batch_size
          self.encoders = []
          for m in self.manifest['members']:
              f = self.bundle / m['file']
              if _sha256(f) != m['sha256']:
                  raise ValueError(f'{f.name}: checksum does not match manifest.json')
              self.encoders.append(torch.jit.load(str(f), map_location=self.device).eval())
          self.members = [m['name'] for m in self.manifest['members']]

      @torch.no_grad()
      def embed_with_ids(self, x):
          arr, ids = load_spectra(x)
          out = np.empty((len(arr), len(self.encoders), 12288), np.float32)
          for i in range(0, len(arr), self.batch_size):
              batch = torch.from_numpy(arr[i:i + self.batch_size]).to(self.device)
              for j, enc in enumerate(self.encoders):
                  out[i:i + len(batch), j] = enc(batch).cpu().numpy()
          return out, ids

      def embed(self, x):
          return self.embed_with_ids(x)[0]


  class EnsembleRegressor:
      """One StandardScaler + RidgeCV per member on (N, M, D) features; predict() averages the members.
      A convenience baseline; the benchmark chose each probe recipe by inner cross-validation."""

      def __init__(self, alphas=np.logspace(-3, 3, 20)):
          self.alphas = alphas

      def fit(self, features, y):
          from sklearn.linear_model import RidgeCV
          from sklearn.pipeline import make_pipeline
          from sklearn.preprocessing import StandardScaler
          self.models_ = [make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas)).fit(features[:, j], y)
                          for j in range(features.shape[1])]
          return self

      def predict(self, features):
          return np.mean([m.predict(features[:, j]) for j, m in enumerate(self.models_)], 0)
  ```
- [ ] **Step 4:** Run the tests. Expected: all pass. The single-spectrum, float64, CPU `map_location`, `batch_size=2` and `notes.txt` cases cover Review Focus items 1–5.
- [ ] **Step 5:** Commit: `feat: self-contained inference module for the robust-ensemble embedding`.

### Task 3: `embed` CLI

**Files:**
- Modify: `src/spectral_lejepa/inference.py` (add `main(argv=None)`) and `src/spectral_lejepa/__main__.py` (add `'embed': 'spectral_lejepa.inference'`).
- Test: `tests/test_inference.py` (append).

- [ ] **Step 1: Write the failing test** (append):
  ```python
  def test_cli_writes_the_documented_npz(bundle, tmp_path):
      np.save(tmp_path / 'x.npy', spectra(4))
      inf.main([str(tmp_path / 'x.npy'), '--output', str(tmp_path / 'out.npz'), '--bundle', str(bundle[0]),
                '--device', 'cpu'])
      z = np.load(tmp_path / 'out.npz')
      assert sorted(z.files) == ['bundle_manifest', 'embeddings', 'ids', 'members']
      assert z['embeddings'].shape == (4, 2, 12288) and list(z['members']) == ['a', 'b']
      assert json.loads(str(z['bundle_manifest']))['format_version'] == 1


  def test_cli_writes_nothing_on_bad_input(bundle, tmp_path):
      np.save(tmp_path / 'bad.npy', np.zeros((2, 10), np.float32))
      with pytest.raises(ValueError):
          inf.main([str(tmp_path / 'bad.npy'), '--output', str(tmp_path / 'out.npz'), '--bundle', str(bundle[0])])
      assert not (tmp_path / 'out.npz').exists()
  ```
- [ ] **Step 2:** Run them. Expected: FAIL with `AttributeError: main`.
- [ ] **Step 3: Implement** (append to `inference.py`):
  ```python
  def main(argv=None):
      p = argparse.ArgumentParser(description='Embed raw spectra with the robust ensemble -> .npz '
                                              '(embeddings (N, M, 12288), ids, members, bundle_manifest).')
      p.add_argument('input', help='.npy (N, 245), a .wav, or a folder of .wav files')
      p.add_argument('--output', required=True, type=Path)
      p.add_argument('--bundle')
      p.add_argument('--device', default='auto')
      p.add_argument('--batch-size', type=int, default=256)
      a = p.parse_args(argv)
      embedder = SpectralEmbedder(a.bundle, a.device, a.batch_size)
      emb, ids = embedder.embed_with_ids(a.input)
      a.output.parent.mkdir(parents=True, exist_ok=True)
      np.savez(a.output, embeddings=emb, ids=np.array(ids), members=np.array(embedder.members),
               bundle_manifest=json.dumps(embedder.manifest))
      print(f'{a.output}: embeddings {emb.shape} from members {embedder.members}')


  if __name__ == '__main__':
      main()
  ```
  Add `'embed': 'spectral_lejepa.inference',` to `COMMANDS`.
- [ ] **Step 4:** Run `tests/test_inference.py` and `tests/test_cli.py`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: python -m spectral_lejepa embed writes the ensemble embedding`.

### Task 4: Real bundle, slow check, docs

**Files:**
- Test: `tests/test_inference_real.py` (slow).
- Modify: `README.md` (Inference section) and `CLAUDE.md` (Evaluate list); mark the spec status as implemented.

- [ ] **Step 1: Write the slow test:**
  ```python
  import glob

  import numpy as np
  import pytest

  from spectral_lejepa.inference import SpectralEmbedder

  pytestmark = pytest.mark.slow
  EVAL = {'robust_s401': 'outputs/robustness-continuation/robust_s401/eval/*_step49799',
          'robust_p0': 'outputs/robust-ensemble/robust_p0/eval/*_step49799',
          'robust_p1': 'outputs/robust-ensemble/robust_p1/eval/*_step49799',
          'robust_s101': 'outputs/robust-ensemble/robust_s101/eval/*_step49799'}


  def test_real_bundle_reproduces_each_members_evaluation_features():
      emb = SpectralEmbedder(device='cpu')
      for j, name in enumerate(emb.members):
          bank = np.load(glob.glob(EVAL[name])[0] + '/dataset0055/bank.npz', allow_pickle=True)
          got = emb.embed(bank['input_raw'][:, 0].astype(np.float32))[:, j]
          assert np.allclose(got, bank['bank__layer2__flat'][:, 0, 0], atol=1e-4), name
  ```
- [ ] **Step 2:** Run it before the bundle exists. Expected: FAIL with `FileNotFoundError ... export-ensemble`.
- [ ] **Step 3:** Build the bundle.
  Run: `PYTHONPATH=$PWD/src .venv/bin/python -m spectral_lejepa export-ensemble`
  Expected: `Bundle: outputs/inference/robust-ensemble (4 members)`.
- [ ] **Step 4:** Run the slow test again. Expected: PASS. Then smoke-test the CLI on a real wav folder from outside the repo checkout, to prove the documented usage:
  `cd /tmp && SPECTRALFM_BUNDLE=/mnt5/home/hadar/nova/SpectralFM-jepa/outputs/inference/robust-ensemble /mnt5/home/hadar/nova/SpectralFM-jepa/.venv/bin/python -m spectral_lejepa embed /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0055 --output /tmp/claude-1040/emb0055.npz --device cpu`
  Expected: `embeddings (N, 4, 12288)`, where N is the number of wavs in that folder (it may include components 1–2). Delete `/tmp/claude-1040/emb0055.npz` afterwards.
- [ ] **Step 5: Docs.**
  - README: add an "Inference" section after "Robustness tests":

    ````markdown
    ## Inference

    The trained encoders stay outside git. Build the bundle once from the member checkpoints in `configs/inference.yaml`:
    `python -m spectral_lejepa export-ensemble` writes `outputs/inference/robust-ensemble/` (four TorchScript encoders and a checksummed manifest).

    ```python
    from spectral_lejepa.inference import SpectralEmbedder, EnsembleRegressor
    emb = SpectralEmbedder()                    # bundle: argument, $SPECTRALFM_BUNDLE, or the default path
    F = emb.embed('path/to/wavs/')              # (N, 4, 12288): block-2 flat features per member
    model = EnsembleRegressor().fit(F, y)       # one ridge per member; predictions are averaged
    y_hat = model.predict(emb.embed(new_spectra))
    ```

    Command line: `python -m spectral_lejepa embed INPUT --output OUT.npz`. INPUT is a `.npy` of shape (N, 245), one `.wav`, or a folder of `.wav` files. The `.npz` holds `embeddings`, `ids`, `members` and `bundle_manifest`. Keep the member axis: the encoders are independent, so fit one model per member and average, rather than averaging embeddings. `src/spectral_lejepa/inference.py` needs only torch, numpy, soundfile and scikit-learn, so it can be copied with a bundle to another machine.
    ````
  - CLAUDE.md, under Evaluate, add:
    `- Inference: \`export-ensemble\` builds \`outputs/inference/robust-ensemble/\` from \`configs/inference.yaml\`; \`embed INPUT --output OUT.npz\` or \`spectral_lejepa.inference.SpectralEmbedder\` gives (N, 4, 12288).`
  - Spec status line: `Status: implemented 2026-10-14.`
- [ ] **Step 6:** Run the full fast and slow suites; all must pass. Commit (`docs: inference usage in README and CLAUDE.md`), then push.
