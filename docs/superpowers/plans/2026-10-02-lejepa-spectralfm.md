# LeJEPA SpectralFM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small, clean repository that pretrains a 1-D ViT on 245-point SpectralFM spectra with a masked-latent LeJEPA objective (MSE + λ·SIGReg, one shared encoder, no EMA) and scores it with a verified copy of the clean-eval nested-CV label probe against the data2vec baseline `ref_feb25`.

**Architecture:** `src/spectral_lejepa/` holds one file per scientific step: data loading → 24-patch tokenizer → random masking → 1-D ViT encoder/predictor → loss → trainer (with diagnostics, checkpoints, W&B) → evaluation (copied from clean-eval and checked to reproduce the parent's saved numbers). Scripts in `scripts/` are thin entry points. Configuration is plain YAML plus `key.sub=value` overrides.

**Tech Stack:** Python 3.10, uv, PyTorch 2.8 (cu128), lightly 1.5.26 (`SIGReg`, `LeJEPAProjectionHead`), NumPy, scikit-learn, SciPy, soundfile, matplotlib, W&B, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-lejepa-spectralfm-design.md`

## Global Constraints

- Repo: `/mnt5/home/hadar/nova/SpectralFM-jepa`. Parent (read-only reference): `/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged`, branch `label-regression/clean-eval`, commit `6237feb`. Never modify the parent.
- Python `>=3.10,<3.11`. Pins: `torch==2.8.0` and `torchvision==0.23.0` from the cu128 index; `numpy==2.2.6`, `scikit-learn==1.6.1`, `scipy==1.15.3`, `threadpoolctl==3.6.0` (identical to the parent eval env, so the equivalence tests can be exact); `lightly==1.5.26`.
- Do not add torchaudio, pandas, transformers, fairseq, hydra or omegaconf as our own dependencies. lightly pulls hydra-core in transitively; that is accepted.
- `import lightly` must never phone home: `src/spectral_lejepa/__init__.py` sets `LIGHTLY_DID_VERSION_CHECK=True` before anything imports lightly.
- Input: 245 samples per spectrum, 24 patches with widths `[11]*5 + [10]*19`, no overlap, no sample dropped.
- Training normalization is per-sample `F.layer_norm(x, x.shape[-1:])` (eps 1e-5, the fairseq convention). Evaluation normalization is `(x - mean) / (std + 1e-8)` per row (the clean-eval convention). Do not unify them.
- One encoder instance serves both the context and the target path. No EMA, no momentum encoder, no `.detach()`, no second copy.
- SIGReg input is `target.float().transpose(0, 1)`, shape `[24, B, D]`, computed with autocast disabled.
- Baseline hyperparameters: `mask_ratio 0.5`, `batch_size 256`, AdamW `lr 5.0e-4`, `weight_decay 0.05`, `lambda_sigreg 0.05`, `projector none`, 20 epochs, seed 0.
- Evaluation: seed 42, the 12 `RECIPES`, nested 2 repeats × 5 outer folds, 5 inner folds, `MIN_N = 20`, `max_samples 5000`, component 0 only.
- Write every float in YAML with a decimal point (`5.0e-4`, not `5e-4`): PyYAML reads `5e-4` as a string.
- The W&B key comes only from the `WANDB_API_KEY` environment variable or `~/.netrc`. Never write it to a file, a commit, a config or a command line. Tests run with `WANDB_MODE=disabled`.
- Representation diagnostics run on the **whole** 1,000-sample valid split of `single_channel_one`. The spec said 2,048, but that split only has 1,000.
- Run every command from the repo root through `uv run` (after Task 1 puts `uv` on PATH: `export PATH="$HOME/.local/bin:$PATH"`).
- Git identity is already set locally: `Hadar <hal.nls@gmail.com>`. Every commit message ends with these two lines (pass them as a second `-m`):
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt
  ```
- Stop and report to the user at three gates: after Task 3 (W&B smoke), after Task 14 (first real pretraining) and after Task 15 (evaluation).

## Review Focus

1. **A CLI override with a typo or scientific notation.** `training.learning_rat=1e-3` must fail loudly. `training.learning_rate=1e-3` must become the float `0.001`, not the string `"1e-3"`. Tests are in Task 1.
2. **W&B enabled but no credentials.** The run must stop immediately with a message saying to export `WANDB_API_KEY`. It must not hang on an interactive login prompt halfway through setup. Test is in Task 3.
3. **A corrupt or wrong-length spectrum in a manifest.** This means a 200-point wav or NaNs. The run must raise an error that names the file; it must not crash later with a shape error or train on NaNs. Tests are in Task 2.
4. **Fully collapsed embeddings in diagnostics.** All-zero or rank-1 outputs must give finite statistics (condition = inf is allowed) and must not raise. They must also trip the collapse alert. Tests are in Task 8.
5. **Pairing with the baseline on different rows.** If our label-set rows differ from the parent's, the paired comparison must refuse with an error rather than report a meaningless gap. Test is in Task 12.

---

## File map

```
pyproject.toml, uv.lock, .gitignore, README.md
configs/pretrain.yaml            pretraining experiment (all values logged to W&B)
configs/eval.yaml                label sets, baseline paths, eval settings
src/spectral_lejepa/
  __init__.py                    disables lightly's version check
  config.py                      YAML + overrides
  data/loader.py                 manifests, wav reading, per-sample layer_norm, DataLoader
  models/tokenizer.py            245 → 24 patches → tokens
  models/masking.py              random unstructured 1-D masking
  models/vit_1d.py               Encoder, Predictor, LeJEPA, EvalBackbone, build_model
  training/loss.py               MSE + λ·SIGReg
  training/diagnostics.py        collapse statistics, figures
  training/checkpoint.py         save/load checkpoints, load_model
  training/trainer.py            the pretraining loop
  evaluation/data.py             labeled sets (copied semantics from clean-eval)
  evaluation/bank.py             hidden-state extraction → bank.npz (parent format)
  evaluation/probe.py            normalizers, probes, r2 (copied)
  evaluation/nested.py           nested CV, ladder, canary, pairing, comparisons (copied)
  utils/wandb.py                 run init, git metadata, logging helpers
scripts/pretrain.py, evaluate.py, baseline.py, wandb_smoke.py
tests/ conftest.py + one test file per module
```

---

### Task 1: Project skeleton, environment, config loader

**Files:**
- Create: `pyproject.toml`, `src/spectral_lejepa/__init__.py`, `src/spectral_lejepa/{data,models,training,evaluation,utils}/__init__.py`, `src/spectral_lejepa/config.py`, `configs/pretrain.yaml`, `tests/conftest.py`, `tests/test_config.py`
- Modify: `.gitignore` (append)

**Interfaces:**
- Produces: `load_config(path: str, overrides: Sequence[str] = ()) -> dict`, `parse_value(raw: str) -> Any`.

- [ ] **Step 1: Install uv and Python 3.10**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.10
```
Expected: `uv --version` prints a version; Python 3.10 is installed.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "spectral-lejepa"
version = "0.1.0"
description = "LeJEPA pretraining for SpectralFM 245-point spectra, evaluated with the clean-eval label probe"
requires-python = ">=3.10,<3.11"
dependencies = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "numpy==2.2.6",
    "scikit-learn==1.6.1",
    "scipy==1.15.3",
    "threadpoolctl==3.6.0",
    "joblib",
    "soundfile",
    "matplotlib",
    "pyyaml",
    "wandb",
    "lightly==1.5.26",
]

[dependency-groups]
dev = ["pytest"]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/spectral_lejepa"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs the parent repo's outputs or real data; run with -m slow"]
addopts = "-m 'not slow'"
```

- [ ] **Step 3: Create the package files**

`src/spectral_lejepa/__init__.py`:
```python
"""LeJEPA pretraining for SpectralFM 245-point spectra."""
import os

# lightly starts a background PyPI version check on import unless told it already ran.
os.environ.setdefault("LIGHTLY_DID_VERSION_CHECK", "True")
```

Create empty `__init__.py` files in `src/spectral_lejepa/data/`, `models/`, `training/`, `evaluation/` and `utils/`.

`tests/conftest.py`:
```python
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("LIGHTLY_DID_VERSION_CHECK", "True")
```

Append to `.gitignore`:
```
# --- spectral-lejepa ---
outputs/
wandb/
artifacts/
*.pt
*.npz
.env
.env.*
```

- [ ] **Step 4: Sync the environment**

Run: `uv sync`
Expected: it creates `.venv/` and `uv.lock`, and installs torch `2.8.0+cu128` and lightly 1.5.26. Check with:
```bash
uv run python -c "import torch, lightly.loss, sklearn, numpy; print(torch.__version__, torch.cuda.is_available(), sklearn.__version__, numpy.__version__)"
```
Expected: `2.8.0+cu128 True 1.6.1 2.2.6`. A NVML warning from the broken GPU 3 is harmless.

- [ ] **Step 5: Write the failing config tests**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest
import yaml

from spectral_lejepa.config import load_config, parse_value

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"training": {"learning_rate": 0.1, "name": "a", "tags": ["x"]}}))
    return path


def test_override_nested_value(cfg_file):
    assert load_config(cfg_file, ["training.learning_rate=0.5"])["training"]["learning_rate"] == 0.5


def test_scientific_notation_becomes_float(cfg_file):
    value = load_config(cfg_file, ["training.learning_rate=1e-3"])["training"]["learning_rate"]
    assert isinstance(value, float) and value == pytest.approx(1e-3)


def test_strings_lists_and_null(cfg_file):
    cfg = load_config(cfg_file, ["training.name=mlp", "training.tags=[a,b]", "training.learning_rate=null"])
    assert cfg["training"]["name"] == "mlp"
    assert cfg["training"]["tags"] == ["a", "b"]
    assert cfg["training"]["learning_rate"] is None


def test_unknown_key_is_rejected(cfg_file):
    with pytest.raises(KeyError, match="learning_rat"):
        load_config(cfg_file, ["training.learning_rat=1e-3"])


def test_unknown_section_is_rejected(cfg_file):
    with pytest.raises(KeyError, match="trainig"):
        load_config(cfg_file, ["trainig.learning_rate=1.0"])


def test_malformed_override_is_rejected(cfg_file):
    with pytest.raises(ValueError, match="key.sub=value"):
        load_config(cfg_file, ["training.learning_rate"])


def test_parse_value():
    assert parse_value("true") is True
    assert parse_value("256") == 256
    assert parse_value("5e-4") == pytest.approx(5e-4)
    assert parse_value("none") == "none"


def test_pretrain_yaml_numbers_are_numbers():
    cfg = load_config(REPO / "configs" / "pretrain.yaml")
    t = cfg["training"]
    for key in ("learning_rate", "weight_decay", "lambda_sigreg", "final_lr_ratio", "sigreg_t_max"):
        assert isinstance(t[key], float), key
    assert t["lambda_sigreg"] == 0.05
    assert t["batch_size"] == 256
    assert cfg["masking"]["mask_ratio"] == 0.5
    assert cfg["model"]["num_patches"] == 24
    assert cfg["data"]["sequence_length"] == 245
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spectral_lejepa.config'`.

- [ ] **Step 7: Write `src/spectral_lejepa/config.py`**

```python
"""Experiment configuration: one YAML file plus `key.sub=value` command-line overrides.

The resolved dictionary is the single source of truth for a run: it is saved next to
the checkpoints and logged to W&B in full.
"""
from __future__ import annotations

from typing import Any, Sequence

import yaml


def parse_value(raw: str) -> Any:
    """YAML-parse an override value. PyYAML reads `1e-3` as a string, so retry as float."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_config(path, overrides: Sequence[str] = ()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        *sections, leaf = key.split(".")
        node = cfg
        for section in sections:
            if not isinstance(node.get(section), dict):
                raise KeyError(f"unknown config section {section!r} in override {item!r}")
            node = node[section]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r} (typo?)")
        node[leaf] = parse_value(raw)
    return cfg
```

- [ ] **Step 8: Write `configs/pretrain.yaml`**

```yaml
# LeJEPA baseline pretraining. Every value here is logged to W&B.
# Floats need a decimal point (5.0e-4): PyYAML reads 5e-4 as a string.
experiment:
  name: lejepa_baseline
  seed: 0
  output_dir: outputs

data:
  # fairseq manifests (train.tsv / valid.tsv); /storage/noy roots are remapped to /mnt5/noy
  manifest_dir: /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_one
  sequence_length: 245
  max_train_samples: null      # null = all 999,000; set an int for dev runs
  num_workers: 8

model:
  num_patches: 24
  dim: 256
  depth: 6
  heads: 8
  mlp_dim: 1024
  dropout: 0.0
  predictor_dim: 192
  predictor_depth: 4
  predictor_heads: 6
  predictor_mlp_dim: 768
  projector: none              # none | mlp (lightly LeJEPAProjectionHead)
  projector_hidden_dim: 1024
  projector_dim: 128

masking:
  strategy: random
  mask_ratio: 0.5

training:
  batch_size: 256
  epochs: 20
  max_steps: null              # overrides epochs when set
  optimizer: adamw
  learning_rate: 5.0e-4
  weight_decay: 0.05
  warmup_epochs: 1.0
  final_lr_ratio: 0.001        # cosine decays to learning_rate * final_lr_ratio
  lambda_sigreg: 0.05
  sigreg_num_slices: 1024
  sigreg_knots: 17
  sigreg_t_max: 3.0
  amp: true                    # fp16 autocast (RTX 2080 Ti has no bf16)
  grad_clip: null
  log_every: 50
  val_every: 1000
  diag_every: 1000
  ckpt_every: 5000
  device: cuda

wandb:
  enabled: true
  entity: null
  project: spectralfm-lejepa
  group: null
  job_type: pretrain
  tags: [lejepa, spectralfm, 1d, baseline]
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all 8 PASS.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src configs/pretrain.yaml tests
git commit -m "Initial project structure, uv environment and config loader" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 2: Data pipeline

**Files:**
- Create: `src/spectral_lejepa/data/loader.py`, `tests/test_loader.py`

**Interfaces:**
- Produces:
  - `SEQUENCE_LENGTH = 245`
  - `remap_root(root: str) -> str`
  - `read_manifest(path) -> list[str]`
  - `manifest_fingerprint(path) -> dict` with keys `path`, `sha256`, `rows`
  - `normalize_signal(x: Tensor) -> Tensor`
  - `SpectraDataset(paths)`, where `__getitem__` returns `float32 [245]`
  - `subsample(paths, max_samples: int | None, seed: int) -> list[str]`
  - `make_loader(paths, batch_size, shuffle, seed, num_workers, drop_last) -> DataLoader`

- [ ] **Step 1: Write the failing tests**

`tests/test_loader.py`:
```python
import os

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from spectral_lejepa.data.loader import (
    SEQUENCE_LENGTH, SpectraDataset, make_loader, manifest_fingerprint, normalize_signal,
    read_manifest, remap_root, subsample,
)

REAL_VALID = "/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_one/valid.tsv"


def write_set(tmp_path, n=10, length=245, bad=None):
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    rng = np.random.default_rng(0)
    names = []
    for i in range(n):
        x = rng.uniform(0.0, 0.8, length).astype(np.float32)
        if bad == "nan" and i == 3:
            x[10] = np.nan
        if bad == "short" and i == 3:
            x = x[:200]
        name = f"spec_{i}.wav"
        sf.write(wav_dir / name, x, 16000, subtype="FLOAT")
        names.append(name)
    manifest = tmp_path / "train.tsv"
    manifest.write_text(str(wav_dir) + "\n" + "".join(f"{n}\t{length}\n" for n in names))
    return manifest


def test_remap_storage_root():
    assert remap_root("/storage/noy/SpectralFM/x/wav") == "/mnt5/noy/SpectralFM/x/wav"
    assert remap_root("/mnt5/noy/SpectralFM/x/wav") == "/mnt5/noy/SpectralFM/x/wav"


def test_read_manifest(tmp_path):
    paths = read_manifest(write_set(tmp_path, n=4))
    assert len(paths) == 4
    assert paths[0] == os.path.join(str(tmp_path / "wav"), "spec_0.wav")


def test_batch_shape_dtype_and_normalization(tmp_path):
    loader = make_loader(read_manifest(write_set(tmp_path)), batch_size=4, shuffle=False,
                         seed=0, num_workers=0, drop_last=True)
    batch = next(iter(loader))
    assert batch.shape == (4, SEQUENCE_LENGTH)
    assert batch.dtype == torch.float32
    assert torch.isfinite(batch).all()
    assert batch.mean(dim=1).abs().max() < 1e-5
    assert torch.allclose(batch.var(dim=1, unbiased=False), torch.ones(4), atol=1e-3)


def test_normalization_matches_fairseq(tmp_path):
    path = read_manifest(write_set(tmp_path, n=1))[0]
    raw = torch.from_numpy(sf.read(path, dtype="float32")[0])
    assert torch.equal(SpectraDataset([path])[0], F.layer_norm(raw, raw.shape))
    assert torch.equal(normalize_signal(raw[None])[0], F.layer_norm(raw, raw.shape))


def test_wrong_length_names_the_file(tmp_path):
    ds = SpectraDataset(read_manifest(write_set(tmp_path, bad="short")))
    with pytest.raises(ValueError, match="spec_3.wav"):
        ds[3]


def test_non_finite_names_the_file(tmp_path):
    ds = SpectraDataset(read_manifest(write_set(tmp_path, bad="nan")))
    with pytest.raises(ValueError, match="spec_3.wav"):
        ds[3]


def test_subsample_is_deterministic_and_sorted():
    paths = [f"p{i}" for i in range(100)]
    a, b = subsample(paths, 10, seed=1), subsample(paths, 10, seed=1)
    assert a == b and len(a) == 10
    assert [paths.index(p) for p in a] == sorted(paths.index(p) for p in a)
    assert subsample(paths, None, seed=1) == paths
    assert subsample(paths, 1000, seed=1) == paths


def test_fingerprint(tmp_path):
    fp = manifest_fingerprint(write_set(tmp_path, n=3))
    assert fp["rows"] == 3 and len(fp["sha256"]) == 64


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(REAL_VALID), reason="real data not mounted")
def test_real_valid_split():
    paths = read_manifest(REAL_VALID)
    assert len(paths) == 1000 and paths[0].startswith("/mnt5/noy/")
    x = SpectraDataset(paths)[0]
    assert x.shape == (SEQUENCE_LENGTH,) and torch.isfinite(x).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: ... data.loader`.

- [ ] **Step 3: Write `src/spectral_lejepa/data/loader.py`**

```python
"""Unlabeled SpectralFM spectra for pretraining.

Data definition (inherited from the parent project, unchanged):
- one spectrum = one mono float32 WAV of exactly 245 samples (the 16 kHz rate is nominal);
- fairseq manifests: line 1 is the wav root, then `filename<TAB>num_samples`;
- preprocessing = per-sample z-score via F.layer_norm (eps 1e-5), exactly what fairseq's
  FileAudioDataset(normalize=True) applied for data2vec. No global statistics, no augmentation.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEQUENCE_LENGTH = 245
STORAGE_PREFIX = "/storage/noy/"  # how RunAI jobs see the volume
LOCAL_PREFIX = "/mnt5/noy/"       # how this server sees it


def remap_root(root: str) -> str:
    return LOCAL_PREFIX + root[len(STORAGE_PREFIX):] if root.startswith(STORAGE_PREFIX) else root


def read_manifest(path) -> list[str]:
    with open(path) as f:
        root = remap_root(f.readline().strip())
        names = [line.split("\t")[0] for line in f if line.strip()]
    return [os.path.join(root, name) for name in names]


def manifest_fingerprint(path) -> dict:
    """Identifies the exact input list a run used: content hash + row count."""
    data = open(path, "rb").read()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "rows": data.count(b"\n") - 1}


def normalize_signal(x: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score over the last axis (population variance, eps 1e-5 inside the sqrt)."""
    return F.layer_norm(x, x.shape[-1:])


class SpectraDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        x, _ = sf.read(path, dtype="float32")
        if x.ndim != 1 or x.shape[0] != SEQUENCE_LENGTH:
            raise ValueError(f"{path}: expected shape ({SEQUENCE_LENGTH},), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: contains NaN or inf")
        return normalize_signal(torch.from_numpy(x))


def subsample(paths, max_samples, seed):
    """A seeded subset (kept in manifest order) for dev runs; all paths when max_samples is None."""
    if max_samples is None or max_samples >= len(paths):
        return list(paths)
    idx = np.sort(np.random.default_rng(seed).choice(len(paths), max_samples, replace=False))
    return [paths[i] for i in idx]


def make_loader(paths, batch_size, shuffle, seed, num_workers, drop_last) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)  # makes the shuffle order reproducible
    return DataLoader(SpectraDataset(paths), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers, generator=generator,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_loader.py -v && uv run pytest tests/test_loader.py -m slow -v`
Expected: 8 PASS, then the slow real-data test PASSES.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_lejepa/data/loader.py tests/test_loader.py
git commit -m "Add data pipeline for unlabeled 245-point spectra" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 3: W&B utilities and smoke test (GATE)

**Files:**
- Create: `src/spectral_lejepa/utils/wandb.py`, `scripts/wandb_smoke.py`, `tests/test_wandb_utils.py`

**Interfaces:**
- Produces:
  - `git_info(repo_dir=None) -> dict` with keys `git_commit`, `git_branch`, `git_dirty`
  - `software_versions() -> dict`
  - `wandb_enabled(cfg) -> bool`
  - `init_run(cfg, job_type, extra_config=None, name=None) -> wandb Run | None`
  - Helpers that are no-ops when `run is None`: `log(run, metrics, step=None)`, `log_figure(run, key, fig, step=None)`, `log_checkpoint(run, path, artifact_name, aliases, metadata)`, `alert(run, title, text)`, `finish(run)`

- [ ] **Step 1: Write the failing tests**

`tests/test_wandb_utils.py`:
```python
import pytest

from spectral_lejepa.utils import wandb as wb

CFG = {"wandb": {"enabled": True, "entity": None, "project": "p", "group": None,
                 "job_type": "pretrain", "tags": ["t"]}}


def test_git_info_in_this_repo():
    info = wb.git_info()
    assert len(info["git_commit"]) == 40
    assert isinstance(info["git_dirty"], bool)
    assert info["git_branch"]


def test_git_info_outside_a_repo(tmp_path):
    assert wb.git_info(tmp_path)["git_commit"] == "unknown"


def test_disabled_by_env_returns_none():
    # conftest sets WANDB_MODE=disabled
    assert wb.init_run(CFG, job_type="pretrain") is None
    wb.log(None, {"a": 1.0}, step=0)
    wb.finish(None)


def test_disabled_by_config(monkeypatch):
    monkeypatch.delenv("WANDB_MODE")
    assert wb.init_run({"wandb": {**CFG["wandb"], "enabled": False}}, job_type="pretrain") is None


def test_missing_credentials_fail_fast(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_MODE")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        wb.init_run(CFG, job_type="pretrain")


def test_software_versions():
    v = wb.software_versions()
    assert v["torch"].startswith("2.8.0") and v["lightly"] == "1.5.26"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_wandb_utils.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/spectral_lejepa/utils/wandb.py`**

```python
"""W&B integration: every run records its resolved config, code version and environment.

The API key is never read from or written to the repository. W&B finds it in the
WANDB_API_KEY environment variable or ~/.netrc. WANDB_MODE=disabled (set by the tests)
or `wandb.enabled: false` turns every helper here into a no-op.
"""
from __future__ import annotations

import os
import platform
import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def git_info(repo_dir=None) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo_dir or PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "HEAD"),
                "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": True}


def software_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "numpy", "scikit-learn", "scipy", "lightly", "wandb"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def wandb_enabled(cfg) -> bool:
    return bool(cfg["wandb"]["enabled"]) and os.environ.get("WANDB_MODE", "").lower() != "disabled"


def _check_credentials():
    """Fail before any work starts instead of hanging on an interactive login prompt."""
    if os.environ.get("WANDB_MODE", "").lower() == "offline" or os.environ.get("WANDB_API_KEY"):
        return
    netrc = Path.home() / ".netrc"
    if netrc.is_file() and "api.wandb.ai" in netrc.read_text():
        return
    raise RuntimeError("W&B is enabled but no credentials were found: export WANDB_API_KEY in "
                       "your shell (never put it in the repo), or set wandb.enabled=false.")


def init_run(cfg, job_type, extra_config=None, name=None):
    if not wandb_enabled(cfg):
        return None
    _check_credentials()
    import wandb

    w = cfg["wandb"]
    config = {**cfg, **git_info(), "software": software_versions(), "hostname": socket.gethostname(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              **(extra_config or {})}
    return wandb.init(project=w["project"], entity=w["entity"], group=w["group"], job_type=job_type,
                      tags=list(w["tags"]), name=name, config=config)


def log(run, metrics, step=None):
    if run is not None:
        run.log(metrics, step=step)


def log_figure(run, key, fig, step=None):
    if run is not None:
        import wandb
        run.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_checkpoint(run, path, artifact_name, aliases, metadata):
    if run is None:
        return
    import wandb
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=list(aliases))


def alert(run, title, text):
    if run is not None:
        run.alert(title=title, text=text)


def finish(run):
    if run is not None:
        run.finish()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_wandb_utils.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Write `scripts/wandb_smoke.py`**

```python
"""W&B smoke test: authenticate from the environment, log git info, a config and a fake loss curve.

  WANDB_API_KEY must already be exported in your shell.
  uv run python scripts/wandb_smoke.py [--entity ENTITY]
"""
import argparse
import math
import sys

from spectral_lejepa.utils import wandb as wb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entity", default=None)
    ap.add_argument("--project", default="spectralfm-lejepa")
    a = ap.parse_args()
    cfg = {"wandb": {"enabled": True, "entity": a.entity, "project": a.project, "group": None,
                     "job_type": "diagnostic", "tags": ["smoke"]},
           "smoke": {"steps": 20}}
    run = wb.init_run(cfg, job_type="diagnostic", name="wandb-smoke")
    if run is None:
        sys.exit("W&B is disabled (WANDB_MODE=disabled?) -- nothing to test")
    for step in range(cfg["smoke"]["steps"]):
        wb.log(run, {"train/loss": math.exp(-step / 5) + 0.05}, step=step)
    print("run url:   ", run.url)
    print("git_commit:", run.config["git_commit"], "dirty:", run.config["git_dirty"])
    wb.finish(run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/spectral_lejepa/utils/wandb.py scripts/wandb_smoke.py tests/test_wandb_utils.py
git commit -m "Add W&B integration and smoke test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

- [ ] **Step 7: Run the smoke test (GATE: needs the user)**

Ask the user to export the key in their own shell first. For example, they type `! export WANDB_API_KEY=...` themselves; never echo or store it. Then run:
```bash
uv run python scripts/wandb_smoke.py
```
Expected: it prints a run URL and a 40-character `git_commit`, and the W&B page shows a decreasing `train/loss` curve. **If this fails, stop and report the error. Do not continue to Task 4.**

---

### Task 4: Tokenizer

**Files:**
- Create: `src/spectral_lejepa/models/tokenizer.py`, `tests/test_tokenizer.py`

**Interfaces:**
- Produces:
  - `patch_bounds(sequence_length=245, num_patches=24) -> list[tuple[int, int]]`, half-open `[lo, hi)`
  - `PatchTokenizer(sequence_length, num_patches, dim)`, with:
    - `.bounds`
    - `.patch_width` (= 11)
    - `.patchify(x [B,245]) -> [B,24,11]`
    - `forward(x) -> [B,24,dim]` (positional embedding already added)
    - `.pos_embed` Parameter `[24, dim]`

- [ ] **Step 1: Write the failing tests**

`tests/test_tokenizer.py`:
```python
import pytest
import torch

from spectral_lejepa.models.tokenizer import PatchTokenizer, patch_bounds


def test_patch_widths():
    assert [hi - lo for lo, hi in patch_bounds()] == [11] * 5 + [10] * 19


def test_every_sample_in_exactly_one_patch():
    covered = [i for lo, hi in patch_bounds() for i in range(lo, hi)]
    assert covered == list(range(245))


def test_patchify_values_and_padding():
    tok = PatchTokenizer(dim=8)
    x = torch.arange(245, dtype=torch.float32)[None]
    p = tok.patchify(x)
    assert p.shape == (1, 24, 11)
    assert torch.equal(p[0, 0], torch.arange(0, 11).float())
    assert torch.equal(p[0, 5, :10], torch.arange(55, 65).float()) and p[0, 5, 10] == 0
    assert torch.equal(p[0, -1, :10], torch.arange(235, 245).float()) and p[0, -1, 10] == 0


def test_output_shape():
    assert PatchTokenizer(dim=32)(torch.randn(3, 245)).shape == (3, 24, 32)


def test_wrong_length_raises():
    with pytest.raises(ValueError, match="245"):
        PatchTokenizer(dim=8)(torch.randn(2, 240))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/spectral_lejepa/models/tokenizer.py`**

```python
"""245-point signal -> 24 patch tokens.

245 = 24 x 10 + 5, so equal patches are impossible without dropping or inventing samples.
We split the signal into 24 contiguous, non-overlapping patches with np.array_split:
5 patches of 11 samples (indices 0..54), then 19 patches of 10 (55..244). Every sample
lands in exactly one patch, so masking a patch hides exactly its samples and no
neighbouring token can leak them. Each 10-sample patch is right-padded with one zero to
width 11 so one shared Linear(11 -> dim) embeds all patches.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_bounds(sequence_length: int = 245, num_patches: int = 24) -> list[tuple[int, int]]:
    parts = np.array_split(np.arange(sequence_length), num_patches)
    return [(int(p[0]), int(p[-1]) + 1) for p in parts]


class PatchTokenizer(nn.Module):
    def __init__(self, sequence_length: int = 245, num_patches: int = 24, dim: int = 256):
        super().__init__()
        self.sequence_length = sequence_length
        self.bounds = patch_bounds(sequence_length, num_patches)
        self.patch_width = max(hi - lo for lo, hi in self.bounds)
        # index[p, j] = sample index of position j in patch p; positions past a patch's end
        # point at `sequence_length`, i.e. at the zero appended in patchify().
        index = torch.full((num_patches, self.patch_width), sequence_length, dtype=torch.long)
        for p, (lo, hi) in enumerate(self.bounds):
            index[p, : hi - lo] = torch.arange(lo, hi)
        self.register_buffer("index", index, persistent=False)
        self.proj = nn.Linear(self.patch_width, dim)
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 245] -> [B, 24, 11]."""
        if x.shape[-1] != self.sequence_length:
            raise ValueError(f"expected signals of length {self.sequence_length}, got {x.shape[-1]}")
        return F.pad(x, (0, 1))[:, self.index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 245] -> [B, 24, dim] tokens, positional embedding included."""
        return self.proj(self.patchify(x)) + self.pos_embed
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_lejepa/models/tokenizer.py tests/test_tokenizer.py
git commit -m "Add 1-D patch tokenizer (245 -> 24 non-overlapping patches)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 5: Random masking

**Files:**
- Create: `src/spectral_lejepa/models/masking.py`, `tests/test_masking.py`

**Interfaces:**
- Produces:
  - `num_masked(num_patches: int, mask_ratio: float) -> int`
  - `random_mask(batch_size, num_patches, mask_ratio, generator=None, device=None) -> (masked_idx [B, n_mask] long, visible_idx [B, P - n_mask] long)`. Both are sorted ascending. The generator is a CPU `torch.Generator`.

- [ ] **Step 1: Write the failing tests**

`tests/test_masking.py`:
```python
import pytest
import torch

from spectral_lejepa.models.masking import num_masked, random_mask


def gen(seed):
    return torch.Generator().manual_seed(seed)


def test_shapes_and_counts():
    m, v = random_mask(8, 24, 0.5, gen(0))
    assert m.shape == (8, 12) and v.shape == (8, 12)
    assert m.dtype == torch.long and v.dtype == torch.long
    assert num_masked(24, 0.45) == 11 and num_masked(24, 0.4) == 10


def test_disjoint_complete_sorted():
    m, v = random_mask(16, 24, 0.5, gen(1))
    for mi, vi in zip(m.tolist(), v.tolist()):
        assert sorted(mi + vi) == list(range(24))
        assert not set(mi) & set(vi)
        assert mi == sorted(mi) and vi == sorted(vi)


def test_masks_differ_between_samples():
    m, _ = random_mask(16, 24, 0.5, gen(2))
    assert len({tuple(row) for row in m.tolist()}) > 1


def test_reproducible_with_seed():
    a = random_mask(4, 24, 0.5, gen(3))
    b = random_mask(4, 24, 0.5, gen(3))
    c = random_mask(4, 24, 0.5, gen(4))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.99, 1.0])
def test_degenerate_ratios_rejected(ratio):
    with pytest.raises(ValueError, match="mask_ratio"):
        random_mask(2, 24, ratio, gen(0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_masking.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/spectral_lejepa/models/masking.py`**

```python
"""Random unstructured masking over the 24 tokens.

Each sample independently gets round(mask_ratio * 24) masked positions, drawn uniformly
without replacement (argsort of uniform noise). Indices are returned sorted, so token
order is preserved. The count is the same for every sample, so no padding masks are
needed. Pass a seeded CPU generator for reproducibility on any device.
"""
from __future__ import annotations

import torch


def num_masked(num_patches: int, mask_ratio: float) -> int:
    n = round(mask_ratio * num_patches)
    if not 1 <= n <= num_patches - 1:
        raise ValueError(f"mask_ratio={mask_ratio} masks {n} of {num_patches} tokens; "
                         "need at least one masked and one visible token")
    return n


def random_mask(batch_size: int, num_patches: int, mask_ratio: float,
                generator: torch.Generator | None = None, device=None):
    """Returns (masked_idx [B, n_mask], visible_idx [B, num_patches - n_mask]), both sorted."""
    n_mask = num_masked(num_patches, mask_ratio)
    order = torch.rand(batch_size, num_patches, generator=generator).argsort(dim=1)
    masked_idx = order[:, :n_mask].sort(dim=1).values
    visible_idx = order[:, n_mask:].sort(dim=1).values
    return masked_idx.to(device), visible_idx.to(device)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_masking.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_lejepa/models/masking.py tests/test_masking.py
git commit -m "Add random unstructured 1-D token masking" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 6: 1-D ViT encoder, predictor, LeJEPA model, eval adapter

**Files:**
- Create: `src/spectral_lejepa/models/vit_1d.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `PatchTokenizer` (Task 4).
- Produces:
  - `gather_tokens(x [B,N,D], idx [B,K]) -> [B,K,D]`
  - `Encoder(dim, depth, heads, mlp_dim, dropout)`, where `forward(tokens, return_all=False)` returns `out` or `(out, [block outputs])`
  - `Predictor(num_patches, in_dim, dim, depth, heads, mlp_dim, dropout)`, where `forward(context, visible_idx, masked_idx) -> [B, n_mask, in_dim]`
  - `LeJEPA(...)` with attributes `.tokenizer`, `.encoder`, `.predictor`, `.projector` (or `None`); `forward(x, masked_idx, visible_idx) -> {"predicted", "target", "target_masked"}`
  - `EvalBackbone(model)`, where `forward(input_values, output_hidden_states=True) -> SimpleNamespace(hidden_states=tuple, last_hidden_state)`
  - `build_model(model_cfg: dict, sequence_length: int) -> LeJEPA`

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import pytest
import torch

from spectral_lejepa.models.masking import random_mask
from spectral_lejepa.models.vit_1d import EvalBackbone, Encoder, build_model, gather_tokens

SMALL = dict(num_patches=24, dim=32, depth=2, heads=4, mlp_dim=64, dropout=0.0,
             predictor_dim=24, predictor_depth=2, predictor_heads=4, predictor_mlp_dim=48,
             projector="none", projector_hidden_dim=64, projector_dim=16)


def setup(projector="none", B=4, seed=0):
    torch.manual_seed(seed)
    model = build_model({**SMALL, "projector": projector}, 245).eval()
    x = torch.randn(B, 245)
    m, v = random_mask(B, 24, 0.5, torch.Generator().manual_seed(seed))
    return model, x, m, v


def test_forward_shapes():
    model, x, m, v = setup()
    out = model(x, m, v)
    assert out["predicted"].shape == (4, 12, 32)
    assert out["target"].shape == (4, 24, 32)
    assert out["target_masked"].shape == (4, 12, 32)


def test_target_masked_is_target_at_masked_positions():
    model, x, m, v = setup()
    out = model(x, m, v)
    for b in range(4):
        assert torch.equal(out["target_masked"][b], out["target"][b, m[b]])
    assert torch.equal(gather_tokens(out["target"], m), out["target_masked"])


def test_one_shared_encoder_no_ema():
    model, x, m, v = setup()
    assert sum(isinstance(mod, Encoder) for mod in model.modules()) == 1
    assert not any("ema" in name or "momentum" in name or "teacher" in name
                   for name, _ in model.named_parameters())
    assert len(list(model.buffers())) == 1  # only the tokenizer's (non-persistent) gather index
    calls = []
    model.encoder.register_forward_hook(lambda mod, inp, out: calls.append(tuple(inp[0].shape)))
    model(x, m, v)
    assert calls == [(4, 12, 32), (4, 24, 32)]  # context (visible tokens) then target (all tokens)


def test_gradient_flows_through_target_path():
    model, x, m, v = setup()
    model.train()
    out = model(x, m, v)
    assert out["target"].requires_grad
    out["target"].square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.predictor.parameters())


def test_gradient_flows_through_context_path():
    model, x, m, v = setup()
    model.train()
    model(x, m, v)["predicted"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.predictor.parameters())


def test_prediction_never_sees_masked_samples():
    model, x, m, v = setup()
    x2 = x.clone()
    for b in range(len(x)):
        for p in m[b].tolist():
            lo, hi = model.tokenizer.bounds[p]
            x2[b, lo:hi] = torch.randn(hi - lo) * 10
    with torch.no_grad():
        assert torch.allclose(model(x, m, v)["predicted"], model(x2, m, v)["predicted"], atol=1e-6)


def test_projector_mlp():
    model, x, m, v = setup(projector="mlp")
    model.train()
    out = model(x, m, v)
    assert out["predicted"].shape == (4, 12, 16) and out["target"].shape == (4, 24, 16)


def test_unknown_projector_rejected():
    with pytest.raises(ValueError, match="projector"):
        build_model({**SMALL, "projector": "linear"}, 245)


def test_eval_backbone_hidden_states():
    model, x, _, _ = setup()
    backbone = EvalBackbone(model)
    out = backbone(input_values=x, output_hidden_states=True)
    hs = out.hidden_states
    assert len(hs) == SMALL["depth"] + 1
    assert all(h.shape == (4, 24, 32) for h in hs)
    with torch.no_grad():
        assert torch.allclose(hs[-1], model.encoder(model.tokenizer(x)))
    assert not hasattr(backbone, "feature_extractor") and not hasattr(backbone, "feature_projection")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/spectral_lejepa/models/vit_1d.py`**

```python
"""1-D ViT pieces for LeJEPA on 24-token spectra.

    tokens = tokenizer(signal)                         [B, 24, D]
    context = encoder(tokens[visible])                 [B, n_vis, D]   only visible tokens
    predicted = predictor(context, masked positions)   [B, n_mask, D]
    target = encoder(tokens)                           [B, 24, D]      SAME encoder, all tokens
    target_masked = target[masked]                     [B, n_mask, D]

There is no EMA/teacher network and no stop-gradient: gradients reach the encoder through
both the context and the target path. SIGReg (training/loss.py) prevents the collapse that
stop-gradients exist to prevent elsewhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from .tokenizer import PatchTokenizer


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, N, D], idx [B, K] -> [B, K, D]: per-sample token selection."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def transformer_blocks(dim, depth, heads, mlp_dim, dropout) -> nn.ModuleList:
    return nn.ModuleList([
        nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout, activation="gelu",
                                   batch_first=True, norm_first=True)
        for _ in range(depth)])


class Encoder(nn.Module):
    """Pre-LN transformer over any subset of tokens (positions already embedded)."""

    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.blocks = transformer_blocks(dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, return_all=False):
        hidden = []
        x = tokens
        for block in self.blocks:
            x = block(x)
            hidden.append(x)
        out = self.norm(x)
        return (out, hidden) if return_all else out


class Predictor(nn.Module):
    """Predicts encoder embeddings at masked positions from the context embeddings.

    Context tokens and one learned mask token per masked position (plus that position's
    embedding) are processed jointly; the outputs at the mask-token slots are the predictions.
    """

    def __init__(self, num_patches, in_dim, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = transformer_blocks(dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_dim)

    def forward(self, context, visible_idx, masked_idx):
        B, n_mask = masked_idx.shape
        ctx = self.embed(context) + self.pos_embed[visible_idx]                      # [B, n_vis, dim]
        queries = self.mask_token.expand(B, n_mask, -1) + self.pos_embed[masked_idx]  # [B, n_mask, dim]
        x = torch.cat([ctx, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, -n_mask:]))                                   # [B, n_mask, in_dim]


class LeJEPA(nn.Module):
    def __init__(self, sequence_length, num_patches, dim, depth, heads, mlp_dim, dropout,
                 predictor_dim, predictor_depth, predictor_heads, predictor_mlp_dim,
                 projector="none", projector_hidden_dim=1024, projector_dim=128):
        super().__init__()
        self.tokenizer = PatchTokenizer(sequence_length, num_patches, dim)
        self.encoder = Encoder(dim, depth, heads, mlp_dim, dropout)
        self.predictor = Predictor(num_patches, dim, predictor_dim, predictor_depth,
                                   predictor_heads, predictor_mlp_dim, dropout)
        if projector == "none":
            self.projector = None
        elif projector == "mlp":
            # imported here so the baseline (projector: none) does not need it
            from lightly.models.modules import LeJEPAProjectionHead
            self.projector = LeJEPAProjectionHead(dim, projector_hidden_dim, projector_dim)
        else:
            raise ValueError(f"unknown projector {projector!r}; use 'none' or 'mlp'")

    def forward(self, x, masked_idx, visible_idx):
        tokens = self.tokenizer(x)                                            # [B, 24, D]
        context = self.encoder(gather_tokens(tokens, visible_idx))            # [B, n_vis, D]
        predicted = self.predictor(context, visible_idx, masked_idx)          # [B, n_mask, D]
        target = self.encoder(tokens)                                         # [B, 24, D]
        if self.projector is not None:
            predicted, target = self._project(predicted, target)
        return {"predicted": predicted, "target": target,
                "target_masked": gather_tokens(target, masked_idx)}

    def _project(self, predicted, target):
        """Project predictions and targets in ONE call so BatchNorm sees one set of batch statistics."""
        B, n_mask, D = predicted.shape
        out = self.projector(torch.cat([predicted.reshape(-1, D), target.reshape(-1, D)]))
        return out[: B * n_mask].reshape(B, n_mask, -1), out[B * n_mask:].reshape(B, target.shape[1], -1)


class EvalBackbone(nn.Module):
    """The interface the clean-eval extraction expects:
    model(input_values=[B, 245], output_hidden_states=True).hidden_states -> tuple of [B, T, D].

    hidden_states = (tokens, block 1, ..., block depth-1, final normed output); the last entry
    is exactly the target representation used in training. The projector is never used here.
    Deliberately has no `feature_extractor` / `feature_projection` attributes: the parent eval
    treats those names as data2vec conv taps.
    """

    def __init__(self, model: LeJEPA):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.encoder = model.encoder

    def forward(self, input_values, output_hidden_states=True):
        tokens = self.tokenizer(input_values)
        out, hidden = self.encoder(tokens, return_all=True)
        return SimpleNamespace(hidden_states=(tokens, *hidden[:-1], out), last_hidden_state=out)


def build_model(model_cfg: dict, sequence_length: int) -> LeJEPA:
    keys = ("num_patches", "dim", "depth", "heads", "mlp_dim", "dropout", "predictor_dim",
            "predictor_depth", "predictor_heads", "predictor_mlp_dim", "projector",
            "projector_hidden_dim", "projector_dim")
    return LeJEPA(sequence_length=sequence_length, **{k: model_cfg[k] for k in keys})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_lejepa/models/vit_1d.py tests/test_model.py
git commit -m "Add 1-D ViT encoder, predictor and shared-encoder LeJEPA model" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 7: LeJEPA loss

**Files:**
- Create: `src/spectral_lejepa/training/loss.py`, `tests/test_loss.py`

**Interfaces:**
- Produces: `LeJEPAObjective(lambda_sigreg, num_slices=1024, knots=17, t_max=3.0)`. Its `forward(predicted, target_masked, target)` returns `{"loss", "mse_loss", "sigreg_loss"}`, all fp32 scalar tensors.

- [ ] **Step 1: Write the failing tests**

`tests/test_loss.py`:
```python
import torch

from spectral_lejepa.models.vit_1d import gather_tokens
from spectral_lejepa.training.loss import LeJEPAObjective


def test_components_and_total():
    torch.manual_seed(0)
    obj = LeJEPAObjective(lambda_sigreg=0.05, num_slices=64)
    p, tm, t = torch.randn(8, 12, 16), torch.randn(8, 12, 16), torch.randn(8, 24, 16)
    out = obj(p, tm, t)
    assert set(out) == {"loss", "mse_loss", "sigreg_loss"}
    assert torch.allclose(out["loss"], out["mse_loss"] + 0.05 * out["sigreg_loss"])


def test_mse_is_exact():
    out = LeJEPAObjective(0.0, num_slices=8)(torch.zeros(2, 3, 4), torch.ones(2, 3, 4), torch.randn(2, 24, 4))
    assert out["mse_loss"].item() == 1.0
    assert out["loss"].item() == 1.0


def test_sigreg_small_for_gaussian_large_for_collapse():
    torch.manual_seed(0)
    obj = LeJEPAObjective(1.0, num_slices=256)
    zeros = torch.zeros(256, 12, 32)
    gaussian = obj(zeros, zeros, torch.randn(256, 24, 32))["sigreg_loss"].item()
    collapsed = obj(zeros, zeros, 0.01 * torch.randn(256, 24, 32))["sigreg_loss"].item()
    assert gaussian < 3.0
    assert collapsed > 20.0


def test_gradients_reach_target_through_both_terms():
    torch.manual_seed(0)
    target = torch.randn(16, 24, 8, requires_grad=True)
    masked = torch.arange(12).repeat(16, 1)
    out = LeJEPAObjective(0.05, num_slices=32)(torch.zeros(16, 12, 8), gather_tokens(target, masked), target)
    out["loss"].backward()
    assert target.grad[:, :12].abs().sum() > 0   # MSE + SIGReg
    assert target.grad[:, 12:].abs().sum() > 0   # SIGReg only (unmasked positions)


def test_sigreg_runs_in_fp32_under_autocast():
    obj = LeJEPAObjective(0.05, num_slices=16)
    t = torch.randn(32, 24, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = obj(t[:, :12], t[:, :12], t)
    assert out["sigreg_loss"].dtype == torch.float32 and out["loss"].dtype == torch.float32
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_loss.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write `src/spectral_lejepa/training/loss.py`**

```python
"""LeJEPA objective for masked latent prediction:

    loss = MSE(predicted, target_masked) + lambda_sigreg * SIGReg(target)

MSE compares predictions only with the target embeddings at the SAME masked positions.
SIGReg (lightly's implementation of the LeJEPA regularizer) tests whether embeddings look
like an isotropic Gaussian. It is applied per token position: target [B, 24, D] is
transposed to [24, B, D], so each position is tested across the B independent samples of
the batch (N = B), and the 24 statistics are averaged. SIGReg sums cos/sin over the batch
and scales by N, so it runs in fp32 with autocast off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import SIGReg


class LeJEPAObjective(nn.Module):
    def __init__(self, lambda_sigreg: float, num_slices: int = 1024, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.sigreg = SIGReg(knots=knots, t_max=t_max, num_vectors=num_slices)

    def forward(self, predicted, target_masked, target) -> dict:
        with torch.autocast(device_type=target.device.type, enabled=False):
            mse = F.mse_loss(predicted.float(), target_masked.float())
            sigreg = self.sigreg(target.float().transpose(0, 1))   # [24, B, D]
        return {"loss": mse + self.lambda_sigreg * sigreg, "mse_loss": mse, "sigreg_loss": sigreg}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_loss.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_lejepa/training/loss.py tests/test_loss.py
git commit -m "Add LeJEPA loss: MSE on masked tokens + lambda * per-position SIGReg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 8: Representation and masking diagnostics

**Files:**
- Create: `src/spectral_lejepa/training/diagnostics.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `LeJEPA.tokenizer` and `LeJEPA.encoder` (Task 6).
- Produces:
  - `encode(model, signals [N,245], device, batch_size=256) -> Tensor [N, 24, D]` (float32, CPU)
  - `representation_stats(Z [N, D], max_cosine_rows=1024) -> (dict[str, float], singular_values Tensor)`. The dict keys are `mean_norm`, `std`, `effective_rank`, `covariance_trace`, `covariance_condition`, `mean_pairwise_cosine`.
  - `collapse_alert(stats, reference_rank, std_floor=0.1, rank_fraction=0.25) -> bool`
  - `singular_value_figure(sv) -> Figure`
  - `masking_figure(signals np [n,245], bounds, masked_idx np [n,K]) -> Figure`

- [ ] **Step 1: Write the failing tests**

`tests/test_diagnostics.py`:
```python
import math

import numpy as np
import torch
from matplotlib.figure import Figure

from spectral_lejepa.models.tokenizer import patch_bounds
from spectral_lejepa.models.vit_1d import build_model
from spectral_lejepa.training import diagnostics as diag
from tests.test_model import SMALL


def test_isotropic_gaussian():
    torch.manual_seed(0)
    s, sv = diag.representation_stats(torch.randn(2000, 16))
    assert s["effective_rank"] > 14
    assert s["covariance_condition"] < 2
    assert abs(s["mean_pairwise_cosine"]) < 0.05
    assert abs(s["std"] - 1) < 0.05
    assert abs(s["covariance_trace"] - 16) < 1.5
    assert sv.shape == (16,)


def test_low_rank_is_detected():
    torch.manual_seed(0)
    s, _ = diag.representation_stats(torch.randn(2000, 2) @ torch.randn(2, 16))
    assert s["effective_rank"] < 3


def test_nearly_low_rank_has_large_condition():
    # exactly-dead directions are excluded from the condition number; tiny-but-alive ones are not
    torch.manual_seed(0)
    Z = torch.randn(2000, 2) @ torch.randn(2, 16) + 1e-4 * torch.randn(2000, 16)
    s, _ = diag.representation_stats(Z)
    assert s["covariance_condition"] > 1e6


def test_fully_collapsed_is_finite_and_alerts():
    s, _ = diag.representation_stats(torch.zeros(500, 16))
    assert s["effective_rank"] == 0.0 and s["covariance_condition"] == math.inf
    assert all(math.isfinite(v) for k, v in s.items() if k != "covariance_condition")
    assert diag.collapse_alert(s, reference_rank=10.0)


def test_collapse_alert_thresholds():
    healthy = {"std": 1.0, "effective_rank": 10.0}
    assert not diag.collapse_alert(healthy, reference_rank=12.0)
    assert diag.collapse_alert({"std": 0.05, "effective_rank": 10.0}, reference_rank=12.0)
    assert diag.collapse_alert({"std": 1.0, "effective_rank": 2.0}, reference_rank=12.0)


def test_encode_shape():
    model = build_model(SMALL, 245).eval()
    out = diag.encode(model, torch.randn(10, 245), device="cpu", batch_size=4)
    assert out.shape == (10, 24, 32) and out.dtype == torch.float32


def test_figures():
    sv = torch.linspace(5, 0.1, 16)
    assert isinstance(diag.singular_value_figure(sv), Figure)
    signals = np.random.default_rng(0).normal(size=(4, 245))
    masked = np.stack([np.sort(np.random.default_rng(i).choice(24, 12, replace=False)) for i in range(4)])
    fig = diag.masking_figure(signals, patch_bounds(), masked)
    assert isinstance(fig, Figure) and len(fig.axes) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: FAIL (module missing). Also create an empty `tests/__init__.py` so that `from tests.test_model import SMALL` resolves.

- [ ] **Step 3: Write `src/spectral_lejepa/training/diagnostics.py`**

```python
"""Diagnostics used to decide lambda_sigreg from measurements rather than assumptions.

representation_stats() summarizes a set of embeddings [N, D]:
  mean_norm, std (mean per-dimension std), effective_rank (exp of the entropy of the
  normalized singular values; D for isotropic data, ~1 for collapse), covariance_trace,
  covariance_condition (largest / smallest non-negligible covariance eigenvalue; inf when
  everything collapsed) and mean_pairwise_cosine (1.0 when all embeddings point the same way).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def encode(model, signals, device, batch_size=256) -> torch.Tensor:
    """Unmasked encoder output for every signal: [N, 245] -> [N, num_patches, D], float32 on CPU."""
    outs = []
    for i in range(0, len(signals), batch_size):
        x = signals[i:i + batch_size].to(device)
        outs.append(model.encoder(model.tokenizer(x)).float().cpu())
    return torch.cat(outs)


@torch.no_grad()
def representation_stats(Z: torch.Tensor, max_cosine_rows: int = 1024):
    Z = Z.double()
    centered = Z - Z.mean(dim=0)
    sv = torch.linalg.svdvals(centered)                     # descending
    eig = sv.square() / max(len(Z) - 1, 1)                   # covariance eigenvalues
    total = sv.sum()
    if total > 0:
        p = sv[sv > 0] / total
        effective_rank = float(torch.exp(-(p * p.log()).sum()))
    else:
        effective_rank = 0.0
    alive = eig[eig > 1e-12 * eig[0]] if eig[0] > 0 else eig[:0]
    condition = float(alive[0] / alive[-1]) if len(alive) else float("inf")
    rows = F.normalize(Z[:max_cosine_rows], dim=1)
    cos = rows @ rows.T
    n = len(rows)
    mean_cos = float((cos.sum() - cos.diagonal().sum()) / max(n * (n - 1), 1))
    stats = {"mean_norm": float(Z.norm(dim=1).mean()), "std": float(Z.std(dim=0).mean()),
             "effective_rank": effective_rank, "covariance_trace": float(eig.sum()),
             "covariance_condition": condition, "mean_pairwise_cosine": mean_cos}
    return stats, sv.float()


def collapse_alert(stats, reference_rank, std_floor=0.1, rank_fraction=0.25) -> bool:
    """True when embeddings shrink (std) or lose rank relative to the step-0 reference."""
    return stats["std"] < std_floor or stats["effective_rank"] < rank_fraction * reference_rank


def singular_value_figure(sv):
    sv = np.asarray(sv, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(np.arange(1, len(sv) + 1), sv / max(sv[0], 1e-30))
    ax.set_xlabel("component")
    ax.set_ylabel("singular value / largest")
    ax.set_title("mean-pooled valid embeddings")
    fig.tight_layout()
    return fig


def masking_figure(signals, bounds, masked_idx):
    """Signals with patch boundaries; masked patches orange, visible patches blue."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 1.8 * len(signals)), sharex=True, squeeze=False)
    for ax, x, masked in zip(axes[:, 0], signals, masked_idx):
        masked = {int(i) for i in masked}
        for p, (lo, hi) in enumerate(bounds):
            is_masked = p in masked
            ax.axvspan(lo - 0.5, hi - 0.5, color="tab:orange" if is_masked else "tab:blue",
                       alpha=0.3 if is_masked else 0.07, lw=0)
            ax.axvline(lo - 0.5, color="0.75", lw=0.5)
        ax.plot(np.arange(len(x)), x, color="k", lw=0.8)
        ax.set_ylabel("z-score")
    axes[0, 0].set_title(f"orange = masked ({len(masked_idx[0])} of {len(bounds)} patches), blue = visible")
    axes[-1, 0].set_xlabel(f"sample index ({len(signals[0])} points, {len(bounds)} patches)")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spectral_lejepa/training/diagnostics.py tests/__init__.py tests/test_diagnostics.py
git commit -m "Add representation-collapse and masking diagnostics" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 9: Checkpoints, training loop, pretrain script, smoke runs

**Files:**
- Create: `src/spectral_lejepa/training/checkpoint.py`, `src/spectral_lejepa/training/trainer.py`, `scripts/pretrain.py`, `tests/test_train_smoke.py`

**Interfaces:**
- Consumes:
  - From Task 2: `read_manifest`, `subsample`, `SpectraDataset`, `make_loader`, `manifest_fingerprint`.
  - From Task 5: `random_mask`.
  - From Task 6: `build_model`.
  - From Task 7: `LeJEPAObjective`.
  - From Task 8: the `diagnostics` functions.
  - From Task 3: the W&B helpers in `utils.wandb`.
- Produces:
  - `save_checkpoint(path, *, model, optimizer, scaler, step, epoch, config, metadata, metrics)`
  - `load_checkpoint(path, map_location="cpu") -> dict`, with keys `model`, `optimizer`, `scaler`, `step`, `epoch`, `config`, `metrics`, `git_commit`, `git_branch`, `git_dirty`, `wandb_run_id`
  - `load_model(path, map_location="cpu") -> (LeJEPA in eval mode, ckpt dict)`
  - `train(cfg: dict) -> dict` with keys `output_dir`, `final_checkpoint`, `history` (a list of logged train metric dicts) and `wandb_run_id`
  - Checkpoint files `checkpoint_step{N}.pt` and `checkpoint_last.pt` under `<output_dir>/<name>_<YYYYmmdd-HHMMSS>/`
  - W&B artifact `lejepa-<run_id>` with aliases `latest` and `step-<N>`

- [ ] **Step 1: Write the failing tests**

`tests/test_train_smoke.py`:
```python
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from spectral_lejepa.config import load_config
from spectral_lejepa.training.checkpoint import load_checkpoint, load_model
from spectral_lejepa.training.trainer import lr_factor, schedule_lengths, train

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_dir(tmp_path):
    """Sinusoids with random frequency/phase: learnable structure for a tiny overfit run."""
    rng = np.random.default_rng(0)
    t = np.arange(245) / 245
    wav = tmp_path / "data" / "wav"
    wav.mkdir(parents=True)
    names = []
    for i in range(80):
        x = np.sin(2 * np.pi * rng.uniform(1, 5) * t + rng.uniform(0, 2 * np.pi)) + 0.05 * rng.normal(size=245)
        sf.write(wav / f"spec_{i}.wav", x.astype(np.float32), 16000, subtype="FLOAT")
        names.append(f"spec_{i}.wav")
    for split, part in (("train", names[:64]), ("valid", names[64:])):
        (tmp_path / "data" / f"{split}.tsv").write_text(f"{wav}\n" + "".join(f"{n}\t245\n" for n in part))
    return tmp_path / "data"


def tiny_cfg(data_dir, out_dir, name="smoke"):
    return load_config(REPO / "configs" / "pretrain.yaml", [
        f"experiment.name={name}", f"experiment.output_dir={out_dir}",
        f"data.manifest_dir={data_dir}", "data.num_workers=0",
        "model.dim=32", "model.depth=2", "model.heads=4", "model.mlp_dim=64",
        "model.predictor_dim=24", "model.predictor_depth=2", "model.predictor_heads=4",
        "model.predictor_mlp_dim=48",
        "training.batch_size=8", "training.max_steps=60", "training.learning_rate=1.0e-3",
        "training.log_every=5", "training.val_every=20", "training.diag_every=20",
        "training.ckpt_every=30", "training.device=cpu", "training.amp=false",
        "training.sigreg_num_slices=64", "wandb.enabled=false",
    ])


def test_schedule():
    assert schedule_lengths(100, {"max_steps": None, "epochs": 20, "warmup_epochs": 1.0}) == (2000, 100)
    assert schedule_lengths(8, {"max_steps": 60, "epochs": 20, "warmup_epochs": 1.0}) == (60, 6)
    assert lr_factor(0, 10, 100, 0.001) == pytest.approx(0.1)
    assert lr_factor(10, 10, 100, 0.001) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100, 0.001) == pytest.approx(0.001)


def test_tiny_training_run(data_dir, tmp_path):
    with pytest.warns(UserWarning, match="batch_size=8"):
        result = train(tiny_cfg(data_dir, tmp_path / "out"))
    out = Path(result["output_dir"])
    assert (out / "config.yaml").exists()
    for name in ("checkpoint_step30.pt", "checkpoint_step60.pt", "checkpoint_last.pt"):
        assert (out / name).exists(), name

    history = result["history"]
    assert len(history) == 12
    assert all(math.isfinite(h["train/loss"]) for h in history)
    first = np.mean([h["train/loss"] for h in history[:3]])
    last = np.mean([h["train/loss"] for h in history[-3:]])
    assert last < first
    assert {"train/mse_loss", "train/sigreg_loss", "train/learning_rate", "train/epoch",
            "train/global_step", "train/samples_seen"} <= set(history[0])

    ckpt = load_checkpoint(out / "checkpoint_last.pt")
    assert ckpt["step"] == 60
    assert ckpt["config"]["training"]["lambda_sigreg"] == 0.05
    assert len(ckpt["git_commit"]) == 40 and ckpt["wandb_run_id"] is None
    assert "valid/loss" in ckpt["metrics"] and "representation/effective_rank" in ckpt["metrics"]
    model, _ = load_model(out / "checkpoint_last.pt")
    assert model.encoder(model.tokenizer(torch.randn(2, 245))).shape == (2, 24, 32)


def test_runs_are_reproducible(data_dir, tmp_path):
    with pytest.warns(UserWarning):
        a = train(tiny_cfg(data_dir, tmp_path / "a", name="rep_a"))
        b = train(tiny_cfg(data_dir, tmp_path / "b", name="rep_b"))
    assert [h["train/loss"] for h in a["history"]] == [h["train/loss"] for h in b["history"]]


def test_cuda_requested_without_gpu_fails(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = tiny_cfg(data_dir, tmp_path / "out")
    cfg["training"]["device"] = "cuda"
    with pytest.raises(RuntimeError, match="no GPU"):
        train(cfg)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train_smoke.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Write `src/spectral_lejepa/training/checkpoint.py`**

```python
"""Checkpoints carry everything needed to identify and rebuild a model: weights, optimizer
state, step/epoch, the resolved config, git commit and the W&B run id."""
from __future__ import annotations

import os

import torch

from ..models.vit_1d import build_model


def save_checkpoint(path, *, model, optimizer, scaler, step, epoch, config, metadata, metrics):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "config": config,
               "metrics": metrics, **metadata}
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # never leave a truncated checkpoint behind


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model(path, map_location="cpu"):
    ckpt = load_checkpoint(path, map_location)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["data"]["sequence_length"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt
```

- [ ] **Step 4: Write `src/spectral_lejepa/training/trainer.py`**

```python
"""Pretraining loop: spectra -> random mask -> LeJEPA forward -> MSE + lambda*SIGReg -> AdamW.

Linear warmup then cosine decay (the LeJEPA recipe), fp16 autocast on GPU, periodic
validation losses, representation/masking diagnostics, checkpoints and W&B logging.
"""
from __future__ import annotations

import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR

from ..data.loader import SpectraDataset, make_loader, manifest_fingerprint, read_manifest, subsample
from ..models.masking import random_mask
from ..models.vit_1d import build_model
from ..utils import wandb as wb
from . import diagnostics as diag
from .checkpoint import save_checkpoint
from .loss import LeJEPAObjective

RECOMMENDED_MIN_BATCH = 256   # SIGReg's Epps-Pulley statistic is estimated from the batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("training.device is cuda but no GPU is visible; "
                           "set training.device=cpu explicitly for a CPU run")
    return torch.device(name)


def param_groups(model, weight_decay):
    """No weight decay on biases, norms, positional embeddings and the mask token."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or name.endswith("pos_embed") or name.endswith("mask_token"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def schedule_lengths(steps_per_epoch: int, tcfg: dict) -> tuple[int, int]:
    """(total_steps, warmup_steps). Warmup is warmup_epochs, capped at 10% of a short dev run."""
    total = tcfg["max_steps"] or tcfg["epochs"] * steps_per_epoch
    warmup = min(int(tcfg["warmup_epochs"] * steps_per_epoch), total // 10)
    return total, max(1, warmup)


def lr_factor(step, warmup_steps, total_steps, final_ratio):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validation_losses(model, objective, signals, num_patches, mask_ratio, batch_size, seed, device, amp):
    """Mean losses over full batches of the fixed valid set; the mask seed is fixed so values
    are comparable across steps."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(batch_size, len(signals))
    totals = {"loss": 0.0, "mse_loss": 0.0, "sigreg_loss": 0.0}
    n_batches = 0
    for i in range(0, len(signals) - batch_size + 1, batch_size):
        x = signals[i:i + batch_size].to(device)
        masked_idx, visible_idx = random_mask(len(x), num_patches, mask_ratio, generator, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x, masked_idx, visible_idx)
        losses = objective(out["predicted"], out["target_masked"], out["target"])
        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1
    model.train()
    return {f"valid/{k}": v / n_batches for k, v in totals.items()}


def run_diagnostics(run, model, valid_signals, mask_ratio, step, seed, device, reference_rank):
    """Representation statistics on the whole valid set (+ figures when W&B is on).
    Returns (metrics, reference_rank); the reference is the step-0 effective rank."""
    model.eval()
    reps = diag.encode(model, valid_signals, device)                       # [N, 24, D]
    model.train()
    pooled, sv = diag.representation_stats(reps.mean(dim=1))               # what the eval reads
    token, _ = diag.representation_stats(reps.reshape(-1, reps.shape[-1]))
    metrics = {**{f"representation/{k}": v for k, v in pooled.items()},
               **{f"representation/token_{k}": v for k, v in token.items()}}
    if reference_rank is None:
        reference_rank = pooled["effective_rank"]
    alert = diag.collapse_alert(pooled, reference_rank)
    metrics["representation/collapse_alert"] = int(alert)
    if alert:
        text = (f"step {step}: std={pooled['std']:.3g}, effective_rank={pooled['effective_rank']:.3g} "
                f"(step-0 rank {reference_rank:.3g})")
        print(f"[diagnostics] WARNING possible collapse -- {text}", flush=True)
        wb.alert(run, "Possible representation collapse", text)
    wb.log(run, metrics, step=step)
    if run is not None:
        wb.log_figure(run, "representation/singular_values", diag.singular_value_figure(sv), step)
        masked_idx, _ = random_mask(4, len(model.tokenizer.bounds), mask_ratio,
                                    torch.Generator().manual_seed(seed + 3 + step))
        wb.log_figure(run, "masking/examples", diag.masking_figure(
            valid_signals[:4].numpy(), model.tokenizer.bounds, masked_idx.numpy()), step)
    return metrics, reference_rank


def train(cfg: dict) -> dict:
    ecfg, dcfg, mcfg, kcfg, tcfg = (cfg[k] for k in ("experiment", "data", "model", "masking", "training"))
    if kcfg["strategy"] != "random":
        raise ValueError(f"masking.strategy={kcfg['strategy']!r}: only 'random' is implemented")
    if tcfg["optimizer"] != "adamw":
        raise ValueError(f"training.optimizer={tcfg['optimizer']!r}: only 'adamw' is implemented")
    seed = ecfg["seed"]
    set_seed(seed)
    device = resolve_device(tcfg["device"])
    amp = bool(tcfg["amp"]) and device.type == "cuda"

    # --- data ---
    train_manifest = os.path.join(dcfg["manifest_dir"], "train.tsv")
    valid_manifest = os.path.join(dcfg["manifest_dir"], "valid.tsv")
    train_paths = subsample(read_manifest(train_manifest), dcfg["max_train_samples"], seed)
    valid_ds = SpectraDataset(read_manifest(valid_manifest))
    valid_signals = torch.stack([valid_ds[i] for i in range(len(valid_ds))])   # small, fixed
    batch_size = tcfg["batch_size"]
    loader = make_loader(train_paths, batch_size, shuffle=True, seed=seed,
                         num_workers=dcfg["num_workers"], drop_last=True)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError(f"{len(train_paths)} training samples is fewer than one batch ({batch_size})")
    total_steps, warmup_steps = schedule_lengths(steps_per_epoch, tcfg)
    below = batch_size < RECOMMENDED_MIN_BATCH
    if below:
        warnings.warn(f"batch_size={batch_size} < {RECOMMENDED_MIN_BATCH}: SIGReg's batch statistic is "
                      "estimated from fewer samples (logged as derived.batch_size_below_256=true)")

    # --- run bookkeeping ---
    run_name = f"{ecfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(ecfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    derived = {"derived": {
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps,
        "train_samples": len(train_paths), "valid_samples": len(valid_ds),
        "batch_size_below_256": below, "output_dir": str(out_dir),
        "train_manifest": manifest_fingerprint(train_manifest),
        "valid_manifest": manifest_fingerprint(valid_manifest)}}
    resolved = {**cfg, **derived}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    run = wb.init_run(cfg, job_type=cfg["wandb"]["job_type"], extra_config=derived, name=run_name)
    metadata = {**wb.git_info(), "wandb_run_id": run.id if run is not None else None}

    # --- model and optimization ---
    model = build_model(mcfg, dcfg["sequence_length"]).to(device)
    objective = LeJEPAObjective(tcfg["lambda_sigreg"], tcfg["sigreg_num_slices"],
                                tcfg["sigreg_knots"], tcfg["sigreg_t_max"]).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, tcfg["weight_decay"]), lr=tcfg["learning_rate"])
    scheduler = LambdaLR(optimizer, lambda s: lr_factor(s, warmup_steps, total_steps, tcfg["final_lr_ratio"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    mask_generator = torch.Generator().manual_seed(seed + 1)
    num_patches, mask_ratio = mcfg["num_patches"], kcfg["mask_ratio"]
    max_norm = tcfg["grad_clip"] or float("inf")

    latest, reference_rank = {}, None
    metrics, reference_rank = run_diagnostics(run, model, valid_signals, mask_ratio, 0, seed, device, reference_rank)
    latest.update(metrics)

    def checkpoint(step, epoch):
        meta = {"config": resolved, "epoch": epoch, "global_step": step, "metrics": latest, **metadata}
        path = out_dir / f"checkpoint_step{step}.pt"
        for target in (path, out_dir / "checkpoint_last.pt"):
            save_checkpoint(target, model=model, optimizer=optimizer, scaler=scaler, step=step,
                            epoch=epoch, config=resolved, metadata=metadata, metrics=dict(latest))
        if run is not None:
            wb.log_checkpoint(run, path, f"lejepa-{run.id}", ["latest", f"step-{step}"], meta)

    step, samples_seen, history = 0, 0, []
    model.train()
    while step < total_steps:
        for x in loader:
            x = x.to(device, non_blocking=True)
            masked_idx, visible_idx = random_mask(len(x), num_patches, mask_ratio, mask_generator, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(x, masked_idx, visible_idx)
            losses = objective(out["predicted"], out["target_masked"], out["target"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            samples_seen += len(x)
            epoch = step / steps_per_epoch
            last = step == total_steps

            if step % tcfg["log_every"] == 0 or last:
                record = {f"train/{k}": v.item() for k, v in losses.items()}
                if not math.isfinite(record["train/loss"]):
                    raise FloatingPointError(f"non-finite loss at step {step}: {record}")
                record.update({"train/learning_rate": scheduler.get_last_lr()[0], "train/epoch": epoch,
                               "train/global_step": step, "train/samples_seen": samples_seen,
                               "train/grad_norm": float(grad_norm)})
                history.append(record)
                wb.log(run, record, step=step)
            if step % tcfg["val_every"] == 0 or last:
                valid = validation_losses(model, objective, valid_signals, num_patches, mask_ratio,
                                          batch_size, seed + 2, device, amp)
                latest.update(valid)
                wb.log(run, valid, step=step)
            if step % tcfg["diag_every"] == 0 or last:
                metrics, reference_rank = run_diagnostics(run, model, valid_signals, mask_ratio, step,
                                                          seed, device, reference_rank)
                latest.update(metrics)
            if step % tcfg["ckpt_every"] == 0 or last:
                checkpoint(step, epoch)
            if last:
                break

    wb.finish(run)
    return {"output_dir": str(out_dir), "final_checkpoint": str(out_dir / "checkpoint_last.pt"),
            "history": history, "wandb_run_id": metadata["wandb_run_id"]}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train_smoke.py -v`
Expected: 4 PASS (the run takes under a minute on CPU). If `last < first` fails, print the history before changing anything. That assertion should hold for structured sinusoids; investigate with superpowers:systematic-debugging rather than weakening it.

- [ ] **Step 6: Write `scripts/pretrain.py`**

```python
"""Pretrain LeJEPA on SpectralFM spectra.

  uv run python scripts/pretrain.py [--config configs/pretrain.yaml] [key.sub=value ...]
  e.g. uv run python scripts/pretrain.py training.max_steps=300 data.max_train_samples=20000
"""
import argparse

from spectral_lejepa.config import load_config
from spectral_lejepa.training.trainer import train


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    a = ap.parse_args()
    result = train(load_config(a.config, a.overrides))
    print("output dir:", result["output_dir"])
    print("final checkpoint:", result["final_checkpoint"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the whole test suite**

Run: `uv run pytest -v`
Expected: every non-slow test PASSES.

- [ ] **Step 8: Commit**

```bash
git add src/spectral_lejepa/training/checkpoint.py src/spectral_lejepa/training/trainer.py scripts/pretrain.py tests/test_train_smoke.py
git commit -m "Add training loop, checkpoints and pretrain script" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

- [ ] **Step 9: Short real-data GPU dev run with W&B**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/pretrain.py experiment.name=dev_smoke \
  data.max_train_samples=20000 training.max_steps=300 training.val_every=100 \
  training.diag_every=100 training.ckpt_every=300 "wandb.tags=[lejepa,spectralfm,1d,dev]"
```
Expected:
- It finishes without errors.
- W&B shows all of: the `train/*` curves, the `valid/*` curves, the `representation/*` metrics, the `masking/examples` image, the `representation/singular_values` image, and the artifact `lejepa-<run_id>` with alias `step-300`.

Record from the run: steps per second (from the W&B step rate), peak GPU memory (`nvidia-smi -i 0 --query-gpu=memory.used --format=csv`, run while training), and the `train/mse_loss` and `train/sigreg_loss` magnitudes. You'll need these for the Task 14 time estimate. No commit, since nothing in the repo changes.

---

### Task 10: Evaluation data and representation bank

**Files:**
- Create: `src/spectral_lejepa/evaluation/data.py`, `src/spectral_lejepa/evaluation/bank.py`, `tests/test_eval_data.py`

**Interfaces:**
- Produces:
  - `load_labeled_data(set_dirs: list[str], max_samples=5000, seed=42, target_length=245) -> (raw float32 [N,245], y float64 [N])`. This is component 0 only, ordered by (dataset, spec), and is the same as the parent's `load_labeled_data(comps=(0,))` on a merged set.
  - `normalize_like_fairseq(arr [N,L]) -> float32 [N,L]`
  - `extract_mean_bank(model, signals_z, device="cuda", batch_size=64) -> {"layer0": [N,D], ...}`
  - `save_bank(path, bank, input_raw [N,245], y, meta: dict)`
  - `load_bank(path) -> (bank {stage: float32 [N,D]} in file order, input_raw float32 [N,245], y float64 [N], meta dict)`. This reads both our banks and the parent's 10-statistic banks.

- [ ] **Step 1: Write the failing tests**

`tests/test_eval_data.py`:
```python
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from spectral_lejepa.evaluation.bank import extract_mean_bank, load_bank, save_bank
from spectral_lejepa.evaluation.data import load_labeled_data, normalize_like_fairseq
from spectral_lejepa.models.vit_1d import EvalBackbone, build_model
from tests.test_model import SMALL

NOVA = "/mnt5/noy/SpectralFM/fairseq/data/nova_data"
PARENT_OUT = "/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged/code/eval_outputs"
SMALL_SETS = ["dataset0055", "dataset0106", "dataset0109", "dataset0112", "dataset0113",
              "dataset0114", "dataset0115", "dataset0117", "dataset0118", "dataset0120"]


def make_set(root, rows, wav_subdir, lengths=None):
    root.mkdir(parents=True)
    wav_dir = root / wav_subdir if wav_subdir else root
    wav_dir.mkdir(exist_ok=True)
    lines = []
    for name, label in rows:
        n = (lengths or {}).get(name, 245)
        sf.write(wav_dir / name, np.full(n, float(len(lines) + 1), np.float32), 16000, subtype="FLOAT")
        lines.append(f"{name}\t{label}")
    (root / "labels.tsv").write_text("\n".join(lines) + "\n")


def test_order_component_filter_and_merge(tmp_path):
    make_set(tmp_path / "a", [("dataset0007_comp0_spec_2.wav", 0.5), ("dataset0007_comp1_spec_2.wav", 0.5),
                              ("dataset0007_comp0_spec_1.wav", -1.0), ("dataset0007_comp1_spec_9.wav", 2.0)],
             "wavs", lengths={"dataset0007_comp0_spec_1.wav": 200})
    make_set(tmp_path / "b", [("dataset0003_comp0_spec_4.wav", 3.0)], None)
    raw, y = load_labeled_data([str(tmp_path / "a"), str(tmp_path / "b")])
    assert raw.shape == (3, 245) and raw.dtype == np.float32 and y.dtype == np.float64
    assert y.tolist() == [3.0, -1.0, 0.5]           # sorted by (dataset, spec); comp1-only spec dropped
    assert (raw[1, 200:] == 0).all() and (raw[1, :200] != 0).all()   # short wav zero-padded


def test_duplicate_spectrum_across_sets_rejected(tmp_path):
    make_set(tmp_path / "a", [("dataset0007_comp0_spec_2.wav", 0.5)], None)
    make_set(tmp_path / "b", [("dataset0007_comp0_spec_2.wav", 0.5)], None)
    with pytest.raises(ValueError, match="dataset 7, spec 2"):
        load_labeled_data([str(tmp_path / "a"), str(tmp_path / "b")])


def test_max_samples_subsample_is_seeded(tmp_path):
    make_set(tmp_path / "a", [(f"dataset0001_comp0_spec_{i}.wav", float(i)) for i in range(30)], None)
    a = load_labeled_data([str(tmp_path / "a")], max_samples=10, seed=42)[1]
    b = load_labeled_data([str(tmp_path / "a")], max_samples=10, seed=42)[1]
    assert len(a) == 10 and np.array_equal(a, b)


def test_normalize_like_fairseq():
    x = np.random.default_rng(0).normal(3, 2, size=(5, 245))
    z = normalize_like_fairseq(x)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z, (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8), rtol=1e-6)


def test_bank_roundtrip_and_extraction(tmp_path):
    torch.manual_seed(0)
    backbone = EvalBackbone(build_model(SMALL, 245))
    z = normalize_like_fairseq(np.random.default_rng(0).normal(size=(10, 245)))
    bank = extract_mean_bank(backbone, z, device="cpu", batch_size=4)
    assert list(bank) == ["layer0", "layer1", "layer2"]
    assert all(v.shape == (10, 32) and v.dtype == np.float32 for v in bank.values())
    with torch.no_grad():
        hs = backbone(input_values=torch.from_numpy(z)).hidden_states
    np.testing.assert_allclose(bank["layer2"], hs[2].mean(1).numpy(), rtol=1e-5, atol=1e-6)

    raw, y = np.random.default_rng(1).normal(size=(10, 245)).astype(np.float32), np.arange(10.0)
    save_bank(tmp_path / "bank.npz", bank, raw, y, {"checkpoint": "x", "backbone": "EvalBackbone"})
    bank2, raw2, y2, meta = load_bank(tmp_path / "bank.npz")
    assert list(bank2) == list(bank) and all(np.array_equal(bank[k], bank2[k]) for k in bank)
    assert np.array_equal(raw2, raw) and np.array_equal(y2, y) and meta["backbone"] == "EvalBackbone"


def test_load_parent_format_bank(tmp_path):
    """Parent banks are [N, K comps, 10 stats, D]; we take component 0, statistic 'mean'."""
    stats = ("mean", "std", "max", "min", "first", "last", "seg0", "seg1", "seg2", "seg3")
    arr = np.random.default_rng(0).normal(size=(6, 1, 10, 4)).astype(np.float32)
    np.savez(tmp_path / "bank.npz", bank__fe=arr, bank__layer0=arr + 1, input_raw=np.ones((6, 1, 245), np.float32),
             input_z=np.ones((6, 1, 245), np.float32), y=np.arange(6.0),
             _meta=np.array([repr({"comps": (0,), "pool_stats": stats})]))
    bank, raw, y, _ = load_bank(tmp_path / "bank.npz")
    assert list(bank) == ["fe", "layer0"]
    assert np.array_equal(bank["fe"], arr[:, 0, 0, :]) and raw.shape == (6, 245)


def _parent_bank(rel):
    path = os.path.join(PARENT_OUT, rel, "bank.npz")
    if not os.path.exists(path):
        pytest.skip("parent outputs not available")
    data = np.load(path)
    return data["input_raw"][:, 0, :], data["y"]


@pytest.mark.slow
@pytest.mark.parametrize("set_dirs,rel", [
    ([f"{NOVA}/labeled_regression/dataset0120"], "label_probe_regression_ref_feb25/label_probe/dataset0120"),
    ([f"{NOVA}/labeled_regression/{s}" for s in SMALL_SETS],
     "label_probe_regression_merged_ref_feb25/label_probe/labeled_regression_all"),
    ([f"{NOVA}/labeled_data"], "label_probe_regression_labeled_data_ref_feb25/label_probe/labeled_data"),
])
def test_same_rows_as_parent(set_dirs, rel):
    parent_raw, parent_y = _parent_bank(rel)
    raw, y = load_labeled_data(set_dirs)
    assert np.array_equal(y, parent_y)
    assert np.array_equal(raw, parent_raw)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_data.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Write `src/spectral_lejepa/evaluation/data.py`**

```python
"""Labeled spectra for the downstream parameter_0 probe.

Copied semantics of clean-eval `data_loader.load_labeled_data(comps=(0,))` (+ the merged
set built by `merge_label_sets.py`), commit 6237feb:
- each set dir has labels.tsv (`filename<TAB>parameter_0`) and wavs under wav/, wavs/ or itself;
- filenames are `dataset<D>_comp<C>_spec_<S>.wav`; component 0 is the probe input;
- spectra are ordered by (dataset, spec) across ALL given dirs (merging = union of sets);
- more than max_samples spectra -> seeded subsample (default_rng(seed).choice, sorted);
- each wav is zero-padded / truncated to 245 points; spectra whose wav is missing are skipped.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import soundfile as sf

LABELED_PATTERN = re.compile(r"dataset(\d+)_comp(\d+)_spec_(\d+)\.wav")
COMPONENT = 0


def _wav_root(set_dir):
    for sub in ("wav", "wavs"):
        cand = os.path.join(set_dir, sub)
        if glob.glob(os.path.join(cand, "*.wav")):
            return cand
    return set_dir


def normalize_like_fairseq(arr: np.ndarray) -> np.ndarray:
    """The eval's per-row z-score (eps 1e-8 on the std), exactly as clean-eval feeds backbones."""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32)


def load_labeled_data(set_dirs, max_samples=5000, seed=42, target_length=245):
    files, labels, owner = {}, {}, {}   # (dataset, spec) -> wav path / label / set dir
    for set_dir in set_dirs:
        wav_root = _wav_root(set_dir)
        with open(os.path.join(set_dir, "labels.tsv")) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2 or (m := LABELED_PATTERN.match(parts[0])) is None:
                    continue
                ds, comp, spec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if comp != COMPONENT:
                    continue
                key = (ds, spec)
                if key in owner and owner[key] != set_dir:
                    raise ValueError(f"dataset {ds}, spec {spec} appears in both {owner[key]} and {set_dir}")
                owner[key] = set_dir
                path = os.path.join(wav_root, parts[0])
                files[key] = path if os.path.isfile(path) else os.path.join(set_dir, parts[0])
                labels[key] = float(parts[1])

    keys = sorted(files)
    if not keys:
        raise RuntimeError(f"no component-{COMPONENT} spectra under {set_dirs}")
    if len(keys) > max_samples:
        idx = np.random.default_rng(seed).choice(len(keys), max_samples, replace=False)
        idx.sort()
        keys = [keys[i] for i in idx]

    rows, ys, missing = [], [], 0
    for key in keys:
        if not os.path.isfile(files[key]):
            missing += 1
            continue
        data = np.asarray(sf.read(files[key], dtype="float32")[0]).flatten()
        row = np.zeros(target_length, dtype=np.float32)
        row[: min(len(data), target_length)] = data[:target_length]
        rows.append(row)
        ys.append(labels[key])
    if not rows:
        raise RuntimeError(f"no labeled wavs found under {set_dirs}")
    print(f"[eval data] {len(rows)} spectra from {len(set_dirs)} set(s); {missing} labels without a wav skipped")
    return np.stack(rows), np.array(ys, dtype=np.float64)
```

- [ ] **Step 4: Write `src/spectral_lejepa/evaluation/bank.py`**

```python
"""Representation bank: one mean-pooled vector per encoder stage per spectrum.

Extraction uses the clean-eval interface: model(input_values=x, output_hidden_states=True)
.hidden_states -> tuple of [B, T, D]; stage i is named f"layer{i}" and pooled by the mean
over T (every headline clean-eval number uses mean pooling).

bank.npz keeps the parent's layout so either side can read the other's files:
bank__<stage> [N, K=1 component, S stats, D], input_raw / input_z [N, 1, 245], y [N],
_meta = repr(dict). Ours store S=1 statistic ("mean"); the parent's store 10.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import torch

from .data import normalize_like_fairseq


@torch.no_grad()
def extract_mean_bank(model, signals_z, device="cuda", batch_size=64) -> dict:
    model.eval()
    model.to(device)
    t = torch.from_numpy(np.asarray(signals_z, dtype=np.float32))
    out = None
    for i in range(0, len(t), batch_size):
        hidden = model(input_values=t[i:i + batch_size].to(device), output_hidden_states=True).hidden_states
        if out is None:
            out = {f"layer{li}": [] for li in range(len(hidden))}
        if len(hidden) != len(out):
            raise RuntimeError(f"hidden_states length changed mid-run: {len(hidden)} vs {len(out)}")
        for li, h in enumerate(hidden):
            out[f"layer{li}"].append(h.mean(dim=1).float().cpu().numpy())
    return {stage: np.concatenate(v).astype(np.float32) for stage, v in out.items()}


def save_bank(path, bank, input_raw, y, meta: dict):
    n = len(y)
    payload = {f"bank__{s}": arr.reshape(n, 1, 1, arr.shape[-1]) for s, arr in bank.items()}
    payload["input_raw"] = np.asarray(input_raw, dtype=np.float32).reshape(n, 1, -1)
    payload["input_z"] = normalize_like_fairseq(payload["input_raw"][:, 0]).reshape(n, 1, -1)
    payload["y"] = np.asarray(y, dtype=np.float64)
    payload["_meta"] = np.array([repr({"comps": (0,), "pool_stats": ("mean",), **meta})])
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, path)


def load_bank(path):
    data = np.load(path, allow_pickle=False)
    meta = ast.literal_eval(str(data["_meta"][0]))
    if tuple(meta.get("comps", (0,)))[0] != 0:
        raise ValueError(f"{path}: first stored component is {meta['comps'][0]}, expected 0")
    mean_i = tuple(meta["pool_stats"]).index("mean")
    bank = {k[len("bank__"):]: np.ascontiguousarray(data[k][:, 0, mean_i, :], dtype=np.float32)
            for k in data.files if k.startswith("bank__")}   # file order = parent's stage order
    input_raw = np.ascontiguousarray(data["input_raw"][:, 0, :], dtype=np.float32)
    return bank, input_raw, np.asarray(data["y"], dtype=np.float64), meta
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_data.py -v && uv run pytest tests/test_eval_data.py -m slow -v`
Expected: 6 PASS, then 3 slow PASS (rows identical to the parent's banks, including n=695 for the merged set). If a slow case fails, stop and report the mismatch rather than adjusting the test.

- [ ] **Step 6: Commit**

```bash
git add src/spectral_lejepa/evaluation/data.py src/spectral_lejepa/evaluation/bank.py tests/test_eval_data.py
git commit -m "Add labeled-data loading and representation bank (clean-eval format)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 11: Nested-CV probe, ladder, canary (verified copy)

**Files:**
- Create: `src/spectral_lejepa/evaluation/probe.py`, `src/spectral_lejepa/evaluation/nested.py`, `tests/test_nested.py`, `tests/test_eval_equivalence.py`

**Interfaces:**
- Consumes: `load_bank` (Task 10).
- Produces:
  - In `probe.py`:
    - `RECIPES` (12 tuples)
    - `r2(y, p) -> float`
    - `fold_transforms(X_tr, X_te, seed=42) -> {norm: (A, B)}`
    - `fit_predict(X_tr, y_tr, X_te, probe, seed) -> np.ndarray`
  - In `nested.py`:
    - `MIN_N = 20`, `ENSEMBLE_K = 3`
    - `nested_cv(...)`
    - `run_nested(bank, input_raw, y, n_repeats=2, n_folds=5, n_inner=5, seed=42, n_jobs=1) -> (results dict, oof dict)`
    - `bootstrap_sd`, `paired_delta(y, PA, PB, n_boot=1000, seed=0) -> {"delta", "sd", "p_a_better"}`
    - `nested_ladder(arms, y, n_trains=LADDER_N, n_folds=5, n_inner=5, seed=42, n_jobs=1, max_draws=None) -> dict`
    - `ladder_for_set(bank, input_raw, y, block, seed=42, n_jobs=1) -> dict`
    - `best_block(results) -> str`
    - `run_canary(X_block, input_raw, y, seed=42) -> dict`
    - `compare_with_reference(ours, ref) -> {"max_abs_diff", "choice_mismatches"}`
    - `compare_ladders(ours, ref) -> {"max_abs_diff", "recipe_mismatches"}`
    - `pair_with_baseline(y, oof, baseline_dir) -> dict`
    - `write_json(path, obj)`

- [ ] **Step 1: Write the failing unit tests**

`tests/test_nested.py`:
```python
import numpy as np
import pytest

from spectral_lejepa.evaluation import nested as nc
from spectral_lejepa.evaluation.probe import RECIPES, fold_transforms, r2


def synthetic(n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = X[:, 0] * 2 + 0.1 * rng.normal(size=n)
    return X, y


def test_recipes_and_r2():
    assert len(RECIPES) == 12 and RECIPES[0] == ("none", "ridgecv")
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, y) == pytest.approx(1.0)


def test_fold_transforms_fit_on_train_only():
    X, _ = synthetic()
    T = fold_transforms(X[:40], X[40:])
    assert set(T) == {"none", "standardize", "whiten", "whiten8", "whiten32", "whiten128"}
    a, _ = T["standardize"]
    np.testing.assert_allclose(a.mean(0), 0, atol=1e-5)
    w, _ = T["whiten"]
    np.testing.assert_allclose(np.cov(w.T, bias=False), np.eye(w.shape[1]), atol=1e-3)
    assert np.array_equal(T["whiten8"][0], w[:, :8])


def test_run_nested_schema_and_signal():
    X, y = synthetic()
    bank = {"layer0": X, "layer1": np.random.default_rng(1).normal(size=X.shape).astype(np.float32)}
    raw = np.random.default_rng(2).normal(size=(60, 245)).astype(np.float32)
    res, oof = nc.run_nested(bank, raw, y, seed=42)
    assert list(res["families"]) == ["raw", "embedding", "layer0", "layer1", "embedding_top3"]
    assert len(res["raw_fixed_recipes"]) == 12
    assert res["families"]["layer0"]["r2_mean"] > 0.9
    assert res["families"]["embedding"]["r2_mean"] > 0.9
    assert oof["embedding"].shape == (2, 60)
    assert nc.best_block(res) == "layer0"
    again, _ = nc.run_nested(bank, raw, y, seed=42)
    assert again["families"]["embedding"]["r2_mean"] == res["families"]["embedding"]["r2_mean"]


def test_paired_delta_identical_is_zero():
    X, y = synthetic()
    P = np.stack([y + 0.1, y - 0.1])
    d = nc.paired_delta(y, P, P)
    assert d["delta"] == 0 and d["sd"] == 0


def test_canary_passes_on_real_signal_and_noise():
    X, y = synthetic(n=80)
    c = nc.run_canary(X, X, y)
    assert c["block"]["real_r2"] > 0.9 and c["block"]["passed"] and c["raw"]["passed"]


def test_ladder_schema():
    X, y = synthetic()
    out = nc.nested_ladder({"raw": X[:, 1:], "block": X}, y, max_draws=2)
    assert out["rungs"] == [10, 20, 48]
    assert set(out["arms"]) == {"raw", "block"} and set(out["gaps"]) == {"block"}
    assert out["arms"]["block"]["48"]["median"] > 0.9


def test_compare_with_reference_detects_differences():
    X, y = synthetic()
    bank = {"layer0": X}
    raw = X[:, ::-1].copy()
    res, _ = nc.run_nested(bank, raw, y)
    assert nc.compare_with_reference(res, res) == {"max_abs_diff": 0.0, "choice_mismatches": 0}
    other = {**res, "embedding_minus_raw": {**res["embedding_minus_raw"], "delta": res["embedding_minus_raw"]["delta"] + 1}}
    assert nc.compare_with_reference(res, other)["max_abs_diff"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nested.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Write `src/spectral_lejepa/evaluation/probe.py`**

These are copied from clean-eval 6237feb: `normalize.py` (none, standardize, whiten), `regressors.make_regressor` (ridgecv, ols), `protocol.r2` and `nested.fold_transforms`. The numerics must stay identical, including the float32/float64 casts.

```python
"""Feature normalizers, linear probes and R² -- copied from clean-eval (6237feb) with behaviour
preserved: normalize.py (none/standardize/whiten), regressors.make_regressor (ridgecv/ols),
protocol.r2, nested.fold_transforms. Every normalizer is fit on training rows only.

Whitening is exact float64 PCA, keeping at most one direction per 2 training samples and
dropping only numerically dead directions: on this task the label signal lives in
low-variance directions, so "explains ~all the variance" is not a reason to drop one.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, RidgeCV

# RidgeCV's small-alpha end is ill-conditioned by construction at high p/n; CV picks a
# well-conditioned alpha. Silence the warnings, not the numerics (as the parent does).
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

NORMALIZERS = ("none", "standardize", "whiten", "whiten8", "whiten32", "whiten128")
PROBES = ("ridgecv", "ols")
RECIPES = tuple((n, p) for n in NORMALIZERS for p in PROBES)
WHITEN_SAMPLES_PER_DIRECTION = 2
EXACT_SVD_MAX_DIM = 6000


def r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _standardize(X_tr, X_te):
    X_tr32 = np.asarray(X_tr).astype(np.float32, copy=False)
    mu = X_tr32.mean(axis=0).astype(np.float32)
    sd = X_tr32.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return [((np.asarray(X, dtype=np.float32) - mu) / sd).astype(np.float32) for X in (X_tr, X_te)]


def _whiten(X_tr, X_te, seed):
    """Full-rank whitening fit on X_tr: returns (Z_tr, Z_te) with the k_keep alive directions."""
    X64 = np.asarray(X_tr, dtype=np.float64)
    n, d = X64.shape
    k = min(d, max(1, n // WHITEN_SAMPLES_PER_DIRECTION))
    solver = "full" if min(n, d) <= EXACT_SVD_MAX_DIM else "randomized"
    pca = PCA(n_components=k, whiten=True, svd_solver=solver, random_state=seed).fit(X64)
    ev = pca.explained_variance_
    tol = (np.finfo(np.float64).eps * max(n, d)) ** 2
    k_keep = max(1, int(np.sum(ev > tol * ev[0])) if ev.size else 0)
    return [pca.transform(np.asarray(X, dtype=np.float64))[:, :k_keep].astype(np.float32)
            for X in (X_tr, X_te)]


def fold_transforms(X_tr, X_te, seed=42) -> dict:
    """{normalizer: (X_tr', X_te')}, every normalizer fit on X_tr only. whitenK = the first K
    columns of one full-rank whitening (identical to fitting each separately with exact SVD)."""
    out = {"none": (np.asarray(X_tr, dtype=np.float32), np.asarray(X_te, dtype=np.float32)),
           "standardize": tuple(_standardize(X_tr, X_te))}
    Ztr, Zte = _whiten(X_tr, X_te, seed)
    for name in NORMALIZERS:
        if name.startswith("whiten"):
            k = Ztr.shape[1] if name == "whiten" else min(Ztr.shape[1], int(name[len("whiten"):]))
            out[name] = (Ztr[:, :k], Zte[:, :k])
    return out


def make_regressor(name):
    if name == "ridgecv":
        # cv=None: efficient leave-one-out over the alpha grid via one SVD
        return RidgeCV(alphas=np.logspace(-3, 3, 20))
    if name == "ols":
        return LinearRegression()
    raise ValueError(f"unknown probe {name!r}")


def fit_predict(X_tr, y_tr, X_te, probe, seed):
    m = make_regressor(probe)
    m.fit(X_tr, y_tr)
    return np.asarray(m.predict(X_te)).ravel()
```

> **Equivalence note:** the parent's `_Standardize` and `_Identity` cast inputs with `np.asarray(X, dtype=np.float32)`, and its standardize statistics come from `X_all.astype(np.float32)`. Our `_standardize` reproduces both. The parent's whitening promotes the raw (uncast) training array to float64, which `_whiten` also does. `seed` is accepted by `fit_predict` only to keep the call signature; RidgeCV and OLS are deterministic.

- [ ] **Step 4: Write `src/spectral_lejepa/evaluation/nested.py`**

```python
"""Nested cross-validation label probe -- copied from clean-eval (6237feb) nested.py,
nested_ladder.py and canary.py, behaviour preserved.

Every arm (raw input, each encoder block) picks its normalizer+probe recipe by inner K-fold
CV INSIDE each outer training fold; the winner is refit on the whole training fold and only
then predicts the held-out fold. All arms share the outer folds (KFold, shuffle,
random_state=seed+r), so out-of-fold predictions of two arms -- or two backbones on the same
rows -- are paired, and paired_delta resamples the same spectra for both.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .probe import RECIPES, fit_predict, fold_transforms, r2

ENSEMBLE_K = 3      # the block-average readout: the 3 best blocks by inner CV, predictions averaged
MIN_N = 20          # below this, inner folds would fit on a handful of rows; such sets are skipped
LADDER_N = (10, 20, 50, 100, 200, 500, 1000, 2000)


def _inner_scores(X, y, n_inner, seed):
    """Inner-CV R² of every recipe, on these (training) rows only."""
    preds = {rc: np.zeros(len(y)) for rc in RECIPES}
    for itr, ite in KFold(n_inner, shuffle=True, random_state=seed).split(X):
        T = fold_transforms(X[itr], X[ite], seed=seed)
        for norm, probe in RECIPES:
            a, b = T[norm]
            preds[(norm, probe)][ite] = fit_predict(a, y[itr], b, probe, seed)
    return {rc: r2(y, p) for rc, p in preds.items()}


def _outer_fold(arms, families, y, tr, te, n_inner, seed, fixed_arms, ensembles):
    with threadpool_limits(limits=1):
        needed = sorted({a for names in families.values() for a in names}
                        | {a for names, _ in ensembles.values() for a in names})
        inner = {a: _inner_scores(arms[a][tr], y[tr], n_inner, seed) for a in needed}
        cache = {}

        def transforms(arm):
            if arm not in cache:
                cache[arm] = fold_transforms(arms[arm][tr], arms[arm][te], seed=seed)
            return cache[arm]

        chosen = {}
        for fam, names in families.items():
            arm, (norm, probe) = max(((a, rc) for a in names for rc in RECIPES),
                                     key=lambda k: inner[k[0]][k[1]])
            a, b = transforms(arm)[norm]
            chosen[fam] = {"pred": fit_predict(a, y[tr], b, probe, seed), "arm": arm,
                           "norm": norm, "probe": probe, "inner_r2": inner[arm][(norm, probe)]}
        for name, (names, k) in ensembles.items():
            best = {a: max(RECIPES, key=lambda rc: inner[a][rc]) for a in names}
            top = sorted(names, key=lambda a: -inner[a][best[a]])[:k]
            preds = []
            for a_ in top:
                norm, probe = best[a_]
                a, b = transforms(a_)[norm]
                preds.append(fit_predict(a, y[tr], b, probe, seed))
            chosen[name] = {"pred": np.mean(preds, axis=0),
                            "members": [{"arm": a_, "norm": best[a_][0], "probe": best[a_][1],
                                         "inner_r2": inner[a_][best[a_]]} for a_ in top]}
        fixed = {}
        for arm in fixed_arms:
            for norm, probe in RECIPES:
                a, b = transforms(arm)[norm]
                fixed[(arm, norm, probe)] = fit_predict(a, y[tr], b, probe, seed)
    return chosen, fixed


def bootstrap_sd(y, P, n_boot=500, seed=0):
    """SD of the repeat-averaged R² under resampling spectra. P: [R, n]."""
    rng = np.random.default_rng(seed)
    vals = [np.mean([r2(y[i], p[i]) for p in P])
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.std(vals))


def paired_delta(y, PA, PB, n_boot=1000, seed=0):
    """R²(A) - R²(B) on the same folds, with its bootstrap SD (both rescored on the SAME
    resampled spectra, so shared difficulty cancels). PA, PB: [R, n]."""
    def diff(i):
        return float(np.mean([r2(y[i], a[i]) - r2(y[i], b[i]) for a, b in zip(PA, PB)]))
    rng = np.random.default_rng(seed)
    boots = np.array([diff(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    return {"delta": diff(np.arange(len(y))), "sd": float(boots.std()),
            "p_a_better": float((boots > 0).mean())}


def nested_cv(arms, families, y, n_repeats=2, n_folds=5, n_inner=5, seed=42,
              fixed_arms=(), ensembles=None, n_jobs=1):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    ensembles = ensembles or {}
    jobs = []
    for r in range(n_repeats):
        for f, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed + r).split(y)):
            jobs.append((r, f, tr, te))
    args = [(arms, families, y, tr, te, n_inner, seed + 1000 + 10 * r + f, fixed_arms, ensembles)
            for r, f, tr, te in jobs]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(*a) for a in args)
    else:
        outs = [_outer_fold(*a) for a in args]

    res = {fam: {"oof": np.zeros((n_repeats, n)), "chosen": []} for fam in [*families, *ensembles]}
    fixed = {k: np.zeros((n_repeats, n)) for k in outs[0][1]}
    test_folds = [[] for _ in range(n_repeats)]
    for (r, f, tr, te), (chosen, fx) in zip(jobs, outs):
        test_folds[r].append(te.tolist())
        for fam, c in chosen.items():
            res[fam]["oof"][r, te] = c["pred"]
            res[fam]["chosen"].append({"repeat": r, "fold": f, **{k: v for k, v in c.items() if k != "pred"}})
        for k, p in fx.items():
            fixed[k][r, te] = p
    for v in res.values():
        v["r2_per_repeat"] = [r2(y, p) for p in v["oof"]]
        v["r2_mean"] = float(np.mean(v["r2_per_repeat"]))
        v["bootstrap_sd"] = bootstrap_sd(y, v["oof"])
    res["fixed"] = fixed
    res["test_folds"] = test_folds
    return res


def run_nested(bank, input_raw, y, n_repeats=2, n_folds=5, n_inner=5, seed=42, n_jobs=1):
    """Raw input vs the embedding (every block, mean-pooled), each at its nested-selected
    recipe; the top-3 block average; every block alone (depth profile / fixed-block readout);
    raw input at every fixed recipe. Returns (json-able results, {family: oof [R, n]})."""
    arms = {"raw": np.asarray(input_raw, dtype=np.float32), **bank}
    stages = list(bank)
    families = {"raw": ["raw"], "embedding": stages, **{s: [s] for s in stages}}
    ensembles = {f"embedding_top{ENSEMBLE_K}": (stages, ENSEMBLE_K)}
    res = nested_cv(arms, families, y, n_repeats=n_repeats, n_folds=n_folds, n_inner=n_inner,
                    seed=seed, fixed_arms=("raw",), ensembles=ensembles, n_jobs=n_jobs)
    families = {**families, **ensembles}
    y64 = np.asarray(y, dtype=np.float64)
    out = {
        "protocol": {"n": len(y), "n_repeats": n_repeats, "n_folds": n_folds, "n_inner": n_inner,
                     "seed": seed, "n_comp": 1, "recipes": [f"{a}+{b}" for a, b in RECIPES]},
        "families": {fam: {k: v for k, v in res[fam].items() if k != "oof"} for fam in families},
        "raw_fixed_recipes": {f"{norm}+{probe}": {"r2_mean": float(np.mean([r2(y64, p) for p in P])),
                                                  "bootstrap_sd": bootstrap_sd(y64, P, n_boot=200)}
                              for (_, norm, probe), P in res["fixed"].items()},
        "embedding_minus_raw": paired_delta(y64, res["embedding"]["oof"], res["raw"]["oof"]),
        "test_folds": res["test_folds"],
    }
    return out, {fam: res[fam]["oof"] for fam in families}


def best_block(results) -> str:
    """The block with the highest nested-CV score on its own."""
    fam = results["families"]
    blocks = {k: v for k, v in fam.items() if k != "raw" and not k.startswith("embedding")}
    return max(blocks, key=lambda k: blocks[k]["r2_mean"])


# --- label-budget ladder (nested_ladder.py) ---

def n_draws(n, n_full, max_draws=None):
    """More draws where one draw is noisy and cheap; one at the full fold."""
    if n >= n_full:
        return 1
    d = 20 if n <= 100 else 10 if n <= 500 else 5
    return min(d, max_draws) if max_draws else d


def _one_draw(arms, y, sub, te, n_inner, seed):
    out = {}
    with threadpool_limits(limits=1):
        for name, X in arms.items():
            inner = _inner_scores(X[sub], y[sub], min(n_inner, len(sub)), seed)
            norm, probe = max(RECIPES, key=lambda rc: inner[rc])
            a, b = fold_transforms(X[sub], X[te], seed=seed)[norm]
            out[name] = {"r2": r2(y[te], fit_predict(a, y[sub], b, probe, seed)), "recipe": f"{norm}+{probe}"}
    return out


def nested_ladder(arms, y, n_trains=LADDER_N, n_folds=5, n_inner=5, seed=42, n_jobs=1, max_draws=None):
    """Per outer fold and label budget n: random n-row subsets of the training fold; inner CV on
    those n rows picks the recipe; scored on the held-out fold. Arms share subsets (paired)."""
    y = np.asarray(y, dtype=np.float64)
    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(y))
    n_full = min(len(tr) for tr, _ in folds)
    rungs = [n for n in n_trains if n < n_full] + [n_full]
    jobs = []
    for f, (tr, te) in enumerate(folds):
        rng = np.random.default_rng(seed + 100 * f)
        for n in rungs:
            for d in range(n_draws(n, n_full, max_draws)):
                sub = tr if n >= n_full else rng.choice(tr, size=n, replace=False)
                jobs.append((n, (arms, y, np.sort(sub), te, n_inner, seed + 7 * d + 1000 * f)))
    if n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_one_draw)(*a) for _, a in jobs)
    else:
        outs = [_one_draw(*a) for _, a in jobs]

    names = list(arms)
    per = {a: collections.defaultdict(list) for a in names}
    rec = {a: collections.defaultdict(collections.Counter) for a in names}
    gap = {a: collections.defaultdict(list) for a in names[1:]}
    for (n, _), o in zip(jobs, outs):
        for a in names:
            per[a][n].append(o[a]["r2"])
            rec[a][n][o[a]["recipe"]] += 1
        for a in names[1:]:
            gap[a][n].append(o[a]["r2"] - o[names[0]]["r2"])

    def summ(v):
        return {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}
    return {
        "rungs": rungs,
        "draws": {str(n): n_folds * n_draws(n, n_full, max_draws) for n in rungs},
        "arms": {a: {str(n): {**summ(per[a][n]), "recipes": dict(rec[a][n].most_common())} for n in rungs}
                 for a in names},
        "gaps": {a: {str(n): {**summ(gap[a][n]), "frac_positive": float(np.mean(np.array(gap[a][n]) > 0))}
                     for n in rungs} for a in names[1:]},
    }


def ladder_for_set(bank, input_raw, y, block, seed=42, n_jobs=1):
    """Raw input vs ONE block fixed in advance (the fixed-block readout)."""
    arms = {"raw": np.asarray(input_raw, dtype=np.float32), "block": bank[block]}
    return {**nested_ladder(arms, y, seed=seed, n_jobs=n_jobs), "block": block}


# --- shuffled-label canary (canary.py) ---

def _kfold_oof(X, y, seed, n_folds=5):
    pred = np.zeros(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        a, b = fold_transforms(X[tr], X[te], seed=seed)["standardize"]
        pred[te] = fit_predict(a, y[tr], b, "ridgecv", seed)
    return pred


def shuffled_label_canary(X, y, seed=42, threshold=0.02):
    """Permute y and rerun the same probe: an honest pipeline scores R² ~ 0."""
    y = np.asarray(y, dtype=np.float64)
    y_shuffled = np.random.default_rng(seed).permutation(y)
    real = r2(y, _kfold_oof(X, y, seed))
    shuffled = r2(y_shuffled, _kfold_oof(X, y_shuffled, seed))
    return {"real_r2": real, "shuffled_r2": shuffled, "passed": bool(shuffled <= threshold)}


def run_canary(X_block, input_raw, y, seed=42):
    return {"block": shuffled_label_canary(X_block, y, seed), "raw": shuffled_label_canary(input_raw, y, seed)}


# --- comparisons and pairing ---

def compare_with_reference(ours, ref):
    """Largest absolute numeric difference and number of differing recipe choices between two
    nested results (ours vs the parent's nested_results.json)."""
    if list(ours["families"]) != list(ref["families"]):
        raise ValueError(f"family lists differ: {list(ours['families'])} vs {list(ref['families'])}")
    diffs, mismatches = [], 0
    for fam, o in ours["families"].items():
        r = ref["families"][fam]
        diffs += [abs(o["r2_mean"] - r["r2_mean"]), abs(o["bootstrap_sd"] - r["bootstrap_sd"])]
        diffs += [abs(a - b) for a, b in zip(o["r2_per_repeat"], r["r2_per_repeat"])]
        for co, cr in zip(o["chosen"], r["chosen"]):
            pairs = list(zip(co["members"], cr["members"])) if "members" in co else [(co, cr)]
            mismatches += "members" in co and len(co["members"]) != len(cr["members"])
            for mo, mr in pairs:
                mismatches += (mo["arm"], mo["norm"], mo["probe"]) != (mr["arm"], mr["norm"], mr["probe"])
                diffs.append(abs(mo["inner_r2"] - mr["inner_r2"]))
    for k, o in ours["raw_fixed_recipes"].items():
        r = ref["raw_fixed_recipes"][k]
        diffs += [abs(o["r2_mean"] - r["r2_mean"]), abs(o["bootstrap_sd"] - r["bootstrap_sd"])]
    for k in ("delta", "sd", "p_a_better"):
        diffs.append(abs(ours["embedding_minus_raw"][k] - ref["embedding_minus_raw"][k]))
    return {"max_abs_diff": float(max(diffs)), "choice_mismatches": int(mismatches)}


def compare_ladders(ours, ref):
    if ours["rungs"] != ref["rungs"] or ours["block"] != ref["block"]:
        raise ValueError(f"ladders differ in rungs/block: {ours['rungs']}/{ours['block']} vs "
                         f"{ref['rungs']}/{ref['block']}")
    diffs, mismatches = [], 0
    for arm, rungs in ours["arms"].items():
        for n, o in rungs.items():
            r = ref["arms"][arm][n]
            diffs += [abs(o[k] - r[k]) for k in ("median", "p25", "p75")]
            mismatches += o["recipes"] != r["recipes"]
    for arm, rungs in ours["gaps"].items():
        for n, o in rungs.items():
            diffs += [abs(o[k] - ref["gaps"][arm][n][k]) for k in ("median", "p25", "p75", "frac_positive")]
    return {"max_abs_diff": float(max(diffs)), "recipe_mismatches": int(mismatches)}


def pair_with_baseline(y, oof, baseline_dir):
    """Paired R² gaps (ours - baseline) on identical spectra and folds, read from the
    baseline's nested_oof.npz. Refuses if the rows differ: a gap on different rows is meaningless."""
    path = os.path.join(baseline_dir, "nested_oof.npz")
    base = np.load(path)
    if base["y"].shape != np.shape(y) or not np.array_equal(base["y"], y):
        raise ValueError(f"baseline rows in {path} differ from ours (n={len(base['y'])} vs {len(y)}); "
                         "refusing to pair")
    families = ("embedding", f"embedding_top{ENSEMBLE_K}", "raw")
    return {fam: paired_delta(np.asarray(y, dtype=np.float64), oof[fam], base[fam])
            for fam in families if fam in base.files and fam in oof}


def write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
```

> **Equivalence note:** we put `test_folds` inside `out` so it can be compared with the parent. The parent's JSON doesn't have it at the top level, so `compare_with_reference` ignores it.

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_nested.py -v`
Expected: 7 PASS.

- [ ] **Step 6: Write the equivalence tests**

`tests/test_eval_equivalence.py`:
```python
"""Our copied nested CV must reproduce the parent's saved ref_feb25 numbers exactly."""
import json
import os

import numpy as np
import pytest

from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.nested import compare_ladders, compare_with_reference, ladder_for_set, run_nested

PARENT_OUT = "/mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged/code/eval_outputs"
SETS = {
    "dataset0055": "label_probe_regression_ref_feb25/label_probe/dataset0055",
    "dataset0120": "label_probe_regression_ref_feb25/label_probe/dataset0120",
    "labeled_regression_all": "label_probe_regression_merged_ref_feb25/label_probe/labeled_regression_all",
}
TOL = 1e-10


def parent(rel):
    d = os.path.join(PARENT_OUT, rel)
    if not os.path.exists(os.path.join(d, "nested_results.json")):
        pytest.skip("parent outputs not available")
    return d


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SETS))
def test_nested_matches_parent(name):
    d = parent(SETS[name])
    bank, raw, y, _ = load_bank(os.path.join(d, "bank.npz"))
    ours, oof = run_nested(bank, raw, y, seed=42, n_jobs=8)
    check = compare_with_reference(ours, json.load(open(os.path.join(d, "nested_results.json"))))
    assert check["choice_mismatches"] == 0, check
    assert check["max_abs_diff"] <= TOL, check
    ref_oof = np.load(os.path.join(d, "nested_oof.npz"))
    for fam, P in oof.items():
        np.testing.assert_allclose(P, ref_oof[fam], rtol=0, atol=TOL, err_msg=fam)


@pytest.mark.slow
def test_ladder_matches_parent_on_merged_set():
    d = parent(SETS["labeled_regression_all"])
    ref = json.load(open(os.path.join(d, "nested_ladder.json")))
    bank, raw, y, _ = load_bank(os.path.join(d, "bank.npz"))
    ours = ladder_for_set(bank, raw, y, ref["block"], seed=42, n_jobs=8)
    check = compare_ladders(ours, ref)
    assert check["recipe_mismatches"] == 0 and check["max_abs_diff"] <= TOL, check
```

- [ ] **Step 7: Run the equivalence tests**

Run: `uv run pytest tests/test_eval_equivalence.py -m slow -v`
Expected: 4 PASS. dataset0055 and dataset0120 take minutes; the merged set and the ladder can take tens of minutes.

If a case fails:
- Do **not** loosen `TOL` or change the copied code until you understand why.
- Use superpowers:systematic-debugging. Compare against the parent's code run in its own env (`/mnt5/home/hadar/nova/.venv-labelprobe/bin/python`, with `PYTHONPATH=<parent>/code`).
- If the only cause is a BLAS-level difference, document it in the README and stop to report to the user before continuing.

- [ ] **Step 8: Commit**

```bash
git add src/spectral_lejepa/evaluation/probe.py src/spectral_lejepa/evaluation/nested.py tests/test_nested.py tests/test_eval_equivalence.py
git commit -m "Add nested-CV label probe, ladder and canary copied from clean-eval, verified against parent outputs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 12: Evaluation and baseline scripts, eval config, baseline W&B run

**Files:**
- Create: `configs/eval.yaml`, `scripts/evaluate.py`, `scripts/baseline.py`, `tests/test_pairing.py`

**Interfaces:**
- Consumes:
  - From Tasks 10–11: everything in `evaluation/*`.
  - From Task 9: `load_model`.
  - From Task 6: `EvalBackbone`.
  - From Task 3: the W&B helpers.
- Produces:
  - `outputs/eval/<label>/<set>/{bank.npz, nested_results.json, nested_oof.npz, nested_ladder.json, summary.json}`
  - `outputs/baseline/ref_feb25/<set>/...`
  - W&B runs with `job_type` `eval` and `baseline`

- [ ] **Step 1: Write the failing pairing test (Review Focus #5)**

`tests/test_pairing.py`:
```python
import numpy as np
import pytest

from spectral_lejepa.evaluation.nested import pair_with_baseline


def test_pairing_requires_identical_rows(tmp_path):
    y = np.arange(30, dtype=np.float64)
    P = np.stack([y + 1, y - 1])
    np.savez(tmp_path / "nested_oof.npz", y=y, embedding=P, raw=P, embedding_top3=P)
    ours = {"embedding": P + 0.5, "raw": P, "embedding_top3": P}
    out = pair_with_baseline(y, ours, tmp_path)
    assert out["raw"]["delta"] == 0.0               # identical raw arms must give exactly zero
    assert out["embedding"]["delta"] < 0
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y[::-1].copy(), ours, tmp_path)
    with pytest.raises(ValueError, match="refusing to pair"):
        pair_with_baseline(y[:20], ours, tmp_path)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_pairing.py -v`
Expected: PASS, because `pair_with_baseline` already exists from Task 11. This test pins the review-focus behaviour. If it fails, fix `pair_with_baseline`.

- [ ] **Step 3: Write `configs/eval.yaml`**

```yaml
# Downstream evaluation: clean-eval nested-CV label probe (seed 42, component 0).
experiment:
  name: lejepa_eval
  seed: 42
  output_dir: outputs/eval

evaluation:
  device: cuda
  batch_size: 64
  max_samples: 5000
  n_jobs: 8
  min_n: 20
  ladder_sets: [labeled_data, labeled_regression_all]
  ladder_block_from: labeled_data      # must be listed first in label_sets

# name -> list of label-set directories (several = merged, as clean-eval's merge_label_sets)
label_sets:
  labeled_data: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data]
  labeled_regression_all:
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0055
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0106
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0109
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0112
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0113
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0114
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0115
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0117
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0118
    - /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0120
  dataset0055: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0055]
  dataset0106: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0106]
  dataset0109: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0109]
  dataset0112: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0112]
  dataset0113: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0113]
  dataset0114: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0114]
  dataset0120: [/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_regression/dataset0120]

baseline:
  tag: ref_feb25
  checkpoint: /mnt5/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt
  parent_repo: /mnt5/home/hadar/nova/SpectralFM-label-regression-eval-merged
  run_dirs:          # relative to parent_repo; each holds bank.npz + nested_results.json + nested_oof.npz
    labeled_data: code/eval_outputs/label_probe_regression_labeled_data_ref_feb25/label_probe/labeled_data
    labeled_regression_all: code/eval_outputs/label_probe_regression_merged_ref_feb25/label_probe/labeled_regression_all
    dataset0055: code/eval_outputs/label_probe_regression_ref_feb25/label_probe/dataset0055
    dataset0106: code/eval_outputs/label_probe_regression_ref_feb25/label_probe/dataset0106
    dataset0109: code/eval_outputs/label_probe_regression_ref_feb25/label_probe/dataset0109
    dataset0112: code/eval_outputs/label_probe_regression_ref_feb25/label_probe/dataset0112
    dataset0113: code/eval_outputs/label_probe_regression_ref_feb25/label_probe/dataset0113
    dataset0114: code/eval_outputs/label_probe_regression_ref_feb25/label_probe/dataset0114
    dataset0120: code/eval_outputs/label_probe_regression_ref_feb25/label_probe/dataset0120

wandb:
  enabled: true
  entity: null
  project: spectralfm-lejepa
  group: null
  job_type: eval
  tags: [lejepa, spectralfm, 1d, eval]
```

- [ ] **Step 4: Write `scripts/evaluate.py`**

```python
"""Evaluate a LeJEPA checkpoint with the clean-eval nested-CV label probe and pair it with the
data2vec baseline (ref_feb25) on identical rows and folds.

  uv run python -m scripts.evaluate --checkpoint outputs/<run>/checkpoint_last.pt
  uv run python -m scripts.evaluate --checkpoint wandb:<entity>/spectralfm-lejepa/lejepa-<run_id>:latest
"""
import argparse
import os
from pathlib import Path

import numpy as np

from spectral_lejepa.config import load_config
from spectral_lejepa.evaluation.bank import extract_mean_bank, save_bank
from spectral_lejepa.evaluation.data import load_labeled_data, normalize_like_fairseq
from spectral_lejepa.evaluation.nested import (ENSEMBLE_K, best_block, ladder_for_set, pair_with_baseline,
                                               run_canary, run_nested, write_json)
from spectral_lejepa.models.vit_1d import EvalBackbone
from spectral_lejepa.training.checkpoint import load_model
from spectral_lejepa.utils import wandb as wb


def flat_metrics(prefix, d):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flat_metrics(key, v))
        elif isinstance(v, (bool, int, float)):
            out[key] = float(v)
    return out


def resolve_checkpoint(ref, run):
    if not ref.startswith("wandb:"):
        return ref
    if run is None:
        raise RuntimeError("a wandb: checkpoint reference needs W&B enabled")
    files = list(Path(run.use_artifact(ref[len("wandb:"):], type="model").download()).glob("*.pt"))
    if len(files) != 1:
        raise RuntimeError(f"expected one .pt file in artifact {ref}, found {files}")
    return str(files[0])


def summarize(nested, block):
    fam = nested["families"]
    return {"n": nested["protocol"]["n"],
            "embedding_r2": fam["embedding"]["r2_mean"], "embedding_sd": fam["embedding"]["bootstrap_sd"],
            "raw_r2": fam["raw"]["r2_mean"], "raw_sd": fam["raw"]["bootstrap_sd"],
            "top3_r2": fam[f"embedding_top{ENSEMBLE_K}"]["r2_mean"],
            "best_block": block, "best_block_r2": fam[block]["r2_mean"],
            "blocks": {k: v["r2_mean"] for k, v in fam.items() if k.startswith("layer")},
            "embedding_minus_raw": nested["embedding_minus_raw"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="local .pt path or wandb:<artifact ref>")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    ecfg, vcfg, bcfg = cfg["experiment"], cfg["evaluation"], cfg["baseline"]
    seed = ecfg["seed"]

    run = wb.init_run(cfg, job_type=cfg["wandb"]["job_type"])
    ckpt_path = resolve_checkpoint(a.checkpoint, run)
    model, ckpt = load_model(ckpt_path)
    lineage = {"pretraining_run_id": ckpt.get("wandb_run_id"), "pretraining_checkpoint": a.checkpoint,
               "pretraining_checkpoint_step": ckpt["step"], "pretraining_git_commit": ckpt.get("git_commit"),
               "pretraining_git_dirty": ckpt.get("git_dirty"), "evaluation_git_commit": wb.git_info()["git_commit"],
               "evaluation_config": cfg}
    if run is not None:
        run.config.update({"lineage": lineage})
        if not a.checkpoint.startswith("wandb:") and ckpt.get("wandb_run_id"):
            run.use_artifact(f"lejepa-{ckpt['wandb_run_id']}:step-{ckpt['step']}")   # W&B lineage edge

    backbone = EvalBackbone(model)
    out_root = Path(ecfg["output_dir"]) / f"{Path(ckpt_path).parent.name}_step{ckpt['step']}"
    sets, ladder_block = {}, None
    for name, dirs in cfg["label_sets"].items():
        raw, y = load_labeled_data(dirs, max_samples=vcfg["max_samples"], seed=seed)
        if len(y) < vcfg["min_n"]:
            print(f"[evaluate] skip {name}: n={len(y)} < {vcfg['min_n']}")
            continue
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        bank = extract_mean_bank(backbone, normalize_like_fairseq(raw), device=vcfg["device"],
                                 batch_size=vcfg["batch_size"])
        save_bank(out_dir / "bank.npz", bank, raw, y, {"checkpoint": a.checkpoint, "backbone": "EvalBackbone",
                                                       "set": name, "n": int(len(y)), "seed": seed,
                                                       "stages": tuple(bank)})
        nested, oof = run_nested(bank, raw, y, seed=seed, n_jobs=vcfg["n_jobs"])
        write_json(out_dir / "nested_results.json", {**nested, "meta": lineage})
        np.savez_compressed(out_dir / "nested_oof.npz", y=y, **oof)
        block = best_block(nested)
        summary = summarize(nested, block)
        summary["canary"] = run_canary(bank[block], raw, y, seed)
        if name in bcfg["run_dirs"]:
            summary["vs_baseline"] = pair_with_baseline(y, oof, os.path.join(bcfg["parent_repo"], bcfg["run_dirs"][name]))
        if name == vcfg["ladder_block_from"]:
            ladder_block = block
        if name in vcfg["ladder_sets"]:
            if ladder_block is None:
                raise ValueError("evaluation.ladder_block_from must come before the ladder sets in label_sets")
            ladder = ladder_for_set(bank, raw, y, ladder_block, seed=seed, n_jobs=vcfg["n_jobs"])
            write_json(out_dir / "nested_ladder.json", ladder)
            summary["ladder"] = {"block": ladder_block,
                                 **{f"{arm}_n{n}": v["median"] for arm, rungs in ladder["arms"].items()
                                    for n, v in rungs.items()}}
        write_json(out_dir / "summary.json", summary)
        sets[name] = summary
        wb.log(run, flat_metrics(name, summary))
        print(f"[evaluate] {name}: embedding R2 {summary['embedding_r2']:.4f} vs raw {summary['raw_r2']:.4f}"
              + (f", vs {bcfg['tag']} {summary['vs_baseline']['embedding']['delta']:+.4f}"
                 if "vs_baseline" in summary else ""))

    write_json(out_root / "summary.json", {"lineage": lineage, "sets": sets})
    if run is not None:
        import wandb
        artifact = wandb.Artifact(f"eval-{run.id}", type="evaluation", metadata={"lineage": lineage})
        for path in out_root.rglob("*.json"):
            artifact.add_file(str(path), name=str(path.relative_to(out_root)))
        run.log_artifact(artifact)
    wb.finish(run)
    print("outputs:", out_root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `scripts/baseline.py`**

```python
"""Record the data2vec baseline (ref_feb25) in W&B by re-scoring the parent's saved banks with
OUR copied evaluation code, and check that it reproduces the parent's own numbers.

  uv run python -m scripts.baseline [--config configs/eval.yaml]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from spectral_lejepa.config import load_config
from spectral_lejepa.evaluation.bank import load_bank
from spectral_lejepa.evaluation.nested import (best_block, compare_ladders, compare_with_reference,
                                               ladder_for_set, run_nested, write_json)
from spectral_lejepa.utils import wandb as wb
from scripts.evaluate import flat_metrics, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    bcfg, vcfg, seed = cfg["baseline"], cfg["evaluation"], cfg["experiment"]["seed"]
    cfg["wandb"]["tags"] = ["baseline", "data2vec", "spectralfm"]
    parent_git = wb.git_info(bcfg["parent_repo"])
    run = wb.init_run(cfg, job_type="baseline", name=f"baseline_{bcfg['tag']}",
                      extra_config={"baseline_parent_git": parent_git, "baseline_checkpoint": bcfg["checkpoint"]})

    out_root = Path("outputs") / "baseline" / bcfg["tag"]
    ladder_block, worst = None, 0.0
    for name, rel in bcfg["run_dirs"].items():
        src = os.path.join(bcfg["parent_repo"], rel)
        bank, raw, y, meta = load_bank(os.path.join(src, "bank.npz"))
        if len(y) < vcfg["min_n"]:
            continue
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        nested, oof = run_nested(bank, raw, y, seed=seed, n_jobs=vcfg["n_jobs"])
        check = compare_with_reference(nested, json.load(open(os.path.join(src, "nested_results.json"))))
        write_json(out_dir / "nested_results.json", {**nested, "meta": {**meta, "source": src}})
        np.savez_compressed(out_dir / "nested_oof.npz", y=y, **oof)
        block = best_block(nested)
        summary = {**summarize(nested, block), "reproduction": check}
        if name == vcfg["ladder_block_from"]:
            ladder_block = block
        if name in vcfg["ladder_sets"] and os.path.exists(os.path.join(src, "nested_ladder.json")):
            ladder = ladder_for_set(bank, raw, y, ladder_block, seed=seed, n_jobs=vcfg["n_jobs"])
            write_json(out_dir / "nested_ladder.json", ladder)
            summary["ladder_reproduction"] = compare_ladders(ladder, json.load(open(os.path.join(src, "nested_ladder.json"))))
        write_json(out_dir / "summary.json", summary)
        worst = max(worst, check["max_abs_diff"])
        wb.log(run, flat_metrics(name, summary))
        print(f"[baseline] {name}: embedding R2 {summary['embedding_r2']:.4f}, raw {summary['raw_r2']:.4f}, "
              f"max |diff| vs parent {check['max_abs_diff']:.2e}, choice mismatches {check['choice_mismatches']}")
    wb.log(run, {"reproduction/max_abs_diff": worst})
    wb.finish(run)


if __name__ == "__main__":
    main()
```

`baseline.py` imports helpers from `scripts.evaluate`, so also create an empty `scripts/__init__.py`; both scripts run as modules from the repo root (`uv run python -m scripts.evaluate`, `uv run python -m scripts.baseline`).

- [ ] **Step 6: Commit**

```bash
git add configs/eval.yaml scripts/__init__.py scripts/evaluate.py scripts/baseline.py tests/test_pairing.py
git commit -m "Add evaluation and baseline scripts with W&B lineage" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

- [ ] **Step 7: Run the baseline once (needs W&B)**

Run: `uv run python -m scripts.baseline`
Expected: for every set it prints `max |diff| vs parent` ≤ 1e-10 and `choice mismatches 0`, and the W&B run `baseline_ref_feb25` (job_type `baseline`) holds the metrics. labeled_data takes about 15–30 minutes; its ladder can take longer.

- [ ] **Step 8: Dry-run the evaluation on the Task 9 dev checkpoint**

Run: `uv run python -m scripts.evaluate --checkpoint outputs/<dev_smoke run dir>/checkpoint_last.pt`
Expected:
- It finishes and writes `outputs/eval/<dev run>_step300/summary.json`.
- `vs_baseline.raw.delta` is exactly `0.0` for every set (identical raw arms, so this is a sanity check of the pairing).
- Both canaries pass.
- The W&B eval run shows its lineage (`pretraining_run_id`), and the artifact graph links it to `lejepa-<dev run>`.
- The R² numbers themselves are meaningless for a 300-step model.

---

### Task 13: README

**Files:**
- Modify: `README.md` (replace the one-line placeholder)

- [ ] **Step 1: Write `README.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add README describing the experiment" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KGh7h4ircUwvzkmxNXKjt"
```

---

### Task 14: First real pretraining run (GATE)

- [ ] **Step 1: Check the tree is clean and the tests pass**

Run: `git status --short && uv run pytest`
Expected: no output from `git status`, and every test PASSES. A clean tree means `git_dirty=false` is logged.

- [ ] **Step 2: Launch**

Use the steps/s measured in Task 9 Step 9 to estimate the run time (78k steps), and tell the user the estimate. Then:
```bash
mkdir -p outputs
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/pretrain.py > outputs/lejepa_baseline.log 2>&1 &
```
Run it in the background, then check `tail outputs/lejepa_baseline.log` and the W&B page.

- [ ] **Step 3: Stop and report to the user**

Report:
- the W&B run URL;
- the curves for `train/mse_loss` and `train/sigreg_loss` (and their ratio);
- valid losses;
- `representation/effective_rank`, `std`, `covariance_condition` and `mean_pairwise_cosine` over time;
- any collapse alerts;
- the final checkpoint path.

Recommend whether λ needs to change, based on those measurements. Do not change λ, and do not start sweeps, without the user's decision.

---

### Task 15: Evaluate the baseline LeJEPA checkpoint (GATE)

- [ ] **Step 1: Run the evaluation**

Run: `uv run python -m scripts.evaluate --checkpoint outputs/<lejepa_baseline run dir>/checkpoint_last.pt`

- [ ] **Step 2: Stop and report to the user**

Report the following for each label set:
- n;
- LeJEPA embedding R² ± bootstrap SD;
- raw R²;
- top-3 R²;
- best block;
- the paired gap against `ref_feb25` (delta ± sd, p_a_better);
- the canary results.

Also report:
- the ladder medians against raw input on labeled_data and labeled_regression_all;
- the W&B eval run URL and its lineage to the pretraining artifact.

State the comparison plainly, including the scale caveat (5M vs 93M parameters, about 20M vs 27.6M samples seen).
