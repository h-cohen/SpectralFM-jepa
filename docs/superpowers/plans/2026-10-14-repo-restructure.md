# Repository Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `scripts/` by moving reusable code into the `spectral_lejepa` package behind one CLI (`python -m spectral_lejepa <cmd>`). Delete finished-study code. Training, evaluation, export, gate and report outputs stay byte-identical.

**Architecture:** Use `git mv` plus a mechanical import rewrite, driven by one module map. No logic edits. There is one extraction: `job_oof` becomes `evaluation/ensemble.py`. `__main__.py` dispatches each subcommand to the moved module's existing `main()` by setting `sys.argv`. Tests move into subfolders that mirror the package.

**Tech Stack:** Python 3.10, PyTorch, pytest, `uv`/`.venv`, git.

**Spec:** `docs/superpowers/specs/2026-10-14-repo-restructure-design.md`

## Global Constraints

- Commit as `Hadar <hal.nls@gmail.com>` on `main`, with the session attribution trailer. Push after the final verification only.
- No logic edits in moved files. Only import lines and module-path strings change. The single extraction is Task 5's `ensemble.py`.
- Never write under `/mnt5/noy` or into `SpectralFM-label-regression-eval-merged`.
- Before starting, `systemctl --user list-units 'spectralfm-*' --state=active --no-legend` must print nothing.
- Run tests with `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q`. Slow tests run with `-m slow`.
- GPU runs use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4` (physical GPU 4). Physical GPU 3 is broken.
- Spec deviation, recorded: `pack_spectra`/`pack_pickles` stay two modules, and `robustness_perturbations`/`robustness_eval` stay two modules (`perturbation.py`, `perturbation_eval.py`). Merging them would be a logic edit. Task 1 updates the spec's mapping table to match.

## Module map (single source of truth)

| Old module | New module |
|---|---|
| `scripts.pretrain` | `spectral_lejepa.training.pretrain` |
| `scripts.gradient_audit` | `spectral_lejepa.training.audit` |
| `scripts.export_production` | `spectral_lejepa.models.export` |
| `scripts.pack_spectra` | `spectral_lejepa.data.pack_spectra` |
| `scripts.pack_pickles` | `spectral_lejepa.data.pack_pickles` |
| `scripts.noise_calibration` | `spectral_lejepa.data.noise_calibration` |
| `scripts.evaluate` | `spectral_lejepa.evaluation.run` |
| `scripts.final_model_gate` | `spectral_lejepa.evaluation.gate` |
| `scripts.permutation_audit` | `spectral_lejepa.evaluation.permutation` |
| `scripts.robustness_perturbations` | `spectral_lejepa.evaluation.perturbation` |
| `scripts.robustness_eval` | `spectral_lejepa.evaluation.perturbation_eval` |
| `scripts.build_results_report` | `spectral_lejepa.reporting.results` |
| `scripts.results_matrix` | `spectral_lejepa.reporting.matrix` |
| `scripts.readout_variants` | `spectral_lejepa.reporting.readouts` |
| `scripts.report_story` | `spectral_lejepa.reporting.story` |
| `scripts.scorecard_figures` | `spectral_lejepa.reporting.scorecard` |
| `scripts.local_wandb` | `spectral_lejepa.reporting.local_wandb` |
| `scripts.architecture_diagram` | `spectral_lejepa.reporting.architecture` |
| `scripts.continuation_report` | `spectral_lejepa.reporting.sections.continuation` |
| `scripts.frozen_feature_report` | `spectral_lejepa.reporting.sections.frozen_features` |
| `scripts.production_analysis_report` | `spectral_lejepa.reporting.sections.production_analysis` |
| `scripts.projection_pilot_report` | `spectral_lejepa.reporting.sections.projection_pilot` |
| `scripts.training_report` | `spectral_lejepa.reporting.training_report` |
| `scripts.training_report_html` | `spectral_lejepa.reporting.training_report_html` |

Deleted: `robust_ensemble`, `robustness_continuation`, `robustness_continuation_analysis`, `seed_ensemble`, `production_policies`, `chemometric_baselines`, `diagnose_0106`, `memorization_check`, `training_progress`, `launch_robust_ensemble.sh`, `launch_robustness_continuation.sh`.

## Review Focus

1. **Module paths inside strings.** Tests use `monkeypatch.setattr("scripts.x.attr", ...)` and docstrings. The rewrite must change strings too; Task 3's grep check catches leftovers.
2. **Relative output paths.** Moved modules use cwd-relative paths such as `outputs/...` and `docs/figures/...`. The CLI must still run from the repo root. Task 7 runs every subcommand from the root.
3. **`python -m spectral_lejepa evaluate --help`.** argparse `prog` names change. Help must still work for every subcommand (Task 6 test).
4. **Name shadowing after test moves.** `tests/training/test_model.py` vs `tests/models/test_model.py`: every test subfolder needs `__init__.py`, or pytest raises "import file mismatch" (Task 6).
5. **The launcher after the move.** `ops/launch_final_model.sh --dry-run` must print the new commands, and the gate/report steps must resolve (Task 6 test).

---

### Task 1: Baseline references

**Files:**
- Modify: `docs/superpowers/specs/2026-10-14-repo-restructure-design.md` (mapping rows for pack and perturbation, per Global Constraints)
- Create (scratchpad, not committed): `$S/before/` with fingerprint, evaluation and report references. `S=/tmp/claude-1040/-mnt5-home-hadar-nova-SpectralFM-jepa/c5eb3939-e776-49c1-a1a7-78ee1c14e4f1/scratchpad`

- [ ] **Step 1:** Confirm that nothing is running and the tree is clean.
  Run: `systemctl --user list-units 'spectralfm-*' --state=active --no-legend; git status --short`
  Expected: both print nothing.
- [ ] **Step 2:** Run the full test suite and record the counts.
  Run: `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q 2>&1 | tail -1; PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q -m slow 2>&1 | tail -1`
  Expected: `199 passed`, `8 passed`.
- [ ] **Step 3:** Fingerprint with the existing `$S/equiv.py`.
  Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 .venv/bin/python $S/equiv.py > $S/before/fingerprint.json`
  Expected: JSON identical to `$S/ref1.json` (`cmp`).
- [ ] **Step 4:** Evaluation reference: one small set through the real CLI.
  Write `$S/before/eval_0055.yaml`: a copy of `outputs/robust-ensemble/robust_p0/eval.yaml`, with `label_sets` reduced to `dataset0055` and `experiment.output_dir: <S>/before/eval` and `wandb.enabled: false`.
  Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m scripts.evaluate --checkpoint outputs/robust-ensemble/training/robust_p0_20261013-172822/checkpoint_last.pt --config $S/before/eval_0055.yaml`
  Expected: `$S/before/eval/*/dataset0055/nested_oof.npz` exists.
- [ ] **Step 5:** Report references.
  Run: `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m scripts.build_results_report --help > $S/before/report_help.txt; MPLCONFIGDIR=/tmp/claude-1040/mpl PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m scripts.training_report && .venv/bin/python -m scripts.training_report_html && git status --short`
  Expected: `git status` is empty, because figure and HTML regeneration is deterministic. If it is not empty, record which files differ in `$S/before/nondeterministic.txt`. Those files are compared by content type in Task 7, not by bytes.
- [ ] **Step 6:** Edit the spec mapping rows for pack and perturbation (two modules each), then commit.
  ```bash
  git add docs/superpowers/specs/2026-10-14-repo-restructure-design.md docs/superpowers/plans/2026-10-14-repo-restructure.md
  git commit -m "docs: restructure plan; keep pack and perturbation modules separate"
  ```

### Task 2: Delete finished-study code

**Files:**
- Delete: the 11 files in "Deleted" above.
- Delete: `tests/test_production_policies.py`.
- Modify: `tests/test_training_metrics.py`. Keep only the first three tests (`test_global_terms_logged_with_actual_weights`, `test_weighted_terms_explain_total`, `test_validation_metrics_do_not_invent_training_loss`) and their imports. Delete the presentation-watcher tests: everything from `test_presentation_excludes_partial_or_merged_scorecards` on, plus `presentation_row`.
- Modify: `scripts/training_report.py`. This prepares Task 5: `from scripts.robust_ensemble import ROOT, job_oof` is about to break.

**Interfaces:**
- Produces: `spectral_lejepa.evaluation.ensemble.member_oof(member_dir: Path | str, dataset: str) -> numpy NpzFile`, and `ensemble_predictions(oofs: list, arm: str = 'layer2/flat') -> np.ndarray`, the mean over members.

- [ ] **Step 1: Write the failing test** `tests/test_ensemble.py`:
  ```python
  import json

  import numpy as np

  from spectral_lejepa.evaluation.ensemble import ensemble_predictions, member_oof


  def make_member(root, name, value):
      d = root / name
      bank = d / 'eval' / 'run_x_step10' / 'dataset0055'
      bank.mkdir(parents=True)
      (d / 'state.json').write_text(json.dumps({'checkpoint': str(root / 'training' / 'run_x' / 'c.pt')}))
      np.savez(bank / 'nested_oof.npz', y=np.arange(3.), raw=np.zeros((2, 3)), **{'layer2/flat': np.full((2, 3), value)})
      return d


  def test_member_oof_reads_the_bank_of_the_trained_checkpoint(tmp_path):
      d = make_member(tmp_path, 'm1', 1.0)
      assert member_oof(d, 'dataset0055')['layer2/flat'].shape == (2, 3)


  def test_ensemble_is_the_equal_weight_mean(tmp_path):
      oofs = [member_oof(make_member(tmp_path, f'm{v}', v), 'dataset0055') for v in (1.0, 3.0)]
      assert np.allclose(ensemble_predictions(oofs), 2.0)
  ```
- [ ] **Step 2:** Run `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q tests/test_ensemble.py`.
  Expected: FAIL, `ModuleNotFoundError: spectral_lejepa.evaluation.ensemble`.
- [ ] **Step 3:** Create `src/spectral_lejepa/evaluation/ensemble.py`:
  ```python
  """Equal-weight ensembles of members' out-of-fold probe predictions."""
  import json
  from pathlib import Path

  import numpy as np


  def member_oof(member_dir, dataset):
      """nested_oof.npz of a pipeline job: <member>/eval/<training run>_step<N>/<dataset>/nested_oof.npz."""
      member_dir = Path(member_dir)
      run = Path(json.loads((member_dir / 'state.json').read_text())['checkpoint']).parent.name
      (bank,) = (member_dir / 'eval').glob(f'{run}_step*')
      return np.load(bank / dataset / 'nested_oof.npz')


  def ensemble_predictions(oofs, arm='layer2/flat'):
      return np.mean([o[arm] for o in oofs], 0)
  ```
- [ ] **Step 4:** Run the test. Expected: 2 passed.
- [ ] **Step 5:** In `scripts/training_report.py`, replace `from scripts.robust_ensemble import ROOT, job_oof` with:
  ```python
  from spectral_lejepa.evaluation.ensemble import ensemble_predictions, member_oof as job_oof

  ROOT = Path('outputs/robust-ensemble')
  ```
  In `ensemble_scores`, replace `np.mean([x['layer2/flat'] for x in o], 0)` with `ensemble_predictions(o)`.
- [ ] **Step 6:** `git rm` the 11 study files and `tests/test_production_policies.py`, then trim `tests/test_training_metrics.py` as listed above.
- [ ] **Step 7:** Run the fast suite, then the figures.
  Run: `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q 2>&1 | tail -1 && MPLCONFIGDIR=/tmp/claude-1040/mpl PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m scripts.training_report && git status --short docs/figures`
  Expected: the suite passes. Before deleting, record the number of collected tests in `tests/test_production_policies.py` and in the removed watcher tests with `pytest --collect-only -q <file>`. The new count must equal 199 + 2 minus those two numbers. Record it as N2. `docs/figures` is unchanged.
- [ ] **Step 8:** Commit: `refactor: delete finished-study orchestration; extract ensemble averaging`.

### Task 3: Move modules into the package

**Files:**
- Create: `src/spectral_lejepa/reporting/__init__.py` and `src/spectral_lejepa/reporting/sections/__init__.py`, both empty.
- Move: every row of the module map, with `git mv scripts/<old>.py src/spectral_lejepa/<new path>.py`.
- Create (scratchpad, not committed): `$S/rewrite_imports.py`.

- [ ] **Step 1:** Write `$S/rewrite_imports.py`:
  ```python
  """Rewrite scripts.<old> module paths (imports and strings) to their package paths, repo-wide."""
  import re
  import subprocess
  import sys

  MAP = {
      'pretrain': 'spectral_lejepa.training.pretrain', 'gradient_audit': 'spectral_lejepa.training.audit',
      'export_production': 'spectral_lejepa.models.export', 'pack_spectra': 'spectral_lejepa.data.pack_spectra',
      'pack_pickles': 'spectral_lejepa.data.pack_pickles', 'noise_calibration': 'spectral_lejepa.data.noise_calibration',
      'evaluate': 'spectral_lejepa.evaluation.run', 'final_model_gate': 'spectral_lejepa.evaluation.gate',
      'permutation_audit': 'spectral_lejepa.evaluation.permutation',
      'robustness_perturbations': 'spectral_lejepa.evaluation.perturbation',
      'robustness_eval': 'spectral_lejepa.evaluation.perturbation_eval',
      'build_results_report': 'spectral_lejepa.reporting.results', 'results_matrix': 'spectral_lejepa.reporting.matrix',
      'readout_variants': 'spectral_lejepa.reporting.readouts', 'report_story': 'spectral_lejepa.reporting.story',
      'scorecard_figures': 'spectral_lejepa.reporting.scorecard', 'local_wandb': 'spectral_lejepa.reporting.local_wandb',
      'architecture_diagram': 'spectral_lejepa.reporting.architecture',
      'continuation_report': 'spectral_lejepa.reporting.sections.continuation',
      'frozen_feature_report': 'spectral_lejepa.reporting.sections.frozen_features',
      'production_analysis_report': 'spectral_lejepa.reporting.sections.production_analysis',
      'projection_pilot_report': 'spectral_lejepa.reporting.sections.projection_pilot',
      'training_report': 'spectral_lejepa.reporting.training_report',
      'training_report_html': 'spectral_lejepa.reporting.training_report_html',
  }
  files = subprocess.check_output(['git', 'ls-files', 'src', 'tests', 'scripts', 'ops'], text=True).split()
  files = [f for f in files if f.endswith(('.py', '.sh'))]
  changed = 0
  for f in files:
      s = open(f).read()
      t = s
      for old, new in MAP.items():
          parent, leaf = new.rsplit('.', 1)
          t = re.sub(rf'from scripts import {old}\b', f'from {parent} import {leaf}', t)
          t = re.sub(rf'\bscripts\.{old}\b', new, t)
          t = re.sub(rf'\bscripts/{old}\.py\b', new.replace('.', '/') + '.py', t)
      if t != s:
          open(f, 'w').write(t)
          changed += 1
  print('rewrote', changed, 'files')
  sys.exit(0)
  ```
- [ ] **Step 2:** Run the moves:
  ```bash
  mkdir -p src/spectral_lejepa/reporting/sections
  touch src/spectral_lejepa/reporting/__init__.py src/spectral_lejepa/reporting/sections/__init__.py
  git mv scripts/pretrain.py src/spectral_lejepa/training/pretrain.py
  git mv scripts/gradient_audit.py src/spectral_lejepa/training/audit.py
  git mv scripts/export_production.py src/spectral_lejepa/models/export.py
  git mv scripts/pack_spectra.py src/spectral_lejepa/data/pack_spectra.py
  git mv scripts/pack_pickles.py src/spectral_lejepa/data/pack_pickles.py
  git mv scripts/noise_calibration.py src/spectral_lejepa/data/noise_calibration.py
  git mv scripts/evaluate.py src/spectral_lejepa/evaluation/run.py
  git mv scripts/final_model_gate.py src/spectral_lejepa/evaluation/gate.py
  git mv scripts/permutation_audit.py src/spectral_lejepa/evaluation/permutation.py
  git mv scripts/robustness_perturbations.py src/spectral_lejepa/evaluation/perturbation.py
  git mv scripts/robustness_eval.py src/spectral_lejepa/evaluation/perturbation_eval.py
  git mv scripts/build_results_report.py src/spectral_lejepa/reporting/results.py
  git mv scripts/results_matrix.py src/spectral_lejepa/reporting/matrix.py
  git mv scripts/readout_variants.py src/spectral_lejepa/reporting/readouts.py
  git mv scripts/report_story.py src/spectral_lejepa/reporting/story.py
  git mv scripts/scorecard_figures.py src/spectral_lejepa/reporting/scorecard.py
  git mv scripts/local_wandb.py src/spectral_lejepa/reporting/local_wandb.py
  git mv scripts/architecture_diagram.py src/spectral_lejepa/reporting/architecture.py
  git mv scripts/continuation_report.py src/spectral_lejepa/reporting/sections/continuation.py
  git mv scripts/frozen_feature_report.py src/spectral_lejepa/reporting/sections/frozen_features.py
  git mv scripts/production_analysis_report.py src/spectral_lejepa/reporting/sections/production_analysis.py
  git mv scripts/projection_pilot_report.py src/spectral_lejepa/reporting/sections/projection_pilot.py
  git mv scripts/training_report.py src/spectral_lejepa/reporting/training_report.py
  git mv scripts/training_report_html.py src/spectral_lejepa/reporting/training_report_html.py
  .venv/bin/python $S/rewrite_imports.py
  ```
- [ ] **Step 3:** Check for leftovers.
  Run: `git grep -n -E "\bscripts[./]" -- src tests scripts ops`
  Expected: only `scripts/launch_final_model.sh` lines that call `scripts/pretrain.py` by file path, which Task 6 handles. Fix any other hit by hand, as an import or string change only.
- [ ] **Step 4:** Run the fast suite.
  Expected: N2 tests, all passing.
- [ ] **Step 5:** Commit: `refactor: move reusable scripts into the spectral_lejepa package`. Confirm with `git show --stat HEAD` that the moved files appear as renames.

### Task 4: Single CLI

**Files:**
- Create: `src/spectral_lejepa/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `spectral_lejepa.__main__.COMMANDS: dict[str, str]` (subcommand to module path), and `main(argv: list[str] | None = None) -> None`.

- [ ] **Step 1: Write the failing test** `tests/test_cli.py`:
  ```python
  import importlib
  import subprocess
  import sys

  import pytest

  from spectral_lejepa.__main__ import COMMANDS


  @pytest.mark.parametrize('name', sorted(COMMANDS))
  def test_every_command_module_has_main(name):
      assert callable(importlib.import_module(COMMANDS[name]).main)


  @pytest.mark.parametrize('name', ['pretrain', 'evaluate', 'export', 'gate', 'report', 'pack'])
  def test_help_works_from_the_repo_root(name):
      r = subprocess.run([sys.executable, '-m', 'spectral_lejepa', name, '--help'], capture_output=True, text=True)
      assert r.returncode == 0, r.stderr
      assert 'usage' in r.stdout.lower()


  def test_unknown_command_lists_commands():
      r = subprocess.run([sys.executable, '-m', 'spectral_lejepa', 'nope'], capture_output=True, text=True)
      assert r.returncode == 2 and 'evaluate' in r.stderr
  ```
- [ ] **Step 2:** Run `PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m pytest -q tests/test_cli.py`.
  Expected: FAIL, `No module named spectral_lejepa.__main__`.
- [ ] **Step 3:** Create `src/spectral_lejepa/__main__.py`:
  ```python
  """python -m spectral_lejepa <command> [args]: one entry point for every pipeline step."""
  import importlib
  import sys

  COMMANDS = {
      'pretrain': 'spectral_lejepa.training.pretrain',
      'gradient-audit': 'spectral_lejepa.training.audit',
      'export': 'spectral_lejepa.models.export',
      'pack': 'spectral_lejepa.data.pack_spectra',
      'pack-pickles': 'spectral_lejepa.data.pack_pickles',
      'noise-calibration': 'spectral_lejepa.data.noise_calibration',
      'evaluate': 'spectral_lejepa.evaluation.run',
      'gate': 'spectral_lejepa.evaluation.gate',
      'permutation-audit': 'spectral_lejepa.evaluation.permutation',
      'perturbation': 'spectral_lejepa.evaluation.perturbation',
      'report': 'spectral_lejepa.reporting.results',
      'training-report': 'spectral_lejepa.reporting.training_report',
      'training-report-html': 'spectral_lejepa.reporting.training_report_html',
  }


  def main(argv=None):
      argv = sys.argv[1:] if argv is None else argv
      if not argv or argv[0] not in COMMANDS:
          sys.stderr.write('usage: python -m spectral_lejepa {' + ','.join(COMMANDS) + '} [args]\n')
          sys.exit(2)
      sys.argv = [f'spectral_lejepa {argv[0]}', *argv[1:]]   # modules parse sys.argv themselves
      importlib.import_module(COMMANDS[argv[0]]).main()


  if __name__ == '__main__':
      main()
  ```
- [ ] **Step 4:** Run the test. Expected: all pass. If a module's `main` takes no `--help` (for example `noise_calibration` or `training_report`), it is not in the help list. Only the six listed commands are checked.
- [ ] **Step 5:** `training_report` dispatches `fetch` on `sys.argv[1:] == ['fetch']` inside its `if __name__` block, not in `main()`. Move that one line into `main()`: rename the current `main` to `figures`, and add `def main(): fetch() if sys.argv[1:] == ['fetch'] else figures()`. This is the only behavioural-code edit allowed in this task, and the `if __name__` block then calls `main()`.
  Re-run: `MPLCONFIGDIR=/tmp/claude-1040/mpl PYTHONPATH=$PWD/src:$PWD .venv/bin/python -m spectral_lejepa training-report && git status --short docs/figures`. Expected: no changes.
- [ ] **Step 6:** Commit: `feat: python -m spectral_lejepa command-line entry point`.

### Task 5: Launcher to ops/, remove scripts/, tests mirror the package

**Files:**
- Move: `scripts/launch_final_model.sh` to `ops/launch_final_model.sh`, and delete `scripts/__init__.py` and `scripts/`.
- Move: tests into `tests/{data,models,training,evaluation,reporting}/`, each with an empty `__init__.py`. `conftest.py`, `__init__.py`, `test_cli.py`, `test_config.py` and `test_wandb_utils.py` stay at the top level.
- Test: `tests/test_launcher.py` (new).

- [ ] **Step 1: Write the failing test** `tests/test_launcher.py`:
  ```python
  import subprocess


  def test_final_launcher_dry_run_uses_the_package_cli():
      out = subprocess.run(['bash', 'ops/launch_final_model.sh', '--dry-run'], capture_output=True, text=True, check=True).stdout
      assert 'python -m spectral_lejepa pretrain --config configs/final_model.yaml' in out
      assert 'python -m spectral_lejepa export' in out and 'python -m spectral_lejepa evaluate' in out
      assert 'scripts' not in out
  ```
- [ ] **Step 2:** Run it. Expected: FAIL, because `ops/launch_final_model.sh` does not exist.
- [ ] **Step 3:** `git mv scripts/launch_final_model.sh ops/launch_final_model.sh`. In the file, replace:
  - `.venv/bin/python scripts/pretrain.py` → `.venv/bin/python -m spectral_lejepa pretrain`, in both the dry-run line and the real line;
  - `.venv/bin/python -m scripts.export_production` → `.venv/bin/python -m spectral_lejepa export`;
  - `.venv/bin/python -m scripts.evaluate` → `.venv/bin/python -m spectral_lejepa evaluate`;
  - `.venv/bin/python -m scripts.final_model_gate` → `.venv/bin/python -m spectral_lejepa gate`;
  - `.venv/bin/python -m scripts.build_results_report` → `.venv/bin/python -m spectral_lejepa report`;
  - in the usage line, `bash scripts/launch_final_model.sh` → `bash ops/launch_final_model.sh`.

  Task 3's rewrite may already have turned some of these into `spectral_lejepa.<module>` paths. Make all five consistent with the CLI form shown. Then `git rm scripts/__init__.py`.
- [ ] **Step 4:** Move the tests. Group them by the package they exercise:
  - data: `test_loader.py`, `test_pack.py`;
  - models: `test_model.py`, `test_tokenizer.py`, `test_masking.py`, `test_final_model.py` (export);
  - training: `test_loss.py`, `test_train_smoke.py`, `test_training_metrics.py`, `test_diagnostics.py`, `test_gradient_audit.py`, `test_robustness_continuation.py` (rename it `test_robust_term.py`), `test_paired_long.py`;
  - evaluation: `test_eval_data.py`, `test_eval_equivalence.py`, `test_evaluate_script.py`, `test_nested.py`, `test_pairing.py`, `test_probe_torch.py`, `test_permutation_audit.py`, `test_final_model_gate.py`, `test_ensemble.py`;
  - reporting: `test_architecture_diagram.py`, `test_continuation_study.py`, `test_figure_logging.py`, `test_frozen_feature_report.py`, `test_local_wandb.py`, `test_production_analysis_report.py`, `test_projection_pilot_report.py`, `test_readout_variants.py`, `test_results_matrix.py`, `test_results_report.py`, `test_scorecard_figures.py`.

  Add an empty `__init__.py` in each folder. Then fix test-to-test imports. For example, `from tests.test_train_smoke import REPO, TINY_OVERRIDES` becomes `from tests.training.test_train_smoke import ...`.
  Find them with: `git grep -n "from tests\." tests`.
  Fix `Path(__file__).parents[1]` repo-root lookups to `parents[2]` in moved tests. Find them with: `git grep -n "parents\[1\]" tests`.
- [ ] **Step 5:** Run the fast and slow suites.
  Expected: everything passes. The fast count is N2 + 21: `test_cli` has 13 + 6 + 1 = 20 cases, plus 1 launcher test. The slow suite still shows `8 passed`.
- [ ] **Step 6:** Commit: `refactor: launcher to ops/, remove scripts/, tests mirror the package`.

### Task 6: Docs follow the code

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/TRAINING_REPORT.md` and its regenerated `.html`, `docs/PRODUCTION_RECOMMENDATION.md`.
- Modify: each `docs/superpowers/specs/*.md` and `docs/superpowers/plans/*.md` that mentions `scripts`. Add one line under the title: `> Paths before 2026-10-14 refer to scripts/; see specs/2026-10-14-repo-restructure-design.md.` Do not add it to the two 2026-10-14 restructure documents.

- [ ] **Step 1:** List the hits.
  Run: `git grep -n -E "scripts[./]" -- README.md CLAUDE.md docs/*.md`
- [ ] **Step 2:** Rewrite them to the CLI form:
  - README commands: `uv run python -m spectral_lejepa evaluate ...`, `... export ...`, `... pretrain --config configs/pretrain.yaml`, and `systemd-run ... bash ops/launch_final_model.sh`.
  - The architecture diagram line becomes `spectral_lejepa.reporting.architecture.render_dataflow()`.
- [ ] **Step 3:** Rewrite the CLAUDE.md repository map:
  ```markdown
  - `src/spectral_lejepa/`: package; `python -m spectral_lejepa <command>` runs pretrain, evaluate, export, gate, report, training-report, pack, audits.
  - `data/`: corpus loaders, packing, perturbations and noise calibration.
  - `models/`: tokenizer, masking, encoder and predictor, TorchScript export.
  - `training/`: objective, loop, checkpointing, diagnostics, gradient terms and audit.
  - `evaluation/`: feature banks, nested probes, ensembles, gate, permutation audit, perturbation suite.
  - `reporting/`: results report and sections, scorecards, training report (figures + HTML), architecture diagram.
  - `ops/launch_final_model.sh`: durable final pipeline. `configs/`: YAML configs. `tests/` mirrors the package.
  ```
  Remove the "2026-10-13 studies" line and the presentation-watcher line. Keep the retired-options line and the `plans/` line.
- [ ] **Step 4:** Add the historical-path note to every older spec and plan that the Step 1 grep lists.
- [ ] **Step 5:** In `docs/TRAINING_REPORT.md`, change `python -m scripts.training_report` to `python -m spectral_lejepa training-report`. Then rebuild the HTML.
  Run: `PYTHONPATH=$PWD/src .venv/bin/python -m spectral_lejepa training-report-html`
- [ ] **Step 6:** Check for leftovers.
  Run: `git grep -n -E "\bscripts[./]" -- src tests ops configs README.md CLAUDE.md docs/*.md`
  Expected: nothing.
- [ ] **Step 7:** Commit: `docs: commands and repository map follow the package layout`.

### Task 7: Verify and push

- [ ] **Step 1:** Fingerprint. Copy `$S/equiv.py` to `$S/equiv_after.py` and change its three imports:
  - `from spectral_lejepa.evaluation.perturbation import block2_flat`
  - `from spectral_lejepa.evaluation.run import model_input`
  - `from tests.training.test_robust_term import packed_cfg, ROBUST`

  Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 .venv/bin/python $S/equiv_after.py > $S/after_fingerprint.json && cmp $S/after_fingerprint.json $S/before/fingerprint.json`
  Expected: no output from `cmp`, meaning identical.
- [ ] **Step 2:** Evaluation reproduction. Use the same config as Task 1, with `output_dir` set to `$S/after/eval`.
  Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/src .venv/bin/python -m spectral_lejepa evaluate --checkpoint outputs/robust-ensemble/training/robust_p0_20261013-172822/checkpoint_last.pt --config $S/after/eval_0055.yaml`
  Then compare every array in the two `nested_oof.npz` files:
  `.venv/bin/python -c "import numpy as np,glob;a,b=[np.load(glob.glob(p)[0]) for p in ('$S/before/eval/*/dataset0055/nested_oof.npz','$S/after/eval/*/dataset0055/nested_oof.npz')];assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files);print('identical')"`
  Expected: `identical`.
- [ ] **Step 3:** Rebuild the reports.
  Run: `PYTHONPATH=$PWD/src .venv/bin/python -m spectral_lejepa report --help | diff - $S/before/report_help.txt`
  Expected: only the `usage:` prog name differs.
  Run: `MPLCONFIGDIR=/tmp/claude-1040/mpl PYTHONPATH=$PWD/src .venv/bin/python -m spectral_lejepa training-report && git status --short`
  Expected: clean.
- [ ] **Step 4:** Run the full fast and slow suites, then `ls scripts 2>&1`.
  Expected: everything passes, and `ls` reports "No such file or directory".
- [ ] **Step 5:** `git push origin main`. Report the commit list and the verification results.
