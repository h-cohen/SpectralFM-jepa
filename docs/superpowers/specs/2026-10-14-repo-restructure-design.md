# Repository restructure — design, 2026-10-14

Status: proposed, awaiting user review.

## Goal

The repository should read as a research-engineering project. Reusable code lives in the `spectral_lejepa` package, and one CLI is the entry point. `scripts/` currently holds 36 files: CLIs, report sections, audits and finished-study orchestration. It is removed. The behaviour of training, evaluation, export, gate and reports does not change.

## Decisions (user, 2026-10-14)

- No `scripts/` folder. The new similarity and noise evaluation is part of the standard evaluation (separate spec: `2026-10-14-embedding-similarity-noise-design.md`).
- One-off study code is deleted. Its results stay in specs and `outputs/`, and git history keeps the code. Reusable parts move into the package first.

## Target layout

```
src/spectral_lejepa/
  __main__.py      python -m spectral_lejepa {pretrain,evaluate,export,gate,report,training-report,pack}
  config.py
  data/            loader.py, perturb.py, pack_spectra.py, pack_pickles.py, noise_calibration.py
  models/          tokenizer.py, masking.py, vit_1d.py, export.py (export_production)
  training/        loss.py, trainer.py, checkpoint.py, diagnostics.py, gradient_terms.py, audit.py (gradient_audit)
  evaluation/      bank.py, data.py, nested.py, probe.py, probe_torch.py, run.py (scripts/evaluate main),
                   ensemble.py (member OOF averaging), gate.py (final_model_gate),
                   permutation.py (permutation_audit), perturbation.py (robustness_perturbations),
                   perturbation_eval.py (robustness_eval)
  reporting/       results.py (build_results_report), matrix.py, readouts.py, story.py, scorecard.py,
                   local_wandb.py, architecture.py,
                   sections/{continuation,frozen_features,production_analysis,projection_pilot}.py,
                   training_report.py (figures), training_report_html.py
  utils/           io.py, wandb.py
ops/               launch_final_model.sh (calls python -m spectral_lejepa ...)
configs/           unchanged
tests/             data/, models/, training/, evaluation/, reporting/ mirroring the package
```

## File mapping

| Current `scripts/` file | Destination |
|---|---|
| pretrain.py, evaluate.py | `__main__.py` subcommands; evaluate body becomes `evaluation/run.py` |
| export_production.py | `models/export.py` |
| final_model_gate.py | `evaluation/gate.py` |
| gradient_audit.py | `training/audit.py` |
| permutation_audit.py | `evaluation/permutation.py` |
| robustness_perturbations.py, robustness_eval.py | `evaluation/perturbation.py`, `evaluation/perturbation_eval.py` |
| noise_calibration.py | `data/noise_calibration.py` |
| pack_spectra.py, pack_pickles.py | `data/pack_spectra.py`, `data/pack_pickles.py` |
| build_results_report.py, results_matrix.py, readout_variants.py, report_story.py, scorecard_figures.py, local_wandb.py, architecture_diagram.py | `reporting/` (names above) |
| continuation_report.py, frozen_feature_report.py, production_analysis_report.py, projection_pilot_report.py | `reporting/sections/` |
| training_report.py, training_report_html.py | `reporting/`; `robust_ensemble.job_oof` moves to `evaluation/ensemble.py` |
| launch_final_model.sh | `ops/` |
| robust_ensemble.py, robustness_continuation.py, robustness_continuation_analysis.py, seed_ensemble.py, production_policies.py, chemometric_baselines.py, diagnose_0106.py, memorization_check.py, training_progress.py, launch_robust_ensemble.sh, launch_robustness_continuation.sh | deleted, along with tests that cover only them |

## Rules

- Moves keep code unchanged except for imports, plus the extraction named above. Each commit is a pure move, or one extraction, with no logic edits.
- Docs follow the code: README commands, CLAUDE.md repo map and execution rules, the training report, and spec and plan pointers to the moved paths. Historical specs keep their text and get one note: "Paths before 2026-10-14 refer to `scripts/`; see the restructure spec."
- No running job depends on `scripts/`. Before the change, `systemctl --user list-units 'spectralfm-*'` must show nothing active.

## Verification

- All tests pass: 199 fast plus 8 slow today, minus the deleted study tests.
- The equivalence fingerprint (block-2 features of three checkpoints plus tiny-train losses) is byte-identical before and after.
- `python -m spectral_lejepa evaluate` on one small set reproduces the saved `nested_oof.npz` exactly.
- `python -m spectral_lejepa report` rebuilds the results report and `training-report` rebuilds the figures and HTML. Outputs match apart from timestamps.
- `grep -r "scripts\." src tests ops configs README.md CLAUDE.md docs/*.md` returns nothing.
