# Frozen feature and downstream head study

Status (2026-10-13): completed. Block2-flat is the best fixed family; no export change. Historical record; current state is in CLAUDE.md.

User approved planning and running the next tests. Run frozen-model experiments before any further pretraining. No automatic export replacement, qualification override or projector training launch.

- Checkpoints: earlier 48p seeds 0/1/2, final 101, and control/low-SIGReg continuation seeds 202/203. Checkpoint SHA256s registered before launch.
- Extract mean/seg4/flat at tokenizer and blocks 1–6. Existing block6 means the output after final LayerNorm; add explicit `layer6_pre_norm` mean/seg4/flat from the input to that same LayerNorm. No weight updates or normalizer refit on evaluation rows.
- Use the existing per-checkpoint banks' raw 245 spectra and parameter_0 labels in exactly their saved order. Confirm overlap features against old banks; verify raw and block2 OOFs against old results. Reuse already measured fixed families only after those checks; new families get the existing linear 12-recipe 2×5 outer/5 inner CV seed42 and matched random-init seed0 controls.
- Missing flat readouts and pre/post LayerNorm diagnostics are fixed family comparisons, not an expanded production selector. Preserve old four-readout menu. Plot all seven outputs and explicit pre/post normalization paired gains with bootstrap SD.
- Secondary product head test on fixed block2-flat only: standardized RidgeCV, PLS (2/8/16 components), RBF kernel ridge with small predeclared alpha/gamma grid. Raw and random block2 get identical head menus. Fit standardization and PLS using inner training rows only; select head/hyperparameters inside 5 inner folds; 2×5 outer seed42. Retain primary linear raw and model OOFs as reference; do not overwrite primary benchmark or claim gain from giving only FM stronger heads.
- All results should include R²±bootstrap SD, paired raw/random/reference gains, source hashes, chosen inner recipes, rows and folds. Small set uncertainty and retrospective benchmark reuse explicit; separate labels/groups still required for release validation.
- Outputs under `outputs/frozen-feature-study/`, W&B group `frozen-feature-study`, local TMP, OMP4, single GPU linear probe worker. Eight jobs queued on seven healthy GPU UUIDs, excluding physical3; mutable per job states and serialized report builds.
- Decide projector pilot after completed frozen checks. If no repeatable fixed/readout/head benefit appears, report that and design a matched projection-space experiment; do not launch it automatically from lower prediction loss.
