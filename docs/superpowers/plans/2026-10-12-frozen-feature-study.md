# Frozen Feature Study Implementation Plan

Status (2026-10-13): completed plan. Historical record; current state is in CLAUDE.md.

> **For agentic workers:** Use superpowers:executing-plans inline; lighter focused head implementation/review.

**Goal:** Complete feature coverage, isolate final LayerNorm effects and test fair downstream heads without retraining.
**Architecture:** Reuse checkpoints, bank layout, readout pooling, nested linear CV, paired uncertainty and offline plots. Separate head module owns its nested selection. Per-job state and registered hashes support durable queueing.
**Tech Stack:** Existing PyTorch/NumPy/scikit-learn/W&B/systemd/matplotlib.
**Spec:** `docs/superpowers/specs/2026-10-12-frozen-feature-study.md`.

## Constraints / review focus

- Original weights, exports, gates and saved scorecards read-only.
- Pre-LayerNorm features captured from same final block as postnorm; stage 6 not renumbered.
- Preserve raw/y ordering and old feature/OOF pairing; cached families require verified lineage.
- All learned head transforms including PLS fit inside each inner training split. Outer-test labels cannot select recipes.
- Primary linear benchmark and secondary head comparison remain separate; matched raw/random treatment and bootstrap versus seed uncertainty explicit.

## Tasks

- [x] Test full feature extraction/pre-postnorm boundary and old-bank pairing; implement frozen worker/registration.
- [x] Implement/test small nested head menu with outer-label perturbation regression (lighter agent).
- [x] Integrate supplemental block/normalization/head figures into offline HTML; pending results stay pending.
- [x] Focused tests, real checkpoint/small set smoke, read-only review; commit registration before launch.
- [x] Launch durable seven GPU queues/eight checkpoint jobs; verify actual progress, W&B IDs, finite values; record status.

## Ledger

- Latest user authorization covers plan and tests. Prior ordered recommendation makes projector pilot conditional on frozen results; not launched in this stage. New study exploratory; no automatic release promotion.
- New feature boundary/row reordering tests observed RED→GREEN; lighter head agent observed RED→GREEN and verified outer-label perturbation/scaler boundaries. 23 focused tests passed. Training/model code unchanged.
- Real GPU prior seed2/0055 smoke passed: 24 fixed arms, block2 R² .6619529858059501 exactly matches cached result. Pre/postnorm gains and head menu finite. Secondary raw head R² .69227 versus trained .70412/random .66396; this is a smoke test, n25, not a promotion result. All old raw/block2 OOFs paired with new calculation.
- Registered source/checkpoint hashes and seven healthy GPU UUIDs in `docs/superpowers/registrations/2026-10-12-frozen-feature-study.json`. Slot0 queues prior0 then low203; others run one checkpoint. Report tracks queued/completed sets and separates new linear readouts from secondary heads.
- Independent lighter review found no methodology/path/leakage/gate-mutation blocker. Sequential slot0 queue confirmed. Secondary RBF heads require substantial CPU work on 4716 rows; OMP/OpenBLAS/MKL caps 4. Host GPU availability verified all seven idle before launch; known physical3 excluded.
- Preregistration/code commit 34ffe23 precedes launch. Seven `spectralfm-frozen-feature-slot0..6` services active; all seven first checkpoint jobs completed 0055 with finite JSON outputs and cache checks; baseline101 also completed 0106. Low203 queued behind prior0 on slot0; no GPU oversubscription.
- W&B prior0 `t1mlyvqp`, prior1 `hu9k300m`, prior2 `z1oil93q`, baseline101 `7bra67dc`, control202 `wtk3pwk2`, control203 `dxxnyjnp`, low202 `hihc5um7`. Report tracks progress and will add full measured plots per completed job. Projector pilot awaits full frozen results; original export/gate unchanged.

## Completed analysis

- All 8 pipelines completed: 64 linear+64 head dataset outputs. Every checkpoint's top fixed readout by raw wins/mean Δ is block2-flat; prior seed2 remains 7/8, +.089366. All other new late flat/pre-norm families failed to replace it. LN changes mixed across seeds/readouts.
- Secondary heads win 4–6/8 against same-head raw. Standardized-only raw is not primary raw: LD .7625 vs .8301; seed101 trained .9588 both. Primary and secondary raw references now both plotted. No encoder improvement inferred from weakened raw baseline.
- Menu selector replay uses fixed-arm inner choices and paired OOFs, not outer test labels. Exact baseline101 reproduction verified. Prior0/1/2 menu 7/6/6 wins; no single 8/8 winner. Saved `menu_replication.json`, `results_summary.json/.csv` under study outputs.
- Independent lighter review confirmed all completed counts, block2 dominance, LN mixed results and raw baseline caveat. Next projection-space pilot design saved `docs/superpowers/specs/2026-10-12-projection-space-pilot.md`; no automatic training/export promotion. Original gate unchanged.
- Expanded head union replay retains primary recipes and same inner fold selection for all arms; 5–6/8 wins, prior seed2 trained mean −.009939 versus primary fixed. Verified all 8 checkpoint SHA256s and 128 result source hashes, shared random fixed2 row identical; 13 focused tests passed. Summary W&B `md2at1yc`; offline HTML has complete primary/secondary/union R²±SD matrices and eight-checkpoint flat depth plot.
