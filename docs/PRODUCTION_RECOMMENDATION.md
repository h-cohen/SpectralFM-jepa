# Production recommendation

## Current status (2026-10-13)

- **Contract.** Winning means beating raw spectra on a majority (at least 5/8) of the eight datasets. Any positive gain counts. The user clarified that the goal is winning a majority, not large margins.
- **Export.** The exported model is seed101 block2-flat: 5/8 wins, mean ΔR² +.0801. It is not gate-qualified: the 0120 single-shuffle canary scored .0204, above the .020 limit.
- **Holdout candidates.** All are retrospective selections and none is gate-qualified.
  - Prior 48-patch seed2 block2-flat: 7/8 wins, mean ΔR² +.0894.
  - Four-seed ensemble (seeds 0/1/2/101, equal-weight predictions): 7/8 wins, mean ΔR² +.0858. It passed its registered criteria.
- **Robustness continuation from seed2 (seeds 401/402).** The registered primary criteria failed on clean mean non-inferiority: mean Δraw fell from .085 to .044. Under the majority contract the robust arms win 7/8 and 8/8 against raw, and 8/8 against random features. Feature drift under 1% input noise is about 6× lower. 0106 becomes a win; the 0112 gain is lost. Details: `docs/superpowers/specs/2026-10-13-robustness-continuation.md`.
- **Robust four-seed ensemble.** Running, built from robust_p0, robust_p1, robust_s101 and robust_s401. Spec: `docs/superpowers/specs/2026-10-13-robust-ensemble.md`.
- **Release blocker.** Independent labelled data with group IDs is still missing. Once it exists, score seed101, seed2, the ensemble (and the robust ensemble if it passes) once each.

## Record of 2026-10-12

The sections below are the 2026-10-12 recommendation, kept as written apart from spacing fixes.

Keep the existing export unchanged. For production validation, prioritize the earlier 48-patch seed2 checkpoint with SIGReg λ=0.05 and block-2 flat features: it remains the strongest measured fixed candidate under the primary linear benchmark. No continuation, normalization change, broader decoder menu or tested projection-space recipe has established a replacement. Do not spend on another duration-only full run from these findings.

This recommendation compares the approved seed101 candidate with its four continuations; it is not a claim that seed101 is the highest-scoring historical fixed embedding. Earlier 48-patch seed2 with block2-flat scores 7/8 wins and mean raw gain +0.08937, versus seed101's 5/8 and +0.08008. It is the strongest fixed candidate within the earlier 48-patch block2 shortlist and deserves locked independent validation alongside seed101. Choosing it now from observed benchmark scores is retrospective selection. The current verified export is seed101, whose qualification remains failed; neither benchmark leadership nor an existing export establishes production readiness.

| Training | Fixed block2: wins vs raw | Mean ΔR² vs raw | Four-readout menu: wins | Menu mean ΔR² |
|---|---:|---:|---:|---:|
| Frozen original |5/8|+.08008|7/8|+.09325|
| Extra 2 epochs, λ=.05, seed202 |6/8|+.08119|6/8|+.08461|
| Extra 2 epochs, λ=.05, seed203 |5/8|+.08021|6/8|+.08891|
| Extra 2 epochs, λ=.01, seed202 |6/8|+.07904|6/8|+.09324|
| Extra 2 epochs, λ=.01, seed203 |6/8|+.07872|7/8|+.08928|

The unchanged-loss continuation improves fixed mean R² by only +0.000622 across two seeds. It passes the literal positive-gain research criterion; this is not a demonstrated material advantage or release qualification. Its menu loses to the original on both seeds. Low-SIGReg fixed performance regresses on both; its menu also fails to exceed the original on both. Extra wins largely reflect tiny 0114 differences; win counts alone overstate progress.

Lower SIGReg improves final validation prediction MSE (~0.00737 vs ~0.01201 for controls) without consistent downstream gains. Final block-6 flat remains worse than raw on 7/8 or 8/8 continuation datasets. Gradient imbalance is diagnostic, not proof that lowering regularization solves the representation problem. Loss convergence does not establish production quality; benefits beyond this tested two-epoch continuation remain unknown.

## Recommended interface and next work

- Keep one frozen 48-patch/d256/six-block encoder. Fixed block2-flat remains the compatible 12,288-feature export. The four-readout menu (block-2 flat/mean, block-3 mean/seg4) is the leading development option for task-specific heads; select readout and probe using training-only inner validation. No per-dataset oracle choice or seed shopping.
- Validate the menu across independent data/groups before replacing the fixed interface. Raw fallback and multiscale concatenation have not demonstrated aggregate improvement. 0109 remains below raw across all four new fixed/menu results; no all-eight winner is established.
- Next required data: separate 245-value spectra and `parameter_0` labels with source/simulation/batch IDs, group-disjoint development splits and a reserved release holdout. Use these to verify readout selection stability and regressions; benchmark labels already informed this research.
- Resolve the original 0120 shuffle-42 qualification failure with a documented prospective validation protocol. It remains failed; the 100-permutation diagnostic never overwrites that result. No continuation has been promoted or exported to production.

All four pipelines completed. Full R² ± bootstrap SD, matched random controls and continuation-seed SD are in `outputs/training-progress/results_report.html`. Preregistered preference remains fixed block2 = `duration_only`, menu = `inconclusive`; practical recommendation is to retain the original until independent evidence warrants replacement.

## Completed frozen-feature follow-up

All eight checkpoints now have mean, seg4 and flat outputs at every block (0–6), plus block 6 before LayerNorm. Block2-flat ranks first by raw-win count and mean gain in every checkpoint; later flat readouts did not replace it. Final LayerNorm effects vary by checkpoint/readout, so removing it is not a reliable fix.

The secondary standardized Ridge/PLS/RBF menu wins 4–6/8 against its same-head raw branch. That branch differs from the primary linear raw benchmark: on labeled_data it scores 0.7625 versus primary raw 0.8301, while seed101 trained scores 0.9588 under both procedures. Keep both raw references visible; this is not a new encoder gain. Seed2's secondary-head mean R² is 0.00567 below its primary fixed embedding score.

Replaying the predeclared four-readout menu from saved inner scores and paired predictions reproduces seed101's previous result. Earlier seeds 0/1/2 yield 7/6/6 wins; the menu is not a consistent upgrade over the 7/8 fixed interface. No 8/8 winner is established.

An exploratory union replay retains the original linear recipes alongside the new heads for raw, trained and random features, selecting each branch by saved inner-fold R². It still wins only 5–6/8 against its union raw comparator; seed2 trained predictions regress by 0.00994 mean R² versus its primary fixed2 score. Extra decoder choices are not a dependable upgrade.

Retain 48p/block2-flat/SIGReg 0.05 as the production architecture shortlist. Seed2 remains the strongest measured primary-linear fixed candidate; the current seed101 export remains unchanged and unqualified. The matched regularization-only projection-space pilot is now complete; see the final findings below.


## Completed projection-space pilot

All four training/evaluation pipelines completed from the earlier seed2 checkpoint, with registered configs/source hashes verified and all 21 block/readout families measured on all eight datasets. One additional epoch, seeds 301/302; only token/pooled SIGReg moved through the training-only 256→512→128 MLP. Masked prediction and pooled invariance remained in the backbone.

| Arm / seed | Fixed2 wins | Mean ΔR² vs raw | Fixed2 change vs matched control | Flat6 wins | Flat6 mean ΔR² vs raw |
|---|---:|---:|---:|---:|---:|
| Starting seed2 |7/8|+.08937|—|—|—|
| Control301 |7/8|+.08648|—|1/8|−.14777|
| Projected301 |7/8|+.08145|−.00503|3/8|−.08032|
| Control302 |7/8|+.08351|—|0/8|−.19656|
| Projected302 |7/8|+.08133|−.00218|3/8|−.02568|

**Stop this tested projector recipe.** The primary criterion (at least +.005 mean R² vs control on both seeds) fails. All four retain 7/8 wins and lose 0106; none improves starting fixed2 mean performance. Block2-flat remains the strongest fixed family in every new run. The declared menu wins 7/6 for controls and 6/6 for projected runs, so it does not provide a consistent alternative.

The secondary result is real: final block6-flat gains +.06745/+.17088 mean R² over its matched controls and reaches 3/8 raw wins on both seeds. This is late-layer recovery, not a better fixed production embedding. At 0113, projected fixed2 loses −.02054±.00837 and −.01921±.01008 vs control (paired row-bootstrap SD), a repeatable concern even though the second seed narrowly misses the predeclared −.02 material-regression cutoff.

Prediction validation MSE improves about 32% (.00793/.00803 vs .01169/.01178), while backbone pooled effective rank decreases (~128–130 vs 145). Final FP32 audit token SIG/prediction gradient ratios rise to 39–44× versus 15–16×, so this head does not demonstrate reduced late regularization pressure. The control head is unused; its geometry is not a trained comparator. Total losses differ in regularization space and should not choose a winner. No claim that projector isotropy helps downstream performance is supported here.

Keep the earlier seed2 fixed2 checkpoint as the retrospective candidate for locked independent validation; keep the existing seed101 export unchanged. Do not extend this projector or start another full-duration run from these results. Next practical work is source/group-disjoint development and release evaluation, including 0106 and the original 0120 canary issue. If further research is authorized, a short, separately registered experiment could investigate gradient balance while preserving the primary interface; these findings do not identify a proven replacement recipe.

Machine analysis and paired uncertainties: `outputs/projection-pilot/results_analysis.json`. The analysis script that produced it was retired in the cleanup, so the saved JSON is the record. The report includes the starting reference, R² ± bootstrap SD, primary paired-gain plot, every block/readout depth plot, geometry/prediction histories and the research stop decision. Bootstrap SD reflects row sampling under the reused benchmark, not uncertainty on unseen populations or seed selection.

Final paired summary and six measured figures published to W&B: https://wandb.ai/hcohen/spectralfm-lejepa/runs/e72aiwt3.
