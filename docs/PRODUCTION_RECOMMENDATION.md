# Production recommendation — 2026-10-12

Keep the existing export unchanged. For production validation, prioritize the earlier 48-patch seed2 checkpoint with SIGReg λ=0.05 and block-2 flat features: it remains the strongest measured fixed candidate under the primary linear benchmark. No continuation, normalization change or broader decoder menu has established a replacement. Do not spend on another duration-only full run from these findings.

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

All four pipelines completed. Full R²±bootstrapSD, matched random controls and continuation-seed SD are in `outputs/training-progress/results_report.html`. Preregistered preference remains fixedblock2=`duration_only`,menu=`inconclusive`; practical recommendation is to retain the original until independent evidence warrants replacement.

## Completed frozen-feature follow-up

All eight checkpoints now have mean, seg4 and flat outputs at every block (0–6), plus block6 before LayerNorm. Block2-flat ranks first by raw-win count and mean gain in every checkpoint; later flat readouts did not replace it. Final LayerNorm effects vary by checkpoint/readout, so removing it is not a reliable fix.

The secondary standardized Ridge/PLS/RBF menu wins 4–6/8 against its same-head raw branch. That branch differs from the primary linear raw benchmark: on labeled_data it scores 0.7625 versus primary raw 0.8301, while seed101 trained scores 0.9588 under both procedures. Keep both raw references visible; this is not a new encoder gain. Seed2's secondary-head mean R² is 0.00567 below its primary fixed embedding score.

Replaying the predeclared four-readout menu from saved inner scores and paired predictions reproduces seed101's previous result. Earlier seeds 0/1/2 yield 7/6/6 wins; the menu is not a consistent upgrade over the 7/8 fixed interface. No 8/8 winner is established.

An exploratory union replay retains the original linear recipes alongside the new heads for raw, trained and random features, selecting each branch by saved inner-fold R². It still wins only 5–6/8 against its union raw comparator; seed2 trained predictions regress by 0.00994 mean R² versus its primary fixed2 score. Extra decoder choices are not a dependable upgrade.

Retain 48p/block2-flat/SIGReg 0.05 as the production architecture shortlist. Seed2 remains the strongest measured primary-linear fixed candidate; the current seed101 export remains unchanged and unqualified. The next training hypothesis is a matched regularization-only projection-space pilot, specified in `docs/superpowers/specs/2026-10-12-projection-space-pilot.md`; it has not been launched.
