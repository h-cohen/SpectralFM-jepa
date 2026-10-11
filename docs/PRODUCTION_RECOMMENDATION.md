# Production recommendation — 2026-10-12

Retain the original seed101 checkpoint at 497,990 steps. Keep SIGReg λ=.05 and the existing block2-flat export. Do not replace it with a continuation or spend on another duration-only full run from these findings.

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
