# Frozen Feature Study Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; lighter focused head implementation/review.

**Goal:** Complete feature coverage, isolate final LayerNorm effects and test fair downstream heads without retraining.
**Architecture:** Reuse checkpoints, banklayout,readoutpooling,nestedlinearCV,paireduncertainty andofflineplots. Separateheadmodule ownsitsnestedselection. Per-jobstateandregisteredhashes supportdurablequeueing.
**Tech Stack:** ExistingPyTorch/NumPy/scikit-learn/W&B/systemd/matplotlib.
**Spec:** `docs/superpowers/specs/2026-10-12-frozen-feature-study.md`.

## Constraints / review focus

- Original weights,exports,gates andsavedscorecards read-only.
- Pre-LayerNormfeatures capturedfromsamefinalblock aspostnorm;stage6notrenumbered.
- Preserve raw/y ordering andoldfeature/OOFpairing;cachedfamiliesrequireverifiedlineage.
- AlllearnedheadtransformsincludingPLSfitinsideeachinnertrainingsplit. Outer-testlabels cannotselectrecipes.
- Primarylinearbenchmark andsecondaryheadcomparison remainseparate;matchedraw/randomtreatmentandbootstrapversusseeduncertainty explicit.

## Tasks

- [x] Testfullfeatureextraction/pre-postnormboundary andold-bankpairing;implementfrozenworker/registration.
- [x] Implement/testsmallnestedheadmenu withouter-labelperturbationregression (lighteragent).
- [x] Integratesupplementalblock/normalization/headfigures intoofflineHTML;pendingresultsstaypending.
- [x] Focusedtests,realcheckpoint/smallsetsmoke,read-onlyreview;commitregistrationbeforelaunch.
- [ ] LaunchdurablesevenGPUqueues/eightcheckpointjobs;verifyactualprogress,W&BIDs,finitevalues;recordstatus.

## Ledger

- Latestuserauthorizationcoversplanandtests. Priororderedrecommendationmakesprojectorpilotconditionalonfrozenresults;notlaunchedinthisstage. Newstudyexploratory;noautomaticreleasepromotion.
- Newfeatureboundary/rowreorderingtestsobservedRED→GREEN;lighterheadagentobservedRED→GREENandverifiedouter-labelperturbation/scaler boundaries.23focusedtests passed. Training/modelcodeunchanged.
- RealGPU priorseed2/0055 smoke passed:24fixedarms,block2R².6619529858059501exactlymatchescachedresult. Pre/postnormgains andheadmenu finite. SecondaryrawheadR².69227versustrained.70412/random.66396;this isasmoketest,n25,notapromotionresult. Alloldraw/block2OOFs pairedwithnewcalculation.
- Registeredsource/checkpointhashes andsevenhealthyGPUUUIDs in`docs/superpowers/registrations/2026-10-12-frozen-feature-study.json`. Slot0queuesprior0thenlow203;othersrunonecheckpoint. Reporttracksqueued/completedsetsandseparatesnewlinearreadoutsfromsecondaryheads.
- Independentlighterreview foundno methodology/path/leakage/gate-mutationblocker. Sequentialslot0queueconfirmed. SecondaryRBFheadsrequire substantialCPUworkon4716rows;OMP/OpenBLAS/MKLcaps4. HostGPUavailabilityverifiedallsevenidlebeforelaunch;knownphysical3excluded.
