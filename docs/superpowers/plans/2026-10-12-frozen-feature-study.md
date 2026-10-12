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
- [x] LaunchdurablesevenGPUqueues/eightcheckpointjobs;verifyactualprogress,W&BIDs,finitevalues;recordstatus.

## Ledger

- Latestuserauthorizationcoversplanandtests. Priororderedrecommendationmakesprojectorpilotconditionalonfrozenresults;notlaunchedinthisstage. Newstudyexploratory;noautomaticreleasepromotion.
- Newfeatureboundary/rowreorderingtestsobservedRED→GREEN;lighterheadagentobservedRED→GREENandverifiedouter-labelperturbation/scaler boundaries.23focusedtests passed. Training/modelcodeunchanged.
- RealGPU priorseed2/0055 smoke passed:24fixedarms,block2R².6619529858059501exactlymatchescachedresult. Pre/postnormgains andheadmenu finite. SecondaryrawheadR².69227versustrained.70412/random.66396;this isasmoketest,n25,notapromotionresult. Alloldraw/block2OOFs pairedwithnewcalculation.
- Registeredsource/checkpointhashes andsevenhealthyGPUUUIDs in`docs/superpowers/registrations/2026-10-12-frozen-feature-study.json`. Slot0queuesprior0thenlow203;othersrunonecheckpoint. Reporttracksqueued/completedsetsandseparatesnewlinearreadoutsfromsecondaryheads.
- Independentlighterreview foundno methodology/path/leakage/gate-mutationblocker. Sequentialslot0queueconfirmed. SecondaryRBFheadsrequire substantialCPUworkon4716rows;OMP/OpenBLAS/MKLcaps4. HostGPUavailabilityverifiedallsevenidlebeforelaunch;knownphysical3excluded.
- Preregistration/codecommit34ffe23precedeslaunch. Seven`spectralfm-frozen-feature-slot0..6`servicesactive;allsevenfirstcheckpointjobscompleted0055withfiniteJSONoutputs andcachechecks;baseline101also completed0106. Low203queued behindprior0 onslot0;noGPUoversubscription.
- W&Bprior0`t1mlyvqp`,prior1`hu9k300m`,prior2`z1oil93q`,baseline101`7bra67dc`,control202`wtk3pwk2`,control203`dxxnyjnp`,low202`hihc5um7`. Reporttracksprogressandwilladdfullmeasuredplotspercompletedjob. Projectorpilot awaitsfullfrozenresults;originalexport/gateunchanged.

## Completed analysis

- All8pipelines completed:64linear+64head datasetoutputs. Everycheckpoint'stopfixedreadoutbyrawwins/meanΔisblock2-flat;priorseed2remains7/8,+.089366. Allothernewlateflat/pre-normfamiliesfailedtoreplaceit. LNchangesmixedacrossseeds/readouts.
- Secondaryheadswin4–6/8againstsame-headraw. Standardized-onlyrawisnotprimaryraw:LD.7625vs.8301;seed101trained.9588both. Primaryandsecondaryrawreferencesnowbothplotted. Noencoderimprovementinferredfromweakenedrawbaseline.
- Menu selector replay uses fixed-arminnerchoices and pairedOOFs,notoutertestlabels. Exactbaseline101reproductionverified. Prior0/1/2menu7/6/6wins;nosingle8/8winner. Saved`menu_replication.json`,`results_summary.json/.csv`understudyoutputs.
- Independentlighterreviewconfirmedallcompletedcounts,block2dominance,LNmixedresultsandrawbaselinecaveat. Nextprojection-spacepilotdesignsaved`docs/superpowers/specs/2026-10-12-projection-space-pilot.md`;no automatictraining/exportpromotion. Originalgateunchanged.
- Expandedheadunion replayretainsprimaryrecipesandsameinnerfoldselectionforallarms;5–6/8wins,priorseed2trainedmean−.009939versusprimaryfixed. Verifiedall8checkpointSHA256s and128resultsourcehashes,sharedrandomfixed2rowidentical;13focusedtests passed. Summary W&B`md2at1yc`;offlineHTMLhascompleteprimary/secondary/unionR²±SDmatrices andeight-checkpointflatdepthplot.
