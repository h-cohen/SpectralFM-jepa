# SpectralFM-LeJEPA

## Contract

- Train a frozen encoder whose nested-CV `parameter_0` R² exceeds raw spectra on a majority of eight datasets: labeled_data, 0055, 0106, 0109, 0112, 0113, 0114, 0120.
- Any positive ΔR² is a win; ≥ +0.05 is a strong win. Keep raw and architecture/readout-matched random controls. Do not change the scoring rule.
- Labels never enter pretraining. All eligible spectra may be used, including evaluation spectra; evaluation is transductive.
- Select probe recipes inside inner CV. Distinguish adaptive layer/readout scores from fixed embeddings. Do not select a production checkpoint using outer-test scores.

## Current candidate

- Plan: `docs/superpowers/plans/2026-10-09-final-production-candidate.md`.
- Configs: `configs/final_model.yaml`, `configs/eval_final_model.yaml`.
- 48 patches, d256, 6 blocks; global normalization; 75% random masking; batch 256; SIGReg 0.05; pooled weight 1; expected 50/50 source sampling. AdamW 5e-4, ten epochs / 497,990 steps, seed 101.
- Export candidate: block-2 flat, 12,288 features; all six blocks train. Other blocks remain available in the full checkpoint for analysis.
- Service: `spectralfm-final-model-s101-recovery`; W&B run `ylymmcnn`, group `final-model-fixed-block2`.
- Output: `outputs/final-model/`. Pipeline: training → TorchScript export → fixed-readout evaluation → benchmark gate → report.
- Gate: eight finite sets, at least five raw wins, positive mean gains versus raw/random, all canaries passing. Passing is benchmark qualification; deployment-domain validation remains separate.
- Seed101 completed: 5/8 wins, mean ΔR² +0.080077 vs raw / +0.072296 vs random. Export verified at 497,990 steps. Not qualified: 0120 shuffled-label R² 0.0204456 exceeds the existing 0.020 limit. Threshold unchanged; final report refresh repaired.
- Prior 48-patch block-2 flat features win 7/8 in each of three seeds; 0106 loses. This retrospective readout nomination is not an architectural requirement; the fresh seed101 replication wins 5/8.
- Production analysis: `docs/superpowers/plans/2026-10-11-production-analysis.md`. Service `spectralfm-production-readouts` evaluates all21readouts, then controlled policies and refreshes the report. Output `outputs/production-analysis/`; W&B group `production-analysis`. Original qualification unchanged.
- Gradient/permutation audits completed: block2prediction/SIGReg gradient norms are comparable; block6SIGReg norms are~18×prediction. Real-label R² exceeds100new permutations for raw/block2-flat; original shuffle42failure repeats and remains unresolved. Separate labelled development/deployment data requested before selecting further training; no new training launched.
- All-block/policy analysis finished successfully. Four-readout menu wins7/8 (+.093253mean raw gain),fixed2flat5/8;rawfallback/multiscale do not improve aggregate menu. New0106/0114wins are small relative to uncertainty;0109still loses. Final6flat wins2/8. Proposed matched continuation tests duration versuslowerSIGReg;independentdevelopmentcriterion missing,no launch/promotion. Policy W&B`hcwixghf`;report updated with conclusions.
- User authorized continuation experiments:fourdurablejobs,seeds202/203×lambda.05/.01,99598steps,freshAdamW,samefrozen101weights,LR5e-5→5e-7/1000warmup. HealthyphysicalGPUs0/1/2/4selectedbyUUID. Plan`docs/superpowers/plans/2026-10-11-continuation-study.md`;outputs`outputs/continuation-study`;W&B group`continuation-study`. Pipelineautomaticallyauditsgradients,evaluatesdepth/fixed/menu+raw/random,andrefreshesreport. Originalexport/gateunchanged;independentlabelsabsent,researchonly.
- Continuationscompleted;recommendation`docs/PRODUCTION_RECOMMENDATION.md`:retainoriginal101/lambda.05/fixedexport,stopduration-onlysweeps. Controlfixedmeanimprovement+.000622only;lowfixedregressesbothseeds;noneimprovesmenureliablyoveroriginal. Literalresearchpreferencefixedduration_only/menuinconclusiveisnotreleaseapproval. Independentgroup-labelledvalidationandoriginalcanaryresolutionstillrequired.
- Production shortlist across earlier fixed readouts:48pseed2/block2flat wins7/8,meanrawgain+.089366;current101fixed5/8+.080077. Priorrecommendationretain101wasversusitscontinuations,notglobalbenchmarkleadership. Seed2shortlistingisretrospective;lockedindependentvalidationneeded. Updatedsignalplotshows101allreadouts0–6andseparatecontinuationplotswithunmeasuredflatgaps.
- Authorizedfrozenfollowupactive:`docs/superpowers/plans/2026-10-12-frozen-feature-study.md`. Eightcheckpointjobs/sevenhealthyGPUUUIDqueues,outputs`outputs/frozen-feature-study`,W&Bgroup`frozen-feature-study`. Fullflat0–6+pre/postfinalLayerNorm;reuseoldfixedscoresonlyafterfeature/OOFproof. Secondaryfixed2Ridge/PLS/RBFmenuappliedequallyraw/model/randomwithnestedtransforms.23tests/realsmoke passed,allseveninitialjobsprogressing;low203queued. Projectorpilotconditional,no newpretraining/releasepromotion.
- Frozenfollowupcompleted:8×8linear/headsets. Bestfixedfamilyblock2-flatforall8;priorseed2still7/8+.089366. LNablationmixed;secondaryheads4–6/8andsometimesweakenrawnormalizationbaseline. Bothrawreferencesexplicit;unionwithprimaryrecipes5–6/8,seed2regresses.00994meanR². Four-readoutmenuprior0/1/2replay7/6/6wins. No8/8winner/exportchange. Projection-spacepilotdesign`docs/superpowers/specs/2026-10-12-projection-space-pilot.md`ready,notlaunched.

## Repository map

- `src/spectral_lejepa/models/`: tokenizer, masking, shared encoder, predictor.
- `src/spectral_lejepa/training/`: objective, loop, checkpointing, diagnostics.
- `src/spectral_lejepa/evaluation/`: feature banks and nested probes.
- `scripts/pretrain.py`, `scripts/evaluate.py`: CLI entry points.
- `scripts/launch_final_model.sh`, `scripts/export_production.py`, `scripts/final_model_gate.py`: final pipeline.
- `scripts/build_results_report.py`: offline HTML, CSV and SVG exports.
- `README.md`: architecture, tensor shapes, commands. `docs/DATASETS.md`: corpus details. Historical decisions stay in `docs/superpowers/plans/`; local status in ignored `plans/SUMMARY.md`.

## Execution rules

- W&B credentials come from the user shell; launch jobs with `bash -ic`. Never print or store keys. Log metrics/figures, not checkpoint artifacts.
- Keep long jobs in user systemd services. Do not overwrite an active launcher or restart training for documentation changes.
- Physical GPU3 is broken. With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, logical CUDA 0–2 map directly; 3–6 map to physical 4–7. Final training uses CUDA 0, evaluation CUDA 6.
- Set `OMP_NUM_THREADS=4`, `JOBLIB_TEMP_FOLDER=/dev/shm`, writable `MPLCONFIGDIR`.
- Never write under `/mnt5/noy` or modify the parent evaluation repository.
- Commit on `main` as `Hadar <hal.nls@gmail.com>`. Push when the user requests it.
- Preserve checkpoint/RNG resume behavior, input normalization, loss weights and W&B axes during cleanup. Keep useful shape/numerical comments; avoid prose that repeats code.
