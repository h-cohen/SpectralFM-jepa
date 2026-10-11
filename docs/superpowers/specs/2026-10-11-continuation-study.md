# Matched continuation study

User approved new experiments after the completed analysis. Execute exploratory research now; independent deployment labels are absent, so do not promote or overwrite the current export/gate.

- Starting checkpoint:seed101step497990,SHA25630486b3e78c354596d7bb5f09ebf351805309b0e9505c88d7ed698145df7fbbe.
- Four runs:paired seeds202/203×SIGReg.05/.01. Fresh optimizer/scheduler/RNG in each;both arms of each seed get the same sample/mask stream. All eligible spectra,existing50/50source mix,globalnorm,48patches,d256,depth6,batch256,mask75%,pooledweight1 unchanged.
-99,598steps,1000warmupsteps,peakLR5e-5,cosinefloor5e-7. Lowerlambda also raises pooled invariance weight.95→.99;masked prediction remains1. Thus this tests the existing lambda control,not an isolated token-only effect.
- Frozen completed checkpoint is the no-training baseline. No intermediate checkpoint selection. Both final seeds must support a benefit before any further full recipe is recommended.
- Evaluate mean/seg4 allblocks andflat2/6,raw/random controls,same2×5outer/5inner CV. Fixedblock2 andpredeclaredfour-arm menu are primary interfaces. Other layers onlydiagnostic. No rawfallback/multiscale promotion from this experiment.
- Predeclare research preference:lowerlambda must improve mean ΔR² versus both frozenbaseline and matchedcontrol on eachseed,not reduce rawwin count,andkeep block2meanR² within.005ofbaseline. Record per-dataset regressions and bootstrap uncertainty;not a statistical release gate. Duration-only benefit evaluated similarly versus frozenbaseline. Otherwise keep currentcandidate andreport inconclusive.
- Retain original canary failure;no thresholdchanges. Independent group-disjoint labelleddevelopment andheldoutrelease data stillrequiredforproductionvalidation.
- DurableGPU0/1/2/3logical (physical0/1/2/4),localTMP,4CPUthreads,singleprobeGPUworker. Per-run atomicstatus,config/checkpoint hashes,W&B groupcontinuation-study,loss/progress/geometryplots,finalgradientaudit,policygainfigures,offlineHTML. Existingtrainingcodeunchanged.
