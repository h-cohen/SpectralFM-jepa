# Matched continuation study

Status (2026-10-13): completed. Duration-only and lower-SIGReg continuations did not improve the fixed export. Historical record; current state is in CLAUDE.md.

User approved new experiments after the completed analysis. Execute exploratory research now; independent deployment labels are absent, so do not promote or overwrite the current export/gate.

- Starting checkpoint: seed101 step 497990, SHA256 30486b3e78c354596d7bb5f09ebf351805309b0e9505c88d7ed698145df7fbbe.
- Four runs: paired seeds 202/203 × SIGReg .05/.01. Fresh optimizer/scheduler/RNG in each; both arms of each seed get the same sample/mask stream. All eligible spectra, existing 50/50 source mix, global norm, 48 patches, d256, depth 6, batch 256, mask 75%, pooled weight 1 unchanged.
- 99,598 steps, 1000 warmup steps, peak LR 5e-5, cosine floor 5e-7. Lower lambda also raises pooled invariance weight .95→.99; masked prediction remains 1. Thus this tests the existing lambda control, not an isolated token-only effect.
- Frozen completed checkpoint is the no-training baseline. No intermediate checkpoint selection. Both final seeds must support a benefit before any further full recipe is recommended.
- Evaluate mean/seg4 all blocks and flat 2/6, raw/random controls, same 2×5 outer/5 inner CV. Fixed block2 and predeclared four-arm menu are primary interfaces. Other layers only diagnostic. No raw fallback/multiscale promotion from this experiment.
- Predeclare research preference: lower lambda must improve mean ΔR² versus both frozen baseline and matched control on each seed, not reduce raw win count, and keep block2 mean R² within .005 of baseline. Record per-dataset regressions and bootstrap uncertainty; not a statistical release gate. Duration-only benefit evaluated similarly versus frozen baseline. Otherwise keep current candidate and report inconclusive.
- Retain original canary failure; no threshold changes. Independent group-disjoint labelled development and heldout release data still required for production validation.
- Durable GPU 0/1/2/3 logical (physical 0/1/2/4), local TMP, 4 CPU threads, single probe GPU worker. Per-run atomic status, config/checkpoint hashes, W&B group continuation-study, loss/progress/geometry plots, final gradient audit, policy gain figures, offline HTML. Existing training code unchanged.
