# Next research design: regularization in projection space

Status: proposed after the completed frozen study; no training launched.

The frozen tests found no better fixed readout than block2-flat, no consistent final-LayerNorm fix, and no robust decoder replacement. The next training hypothesis is that a separate training representation can absorb SIGReg constraints while preserving useful backbone features. Lower total SIGReg weight has already failed to improve consistently.

## Matched pilot

- Lock the earlier48-patch seed2 checkpoint as the starting candidate, SHA256`ec4c041dc4f83aa04e9ec568a335cdf7e27dbc7aa7fac965f2fd1f686b566ff0`. Its nomination is retrospective; the pilot remains exploratory.
- Four jobs:paired seeds301/302,control versus regularization-only projector. Both arms load identical backbone/predictor weights and use fresh AdamW. Instantiate identical unused projector parameters in controls so initialization consumes the same random stream. Keep spectra/mask streams matched.
- One additional epoch /49,799steps;1,000warmup;backbone/head peakLR5e-5,cosinefloor5e-7. Batch256,48patches,d256,6blocks,mask75%,globalnorm,50/50source mix,all eligible spectra,SIGRegλ.05,pooledweight1 unchanged.
- Projector:MLP256→512→128,GELU,noBatchNorm/dropout,FP32. Apply it to full target tokens and pooled views for the two SIGReg terms only. Masked prediction MSE and pooled-view invariance stay in backbone space with their existing coefficients. Gradients flow through the projector into the encoder; no teacher or stop-gradient.
- This tests one concrete projection-space recipe, including its128-dimensional regularization space. It does not isolate dimension from placement. The existing`model.projector=mlp`flag is insufficient:it also changes prediction MSE and leaves pooled regularization in backbone space.

## Evaluation and proof

- TDD pins unchanged backbone MSE/invariance, correct projected SIGReg shapes, gradient flow and exact warm-started backbone weights. Old checkpoints/default objective behavior remain compatible.
- Predeclare fixed block2-flat as primary and block6-flat recovery as secondary. Evaluate all blocks/readouts,raw and matched random controls with the existing nested linear protocol. Do not select checkpoints from intermediate probes or expand the production menu after seeing finals.
- Log actual weighted terms, block2/block6 per-term gradients, backbone and projector covariance/rank/std, sample progress and validation prediction error. A Gaussian-looking projector alone is not success.
- Research continuation requires≥+.005meanR² versus matched control on both seeds,no reduction in raw-win count against control or the starting fixed embedding,and no repeated material dataset regression. Otherwise stop this recipe. These are proposed pilot criteria,not a modification of the existing production gate.
- No export replacement or release claim without independent group-disjoint labelled development/release data and resolution of the existing qualification issue.

Before increasing the downstream head search, include the original full linear recipes as candidates for raw,trained andrandom features. The initial secondary standardized-only head menu was a narrower normalization policy and could weaken raw on labeled_data.
