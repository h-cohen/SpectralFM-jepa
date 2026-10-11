# Production analysis design

User approved the proposed steps: inspect the completed encoder before further training, compare controlled interfaces, investigate the borderline shuffle check, and use measured gradients/downstream development performance to inform any later training experiment.

- Reuse the seed101 checkpoint at497,990. Original qualification remains failed and unchanged.
- Score all7states(tokenizer+6blocks), mean/seg4/flat, including final flattened features. Same8datasets,2×5outer/5inner folds,raw and same-architecture random controls.
- Restricted menu: layer2-flat,layer2-mean,layer3-mean,layer3-seg4. Compare fixed block2-flat,menu,menu+raw fallback,and fixed concatenated menu features(13,824dims). Select arms/recipes inside inner folds only.
- Existing benchmark evidence is reused development analysis. A separate labelled dataset is requested before choosing readouts/duration for an independent release test.
- Audit0120 with100permutations,seeds1000–1099,frozen feature rows,original fixed standardize/RidgeCV5fold canary. Reproduce shuffle42. Preserve failure; report null distribution/empiricalp,duplicates/fold checks and limits.
- Measure scalar contributions and per-term gradients at the final checkpoint, including block2andblock6 features. Diagnostic only: no optimizer updates.
- Separate outputs/W&B group`production-analysis`; durable GPU jobs,local TMPDIR,source hashes,measured plots. No automatic extra training or threshold relaxation.
