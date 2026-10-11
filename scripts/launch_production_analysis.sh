#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TMPDIR="$PWD/.tmp"
export MPLCONFIGDIR="$TMPDIR/mpl"
export JOBLIB_TEMP_FOLDER=/dev/shm
mkdir -p outputs/production-analysis "$MPLCONFIGDIR"
checkpoint=outputs/final-model/production_p48_s101_20261010-024928/checkpoint_last.pt
.venv/bin/python -m scripts.evaluate --checkpoint "$checkpoint" --config configs/eval_production_all_blocks.yaml \
  > outputs/production-analysis/all-blocks.log 2>&1
.venv/bin/python -m scripts.production_policies \
  --bank-root outputs/production-analysis/all-blocks/eval/production_p48_s101_20261010-024928_step497990 \
  --wandb > outputs/production-analysis/policies.log 2>&1
.venv/bin/python -m scripts.build_results_report > outputs/production-analysis/report.log 2>&1
