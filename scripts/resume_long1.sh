#!/usr/bin/env bash
set -euo pipefail
seed="$1"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$2"
export OMP_NUM_THREADS=4
export JOBLIB_TEMP_FOLDER=/dev/shm
train_log=$(mktemp "outputs/long-1/resume_s${seed}_XXXXXX.log")
uv run python scripts/pretrain.py --config "configs/long1_resume_s${seed}.yaml" | tee "$train_log"
checkpoint=$(sed -n 's/^final checkpoint: //p' "$train_log" | tail -1)
[[ -f "$checkpoint" ]]
uv run python -m scripts.evaluate --checkpoint "$checkpoint" --config configs/eval_long1.yaml
