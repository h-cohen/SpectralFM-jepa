#!/usr/bin/env bash
# Serial checkpoint probes use one spare GPU; final evaluation belongs to resume_long1.sh.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=4
export JOBLIB_TEMP_FOLDER=/dev/shm
run_dirs=("$1" "$2")
for step in 250000 350000; do
  for seed in 0 1; do
    checkpoint="${run_dirs[$seed]}/checkpoint_step${step}.pt"
    summary="outputs/long-1/eval/$(basename "${run_dirs[$seed]}")_step${step}/summary.json"
    if [[ -f "$summary" ]]; then
      continue
    fi
    while [[ ! -f "$checkpoint" ]]; do
      if ! systemctl --user is-active --quiet "spectralfm-long1-s${seed}"; then
        echo "Missing $checkpoint and training service inactive" >&2
        exit 1
      fi
      sleep 60
    done
    echo "Evaluating seed=$seed step=$step checkpoint=$checkpoint"
    uv run python -m scripts.evaluate --checkpoint "$checkpoint" --config configs/eval_long1.yaml
  done
done
