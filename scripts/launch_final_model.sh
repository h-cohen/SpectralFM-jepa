#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' 'CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/pretrain.py --config configs/final_model.yaml'
  printf '%s\n' '.venv/bin/python -m scripts.export_production --checkpoint <final-checkpoint> --output outputs/final-model/production'
  printf '%s\n' 'CUDA_VISIBLE_DEVICES=6 flock -x outputs/final-model/evaluation.lock .venv/bin/python -m scripts.evaluate --checkpoint <final-checkpoint> --config configs/eval_final_model.yaml'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'usage: bash scripts/launch_final_model.sh [--dry-run]' >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export JOBLIB_TEMP_FOLDER=/dev/shm
export MPLCONFIGDIR=/tmp/spectralfm-mpl
mkdir -p outputs/final-model "$MPLCONFIGDIR"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "final pipeline failed: exit=$code" >> outputs/final-model/pipeline.log; fi' EXIT
.venv/bin/python scripts/pretrain.py --config configs/final_model.yaml 2>&1 | tee outputs/final-model/training.log
checkpoint=$(sed -n 's/^final checkpoint: //p' outputs/final-model/training.log | tail -1)
[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo 'Final checkpoint unavailable; stopping.' >&2; exit 1; }
.venv/bin/python -m scripts.export_production --checkpoint "$checkpoint" --output outputs/final-model/production 2>&1 | tee outputs/final-model/export.log
CUDA_VISIBLE_DEVICES=6 flock -x outputs/final-model/evaluation.lock .venv/bin/python -m scripts.evaluate \
  --checkpoint "$checkpoint" --config configs/eval_final_model.yaml 2>&1 | tee outputs/final-model/evaluation.log
.venv/bin/python -m scripts.final_model_gate --checkpoint "$checkpoint" 2>&1 | tee outputs/final-model/gate.log
MPLCONFIGDIR=/tmp/spectralfm-mpl .venv/bin/python -m scripts.build_results_report 2>&1 | tee outputs/final-model/report.log
