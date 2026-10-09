#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' 'CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/pretrain.py --config configs/final_model.yaml'
  printf '%s\n' '.venv/bin/python -m scripts.export_production --checkpoint <final-checkpoint> --output outputs/final-model/production'
  printf '%s\n' 'CUDA_VISIBLE_DEVICES=6 flock -x outputs/final-model/evaluation.lock .venv/bin/python -m scripts.evaluate --checkpoint <final-checkpoint> --config configs/eval_final_model.yaml'
  exit 0
fi
train_args=(--config configs/final_model.yaml)
if [[ "${1:-}" == "--resume" && $# -eq 2 && -f "$2" ]]; then
  train_args+=("training.resume=$2")
elif [[ $# -ne 0 ]]; then
  echo 'usage: bash scripts/launch_final_model.sh [--dry-run | --resume CHECKPOINT]' >&2
  exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR="$PWD/.tmp"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p outputs/final-model "$MPLCONFIGDIR"
if [[ ${#train_args[@]} -gt 2 ]]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  [[ ! -f outputs/final-model/training.log ]] || mv outputs/final-model/training.log "outputs/final-model/training-$stamp.log"
fi
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "%s\n" "final pipeline failed: exit=$code" >> outputs/final-model/pipeline.log; fi' EXIT
.venv/bin/python scripts/pretrain.py "${train_args[@]}" 2>&1 | tee outputs/final-model/training.log
checkpoint=$(sed -n 's/^final checkpoint: //p' outputs/final-model/training.log | tail -1)
[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo 'Final checkpoint unavailable; stopping.' >&2; exit 1; }
.venv/bin/python -m scripts.export_production --checkpoint "$checkpoint" --output outputs/final-model/production 2>&1 | tee outputs/final-model/export.log
CUDA_VISIBLE_DEVICES=6 flock -x outputs/final-model/evaluation.lock .venv/bin/python -m scripts.evaluate \
  --checkpoint "$checkpoint" --config configs/eval_final_model.yaml 2>&1 | tee outputs/final-model/evaluation.log
.venv/bin/python -m scripts.final_model_gate --checkpoint "$checkpoint" 2>&1 | tee outputs/final-model/gate.log
.venv/bin/python -m scripts.build_results_report 2>&1 | tee outputs/final-model/report.log
