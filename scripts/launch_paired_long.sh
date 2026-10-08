#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if [[ $# -ne 3 ]]; then
  echo "usage: $0 [--dry-run] <24|48> <seed 0..2> <CUDA index 0..6>" >&2
  exit 2
fi
patches="$1"
seed="$2"
cuda_index="$3"
[[ "$patches" == 24 || "$patches" == 48 ]] || { echo 'variant must be 24 or 48' >&2; exit 2; }
[[ "$seed" =~ ^[0-2]$ ]] || { echo 'seed must be 0, 1, or 2' >&2; exit 2; }
[[ "$cuda_index" =~ ^[0-6]$ ]] || { echo 'CUDA index must be 0..6 (broken physical GPU 3 excluded)' >&2; exit 2; }

name="p${patches}_s${seed}"
run_dir='outputs/paired-long'
common=(--config configs/paired_long.yaml
  "model.num_patches=${patches}"
  "experiment.seed=${seed}"
  "experiment.name=${name}"
  "experiment.output_dir=${run_dir}")
train_cmd=(uv run python scripts/pretrain.py "${common[@]}")
eval_lock="${run_dir}/evaluation.lock"
eval_cuda=6
eval_cmd=(CUDA_VISIBLE_DEVICES="$eval_cuda" flock -x "$eval_lock" uv run python -m scripts.evaluate
  --checkpoint '<final-checkpoint>' --config configs/eval_long1.yaml)

if [[ "$dry_run" == true ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s' "$cuda_index"
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=%s ' "$eval_cuda"
  printf '%q -x %q' "${eval_cmd[1]}" "$eval_lock"
  printf ' %q' "${eval_cmd[@]:4}"
  printf '\n'
  exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$cuda_index"
export OMP_NUM_THREADS=4
export JOBLIB_TEMP_FOLDER=/dev/shm
export MPLCONFIGDIR=/tmp/spectralfm-mpl
mkdir -p "$run_dir" "$MPLCONFIGDIR"
log_file="${run_dir}/${name}.log"
set -o pipefail
"${train_cmd[@]}" 2>&1 | tee "$log_file"
checkpoint=$(sed -n 's/^final checkpoint: //p' "$log_file" | tail -1)
[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo 'final checkpoint missing; refusing evaluation' >&2; exit 1; }
eval_log="${run_dir}/${name}_final_eval.log"
CUDA_VISIBLE_DEVICES="$eval_cuda" flock -x "$eval_lock" uv run python -m scripts.evaluate \
  --checkpoint "$checkpoint" --config configs/eval_long1.yaml 2>&1 | tee "$eval_log"
