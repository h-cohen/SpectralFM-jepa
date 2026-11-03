#!/usr/bin/env bash
# Measured-only stage 1 (spec docs/superpowers/specs/2026-11-03-measured-only-pretraining-design.md):
# seeds 107-110, one per physical GPU 4-7, each a user systemd service run through bash -ic (W&B key).
# usage: bash ops/launch_measured_only.sh [--dry-run] [--resume SEED CHECKPOINT]
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$PWD
declare -A GPU=(
  [107]=GPU-acdb7bbc-75c8-fe29-0597-7db4eaefb248   # physical 4
  [108]=GPU-21b34dcc-0659-c023-ebc5-79493108b99a   # physical 5
  [109]=GPU-7d513fa3-4201-a1b3-7d45-cdd984b56c57   # physical 6
  [110]=GPU-3e8744fb-7aff-6a5a-4d03-52ea0ae8b3a4   # physical 7
)
dry=0; seeds=(107 108 109 110); extra=""
if [[ "${1:-}" == "--dry-run" ]]; then dry=1
elif [[ "${1:-}" == "--resume" && $# -eq 3 && -f "$3" ]]; then seeds=("$2"); extra="training.resume=$3"
elif [[ $# -ne 0 ]]; then echo 'usage: bash ops/launch_measured_only.sh [--dry-run | --resume SEED CHECKPOINT]' >&2; exit 2
fi
mkdir -p outputs/measured-only .tmp/mpl
for s in "${seeds[@]}"; do
  unit=spectralfm-mo-s$s
  log=outputs/measured-only/mo_s$s.log
  cmd="cd $repo && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU[$s]} OMP_NUM_THREADS=4 \
JOBLIB_TEMP_FOLDER=/dev/shm MPLCONFIGDIR=$repo/.tmp/mpl PYTHONPATH=$repo/src .venv/bin/python -m spectral_lejepa pretrain \
--config configs/measured_only.yaml experiment.seed=$s experiment.name=mo_s$s $extra >> $log 2>&1"
  if (( dry )); then echo "$unit: $cmd"; continue; fi
  if systemctl --user is-active --quiet "$unit"; then echo "$unit is already active; not starting" >&2; exit 1; fi
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" bash -ic "$cmd"
done
