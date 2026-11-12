#!/usr/bin/env bash
# Measured-budget stage 1 (spec docs/superpowers/specs/2026-11-12-measured-budget-design.md): 18 runs,
# arms bm (measured only), bms (+ synthetic), bmx (+ blend) x measured fraction x seeds 107/108.
# Runs are queued round-robin over 7 GPUs (physical 0-2, 4-7; GPU3 is broken); each GPU is one user
# systemd unit that runs its queue in order, through bash -ic (W&B key).
# usage: bash ops/launch_budget.sh [--dry-run] [--only NAME] [--resume NAME CHECKPOINT]
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$PWD; out=outputs/measured-budget
GPUS=(
  GPU-69e42c2d-27a4-ff64-073e-f3f6941027b1   # physical 0
  GPU-d54a30ae-7b4d-cc41-7d72-7ebdaf0f4798   # physical 1
  GPU-a867db5d-e391-4ccc-825a-8443e68b59a3   # physical 2
  GPU-acdb7bbc-75c8-fe29-0597-7db4eaefb248   # physical 4
  GPU-21b34dcc-0659-c023-ebc5-79493108b99a   # physical 5
  GPU-7d513fa3-4201-a1b3-7d45-cdd984b56c57   # physical 6
  GPU-3e8744fb-7aff-6a5a-4d03-52ea0ae8b3a4   # physical 7
)
declare -A FRAC=([100]=1.0 [25]=0.25 [6]=0.0625 [1p5]=0.015)
declare -A CONF=([bm]=configs/budget_m.yaml [bms]=configs/budget_ms.yaml [bmx]=configs/budget_mx.yaml)
runs=()
for s in 107 108; do
  for r in bm100 bm25 bm6 bm1p5 bms25 bms6 bms1p5 bmx6 bmx1p5; do runs+=("${r}_s$s"); done
done
dry=0; extra=""
case "${1:-}" in
  --dry-run) dry=1 ;;
  --only) [[ $# -eq 2 ]] || { echo 'usage: --only NAME' >&2; exit 2; }; runs=("$2") ;;
  --resume) [[ $# -eq 3 && -f "$3" ]] || { echo 'usage: --resume NAME CHECKPOINT' >&2; exit 2; }
            runs=("$2"); extra="training.resume=$3" ;;
  "") ;;
  *) echo 'usage: bash ops/launch_budget.sh [--dry-run | --only NAME | --resume NAME CHECKPOINT]' >&2; exit 2 ;;
esac
command_for() {   # $1 = run name like bms6_s107; $2 = GPU UUID
  local name=$1 arm frac seed
  [[ $name =~ ^(bm|bms|bmx)(100|25|6|1p5)_s([0-9]+)$ ]] || { echo "bad run name $name" >&2; exit 2; }
  arm=${BASH_REMATCH[1]}; frac=${FRAC[${BASH_REMATCH[2]}]}; seed=${BASH_REMATCH[3]}
  echo "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 JOBLIB_TEMP_FOLDER=/dev/shm \
MPLCONFIGDIR=$repo/.tmp/mpl PYTHONPATH=$repo/src .venv/bin/python -m spectral_lejepa pretrain --config ${CONF[$arm]} \
data.measured_fraction=$frac experiment.seed=$seed experiment.name=$name $extra >> $out/$name.log 2>&1"
}
mkdir -p $out .tmp/mpl
for g in "${!GPUS[@]}"; do
  queue=""
  for i in "${!runs[@]}"; do
    (( i % ${#GPUS[@]} == g )) || continue
    queue+="$(command_for "${runs[$i]}" "${GPUS[$g]}"); "
  done
  [[ -n $queue ]] || continue
  unit=spectralfm-budget-g$g
  [[ ${#runs[@]} -eq 1 ]] && unit=spectralfm-budget-${runs[0]}
  if (( dry )); then echo "$unit: cd $repo && $queue"; continue; fi
  if systemctl --user is-active --quiet "$unit"; then echo "$unit is already active; not starting" >&2; exit 1; fi
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" bash -ic "cd $repo && $queue"
done
