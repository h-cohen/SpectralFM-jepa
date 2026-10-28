#!/usr/bin/env bash
# All 36 set-adapt runs (4 members x (8 per-set + 1 pooled)) over the given free GPUs, one run per GPU at a time.
# Usage: ops/set_adapt_queue.sh GPU_UUID [GPU_UUID ...]
# Reruns skip finished jobs. Launch under systemd-run --user via bash -ic.
set -euo pipefail
REPO=/mnt5/home/hadar/nova/SpectralFM-jepa
cd "$REPO"
GPUS=("$@")
[[ ${#GPUS[@]} -gt 0 ]] || { echo "usage: $0 GPU_UUID..." >&2; exit 2; }
SETS=(labeled_data dataset0055 dataset0106 dataset0109 dataset0112 dataset0113 dataset0114 dataset0120)
src_of() { [[ $1 == labeled_data ]] && echo labeled_data || echo "labeled_regression/$1"; }
ALL="[$(for s in "${SETS[@]}"; do printf '%s,' "$(src_of "$s")"; done | sed 's/,$//')]"

JOBS=()
for seed in 107 108 109 110; do
  ckpt=$(ls outputs/accuracy-ensemble/training/nr_s${seed}_*/checkpoint_last.pt | head -1)
  for s in "${SETS[@]}"; do
    JOBS+=("$ckpt|ad_s${seed}_$s|[$(src_of "$s")]|$s|$seed")
  done
  JOBS+=("$ckpt|ad_s${seed}_pooled|$ALL|all|$seed")
done

worker() {
  local g=$1 i=$2 j
  for ((j = i; j < ${#JOBS[@]}; j += ${#GPUS[@]})); do
    IFS='|' read -r ckpt name src set seed <<< "${JOBS[$j]}"
    echo "[$(date +%T)] gpu$i start $name"
    bash ops/set_adapt.sh "$ckpt" "$name" "$src" "$set" "$g" "$seed" || echo "FAILED $name"
  done
}
for i in "${!GPUS[@]}"; do worker "${GPUS[$i]}" "$i" & done
wait
echo "queue done"
