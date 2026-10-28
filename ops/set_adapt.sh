#!/usr/bin/env bash
# One set-adapt run: adapt a stage-1 member on SOURCES, then evaluate each saved step.
# Usage: ops/set_adapt.sh MEMBER_CKPT NAME SOURCES SET_OR_ALL GPU_UUID [SEED]
#   SOURCES: e.g. "[labeled_regression/dataset0112]"; SET_OR_ALL: eval set name (dataset0112, labeled_data) or "all".
set -euo pipefail
CKPT=$1 NAME=$2 SOURCES=$3 SET=$4 GPU=$5 SEED=${6:-101}
REPO=/mnt5/home/hadar/nova/SpectralFM-jepa
NOY=/mnt5/noy/SpectralFM/fairseq/data/nova_data
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 JOBLIB_TEMP_FOLDER=/dev/shm MPLCONFIGDIR=$REPO/.tmp/mpl
mkdir -p "$MPLCONFIGDIR" outputs/set-adapt/training outputs/set-adapt/eval
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO/src:$REPO
EVAL_OUT=outputs/set-adapt/eval/$NAME

LABEL=()
if [[ $SET == labeled_data ]]; then
  LABEL=("label_sets={labeled_data: [$NOY/labeled_data]}")
elif [[ $SET != all ]]; then
  LABEL=("label_sets={$SET: [$NOY/labeled_regression/$SET]}")
fi

done_all=1
for N in 250 500 1000 2000; do
  compgen -G "$EVAL_OUT/${NAME}_*_step$N/*/nested_results.json" > /dev/null || done_all=0
done
if [[ $done_all == 1 ]]; then echo "skip $NAME (eval outputs exist)"; exit 0; fi

if ! compgen -G "outputs/set-adapt/training/${NAME}_*/checkpoint_step2000.pt" > /dev/null; then
  $PY -m spectral_lejepa pretrain --config configs/set_adapt.yaml \
    training.init_checkpoint="$CKPT" experiment.seed="$SEED" experiment.name="$NAME" \
    "data.sources=$SOURCES" experiment.output_dir=outputs/set-adapt/training wandb.enabled=false
fi
RUN=$(ls -d outputs/set-adapt/training/${NAME}_* | tail -1)
for N in 250 500 1000 2000; do
  compgen -G "$EVAL_OUT/$(basename "$RUN")_step$N/*/nested_results.json" > /dev/null && continue
  $PY -m spectral_lejepa evaluate --checkpoint "$RUN/checkpoint_step$N.pt" \
    --config configs/eval_production_all_blocks.yaml experiment.output_dir="$EVAL_OUT" \
    "evaluation.readouts=[flat]" "evaluation.flat_blocks=[2]" "${LABEL[@]}" wandb.enabled=false
done
