#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: launch_continuation.sh JOB GPU_UUID' >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$2"
export OMP_NUM_THREADS=4
export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR="$PWD/.tmp"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python -u -m scripts.continuation_study "$1" > "outputs/continuation-study/$1/pipeline.log" 2>&1
