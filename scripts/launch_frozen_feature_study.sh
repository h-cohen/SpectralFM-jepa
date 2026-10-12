#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo 'usage: launch_frozen_feature_study.sh GPU_UUID JOB [JOB...]' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$1"
shift
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR="$PWD/.tmp"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$MPLCONFIGDIR"
failed=0
for name in "$@"; do
  mkdir -p "outputs/frozen-feature-study/$name"
  .venv/bin/python -u -m scripts.frozen_feature_study "$name" > "outputs/frozen-feature-study/$name/pipeline.log" 2>&1 || failed=1
done
exit "$failed"
