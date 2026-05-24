#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env.sh

: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES

IMPL="${IMPL:-triton}"
DTYPE="${DTYPE:-fp16}"
BATCH="${BATCH:-4096}"
HIDDEN="${HIDDEN:-4096}"
ITERS="${ITERS:-50}"
OUT_DIR="${OUT_DIR:-results/profiling}"
mkdir -p "${OUT_DIR}"

REPORT="${OUT_DIR}/nsys_rmsnorm_${IMPL}_${DTYPE}_b${BATCH}_h${HIDDEN}"

nsys profile \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  --sample none \
  --stats true \
  --output "${REPORT}" \
  .venv/bin/python scripts/profile_rmsnorm.py \
    --impl "${IMPL}" \
    --dtype "${DTYPE}" \
    --batch "${BATCH}" \
    --hidden "${HIDDEN}" \
    --iters "${ITERS}" \
    --cuda-profiler-api

echo "Nsight Systems report: ${REPORT}.nsys-rep"

