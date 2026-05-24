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
ITERS="${ITERS:-20}"
OUT_DIR="${OUT_DIR:-results/profiling}"
mkdir -p "${OUT_DIR}"

REPORT="${OUT_DIR}/ncu_rmsnorm_${IMPL}_${DTYPE}_b${BATCH}_h${HIDDEN}"

ncu \
  --force-overwrite \
  --target-processes all \
  --profile-from-start off \
  --set basic \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --export "${REPORT}" \
  .venv/bin/python scripts/profile_rmsnorm.py \
    --impl "${IMPL}" \
    --dtype "${DTYPE}" \
    --batch "${BATCH}" \
    --hidden "${HIDDEN}" \
    --iters "${ITERS}" \
    --cuda-profiler-api

echo "Nsight Compute report: ${REPORT}.ncu-rep"

