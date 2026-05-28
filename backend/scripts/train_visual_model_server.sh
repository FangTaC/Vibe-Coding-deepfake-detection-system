#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash backend/scripts/train_visual_model_server.sh <data_root> <output_dir> <dataset_version> [extra args...]"
  exit 1
fi

DATA_ROOT="$1"
OUTPUT_DIR="$2"
DATASET_VERSION="$3"
shift 3

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/train_visual_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] log file: ${LOG_FILE}"
echo "[info] data root: ${DATA_ROOT}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] dataset version: ${DATASET_VERSION}"

python backend/scripts/train_visual_model.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-version "${DATASET_VERSION}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
