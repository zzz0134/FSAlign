#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/work/was598/miniconda3/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/was598/modilty_gap}"
DATA_ROOT="${DATA_ROOT:-/work/was598/modilty_gap/tools/data}"
MODEL_KEY="${MODEL_KEY:-siglip}"
SIGLIP_NAME="${SIGLIP_NAME:-google/siglip-base-patch16-224}"
OPENCLIP_MODEL="${OPENCLIP_MODEL:-ViT-B-32}"
OPENCLIP_PRETRAINED="${OPENCLIP_PRETRAINED:-openai}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NAS_K="${NAS_K:-100}"
NAS_MAX_ITEMS="${NAS_MAX_ITEMS:-5000}"
INTRA_SAMPLES="${INTRA_SAMPLES:-20000}"
STAMP="${STAMP:-20260327}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/rebuttal_q2_paper_faithful_${MODEL_KEY}_fullbudget_${STAMP}}"

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run.log"

echo "[Run] q2_paper_faithful_benchmarks.py" | tee "${LOG_FILE}"
echo "[Out] ${OUT_DIR}" | tee -a "${LOG_FILE}"
echo "[Model] ${MODEL_KEY}" | tee -a "${LOG_FILE}"
echo "[Budget] IOT=full Flickr train split, GR-CLIP=full Flickr train split" | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/q2_paper_faithful_benchmarks.py" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --model-key "${MODEL_KEY}" \
  --siglip-name "${SIGLIP_NAME}" \
  --openclip-model "${OPENCLIP_MODEL}" \
  --openclip-pretrained "${OPENCLIP_PRETRAINED}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --nas-k "${NAS_K}" \
  --nas-max-items "${NAS_MAX_ITEMS}" \
  --intra-samples "${INTRA_SAMPLES}" \
  --iot-fit-max-flickr 0 \
  --gr-calib-n 0 \
  2>&1 | tee -a "${LOG_FILE}"
