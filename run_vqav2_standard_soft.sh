#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/tools/data}"
VQAV2_ROOT="${VQAV2_ROOT:-$DATA_ROOT/vqav2}"
OUT_BASE="${OUT_BASE:-$ROOT/results/vqav2_standard_soft_$(date +%Y%m%d_%H%M%S)}"
MODELS="${MODELS:-clip,siglip,openclip}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOPK_ANSWERS="${TOPK_ANSWERS:-3129}"
MAX_VQA_TRAIN="${MAX_VQA_TRAIN:-0}"
MAX_VQA_VAL="${MAX_VQA_VAL:-0}"
NAS_K="${NAS_K:-10}"
NAS_MAX_ITEMS="${NAS_MAX_ITEMS:-5000}"
INTRA_SAMPLES="${INTRA_SAMPLES:-20000}"
SEED="${SEED:-42}"

OUR_TRAIN_EPOCHS="${OUR_TRAIN_EPOCHS:-30}"
OUR_TRAIN_LR="${OUR_TRAIN_LR:-1e-3}"
OUR_LORA_RANK="${OUR_LORA_RANK:-8}"
OUR_LORA_ALPHA="${OUR_LORA_ALPHA:-8.0}"
OUR_LORA_MIX="${OUR_LORA_MIX:-0.3}"
OUR_TRAIN_PRINT_EVERY="${OUR_TRAIN_PRINT_EVERY:-1}"
OUR_SAVE_LORA="${OUR_SAVE_LORA:-1}"

GR_CALIB_N="${GR_CALIB_N:-10000}"
GR_CALIB_BATCH="${GR_CALIB_BATCH:-128}"

UNIFIED_MODEL_KEY="${UNIFIED_MODEL_KEY:-clip}"
UNIFIED_MODEL_SIZE="${UNIFIED_MODEL_SIZE:-32}"
UNIFIED_MAX_TRAIN_ITEMS="${UNIFIED_MAX_TRAIN_ITEMS:-0}"
UNIFIED_MAX_VAL_ITEMS="${UNIFIED_MAX_VAL_ITEMS:-0}"
UNIFIED_HEAD_EPOCHS="${UNIFIED_HEAD_EPOCHS:-30}"
UNIFIED_HEAD_LR="${UNIFIED_HEAD_LR:-1e-3}"
UNIFIED_HEAD_WEIGHT_DECAY="${UNIFIED_HEAD_WEIGHT_DECAY:-1e-4}"
UNIFIED_HEAD_VAL_FRAC="${UNIFIED_HEAD_VAL_FRAC:-0.1}"
UNIFIED_FEATURE_CHUNK_SIZE="${UNIFIED_FEATURE_CHUNK_SIZE:-8192}"

RUN_OUR="${RUN_OUR:-1}"
RUN_I0T="${RUN_I0T:-1}"
RUN_GR="${RUN_GR:-1}"
RUN_UNIFIED="${RUN_UNIFIED:-1}"

mkdir -p "$OUT_BASE"

if [[ ! -d "$VQAV2_ROOT" ]]; then
  echo "[ERROR] VQAv2 root not found: $VQAV2_ROOT" >&2
  exit 1
fi

run_step() {
  local name="$1"
  shift
  local log_file="$OUT_BASE/${name}.log"
  echo
  echo "[RUN] $name"
  echo "[LOG] $log_file"
  "$@" 2>&1 | tee "$log_file"
}

if [[ "$RUN_OUR" == "1" ]]; then
  OUR_OUT="$OUT_BASE/our_code_final"
  mkdir -p "$OUR_OUT"
  OUR_ARGS=(
    "$PYTHON_BIN" "$ROOT/our_code_final.py"
    --data-root "$DATA_ROOT"
    --out-dir "$OUR_OUT"
    --models "$MODELS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --eval-vqav2
    --vqav2-root "$VQAV2_ROOT"
    --max-vqa-train "$MAX_VQA_TRAIN"
    --max-vqa-val "$MAX_VQA_VAL"
    --vqav2-topk-answers "$TOPK_ANSWERS"
    --nas-k "$NAS_K"
    --nas-max-items "$NAS_MAX_ITEMS"
    --intra-samples "$INTRA_SAMPLES"
    --train-epochs "$OUR_TRAIN_EPOCHS"
    --train-lr "$OUR_TRAIN_LR"
    --lora-rank "$OUR_LORA_RANK"
    --lora-alpha "$OUR_LORA_ALPHA"
    --lora-mix "$OUR_LORA_MIX"
    --train-print-every "$OUR_TRAIN_PRINT_EVERY"
    --seed "$SEED"
  )
  if [[ "$OUR_SAVE_LORA" == "1" ]]; then
    OUR_ARGS+=(--save-lora)
  fi
  run_step our_code_final "${OUR_ARGS[@]}"
fi

if [[ "$RUN_I0T" == "1" ]]; then
  I0T_OUT="$OUT_BASE/I0T_full"
  mkdir -p "$I0T_OUT"
  run_step I0T_full     "$PYTHON_BIN" "$ROOT/I0T_full.py"     --data-root "$DATA_ROOT"     --out-dir "$I0T_OUT"     --models "$MODELS"     --batch-size "$BATCH_SIZE"     --num-workers "$NUM_WORKERS"     --eval-vqav2     --vqav2-root "$VQAV2_ROOT"     --fit-max-vqa "$MAX_VQA_TRAIN"     --max-vqa-val "$MAX_VQA_VAL"     --vqav2-topk-answers "$TOPK_ANSWERS"     --nas-k "$NAS_K"     --nas-max-items "$NAS_MAX_ITEMS"     --intra-samples "$INTRA_SAMPLES"     --seed "$SEED"
fi

if [[ "$RUN_GR" == "1" ]]; then
  GR_OUT="$OUT_BASE/GR_CLIP"
  GR_CACHE="$GR_OUT/calib_cache"
  mkdir -p "$GR_OUT" "$GR_CACHE"
  run_step GR_CLIP     "$PYTHON_BIN" "$ROOT/GR_CLIP.py"     --data-root "$DATA_ROOT"     --out-dir "$GR_OUT"     --models "$MODELS"     --batch-size "$BATCH_SIZE"     --num-workers "$NUM_WORKERS"     --calib-n "$GR_CALIB_N"     --calib-batch "$GR_CALIB_BATCH"     --calib-cache "$GR_CACHE"     --eval-vqav2     --vqav2-root "$VQAV2_ROOT"     --max-vqa-val "$MAX_VQA_VAL"     --vqav2-topk-answers "$TOPK_ANSWERS"     --nas-k "$NAS_K"     --nas-max-items "$NAS_MAX_ITEMS"     --intra-samples "$INTRA_SAMPLES"     --seed "$SEED"
fi

if [[ "$RUN_UNIFIED" == "1" ]]; then
  UNIFIED_OUT="$OUT_BASE/vqav2_unified_supervised"
  mkdir -p "$UNIFIED_OUT"
  run_step vqav2_unified_supervised     "$PYTHON_BIN" "$ROOT/vqav2_unified_supervised_benchmark.py"     --vqav2-root "$VQAV2_ROOT"     --out-dir "$UNIFIED_OUT"     --model-key "$UNIFIED_MODEL_KEY"     --model-size "$UNIFIED_MODEL_SIZE"     --batch-size "$BATCH_SIZE"     --num-workers "$NUM_WORKERS"     --max-train-items "$UNIFIED_MAX_TRAIN_ITEMS"     --max-val-items "$UNIFIED_MAX_VAL_ITEMS"     --topk-answers "$TOPK_ANSWERS"     --question-template 'Question: {q}'     --answer-template 'Answer: {a}.'     --fusion-mode mean     --feature-chunk-size "$UNIFIED_FEATURE_CHUNK_SIZE"     --head-epochs "$UNIFIED_HEAD_EPOCHS"     --head-lr "$UNIFIED_HEAD_LR"     --head-weight-decay "$UNIFIED_HEAD_WEIGHT_DECAY"     --head-val-frac "$UNIFIED_HEAD_VAL_FRAC"     --seed "$SEED"
fi

echo
echo "[DONE] outputs written to $OUT_BASE"
