#!/usr/bin/env bash
set -euo pipefail

# FSAlign release data bootstrap script
# Usage:
#   bash tools/download_data_release.sh
#   DATA_ROOT=/path/to/data bash tools/download_data_release.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/tools/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${DATA_ROOT}" "${DATA_ROOT}/_tmp"

echo "[FSAlign] DATA_ROOT=${DATA_ROOT}"

echo "[1/5] Download/prepare COCO2017 + CIFAR100 + DTD via prepare_datasets.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/data/prepare_datasets.py" \
  --root "${DATA_ROOT}" \
  --dataset all \
  --download

echo "[2/5] Download Tiny-ImageNet-200 archive (if missing)"
TINY_ZIP="${DATA_ROOT}/tiny-imagenet-200.zip"
if [[ ! -f "${TINY_ZIP}" ]]; then
  curl -L "http://cs231n.stanford.edu/tiny-imagenet-200.zip" -o "${TINY_ZIP}"
else
  echo "  - already exists: ${TINY_ZIP}"
fi

echo "[3/5] Extract Tiny-ImageNet-200 (if missing)"
if [[ ! -d "${DATA_ROOT}/tiny-imagenet-200" ]]; then
  unzip -q "${TINY_ZIP}" -d "${DATA_ROOT}"
else
  echo "  - already extracted: ${DATA_ROOT}/tiny-imagenet-200"
fi

echo "[4/5] Manual datasets reminder"
cat <<'EOF'
  Flickr30k and MSCOCO2014 Karpathy resources are typically license-gated/manual.
  Please ensure these are present:
    tools/data/flickr30k/
    tools/data/mscoco2014/
EOF

echo "[5/5] Run verification"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/data/verify_release_data.py" --root "${DATA_ROOT}"

echo "[FSAlign] Done."

