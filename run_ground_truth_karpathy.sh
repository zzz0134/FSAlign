#!/usr/bin/env bash
set -euo pipefail

# Choose GPU(s)
export CUDA_VISIBLE_DEVICES=0

# Choose CLIP/OpenCLIP backbone size: 16 or 32
MODEL_SIZE=32
# --models clip,siglip,openclip \
# python /work/was598/modilty_gap/ground_truth_karpathy.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir /work/was598/modilty_gap/results/ground_truth_karpathy.128 \
#   --models siglip\
#   --model-size ${MODEL_SIZE} \
#   --batch-size 256 --num-workers 8 \
#   --max-coco 999999 --max-flickr 999999 --max-cls 999999 \
#   --nas-k 100

python /work/was598/modilty_gap/ground_truth_karpathy.py \
  --data-root /work/was598/modilty_gap/tools/data \
  --out-dir /work/was598/modilty_gap/results/ground_truth_karpathy.128 \
  --models siglip \
  --siglip-sanity \
  --siglip-v1-maxlen

