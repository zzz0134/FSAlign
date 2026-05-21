# python baseline_clip.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --device cuda \
#   --model ViT-B-32 \
#   --pretrained laion2b_s34b_b79k \
#   --image-size 224 \
#   --batch-size 64

# python HSMG.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --device cuda \
#   --model ViT-B-32 \
#   --pretrained laion2b_s34b_b79k \
#   --image-size 224 \
#   --batch-size 64 \
#   --curvature 1.0

# python HSMG_train.py --data-root /work/was598/modilty_gap/tools/data --device cuda \
#   --model ViT-B-32 --pretrained laion2b_s34b_b79k \
#   --do-train --do-eval \
#   --total-steps 1200 \
#   --batch-size 512 --grad-accum 4 --lr 5e-4 --warmup-iters 4000 \
#   --weight-decay 0.2 --adam-betas 0.9 0.98 \
#   --image-crop-scale 0.5 1.0 --image-size 224 \
#   --amp --bf16 \
#   --init-curvature 1.0 --init-temperature 0.07 \
#   --save-dir runs/HSMG


# python I0T.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --device cuda \
#   --do-stage1 --stage1-epochs 3 \
#   --do-stage2 --stage2-epochs 2 \
#   --do-eval \
#   --batch-size 256 \
#   --save-dir runs/iot_stage1_async


# python INCL.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --device cuda \
#   --model ViT-B-32 --pretrained laion2b_s34b_b79k \
#   --batch-size 256 \
#   --save-dir runs/ctg_baseline_std


# python OTCL.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --device cuda \
#   --model ViT-B-32 --pretrained laion2b_s34b_b79k \
#   --batch-size 256 \
#   --postproc all \
#   --spectral-k 60 \
#   --ot-k-graph 10 --ot-reg-e 0.0 --ot-reg-lap 1.0 --ot-max-iter 200 \
#   --save-dir runs/fill_the_gap_all

# python DIA.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --device cuda \
#   --do-train \
#   --epochs 30 \
#   --lr 5e-4 \
#   --batch-size-flickr 128 \
#   --batch-size-coco 256 \
#   --w-dim 10.0 --w-inter 0.05 --w-intra 0.1 \
#   --text-enc bert \
#   --save-dir runs/DIA

# python ALIGNCLIP.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --cc12m-mode coco-train \
#   --epochs 1 --batch-size 128 --max-pretrain 10000

# python ground_truth.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir  /work/was598/modilty_gap/results/ground_truth \
#   --models clip,openclip,siglip \
#   --num-workers 0 \
#   --batch-size 128 \
#   --zs-batch 256 \
#   --max-coco-images 5000 \
#   --max-flickr-images 5000 \
#   --auto-download-tiny
# -------------------------------------------------------------------------------------------------
# python ground_truth_karpathy.py \
#     --data-root /work/was598/modilty_gap/tools/data \
#     --out-dir /work/was598/modilty_gap/results/ground_truth_karpathy \
#     --models clip,siglip,openclip \
#     --batch-size 256 --num-workers 8 \
#     --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#     --nas-k 10


# python /work/was598/modilty_gap/I0T_full.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir /work/was598/modilty_gap/results/I0T.124 \
#   --models clip,siglip,openclip \
#   --batch-size 128 --num-workers 8 \
#   --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#   --fit-max-coco 20000 --fit-max-flickr 20000 --fit-max-cls 20000 \
#   --nas-k 100

# python INCL_full.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir /work/was598/modilty_gap/results/incl.124 \
#   --models clip,siglip,openclip \
#   --batch-size 128 --num-workers 8 \
#   --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#   --nas-k 100 \
#   --oti-steps 150 --ovi-steps 1000 --oti-lr 0.02 --ovi-lr 0.02 \
#   --ovi-p 4

# python OTCL_full.py \
#     --data-root /work/was598/modilty_gap/tools/data \
#     --out-dir /work/was598/modilty_gap/results/OTCL.124 \
#     --models clip,siglip,openclip \
#     --postprocs orig,spec60,ot \
#     --spec-graph-topk 50 \
#     --ot-fit-pairs 5000 \
#     --batch-size 128 --num-workers 8 \
#     --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#     --nas-k 100


# python MG.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir /work/was598/modilty_gap/results/MG.124 \
#   --models clip,siglip,openclip \
#   --batch-size 128 --num-workers 8 \
#   --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#   --nas-k 100 \
#   --mg-lambda 0.375

# python ALIGNCLIP_full.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir  /work/was598/modilty_gap/results/alignclip \
#     --models alignclip,sharedclip,imsep,orgclip \
#     --batch-size 128 --num-workers 8 \
#     --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#     --nas-k 10

# python ALIGNCLIP_full.py  \
#     --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir  /work/was598/modilty_gap/results/alignclip.124 \
#     --models clip,siglip,openclip \
#     --do-train \
#     --train-splits train,restval \
#     --train-source coco+flickr \
#     --epochs 1 --lr 1e-5 --batch-size 128 --num-workers 8 \
#     --alpha 0.5 --nas-k 100 --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#     --finetune-scope adapter

# python GR_CLIP.py \
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir  /work/was598/modilty_gap/results/GR_CLIP.124 \
#   --models clip,siglip,openclip \
#   --batch-size 128 --num-workers 8 \
#   --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
#   --nas-k 100 \
#   --calib-n 10000 \
#   --calib-cache /work/was598/modilty_gap/results/baseline6/calib_cache

# python /work/was598/modilty_gap/q2_unified_overhead_benchmark.py \
#   --data-root /work/was598/modilty_gap/tools/data/flickr30k \
#   --out-dir /work/was598/modilty_gap/results/q2_unified_siglip \
#   --device cuda \
#   --model-key siglip \
#   --batch-size 128 \
#   --num-workers 8

# python /work/was598/modilty_gap/vqav2_unified_supervised_benchmark.py \
#   --vqav2-root /work/was598/modilty_gap/tools/data/vqav2 \
#   --out-dir /work/was598/modilty_gap/results/vqav2_unified_supervised_clip \
#   --device cuda \
#   --model-key clip \
#   --model-size 32 \
#   --batch-size 128 \
#   --num-workers 2 \
#   --max-train-items 20000 \
#   --max-val-items 0 \
#   --head-epochs 30 \
#   --head-batch-size 1024 \
#   --head-eval-batch-size 4096


# python longclip_sharegpt4v_fsalign.py \
#   --sharegpt4v-root <sharegpt4v_root> \
#   --longclip-repo <Long-CLIP_repo> \
#   --longclip-checkpoint <longclip-L.pt> \
#   --out-dir /work/was598/modilty_gap/results/longclip_sharegpt4v_long_text


# 只跑 our
# OUT_BASE=/work/was598/modilty_gap/results/vqav2_soft_full \
# RUN_OUR=1 RUN_I0T=1 RUN_GR=1 RUN_UNIFIED=1 \
# bash /work/was598/modilty_gap/run_vqav2_standard_soft.sh

# # 只跑 unified supervised
# RUN_OUR=0 RUN_I0T=0 RUN_GR=0 RUN_UNIFIED=1 \
# bash /work/was598/modilty_gap/run_vqav2_standard_soft.sh

# 自定义输出目录
# OUT_BASE=/work/was598/modilty_gap/results/vqav2_soft_full \
# bash /work/was598/modilty_gap/run_vqav2_standard_soft.sh


python /work/was598/modilty_gap/plot/shared_projection_main_figure.py \
  --data-root /work/was598/modilty_gap/tools/data \
  --model-key clip \
  --clip-model ViT-B-32 \
  --lora-state /work/was598/modilty_gap/results/our_final_train/clip:ViT-B-32:openai_flickr30k_karpathy_test_lora_state.pt \
  --lora-mix 0.3 \
  --figure-samples 400 \
  --pair-line-count 36 \
  --out-prefix /work/was598/modilty_gap/plot/fk30k_shared_projection_main
