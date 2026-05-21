# python -m src.train --emb_dir data/flickr30k/clip_embeddings --epochs 30 --batch_size 1024 --lr 1e-3 --proj_dim 512 --num_scales 8 --r_min 0.1 --r_max 5.0 --alpha_mode spectral --learn_Q true --device cuda
export CUDA_VISIBLE_DEVICES=1
MODEL_SIZE=32

# python /work/was598/modilty_gap/our_code_final.py\
#   --data-root /work/was598/modilty_gap/tools/data \
#   --out-dir /work/was598/modilty_gap/results/our_final_train_1.27d\
#   --models clip,siglip,openclip \
#   --model-size ${MODEL_SIZE} \
#   --batch-size 128 --num-workers 8 \
#   --max-coco 999999 --max-flickr 999999 --max-cls 999999 \
#   --nas-k 100 --nas-max-items 5000 --intra-samples 20000 \
#   --train-epochs 30 --train-anchors 1000 --spectral-samples 1000 \
#   --train-lr 5e-4 --lambda-dbl 0.5 --lambda-spec 0.05 --lambda-match 0.05 \
#   --lambda-align 2.0 --lambda-orth 0.1 --train-reg 1e-3 \
#   --lora-rank 8 --lora-alpha 8.0 --save-lora \
#   --multi-caption --lora-mix 0.3 \
#   --align-samples 2048

# Eval-only with trained LoRA (zero-shot only, CLIP-matching LoRA)
python /work/was598/modilty_gap/our_code_final.py \
  --data-root /work/was598/modilty_gap/tools/data \
  --out-dir /work/was598/modilty_gap/results/our_final_eval_lora_1.27 \
  --models clip \
  --model-size ${MODEL_SIZE} \
  --batch-size 128 --num-workers 8 \
  --max-coco 999999 --max-flickr 999999 --max-cls 999999 \
  --nas-k 100 --nas-max-items 5000 --intra-samples 20000 \
  --train-epochs 0 --only-zeroshot \
  --lora-state /work/was598/modilty_gap/results/our_final_train_1.27/clip_ViT-B-32_openai_flickr30k_karpathy_test_lora_state.pt \
  --lora-mix 0.3
