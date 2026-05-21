# python fractal.py \
#   --images-root /work/was598/modilty_gap/tools/data/flickr30k \
#   --karpathy-json /work/was598/modilty_gap/tools/data/flickr30k/karpathy_splits.json \
#   --split test \
#   --max-samples 6000 \
#   --model ViT-B-32 --pretrained openai \
#   --k-nn 250 \
#   --kde-bw 0.28 \
#   --out-svg fractal_flickr30k.svg \
#   --out-pdf fractal_flickr30k.pdf

# python fractal.py \
#   --mode local \
#   --images-root /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images \
#   --ann-file /work/was598/modilty_gap/tools/data/flickr30k/results_20130124.token \
#   --keep-one-caption \
#   --max-samples 6000 \
#   --model ViT-B-32 --pretrained openai \
#   --k-nn 250 \
#   --kde-bw 0.28 \
#   --out-svg panel2_fractal_flickr30k.svg \
#   --out-pdf panel2_fractal_flickr30k.pdf

# python fractal.py \
#   --images-root /work/was598/modilty_gap/tools/data/flickr30k \
#   --captions /work/was598/modilty_gap/tools/data/flickr30k/results_20130124.token \
#   --max-pairs 9000 \
#   --model ViT-B-32 --pretrained openai \
#   --k-nn 800 \
#   --n-show 120 \
#   --out-pdf panel2_fractal_3d.pdf \
#   --out-svg panel2_fractal_3d.svg

# python fractal.py \
#   --images-root /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images \
#   --captions-file /work/was598/modilty_gap/tools/data/flickr30k/results_20130124.token \
#   --max-images 9000 \
#   --model ViT-B-32 --pretrained openai \
#   --k-nn 480 \
#   --out-svg panel2_fractal.svg \
#   --out-pdf panel2_fractal.pdf


python part_plot.py \
  --images-root /work/was598/modilty_gap/tools/data/flickr30k \
  --token-file /work/was598/modilty_gap/tools/data/flickr30k/results_20130124.token \
  --max-images 12000 \
  --model ViT-B-32 --pretrained openai \
  --out-svg panel1_fk30k.svg --out-pdf panel1_fk30k.pdf

# Stronger LNO CDF run for the paper-style Flickr30k CLIP comparison.
python /work/was598/modilty_gap/plot/local_neighborhood_overlap_cdf.py \
  --data-root /work/was598/modilty_gap/tools/data \
  --dataset flickr30k \
  --model-key clip \
  --clip-model ViT-B-32 \
  --text-variant short \
  --paragraph-sentences 3 \
  --ks 10,50,100 \
  --lora-state /work/was598/modilty_gap/results/our_final_train_1.24/clip_ViT-B-32_openai_flickr30k_karpathy_test_lora_state.pt \
  --lora-mix 0.6 \
  --out-prefix /work/was598/modilty_gap/plot/flickr30k_lno_cdf_full127
