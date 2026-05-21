# python demo_fractal_levels_viz.py \
#   --image /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1000092795.jpg \
#   --caption "Two young guys with shaggy hair look at their hands while hanging out in the yard." \
#   --out_dir outputs_demo \
#   --clip_model ViT-B-32 \
#   --clip_pretrained openai \
#   --device cuda


# python demo_hierarchical_alignment_viz.py \
#   --image  /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1000092795.jpg \
#   --caption "Two young guys with shaggy hair look at their hands while hanging out in the yard." \
#   --out_dir outputs_hier_demo \
#   --device cuda

# python fractal_manifold_demo.py \
#   --img /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1000092795.jpg \
#   --caption "Two young guys with shaggy hair look at their hands while hanging out in the yard." \
#   --out ./fm_out

  # python fm_single_generic.py \
  # --img /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1000092795.jpg \
  # --caption "Two young guys with shaggy hair look at their hands while hanging out in the yard." \
  # --out ./fm_out \
  # --yolo_model yolov8l.pt \
  # --conf 0.30 \
  # --parts 3

  # 最小依赖（YOLO + 通用部件兜底）
# python fm_single_allentities_parts.py \
#   --img /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1000092795.jpg \
#   --caption "Two young guys with shaggy hair look at their hands while hanging out in the yard." \
#   --out ./fm_out \
#   --yolo_model yolov8x.pt --conf 0.25 --imgsz 1024 \
#   --parts 3

# 开启开放词表“语义部件”（推荐，有更强的部件语义）
# python fm_single_allentities_parts.py \
#   --img /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1000092795.jpg \
#   --caption "..." \
#   --out ./fm_out \
#   --yolo_model yolov8x.pt --conf 0.25 --imgsz 1024 \
#   --ov_on --ov_cfg GroundingDINO_SwinT_OGC.py --ov_weights groundingdino_swint_ogc.pth \
#   --parts 3

# # 如需把“背景/区域类”也视为实体（table/sofa/tv 等区域）再打开 DeepLab：
# python fm_single_allentities_parts.py \
#   --img /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1001545525.jpg \
#   --caption "Two men in Germany jumping over a rail at the same time without shirts." \
#   --out ./fm_out \
#   --yolo_model yolov8x.pt --conf 0.25 --imgsz 1024 \
#   --ov_on --ov_cfg GroundingDINO_SwinT_OGC.py --ov_weights groundingdino_swint_ogc.pth \
#   --seg_on --seg_min_area 800 \
#   --parts 3

# python fractal_manifold_demo.py \
#   --image /work/was598/modilty_gap/tools/data/flickr30k/flickr30k-images/1001545525.jpg \
#   --caption "Two young guys with shaggy hair look at their hands while hanging out in the yard." \
#   --out_dir ./fractal_outputs \
#   --device cuda 

python eval_modality_gap_and_zeroshot.py \
  --data_root /work/was598/modilty_gap/tools/data \
  --model_name ViT-B-32 --pretrained openai \
  --batch_size 256 --num_workers 8 \
  --device cuda \
  --out_dir ./fractal_eval_outputs
