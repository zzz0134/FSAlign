python -m src.eval_retrieval --emb_dir data/flickr30k/clip_embeddings --ckpt checkpoints/best.pt --num_scales 8 --r_min 0.1 --r_max 5.0 --alpha_mode spectral --device cuda
