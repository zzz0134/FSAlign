# Fractal Manifold Alignment for Cross-Modal Retrieval & Zero-Shot Classification

This repository implements a **unified modality alignment framework** based on a **Fractal Manifold** and **Fractal Diffusion Kernel**.
It supports:
- **Text-to-Image** retrieval
- **Image-to-Text** retrieval
- **Zero-Shot Classification** (point vs. distribution matching)

Dataset: **Flickr30K** with **CLIP** embeddings.

## Quick Start

### 0) Install
```
pip install -r requirements.txt
```

### 1) Precompute CLIP embeddings
```
python -m src.precompute_embeddings   --data_root data/flickr30k   --images_dir flickr30k-images   --captions_file results_20130124.token   --out_dir data/flickr30k/clip_embeddings   --clip_model ViT-B-32 --clip_pretrained openai   --batch_size 128 --device cuda
```

### 2) Train fractal alignment
```
python -m src.train   --emb_dir data/flickr30k/clip_embeddings   --epochs 30 --batch_size 1024 --lr 1e-3   --proj_dim 512   --num_scales 8   --r_min 0.1 --r_max 5.0   --alpha_mode spectral --learn_Q true   --device cuda
```

### 3) Evaluate retrieval
```
python -m src.eval_retrieval   --emb_dir data/flickr30k/clip_embeddings   --ckpt checkpoints/best.pt   --num_scales 8 --r_min 0.1 --r_max 5.0   --alpha_mode spectral   --device cuda
```

### 4) Evaluate zero-shot classification
Prepare a `prompts.json` like:
```json
{
  "dog": [0, 1, 2],
  "cat": [3, 4, 5]
}
```
> Here the integers are indices into `txt_embeddings.pt` **for demo**. In practice, re-encode your class prompts via CLIP text encoder and pass those embeddings in (you can adapt `eval_zsc.py` accordingly).

Run:
```
python -m src.eval_zsc   --emb_dir data/flickr30k/clip_embeddings   --ckpt checkpoints/best.pt   --prompts_file prompts.json   --device cuda
```

## Notes
- This code trains small projection heads and a learnable **fractal diffusion kernel** (with spectral dimension `Q`) on top of CLIP features.
- Retrieval uses **fractal diffusion distance**; classification uses **fractal MMD energy**.
- The design mirrors the paper's idea: **align distributions across scales** to shrink the modality gap.

