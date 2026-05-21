#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panel 1 (ICML-style): Modality gap shown via a REAL Flickr30k example.

Inputs (no Karpathy JSON):
- Flickr30k images folder (either contains jpgs directly, or has subfolder flickr30k-images/)
- results_20130124.token (each line: "<image>.jpg#k<TAB>caption")

What it does:
1) Load captions mapping from token file
2) Build a list of images that exist on disk
3) Sample up to max_images images for embedding (for speed)
4) Encode images and captions with OpenCLIP (image encoder + text encoder)
5) Choose an anchor image (random or by --anchor-image filename)
6) Find:
   - image neighbors: nearest images to anchor image embedding
   - text neighbors: nearest captions to anchor caption embedding
7) Render Panel 1:
   Left: anchor image center + 8 nearest image thumbnails around
   Right: anchor caption center + 8 nearest caption cards around
8) Save as SVG and PDF (paper-ready).

Install:
  pip install open_clip_torch torch pillow numpy matplotlib

Optional (faster dataloading not needed; this script is simple)

Example:
  python panel1_flickr30k_real_example.py \
    --images-root /data/flickr30k \
    --token-file /data/flickr30k/results_20130124.token \
    --max-images 8000 \
    --model ViT-B-32 --pretrained openai \
    --out-svg panel1_fk30k.svg --out-pdf panel1_fk30k.pdf
"""

import argparse
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_images_folder(images_root: str) -> str:
    """
    Accept either:
      - images_root contains jpgs
      - images_root/flickr30k-images contains jpgs
    """
    sub = os.path.join(images_root, "flickr30k-images")
    if os.path.isdir(sub):
        return sub
    return images_root

def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts):
            paths.append(os.path.join(folder, fn))
    paths.sort()
    return paths

def parse_results_token_file(token_path: str) -> Dict[str, List[str]]:
    """
    Parse results_20130124.token:
      1000092795.jpg#0\tTwo young guys ...
    Return: dict image_filename -> [captions]
    """
    mapping: Dict[str, List[str]] = {}
    pat = re.compile(r"^(.+?\.(jpg|jpeg|png|webp))#\d+$", flags=re.IGNORECASE)
    with open(token_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key, cap = parts[0].strip(), parts[1].strip()
            m = pat.match(key)
            if m:
                img_fn = m.group(1)
            else:
                img_fn = key.split("#")[0]
                if not img_fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    img_fn += ".jpg"
            if cap:
                mapping.setdefault(img_fn, []).append(cap)
    return mapping

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def cosine_knn(emb: np.ndarray, anchor_vec: np.ndarray, k: int) -> np.ndarray:
    """
    emb: [N,D], normalized
    anchor_vec: [D], normalized
    return indices of k nearest by cosine distance
    """
    sim = emb @ anchor_vec.reshape(-1, 1)
    sim = sim.squeeze(1)
    dist = 1.0 - sim
    return np.argsort(dist)[:k]


# -----------------------------
# OpenCLIP
# -----------------------------

def load_openclip(model_name: str, pretrained: str, device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, preprocess, tokenizer

@torch.no_grad()
def encode_images(model, preprocess, image_paths: List[str], device: str, batch_size: int) -> np.ndarray:
    feats = []
    for st in range(0, len(image_paths), batch_size):
        ed = min(len(image_paths), st + batch_size)
        ims = []
        for p in image_paths[st:ed]:
            im = Image.open(p).convert("RGB")
            ims.append(preprocess(im))
        ims = torch.stack(ims).to(device)
        f = model.encode_image(ims)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device: str, batch_size: int) -> np.ndarray:
    feats = []
    for st in range(0, len(texts), batch_size):
        ed = min(len(texts), st + batch_size)
        toks = tokenizer(texts[st:ed]).to(device)
        f = model.encode_text(toks)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


# -----------------------------
# Rendering Panel 1 (clean ICML style)
# -----------------------------

def draw_image_grid(ax, center_img: Image.Image, neighbor_imgs: List[Image.Image], title: str):
    """
    Layout: 3x3 grid with center big-ish and 8 neighbors around.
    """
    ax.set_title(title, pad=10)
    ax.axis("off")

    # positions for 8 neighbors around (normalized axes coords)
    # corners + edges around center
    pos = [
        (0.05, 0.68), (0.35, 0.78), (0.65, 0.68),
        (0.05, 0.40),             (0.65, 0.40),
        (0.05, 0.12), (0.35, 0.02), (0.65, 0.12),
    ]
    # center box
    cx, cy, cw, ch = 0.28, 0.25, 0.44, 0.50

    # draw faint rounded frame around the whole panel region
    frame = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                           boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.0, fill=False, alpha=0.25,
                           transform=ax.transAxes)
    ax.add_patch(frame)

    # place center image
    ax.imshow(center_img, extent=(cx, cx+cw, cy, cy+ch), transform=ax.transAxes, zorder=2)
    # outline center image
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.1, fill=False, alpha=0.55,
                                transform=ax.transAxes, zorder=3))

    # neighbors
    for (px, py), im in zip(pos, neighbor_imgs):
        w, h = 0.22, 0.20
        ax.imshow(im, extent=(px, px+w, py, py+h), transform=ax.transAxes, zorder=2)
        ax.add_patch(FancyBboxPatch((px, py), w, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    linewidth=0.9, fill=False, alpha=0.35,
                                    transform=ax.transAxes, zorder=3))

    # anchor marker
    ax.text(cx + 0.01, cy + ch - 0.02, "anchor", transform=ax.transAxes, fontsize=10, alpha=0.8)


def draw_text_grid(ax, center_text: str, neighbor_texts: List[str], title: str):
    """
    Layout: center caption + 8 nearest captions around (short cards).
    """
    ax.set_title(title, pad=10)
    ax.axis("off")

    frame = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                           boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.0, fill=False, alpha=0.25,
                           transform=ax.transAxes)
    ax.add_patch(frame)

    # center card
    cx, cy, cw, ch = 0.18, 0.32, 0.64, 0.36
    center_box = FancyBboxPatch((cx, cy), cw, ch,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.1, fill=False, alpha=0.55,
                                transform=ax.transAxes)
    ax.add_patch(center_box)
    ax.text(cx + 0.02, cy + ch - 0.06, "anchor caption", transform=ax.transAxes,
            fontsize=10, alpha=0.8)
    ax.text(cx + 0.02, cy + 0.10, wrap_text(center_text, 44), transform=ax.transAxes,
            fontsize=12)

    # 8 neighbor cards around
    pos = [
        (0.05, 0.72), (0.36, 0.76), (0.67, 0.72),
        (0.05, 0.52),             (0.67, 0.52),
        (0.05, 0.12), (0.36, 0.06), (0.67, 0.12),
    ]
    for (px, py), t in zip(pos, neighbor_texts):
        w, h = 0.28, 0.16
        box = FancyBboxPatch((px, py), w, h,
                             boxstyle="round,pad=0.015,rounding_size=0.025",
                             linewidth=0.9, fill=False, alpha=0.35,
                             transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(px + 0.015, py + 0.05, wrap_text(t, 28), transform=ax.transAxes, fontsize=10.5)


def wrap_text(s: str, width: int) -> str:
    s = s.strip().replace("\n", " ")
    words = s.split()
    lines, cur = [], []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) <= width:
            cur.append(w)
            cur_len += len(w) + (1 if cur_len > 0 else 0)
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines[:3]) + (" …" if len(lines) > 3 else "")


def make_panel1_figure(anchor_img_path: str,
                       anchor_caption: str,
                       image_neighbor_paths: List[str],
                       text_neighbors: List[str],
                       out_svg: str,
                       out_pdf: str) -> None:
    # Load images
    def load_thumb(p: str, size=(220, 160)) -> Image.Image:
        im = Image.open(p).convert("RGB")
        im.thumbnail(size, Image.BICUBIC)
        return im

    center_img = load_thumb(anchor_img_path, size=(520, 360))
    neigh_imgs = [load_thumb(p, size=(240, 180)) for p in image_neighbor_paths]

    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(12.2, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.12)

    axL = fig.add_subplot(gs[0, 0])
    draw_image_grid(axL, center_img, neigh_imgs, "Image neighborhood (kNN in image embedding)")

    axR = fig.add_subplot(gs[0, 1])
    draw_text_grid(axR, anchor_caption, text_neighbors, "Text neighborhood (kNN in text embedding)")

    fig.suptitle("Panel 1: Modality gap as mismatched neighborhoods around the same sample", y=1.02, fontsize=13)

    fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, required=True)
    ap.add_argument("--token-file", type=str, required=True)

    ap.add_argument("--max-images", type=int, default=8000,
                    help="Embed at most this many images for speed (random subset).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")

    ap.add_argument("--k-img", type=int, default=9,
                    help="How many images to show including anchor. Recommended 9 -> 1 center + 8 neighbors.")
    ap.add_argument("--k-txt", type=int, default=9,
                    help="How many captions to show including anchor. Recommended 9 -> 1 center + 8 neighbors.")

    ap.add_argument("--anchor-image", type=str, default="",
                    help="Optional: anchor filename like '1000092795.jpg'. If empty, random.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cache-npz", type=str, default="fk30k_panel1_cache.npz")
    ap.add_argument("--out-svg", type=str, default="panel1_fk30k.svg")
    ap.add_argument("--out-pdf", type=str, default="panel1_fk30k.pdf")
    args = ap.parse_args()

    set_seed(args.seed)

    img_folder = find_images_folder(args.images_root)
    img_paths_all = list_images(img_folder)
    if not img_paths_all:
        raise RuntimeError(f"No images found in {img_folder}")

    cap_map = parse_results_token_file(args.token_file)
    # keep only images that have captions
    path_by_fn = {os.path.basename(p): p for p in img_paths_all}
    valid_fns = [fn for fn in cap_map.keys() if fn in path_by_fn]
    if len(valid_fns) < 1000:
        raise RuntimeError(f"Too few matched images with captions: {len(valid_fns)}. Check token file vs images.")

    # sample subset for speed
    rng = random.Random(args.seed)
    rng.shuffle(valid_fns)
    valid_fns = valid_fns[:min(args.max_images, len(valid_fns))]

    image_paths = [path_by_fn[fn] for fn in valid_fns]
    # choose one representative caption per image for neighbor search in caption space (anchor uses its own)
    rep_caps = [cap_map[fn][0] for fn in valid_fns]

    # Determine anchor index
    if args.anchor_image:
        if args.anchor_image not in path_by_fn:
            raise ValueError(f"anchor-image '{args.anchor_image}' not found under images-root")
        # anchor must also be in our sampled list; if not, force include
        if args.anchor_image not in valid_fns:
            valid_fns[0] = args.anchor_image
            image_paths[0] = path_by_fn[args.anchor_image]
            rep_caps[0] = cap_map[args.anchor_image][0]
        anchor_idx = valid_fns.index(args.anchor_image)
    else:
        anchor_idx = rng.randint(0, len(valid_fns) - 1)

    # Load/compute embeddings
    img_emb, txt_emb = None, None
    if os.path.isfile(args.cache_npz):
        cache = np.load(args.cache_npz, allow_pickle=True)
        if "img_emb" in cache and "txt_emb" in cache and cache["img_emb"].shape[0] == len(image_paths):
            img_emb = cache["img_emb"]
            txt_emb = cache["txt_emb"]

    if img_emb is None:
        model, preprocess, tokenizer = load_openclip(args.model, args.pretrained, args.device)

        img_emb = encode_images(model, preprocess, image_paths, args.device, args.batch_size)
        txt_emb = encode_texts(model, tokenizer, rep_caps, args.device, args.batch_size)

        img_emb = l2_normalize(img_emb)
        txt_emb = l2_normalize(txt_emb)

        np.savez_compressed(args.cache_npz, img_emb=img_emb, txt_emb=txt_emb, fns=np.array(valid_fns))

    # --- image neighbors ---
    anchor_img_vec = img_emb[anchor_idx]
    img_nn_idx = cosine_knn(img_emb, anchor_img_vec, k=args.k_img)
    # first is anchor itself; neighbors are next 8
    img_neighbors = [image_paths[i] for i in img_nn_idx[1:args.k_img]]

    # --- text neighbors ---
    # Use anchor caption as: the FIRST caption of anchor image (or you can change to random among 5)
    anchor_fn = valid_fns[anchor_idx]
    anchor_caps = cap_map[anchor_fn]
    anchor_caption = anchor_caps[0]

    # Build a caption pool for text neighbor search:
    # to keep it simple + fast, we search among representative captions of images.
    anchor_txt_vec = txt_emb[anchor_idx]
    txt_nn_idx = cosine_knn(txt_emb, anchor_txt_vec, k=args.k_txt)
    txt_neighbors = [rep_caps[i] for i in txt_nn_idx[1:args.k_txt]]

    # Render panel
    os.makedirs(os.path.dirname(args.out_svg) or ".", exist_ok=True)
    make_panel1_figure(
        anchor_img_path=image_paths[anchor_idx],
        anchor_caption=anchor_caption,
        image_neighbor_paths=img_neighbors,
        text_neighbors=txt_neighbors,
        out_svg=args.out_svg,
        out_pdf=args.out_pdf
    )

    print(f"[Saved] {args.out_svg}")
    print(f"[Saved] {args.out_pdf}")
    print(f"[Anchor] {anchor_fn}")
    print(f"[Caption] {anchor_caption}")

if __name__ == "__main__":
    main()
