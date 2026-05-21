#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICML-style Panel 1 (REAL Flickr30k): show modality gap via mismatched neighborhoods
- Left: anchor image + 8 image neighbors (with connecting lines)
- Right: anchor caption + 8 text neighbors (radial cards + connecting lines + similarity)

NO Karpathy JSON. Uses:
- images under --images-root (or --images-root/flickr30k-images)
- captions under --token-file (results_20130124.token)

Install:
  pip install open_clip_torch torch pillow numpy matplotlib

Run:
  python panel1_fk30k_icml_neighbors.py \
    --images-root /path/to/flickr30k \
    --token-file /path/to/results_20130124.token \
    --max-images 12000 \
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
    sub = os.path.join(images_root, "flickr30k-images")
    return sub if os.path.isdir(sub) else images_root

def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.lower().endswith(exts)]
    paths.sort()
    return paths

def parse_results_token_file(token_path: str) -> Dict[str, List[str]]:
    """
    results_20130124.token:
      1000092795.jpg#0\tcaption ...
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

def cosine_sim_all(emb: np.ndarray, anchor_vec: np.ndarray) -> np.ndarray:
    # emb & anchor_vec assumed normalized
    return (emb @ anchor_vec.reshape(-1, 1)).squeeze(1)

def cosine_knn_indices_and_sims(emb: np.ndarray, anchor_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sim = cosine_sim_all(emb, anchor_vec)
    dist = 1.0 - sim
    idx = np.argsort(dist)[:k]
    return idx, sim[idx]

def wrap_text(s: str, width: int, max_lines: int = 2) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    words = s.split()
    lines, cur = [], []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add <= width:
            cur.append(w)
            cur_len += add
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    out = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        out += " …"
    return out

# simple stopwords (enough for captions)
STOPWORDS = set("""
a an the and or of to in on at for with from by is are was were be been being
this that these those it its as into over under above below around near
man woman people person boy girl men women children child
""".split())

def keyword_phrase(s: str, max_words: int = 7) -> str:
    """
    Make a short, readable neighbor label:
    keep informative tokens, drop stopwords/punctuation, keep order.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    if not toks:
        return wrap_text(s, 24, 1)
    return " ".join(toks[:max_words])


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
# Rendering helpers
# -----------------------------

def add_frame(ax):
    frame = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                           boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.0, fill=False, alpha=0.20,
                           transform=ax.transAxes)
    ax.add_patch(frame)

def draw_image_neighbors_radial(ax, center_img: Image.Image, neighbor_imgs: List[Image.Image],
                               neighbor_sims: List[float], title: str):
    """
    Center image + 8 neighbors around a circle, with faint connecting lines.
    """
    ax.set_title(title, pad=10)
    ax.axis("off")
    add_frame(ax)

    # center placement
    cx, cy, cw, ch = 0.30, 0.26, 0.40, 0.48
    ax.imshow(center_img, extent=(cx, cx+cw, cy, cy+ch), transform=ax.transAxes, zorder=3)
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.1, fill=False, alpha=0.55,
                                transform=ax.transAxes, zorder=4))
    ax.text(cx + 0.01, cy + ch - 0.02, "anchor", transform=ax.transAxes, fontsize=10, alpha=0.85)

    # neighbor ring
    ring_r = 0.34
    thumb_w, thumb_h = 0.22, 0.18
    center_pt = (cx + cw/2, cy + ch/2)

    angles = np.linspace(0, 2*np.pi, num=len(neighbor_imgs)+1)[:-1] + np.pi/10
    for im, sim, ang in zip(neighbor_imgs, neighbor_sims, angles):
        px = center_pt[0] + ring_r * np.cos(ang) - thumb_w/2
        py = center_pt[1] + ring_r * np.sin(ang) - thumb_h/2

        # line (alpha by similarity)
        alpha = 0.15 + 0.35 * float(np.clip((sim - 0.15) / 0.60, 0, 1))
        ax.plot([center_pt[0], px + thumb_w/2], [center_pt[1], py + thumb_h/2],
                transform=ax.transAxes, linewidth=1.0, alpha=alpha, zorder=1)

        ax.imshow(im, extent=(px, px+thumb_w, py, py+thumb_h), transform=ax.transAxes, zorder=2)
        ax.add_patch(FancyBboxPatch((px, py), thumb_w, thumb_h,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    linewidth=0.9, fill=False, alpha=0.30,
                                    transform=ax.transAxes, zorder=3))

def draw_text_neighbors_radial(ax, anchor_caption: str, neighbor_captions: List[str],
                              neighbor_sims: List[float], title: str):
    """
    Center caption card + 8 neighbor cards around, with connecting lines and similarity labels.
    """
    ax.set_title(title, pad=10)
    ax.axis("off")
    add_frame(ax)

    # center caption card
    ccx, ccy, ccw, cch = 0.10, 0.38, 0.80, 0.26
    center_box = FancyBboxPatch((ccx, ccy), ccw, cch,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.1, fill=False, alpha=0.55,
                                transform=ax.transAxes, zorder=3)
    ax.add_patch(center_box)
    ax.text(ccx + 0.02, ccy + cch - 0.06, "anchor caption", transform=ax.transAxes,
            fontsize=10, alpha=0.85)
    ax.text(ccx + 0.02, ccy + 0.06, wrap_text(anchor_caption, 72, max_lines=2),
            transform=ax.transAxes, fontsize=12, va="bottom")

    center_pt = (ccx + ccw/2, ccy + cch/2)

    # neighbor cards around
    ring_r = 0.36
    card_w, card_h = 0.26, 0.15
    angles = np.linspace(0, 2*np.pi, num=len(neighbor_captions)+1)[:-1] + np.pi/14

    for cap, sim, ang in zip(neighbor_captions, neighbor_sims, angles):
        px = center_pt[0] + ring_r * np.cos(ang) - card_w/2
        py = center_pt[1] + ring_r * np.sin(ang) - card_h/2

        # keep inside axes bounds gently
        px = float(np.clip(px, 0.03, 0.97-card_w))
        py = float(np.clip(py, 0.05, 0.95-card_h))

        # connecting line: alpha by similarity
        alpha = 0.18 + 0.40 * float(np.clip((sim - 0.15) / 0.60, 0, 1))
        ax.plot([center_pt[0], px + card_w/2], [center_pt[1], py + card_h/2],
                transform=ax.transAxes, linewidth=1.0, alpha=alpha, zorder=1)

        box = FancyBboxPatch((px, py), card_w, card_h,
                             boxstyle="round,pad=0.014,rounding_size=0.025",
                             linewidth=0.9, fill=False, alpha=0.30,
                             transform=ax.transAxes, zorder=2)
        ax.add_patch(box)

        # label: keyword phrase (readable) + similarity
        phrase = keyword_phrase(cap, max_words=7)
        ax.text(px + 0.012, py + 0.055, wrap_text(phrase, 20, max_lines=2),
                transform=ax.transAxes, fontsize=10.5, va="bottom", zorder=3)
        ax.text(px + card_w - 0.012, py + card_h - 0.035, f"{sim:.2f}",
                transform=ax.transAxes, fontsize=9.5, alpha=0.75,
                ha="right", zorder=3)


def make_panel1(anchor_img_path: str,
                anchor_caption: str,
                image_neighbor_paths: List[str],
                image_neighbor_sims: List[float],
                text_neighbor_caps: List[str],
                text_neighbor_sims: List[float],
                overlap_ratio: float,
                out_svg: str,
                out_pdf: str):
    def load_thumb(p: str, size=(280, 210)) -> Image.Image:
        im = Image.open(p).convert("RGB")
        im.thumbnail(size, Image.BICUBIC)
        return im

    center_img = load_thumb(anchor_img_path, size=(640, 480))
    neigh_imgs = [load_thumb(p, size=(320, 240)) for p in image_neighbor_paths]

    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(13.2, 5.0))
    gs = fig.add_gridspec(1, 2, wspace=0.10)

    axL = fig.add_subplot(gs[0, 0])
    draw_image_neighbors_radial(
        axL, center_img, neigh_imgs, image_neighbor_sims,
        "Image neighborhood (kNN in image embedding)"
    )

    axR = fig.add_subplot(gs[0, 1])
    draw_text_neighbors_radial(
        axR, anchor_caption, text_neighbor_caps, text_neighbor_sims,
        "Text neighborhood (kNN in text embedding)"
    )

    fig.suptitle(
        "Panel 1: Modality gap as mismatched neighborhoods around the same paired sample",
        y=1.03, fontsize=13
    )
    fig.text(0.5, 0.99, f"Overlap@8 (paired IDs) = {overlap_ratio:.2f}",
             ha="center", va="top", fontsize=11, alpha=0.80)

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

    ap.add_argument("--max-images", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")

    ap.add_argument("--k", type=int, default=9,
                    help="Total neighbors including anchor. Use 9 = anchor + 8 neighbors.")
    ap.add_argument("--anchor-image", type=str, default="",
                    help="Optional: specify an image filename like '1000092795.jpg'.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cache-npz", type=str, default="fk30k_panel1_cache_icml.npz")
    ap.add_argument("--out-svg", type=str, default="panel1_fk30k.svg")
    ap.add_argument("--out-pdf", type=str, default="panel1_fk30k.pdf")
    args = ap.parse_args()

    set_seed(args.seed)

    img_folder = find_images_folder(args.images_root)
    all_img_paths = list_images(img_folder)
    if not all_img_paths:
        raise RuntimeError(f"No images found in: {img_folder}")

    cap_map = parse_results_token_file(args.token_file)
    path_by_fn = {os.path.basename(p): p for p in all_img_paths}
    valid_fns = [fn for fn in cap_map.keys() if fn in path_by_fn]
    if len(valid_fns) < 1000:
        raise RuntimeError(f"Too few matched images+captions: {len(valid_fns)}. Check filenames/token file.")

    rng = random.Random(args.seed)
    rng.shuffle(valid_fns)
    valid_fns = valid_fns[:min(args.max_images, len(valid_fns))]

    image_paths = [path_by_fn[fn] for fn in valid_fns]
    rep_caps = [cap_map[fn][0] for fn in valid_fns]  # one caption per image for text pool

    # choose anchor
    if args.anchor_image:
        if args.anchor_image not in path_by_fn:
            raise ValueError(f"anchor-image '{args.anchor_image}' not found under images-root.")
        if args.anchor_image not in valid_fns:
            # force include anchor into sampled subset
            valid_fns[0] = args.anchor_image
            image_paths[0] = path_by_fn[args.anchor_image]
            rep_caps[0] = cap_map[args.anchor_image][0]
        anchor_idx = valid_fns.index(args.anchor_image)
    else:
        anchor_idx = rng.randint(0, len(valid_fns) - 1)

    # embeddings cache
    img_emb, txt_emb = None, None
    if os.path.isfile(args.cache_npz):
        cache = np.load(args.cache_npz, allow_pickle=True)
        if "img_emb" in cache and "txt_emb" in cache and cache["img_emb"].shape[0] == len(image_paths):
            img_emb = cache["img_emb"]
            txt_emb = cache["txt_emb"]

    if img_emb is None:
        model, preprocess, tokenizer = load_openclip(args.model, args.pretrained, args.device)
        img_emb = l2_normalize(encode_images(model, preprocess, image_paths, args.device, args.batch_size))
        txt_emb = l2_normalize(encode_texts(model, tokenizer, rep_caps, args.device, args.batch_size))
        np.savez_compressed(args.cache_npz, img_emb=img_emb, txt_emb=txt_emb, fns=np.array(valid_fns))

    # kNN on image embeddings
    anchor_img_vec = img_emb[anchor_idx]
    img_idx, img_sims = cosine_knn_indices_and_sims(img_emb, anchor_img_vec, k=args.k)
    # skip self
    img_neighbors_idx = img_idx[1:args.k]
    img_neighbors_paths = [image_paths[i] for i in img_neighbors_idx]
    img_neighbors_sims = [float(s) for s in img_sims[1:args.k]]

    # anchor caption
    anchor_fn = valid_fns[anchor_idx]
    anchor_caption = cap_map[anchor_fn][0]

    # kNN on text embeddings (pool = rep captions)
    anchor_txt_vec = txt_emb[anchor_idx]
    txt_idx, txt_sims = cosine_knn_indices_and_sims(txt_emb, anchor_txt_vec, k=args.k)
    txt_neighbors_idx = txt_idx[1:args.k]
    txt_neighbors_caps = [rep_caps[i] for i in txt_neighbors_idx]
    txt_neighbors_sims = [float(s) for s in txt_sims[1:args.k]]

    # Overlap@8 (paired IDs): compare neighbor image IDs and neighbor text IDs (since text pool is per-image)
    k_eff = args.k - 1
    overlap = len(set(img_neighbors_idx[:k_eff]).intersection(set(txt_neighbors_idx[:k_eff])))
    overlap_ratio = overlap / float(k_eff) if k_eff > 0 else 0.0

    # render
    os.makedirs(os.path.dirname(args.out_svg) or ".", exist_ok=True)
    make_panel1(
        anchor_img_path=image_paths[anchor_idx],
        anchor_caption=anchor_caption,
        image_neighbor_paths=img_neighbors_paths,
        image_neighbor_sims=img_neighbors_sims,
        text_neighbor_caps=txt_neighbors_caps,
        text_neighbor_sims=txt_neighbors_sims,
        overlap_ratio=overlap_ratio,
        out_svg=args.out_svg,
        out_pdf=args.out_pdf
    )

    print(f"[Saved] {args.out_svg}")
    print(f"[Saved] {args.out_pdf}")
    print(f"[Anchor] {anchor_fn}")
    print(f"[Caption] {anchor_caption}")
    print(f"[Overlap@{k_eff}] {overlap_ratio:.3f}")

if __name__ == "__main__":
    main()
