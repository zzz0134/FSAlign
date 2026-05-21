#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panel 2 (ICML clean style, NO edges):
- Points only (real Flickr30k embeddings neighborhood)
- 3 dashed multi-scale balls B_{d_theta}(x,r)
- 2 zoom-in lenses to visually show self-similarity (fractal intuition) WITHOUT any graph edges

NO karpathy json. Uses Flickr30k token file (results_20130124.token).

Inputs:
  --images-root
  --captions-file

Outputs:
  panel2_fractal_zoom_style.svg / .pdf

Deps:
  pip install open_clip_torch pillow numpy torch scikit-learn matplotlib
Optional (better 2D):
  pip install umap-learn
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from sklearn.decomposition import PCA


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def cosine_knn(emb: np.ndarray, anchor_idx: int, k: int) -> np.ndarray:
    """Return kNN indices by cosine distance (emb assumed normalized)."""
    a = emb[anchor_idx:anchor_idx+1]
    sim = (emb @ a.T).squeeze(1)
    dist = 1.0 - sim
    return np.argsort(dist)[:k]


# -----------------------------
# Flickr30k token captions loader (NO Karpathy JSON)
# -----------------------------

@dataclass
class Pair:
    image_path: str
    caption: str

def load_flickr30k_tokens(images_root: str,
                          captions_file: str,
                          one_caption_per_image: bool = True,
                          max_images: int = 0,
                          seed: int = 0) -> List[Pair]:
    if not os.path.isfile(captions_file):
        raise FileNotFoundError(f"captions-file not found: {captions_file}")

    caps: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            key, cap = line.split("\t", 1)
            img_name = key.split("#")[0]
            caps.setdefault(img_name, []).append(cap.strip())

    if not caps:
        raise RuntimeError("No captions parsed. Check captions-file format.")

    rng = random.Random(seed)
    items = list(caps.items())
    rng.shuffle(items)

    pairs: List[Pair] = []
    for img_name, cap_list in items:
        img_path = os.path.join(images_root, img_name)
        if not os.path.isfile(img_path):
            continue

        if one_caption_per_image:
            pairs.append(Pair(img_path, cap_list[0]))
        else:
            for cap in cap_list:
                pairs.append(Pair(img_path, cap))

        if max_images > 0 and len(pairs) >= max_images:
            break

    if len(pairs) < 1200:
        raise RuntimeError(f"Only loaded {len(pairs)} usable pairs. Check images-root/captions-file.")
    return pairs


# -----------------------------
# OpenCLIP encoder
# -----------------------------

def load_openclip(model_name: str, pretrained: str, device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, preprocess, tokenizer

@torch.no_grad()
def encode_openclip(model, preprocess, tokenizer,
                    image_paths: List[str], texts: List[str],
                    device: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    img_out, txt_out = [], []
    N = len(image_paths)
    for st in range(0, N, batch_size):
        ed = min(N, st + batch_size)

        imgs = []
        for p in image_paths[st:ed]:
            im = Image.open(p).convert("RGB")
            imgs.append(preprocess(im))
        imgs = torch.stack(imgs).to(device)

        toks = tokenizer(texts[st:ed]).to(device)

        img_f = model.encode_image(imgs)
        txt_f = model.encode_text(toks)

        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

        img_out.append(img_f.cpu().numpy())
        txt_out.append(txt_f.cpu().numpy())

    return np.concatenate(img_out, 0), np.concatenate(txt_out, 0)


# -----------------------------
# 2D layout (UMAP preferred)
# -----------------------------

def to_2d(points: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=0.24,
            metric="cosine",
            random_state=seed
        )
        return reducer.fit_transform(points)
    except Exception:
        return PCA(n_components=2, random_state=seed).fit_transform(points)


# -----------------------------
# Plot helpers (clean ICML style)
# -----------------------------

def choose_radii(pts2d: np.ndarray, anchor_pos: int) -> List[float]:
    c = pts2d[anchor_pos]
    dist = np.linalg.norm(pts2d - c[None, :], axis=1)
    r1 = np.percentile(dist, 35)
    r2 = np.percentile(dist, 62)
    r3 = np.percentile(dist, 86)
    return [r1, r2, r3]

def point_style(pts2d: np.ndarray, anchor_pos: int):
    c = pts2d[anchor_pos]
    d = np.linalg.norm(pts2d - c[None, :], axis=1)
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    sizes = 16 + (1.0 - d) * 44
    alphas = 0.30 + (1.0 - d) * 0.60
    return sizes, alphas

def draw_points_only_panel(ax,
                           pts2d: np.ndarray,
                           anchor_pos: int,
                           title: str,
                           show_math: bool = True):
    ax.set_title(title, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_alpha(0.20)

    c = pts2d[anchor_pos]
    sizes, alphas = point_style(pts2d, anchor_pos)

    # points (alpha per point to create "structure" without edges)
    ax.scatter(pts2d[:, 0], pts2d[:, 1], s=sizes, c="C0",
               alpha=alphas, edgecolor="white", linewidth=0.35)

    # anchor: dark disk + white X (like your example)
    ax.scatter([c[0]], [c[1]], s=300, c="C0", alpha=0.95,
               edgecolor="black", linewidth=1.0, zorder=5)
    ax.scatter([c[0]], [c[1]], s=180, marker="X",
               edgecolor="white", linewidth=1.4, zorder=6)

    # multi-scale dashed balls
    r1, r2, r3 = choose_radii(pts2d, anchor_pos)
    for r in [r1, r2, r3]:
        circ = Circle((c[0], c[1]), radius=r, fill=False,
                      linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.55, edgecolor="C0")
        ax.add_patch(circ)
    ax.text(c[0] + 0.02, c[1] + 0.02, r"$t_1<t_2<t_3$", fontsize=11, alpha=0.80)

    if show_math:
        ax.text(0.02, 0.04, r"multi-scale $B_{d_\theta}(x,r)$",
                transform=ax.transAxes, fontsize=11, alpha=0.80)

    # limits
    xmin, xmax = pts2d[:, 0].min(), pts2d[:, 0].max()
    ymin, ymax = pts2d[:, 1].min(), pts2d[:, 1].max()
    pad = 0.18
    ax.set_xlim(xmin - pad*(xmax-xmin), xmax + pad*(xmax-xmin))
    ax.set_ylim(ymin - pad*(ymax-ymin), ymax + pad*(ymax-ymin))

def add_zoom_lens(fig,
                  parent_ax,
                  pts2d: np.ndarray,
                  anchor_pos: int,
                  lens_radius: float,
                  inset_rect: Tuple[float, float, float, float],
                  seed: int):
    """
    Add a zoom-in inset that shows points within a given radius.
    inset_rect: (x0,y0,w,h) in figure coordinates.
    """
    c = pts2d[anchor_pos]
    dist = np.linalg.norm(pts2d - c[None, :], axis=1)
    mask = dist <= lens_radius
    sub = pts2d[mask]
    if sub.shape[0] < 20:
        # if too few, expand a bit
        mask = dist <= (1.35 * lens_radius)
        sub = pts2d[mask]

    ax_in = fig.add_axes(inset_rect)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for sp in ax_in.spines.values():
        sp.set_alpha(0.25)

    # draw subpoints
    # (keep same style but slightly stronger to emphasize zoom)
    # use alpha from distance within subset
    d = np.linalg.norm(sub - c[None, :], axis=1)
    if d.max() > 1e-9:
        d = d / (d.max() + 1e-9)
    sizes = 14 + (1.0 - d) * 34
    alphas = 0.35 + (1.0 - d) * 0.55

    ax_in.scatter(sub[:, 0], sub[:, 1], s=sizes, c="C0",
                  alpha=alphas, edgecolor="white", linewidth=0.30)
    ax_in.scatter([c[0]], [c[1]], s=220, c="C0", alpha=0.95,
                  edgecolor="black", linewidth=0.9, zorder=5)
    ax_in.scatter([c[0]], [c[1]], s=140, marker="X",
                  edgecolor="white", linewidth=1.2, zorder=6)

    # inset limits tight
    xmin, xmax = sub[:, 0].min(), sub[:, 0].max()
    ymin, ymax = sub[:, 1].min(), sub[:, 1].max()
    pad = 0.22
    ax_in.set_xlim(xmin - pad*(xmax-xmin), xmax + pad*(xmax-xmin))
    ax_in.set_ylim(ymin - pad*(ymax-ymin), ymax + pad*(ymax-ymin))

    # link with a subtle arrow (optional but looks like your example)
    # compute parent-to-inset direction
    # place arrow from near anchor to inset box center in figure coords
    box_cx = inset_rect[0] + inset_rect[2] * 0.5
    box_cy = inset_rect[1] + inset_rect[3] * 0.5
    arrow = FancyArrowPatch(
        (0.0, 0.0), (0.0, 0.0),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        alpha=0.35,
        color="C0"
    )
    # start point: anchor projected into figure coords
    start_disp = parent_ax.transData.transform((c[0], c[1]))
    start_fig = fig.transFigure.inverted().transform(start_disp)
    arrow.set_positions((start_fig[0], start_fig[1]), (box_cx, box_cy))
    fig.patches.append(arrow)

    # small label to hint self-similarity
    ax_in.text(0.05, 0.06, "zoom-in", transform=ax_in.transAxes, fontsize=10.5, alpha=0.75)


def make_panel2(points_txt2d, points_img2d, txt_anchor_pos, img_anchor_pos,
                out_svg, out_pdf, seed):
    plt.rcParams.update({
        "figure.dpi": 260,
        "savefig.dpi": 500,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 0.8,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(10.8, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.16)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_points_only_panel(ax1, points_txt2d, txt_anchor_pos, "Text fractal latent space (multi-scale view)")

    ax2 = fig.add_subplot(gs[0, 1])
    draw_points_only_panel(ax2, points_img2d, img_anchor_pos, "Image fractal latent space (multi-scale view)")

    # add zoom lenses (this is the "fractal intuition" without edges)
    # lens radius chosen from the first dashed ball (small scale)
    txt_r1 = choose_radii(points_txt2d, txt_anchor_pos)[0]
    img_r1 = choose_radii(points_img2d, img_anchor_pos)[0]

    # insets positions: top-right corner inside each subplot region (figure coords)
    # These values are tuned to look like ICML figures.
    add_zoom_lens(fig, ax1, points_txt2d, txt_anchor_pos, lens_radius=txt_r1 * 0.75,
                  inset_rect=(0.33, 0.60, 0.14, 0.26), seed=seed)
    add_zoom_lens(fig, ax2, points_img2d, img_anchor_pos, lens_radius=img_r1 * 0.75,
                  inset_rect=(0.78, 0.60, 0.14, 0.26), seed=seed)

    fig.suptitle("Panel 2: Fractal metric–measure latent spaces (no graph edges; self-similar zoom)",
                 y=1.02, fontsize=13)

    os.makedirs(os.path.dirname(out_svg) or ".", exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, required=True)
    ap.add_argument("--captions-file", type=str, required=True)

    ap.add_argument("--max-images", type=int, default=9000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")

    ap.add_argument("--k-nn", type=int, default=360)
    ap.add_argument("--anchor-idx", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cache-npz", type=str, default="flickr30k_openclip_cache_tokens.npz")
    ap.add_argument("--out-svg", type=str, default="panel2_fractal_zoom_style.svg")
    ap.add_argument("--out-pdf", type=str, default="panel2_fractal_zoom_style.pdf")
    args = ap.parse_args()

    set_seed(args.seed)

    pairs = load_flickr30k_tokens(
        images_root=args.images_root,
        captions_file=args.captions_file,
        one_caption_per_image=True,
        max_images=args.max_images,
        seed=args.seed
    )
    N = len(pairs)
    image_paths = [p.image_path for p in pairs]
    captions = [p.caption for p in pairs]

    if args.anchor_idx < 0:
        anchor = random.randint(0, N - 1)
    else:
        anchor = int(args.anchor_idx)
        if not (0 <= anchor < N):
            raise ValueError("anchor-idx out of range")

    # cache embeddings
    img_emb, txt_emb = None, None
    if os.path.isfile(args.cache_npz):
        cache = np.load(args.cache_npz)
        img_emb = cache.get("img_emb", None)
        txt_emb = cache.get("txt_emb", None)
        if img_emb is None or txt_emb is None or img_emb.shape[0] != N:
            img_emb, txt_emb = None, None

    if img_emb is None:
        model, preprocess, tokenizer = load_openclip(args.model, args.pretrained, args.device)
        img_emb, txt_emb = encode_openclip(
            model, preprocess, tokenizer,
            image_paths, captions,
            device=args.device,
            batch_size=args.batch_size
        )
        img_emb = l2_normalize(img_emb)
        txt_emb = l2_normalize(txt_emb)
        np.savez_compressed(args.cache_npz, img_emb=img_emb, txt_emb=txt_emb)

    # local neighborhoods (within each modality)
    txt_idx = cosine_knn(txt_emb, anchor, args.k_nn)
    img_idx = cosine_knn(img_emb, anchor, args.k_nn)

    # anchor positions in local arrays
    txt_anchor_pos = int(np.where(txt_idx == anchor)[0][0]) if np.any(txt_idx == anchor) else 0
    img_anchor_pos = int(np.where(img_idx == anchor)[0][0]) if np.any(img_idx == anchor) else 0

    txt_local = txt_emb[txt_idx]
    img_local = img_emb[img_idx]

    # 2D layout
    txt2d = to_2d(txt_local, seed=args.seed)
    img2d = to_2d(img_local, seed=args.seed + 1)

    make_panel2(
        points_txt2d=txt2d,
        points_img2d=img2d,
        txt_anchor_pos=txt_anchor_pos,
        img_anchor_pos=img_anchor_pos,
        out_svg=args.out_svg,
        out_pdf=args.out_pdf,
        seed=args.seed
    )

    print(f"[Done] Saved:\n  {args.out_svg}\n  {args.out_pdf}")
    print(f"[Anchor] idx={anchor}")
    print(f"[Anchor] caption={captions[anchor][:140]}...")
    print(f"[Anchor] image={image_paths[anchor]}")


if __name__ == "__main__":
    main()
