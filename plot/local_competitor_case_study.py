#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Representative local-neighborhood case study in shared local projection style."""

import argparse
import json
import random
import textwrap
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for candidate in [CURRENT_DIR, REPO_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import torch
from sklearn.decomposition import PCA

import local_neighborhood_overlap_cdf as lno
import our_code_final as fs


K_DEFAULT = 10
IMAGE_COLOR = "#2563eb"
TEXT_COLOR = "#f97316"
OVERLAP_COLOR = "#16a34a"
PAIR_LINE_COLOR = "#52525b"
BOX_EDGE = "#d4d4d8"
BLOCKER_COLOR = "#dc2626"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@torch.no_grad()
def retrieval_ranks(image_feats: torch.Tensor, paired_text: torch.Tensor, chunk: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    image_feats = image_feats.detach().cpu().float()
    paired_text = paired_text.detach().cpu().float()
    n = int(image_feats.size(0))
    gt_scores = (image_feats * paired_text).sum(dim=1)
    i2t = np.zeros(n, dtype=np.int64)
    t2i = np.zeros(n, dtype=np.int64)

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sims = image_feats[s:e] @ paired_text.t()
        block_gt = gt_scores[s:e][:, None]
        i2t[s:e] = (sims > block_gt).sum(dim=1).cpu().numpy().astype(np.int64) + 1

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sims = paired_text[s:e] @ image_feats.t()
        block_gt = gt_scores[s:e][:, None]
        t2i[s:e] = (sims > block_gt).sum(dim=1).cpu().numpy().astype(np.int64) + 1

    return i2t, t2i


@torch.no_grad()
def topk_neighbors(feats: torch.Tensor, k: int, chunk: int) -> np.ndarray:
    return fs.same_modality_topk_neighbors(feats.detach().cpu().float(), k, device="cpu", chunk=chunk).cpu().numpy()


def overlap_values(img_neighbors: np.ndarray, txt_neighbors: np.ndarray, k: int) -> np.ndarray:
    img_k = img_neighbors[:, :k]
    txt_k = txt_neighbors[:, :k]
    return (img_k[:, :, None] == txt_k[:, None, :]).any(axis=2).sum(axis=1).astype(np.float32) / float(k)


def paired_caption(bundle: fs.RetrievalFeatureBundle, sample_idx: int) -> str:
    cap_idx = int(bundle.pair_map[sample_idx].item())
    return str(bundle.flat_captions[cap_idx])


def strict_retrieval_blockers(image_feats: torch.Tensor, paired_text: torch.Tensor, sample_idx: int) -> Dict[str, Any]:
    image_feats = image_feats.detach().cpu().float()
    paired_text = paired_text.detach().cpu().float()

    i2t_sims = (image_feats[sample_idx:sample_idx + 1] @ paired_text.t()).squeeze(0).cpu().numpy()
    t2i_sims = (paired_text[sample_idx:sample_idx + 1] @ image_feats.t()).squeeze(0).cpu().numpy()

    i2t_gt = float(i2t_sims[sample_idx])
    t2i_gt = float(t2i_sims[sample_idx])
    i2t_order = np.argsort(-i2t_sims)
    t2i_order = np.argsort(-t2i_sims)

    i2t_blockers = [int(j) for j in i2t_order if int(j) != sample_idx and float(i2t_sims[j]) > i2t_gt]
    t2i_blockers = [int(j) for j in t2i_order if int(j) != sample_idx and float(t2i_sims[j]) > t2i_gt]
    return {
        "i2t_blockers": i2t_blockers,
        "t2i_blockers": t2i_blockers,
        "i2t_gt": i2t_gt,
        "t2i_gt": t2i_gt,
    }


def build_blocker_spec(
    sample_idx: int,
    before_image_feats: torch.Tensor,
    before_paired: torch.Tensor,
    after_image_feats: torch.Tensor,
    after_paired: torch.Tensor,
    before_image_neighbors: np.ndarray,
    before_text_neighbors: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    before = strict_retrieval_blockers(before_image_feats, before_paired, sample_idx)
    after = strict_retrieval_blockers(after_image_feats, after_paired, sample_idx)

    image_local = {int(v) for v in before_image_neighbors[sample_idx, :k].tolist()}
    text_local = {int(v) for v in before_text_neighbors[sample_idx, :k].tolist()}

    before_i2t_blockers = [int(v) for v in before["i2t_blockers"]]
    before_t2i_blockers = [int(v) for v in before["t2i_blockers"]]
    after_i2t_blockers = {int(v) for v in after["i2t_blockers"]}
    after_t2i_blockers = {int(v) for v in after["t2i_blockers"]}

    before_i2t_local = [idx for idx in before_i2t_blockers if idx in image_local]
    before_t2i_local = [idx for idx in before_t2i_blockers if idx in text_local]
    after_i2t_surpassed = [idx for idx in before_i2t_blockers if idx not in after_i2t_blockers]
    after_t2i_surpassed = [idx for idx in before_t2i_blockers if idx not in after_t2i_blockers]

    return {
        "before_i2t_blockers": before_i2t_blockers,
        "before_t2i_blockers": before_t2i_blockers,
        "before_i2t_local_blockers": before_i2t_local,
        "before_t2i_local_blockers": before_t2i_local,
        "after_i2t_surpassed": after_i2t_surpassed,
        "after_t2i_surpassed": after_t2i_surpassed,
        "after_i2t_remaining": [idx for idx in before_i2t_blockers if idx in after_i2t_blockers],
        "after_t2i_remaining": [idx for idx in before_t2i_blockers if idx in after_t2i_blockers],
    }


def select_representative_case(
    before_i2t: np.ndarray,
    before_t2i: np.ndarray,
    after_i2t: np.ndarray,
    after_t2i: np.ndarray,
    before_overlap: np.ndarray,
    after_overlap: np.ndarray,
    quantile: float,
) -> Dict[str, Any]:
    improve_i2t = before_i2t - after_i2t
    improve_t2i = before_t2i - after_t2i
    improve_total = improve_i2t + improve_t2i
    overlap_gain = after_overlap - before_overlap

    pool_mask = (improve_i2t > 0) & (improve_t2i > 0) & (overlap_gain > 0)
    rule = "both_ranks_improve_and_overlap_increases"
    if int(pool_mask.sum()) < 20:
        pool_mask = (improve_total > 0) & (overlap_gain >= 0)
        rule = "positive_total_rank_improvement_and_nonnegative_overlap_gain"
    if int(pool_mask.sum()) == 0:
        pool_mask = improve_total > 0
        rule = "positive_total_rank_improvement"
    if int(pool_mask.sum()) == 0:
        pool_mask = np.ones_like(improve_total, dtype=bool)
        rule = "all_samples_fallback"

    pool = np.where(pool_mask)[0]
    target_total = float(np.quantile(improve_total[pool], quantile))
    target_overlap = float(np.quantile(overlap_gain[pool], min(max(quantile, 0.5), 0.9)))
    distance = np.abs(improve_total[pool] - target_total) + 0.35 * np.abs(overlap_gain[pool] - target_overlap)
    after_rank_sum = after_i2t[pool] + after_t2i[pool]
    order = np.lexsort((after_rank_sum, distance))
    selected = int(pool[order[0]])

    sorted_pool = pool[np.argsort(improve_total[pool])]
    selected_pos = int(np.where(sorted_pool == selected)[0][0]) if selected in sorted_pool else 0
    percentile = 100.0 * selected_pos / float(max(len(sorted_pool) - 1, 1))

    return {
        "sample_index": selected,
        "selection_rule": rule,
        "target_quantile": float(quantile),
        "selected_percentile": float(percentile),
        "pool_size": int(len(pool)),
        "improve_i2t": int(improve_i2t[selected]),
        "improve_t2i": int(improve_t2i[selected]),
        "improve_total": int(improve_total[selected]),
        "overlap_gain": float(overlap_gain[selected]),
        "target_total_improvement": target_total,
        "target_overlap_gain": target_overlap,
    }


def build_panel_spec(
    sample_idx: int,
    image_neighbors: np.ndarray,
    text_neighbors: np.ndarray,
    i2t_ranks: np.ndarray,
    t2i_ranks: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    img_ids = image_neighbors[sample_idx, :k].tolist()
    txt_ids = text_neighbors[sample_idx, :k].tolist()
    overlap_ids = sorted(set(img_ids) & set(txt_ids))
    return {
        "sample_index": int(sample_idx),
        "image_neighbors": [int(v) for v in img_ids],
        "text_neighbors": [int(v) for v in txt_ids],
        "overlap_ids": [int(v) for v in overlap_ids],
        "overlap_count": int(len(overlap_ids)),
        "overlap_value": float(len(overlap_ids) / float(k)),
        "i2t_rank": int(i2t_ranks[sample_idx]),
        "t2i_rank": int(t2i_ranks[sample_idx]),
    }


def shared_projection(
    before_img: np.ndarray,
    before_txt: np.ndarray,
    after_img: np.ndarray,
    after_txt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stack = np.concatenate([before_img, before_txt, after_img, after_txt], axis=0)
    coords = PCA(n_components=2, random_state=0).fit_transform(stack)
    n = before_img.shape[0]
    return coords[:n], coords[n:2 * n], coords[2 * n:3 * n], coords[3 * n:]


def panel_limits(*point_sets: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = np.concatenate([points[:, 0] for points in point_sets], axis=0)
    ys = np.concatenate([points[:, 1] for points in point_sets], axis=0)
    x_span = max(xs.max() - xs.min(), 1e-6)
    y_span = max(ys.max() - ys.min(), 1e-6)
    x_pad = 0.16 * x_span
    y_pad = 0.16 * y_span
    return (xs.min() - x_pad, xs.max() + x_pad), (ys.min() - y_pad, ys.max() + y_pad)


def local_limits(
    before_img_xy: np.ndarray,
    before_txt_xy: np.ndarray,
    after_img_xy: np.ndarray,
    after_txt_xy: np.ndarray,
    sample_idx: int,
    before_spec: Dict[str, Any],
    after_spec: Dict[str, Any],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ids = {int(sample_idx)}
    ids.update(before_spec["image_neighbors"])
    ids.update(before_spec["text_neighbors"])
    ids.update(after_spec["image_neighbors"])
    ids.update(after_spec["text_neighbors"])
    ordered = sorted(ids)
    return panel_limits(before_img_xy[ordered], before_txt_xy[ordered], after_img_xy[ordered], after_txt_xy[ordered])




def wrap_caption(text: str, width: int = 74) -> str:
    normalized = " ".join(str(text).split())
    return "\n".join(textwrap.wrap(normalized, width=width))


def draw_anchor_strip(ax, image_path: str, caption: str) -> None:
    ax.axis("off")
    image_ax = ax.inset_axes([0.02, 0.10, 0.16, 0.78])
    image_ax.imshow(Image.open(image_path).convert("RGB"))
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    for spine in image_ax.spines.values():
        spine.set_edgecolor(BOX_EDGE)
        spine.set_linewidth(1.1)

    ax.text(
        0.22,
        0.92,
        "Selected Anchor Pair",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.22,
        0.72,
        wrap_caption(caption),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.2,
        linespacing=1.16,
        bbox={"boxstyle": "round,pad=0.34", "fc": "white", "ec": BOX_EDGE, "alpha": 0.96},
    )


def plot_local_panel(
    ax,
    img_xy: np.ndarray,
    txt_xy: np.ndarray,
    spec: Dict[str, Any],
    title: str,
    k: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    panel_mode: str,
    blocker_spec=None,
) -> None:
    sample_idx = int(spec["sample_index"])
    img_neighbors = spec["image_neighbors"]
    txt_neighbors = spec["text_neighbors"]
    overlap_ids = spec["overlap_ids"]

    ax.scatter(img_xy[:, 0], img_xy[:, 1], s=18, marker="^", color=IMAGE_COLOR, alpha=0.08, edgecolors="none")
    ax.scatter(txt_xy[:, 0], txt_xy[:, 1], s=18, marker="o", color=TEXT_COLOR, alpha=0.08, edgecolors="none")

    if img_neighbors:
        pts = img_xy[img_neighbors]
        ax.scatter(pts[:, 0], pts[:, 1], s=64, marker="^", color=IMAGE_COLOR, alpha=0.92, edgecolors="white", linewidth=0.45)
    if txt_neighbors:
        pts = txt_xy[txt_neighbors]
        ax.scatter(pts[:, 0], pts[:, 1], s=60, marker="o", color=TEXT_COLOR, alpha=0.88, edgecolors="white", linewidth=0.45)

    if overlap_ids:
        ax.scatter(img_xy[overlap_ids, 0], img_xy[overlap_ids, 1], s=148, marker="o", facecolors="none", edgecolors=OVERLAP_COLOR, linewidths=1.6)
        ax.scatter(txt_xy[overlap_ids, 0], txt_xy[overlap_ids, 1], s=140, marker="o", facecolors="none", edgecolors=OVERLAP_COLOR, linewidths=1.6)

    if blocker_spec and panel_mode == "before":
        image_blockers = blocker_spec["before_t2i_local_blockers"]
        text_blockers = blocker_spec["before_i2t_local_blockers"]
        if image_blockers:
            ax.scatter(img_xy[image_blockers, 0], img_xy[image_blockers, 1], s=200, marker="o", facecolors="none", edgecolors=BLOCKER_COLOR, linewidths=1.9, zorder=7)
        if text_blockers:
            ax.scatter(txt_xy[text_blockers, 0], txt_xy[text_blockers, 1], s=196, marker="o", facecolors="none", edgecolors=BLOCKER_COLOR, linewidths=1.9, zorder=7)

    ax.plot(
        [img_xy[sample_idx, 0], txt_xy[sample_idx, 0]],
        [img_xy[sample_idx, 1], txt_xy[sample_idx, 1]],
        linestyle="--",
        linewidth=1.35,
        color=PAIR_LINE_COLOR,
        alpha=0.95,
    )
    ax.scatter([img_xy[sample_idx, 0]], [img_xy[sample_idx, 1]], s=190, marker="^", color=IMAGE_COLOR, edgecolors="black", linewidth=1.0, zorder=5)
    ax.scatter([txt_xy[sample_idx, 0]], [txt_xy[sample_idx, 1]], s=178, marker="o", color=TEXT_COLOR, edgecolors="black", linewidth=1.0, zorder=5)

    dx = 0.025 * (xlim[1] - xlim[0])
    dy = 0.030 * (ylim[1] - ylim[0])
    ax.text(img_xy[sample_idx, 0] - dx, img_xy[sample_idx, 1] + dy, "x_i", fontsize=11, fontweight="bold", color="#1f2937")
    ax.text(txt_xy[sample_idx, 0] + 0.35 * dx, txt_xy[sample_idx, 1] + dy, "y_i", fontsize=11, fontweight="bold", color="#1f2937")

    ax.text(
        0.02,
        0.98,
        f"Overlap@{k} = {spec['overlap_value']:.2f}\nI->T rank = {spec['i2t_rank']}\nT->I rank = {spec['t2i_rank']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": BOX_EDGE, "alpha": 0.94},
    )

    if blocker_spec and panel_mode == "before":
        blocker_text = (
            f"Wrong texts above GT: {len(blocker_spec['before_i2t_blockers'])}\n"
            f"Wrong images above GT: {len(blocker_spec['before_t2i_blockers'])}\n"
            f"Local competitors: {len(blocker_spec['before_i2t_local_blockers'])} text, {len(blocker_spec['before_t2i_local_blockers'])} image\n"
            "Red circles mark those local blockers"
        )
        ax.text(
            0.98,
            0.98,
            blocker_text,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9.6,
            color="#7f1d1d",
            bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": BLOCKER_COLOR, "alpha": 0.95},
        )
    elif blocker_spec and panel_mode == "after":
        blocker_text = (
            "Previous blockers now behind GT\n"
            f"Text side: {len(blocker_spec['after_i2t_surpassed'])}/{len(blocker_spec['before_i2t_blockers'])}\n"
            f"Image side: {len(blocker_spec['after_t2i_surpassed'])}/{len(blocker_spec['before_t2i_blockers'])}"
        )
        ax.text(
            0.98,
            0.06,
            blocker_text,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=9.6,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": BOX_EDGE, "alpha": 0.95},
        )

    ax.set_title(title, fontsize=15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.25)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    parser.add_argument("--dataset", type=str, default="flickr30k", choices=["flickr30k", "mscoco"])
    parser.add_argument("--cache-dir", type=str, default="/work/was598/modilty_gap/plot/cache")
    parser.add_argument("--out-prefix", type=str, default="/work/was598/modilty_gap/plot/before_vs_after")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--model-key", type=str, default="clip", choices=["clip", "openclip", "siglip"])
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", type=str, default="openai")
    parser.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument("--text-variant", type=str, default="short", choices=["short", "paragraph"])
    parser.add_argument("--paragraph-sentences", type=int, default=3)
    parser.add_argument("--neighbor-chunk", type=int, default=1024)

    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--target-quantile", type=float, default=0.75)
    parser.add_argument("--selection-index", type=int, default=-1)

    parser.add_argument("--lora-state", type=str, default="")
    parser.add_argument("--disable-recommended-preset", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--train-anchors", type=int, default=512)
    parser.add_argument("--anchor-batch", type=int, default=128)
    parser.add_argument("--spectral-samples", type=int, default=512)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-dbl", type=float, default=1.0)
    parser.add_argument("--lambda-spec", type=float, default=0.1)
    parser.add_argument("--lambda-match", type=float, default=0.1)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-orth", type=float, default=0.1)
    parser.add_argument("--train-reg", type=float, default=1e-3)
    parser.add_argument("--align-temp", type=float, default=0.07)
    parser.add_argument("--align-samples", type=int, default=0)
    parser.add_argument("--pairwise-row-chunk", type=int, default=0)
    parser.add_argument("--pairwise-col-chunk", type=int, default=0)
    parser.add_argument("--pairwise-checkpoint", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-mix", type=float, default=0.3)
    parser.add_argument("--multi-caption", action="store_true")
    parser.add_argument("--caption-agg", type=str, default="random", choices=["random", "mean"])
    parser.add_argument("--structure-batch-size", type=int, default=0)
    parser.add_argument("--noise-pair-rate", type=float, default=0.0)
    parser.add_argument("--noise-mix", type=float, default=0.5)
    parser.add_argument("--lambda-neighbor-compete", type=float, default=0.0)
    parser.add_argument("--neighbor-compete-k", type=int, default=100)
    parser.add_argument("--neighbor-compete-samples", type=int, default=4000)
    parser.add_argument("--neighbor-compete-margin", type=float, default=0.05)
    parser.add_argument("--neighbor-compete-chunk", type=int, default=1024)
    parser.add_argument("--df", type=float, default=2.0)
    parser.add_argument("--ds", type=float, default=None)
    parser.add_argument("--dw", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--radii-min", type=float, default=0.05)
    parser.add_argument("--radii-max", type=float, default=0.5)
    parser.add_argument("--radii-count", type=int, default=6)
    parser.add_argument("--rho-list", type=str, default="1.5,2.0,3.0")
    parser.add_argument("--diffusion-min", type=float, default=0.01)
    parser.add_argument("--diffusion-max", type=float, default=1.0)
    parser.add_argument("--diffusion-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fs.seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    cache_dir = Path(args.cache_dir)
    out_prefix = Path(args.out_prefix)

    preset_info = lno.resolve_recommended_preset(args)
    effective_lora_state = str(args.lora_state)
    effective_lora_mix = float(args.lora_mix)
    if preset_info:
        effective_lora_state = str(preset_info["lora_state"])
        effective_lora_mix = float(preset_info["lora_mix"])
        print(
            f"[Preset] {preset_info['name']} -> {effective_lora_state} "
            f"(lora_mix={effective_lora_mix:.2f})"
        )

    model_name, model = lno.build_model(args, device)
    test_ds = lno.get_dataset(args.data_root, args.dataset, split="test")
    test_cap = None if args.max_test_images <= 0 else int(args.max_test_images)
    cache_stub = f"{args.dataset}_{lno.safe_model_tag(model_name)}_{args.text_variant}_p{args.paragraph_sentences}"
    test_bundle = lno.get_retrieval_bundle(
        model,
        test_ds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=test_cap,
        text_variant=args.text_variant,
        paragraph_sentences=args.paragraph_sentences,
        cache_path=cache_dir / f"{cache_stub}_test_{test_cap or 'all'}.pt",
    )

    lora_state, method_info = lno.load_or_train_lora_state(
        model,
        args,
        args.dataset,
        device,
        cache_dir,
        model_name,
        effective_lora_state,
        preset_info,
    )
    after_image_feats, after_text_feats = lno.apply_method(test_bundle, lora_state, effective_lora_mix, device)

    before_image_feats = test_bundle.image_feats
    before_paired = test_bundle.text_feats[test_bundle.pair_map]
    after_paired = after_text_feats[test_bundle.pair_map]

    k = int(args.k)
    before_img_neighbors = topk_neighbors(before_image_feats, k, args.neighbor_chunk)
    before_txt_neighbors = topk_neighbors(before_paired, k, args.neighbor_chunk)
    after_img_neighbors = topk_neighbors(after_image_feats, k, args.neighbor_chunk)
    after_txt_neighbors = topk_neighbors(after_paired, k, args.neighbor_chunk)

    before_i2t, before_t2i = retrieval_ranks(before_image_feats, before_paired)
    after_i2t, after_t2i = retrieval_ranks(after_image_feats, after_paired)
    before_overlap = overlap_values(before_img_neighbors, before_txt_neighbors, k)
    after_overlap = overlap_values(after_img_neighbors, after_txt_neighbors, k)

    if int(args.selection_index) >= 0:
        selected = int(args.selection_index)
        selection_info = {
            "sample_index": selected,
            "selection_rule": "manual_override",
            "target_quantile": float(args.target_quantile),
            "selected_percentile": None,
            "pool_size": int(before_image_feats.size(0)),
            "improve_i2t": int(before_i2t[selected] - after_i2t[selected]),
            "improve_t2i": int(before_t2i[selected] - after_t2i[selected]),
            "improve_total": int((before_i2t[selected] - after_i2t[selected]) + (before_t2i[selected] - after_t2i[selected])),
            "overlap_gain": float(after_overlap[selected] - before_overlap[selected]),
            "target_total_improvement": None,
            "target_overlap_gain": None,
        }
    else:
        selection_info = select_representative_case(
            before_i2t,
            before_t2i,
            after_i2t,
            after_t2i,
            before_overlap,
            after_overlap,
            float(args.target_quantile),
        )
        selected = int(selection_info["sample_index"])

    before_spec = build_panel_spec(selected, before_img_neighbors, before_txt_neighbors, before_i2t, before_t2i, k)
    after_spec = build_panel_spec(selected, after_img_neighbors, after_txt_neighbors, after_i2t, after_t2i, k)
    blocker_spec = build_blocker_spec(selected, before_image_feats, before_paired, after_image_feats, after_paired, before_img_neighbors, before_txt_neighbors, k)

    before_img_xy, before_txt_xy, after_img_xy, after_txt_xy = shared_projection(
        before_image_feats.detach().cpu().numpy(),
        before_paired.detach().cpu().numpy(),
        after_image_feats.detach().cpu().numpy(),
        after_paired.detach().cpu().numpy(),
    )
    xlim, ylim = local_limits(before_img_xy, before_txt_xy, after_img_xy, after_txt_xy, selected, before_spec, after_spec)

    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
    })

    fig = plt.figure(figsize=(10.8, 6.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.34, 1.0])
    anchor_ax = fig.add_subplot(grid[0, :])
    draw_anchor_strip(anchor_ax, test_ds.items[selected][0], paired_caption(test_bundle, selected))
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    plot_local_panel(axes[0], before_img_xy, before_txt_xy, before_spec, "Before FSAlign", k, xlim, ylim, panel_mode="before", blocker_spec=blocker_spec)
    plot_local_panel(axes[1], after_img_xy, after_txt_xy, after_spec, "After FSAlign", k, xlim, ylim, panel_mode="after", blocker_spec=blocker_spec)

    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=IMAGE_COLOR, markeredgecolor="none", markersize=8, label="Image"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEXT_COLOR, markeredgecolor="none", markersize=8, label="Text"),
        Line2D([0], [0], marker="o", color=OVERLAP_COLOR, markerfacecolor="none", markersize=8, linewidth=0, markeredgewidth=1.5, label="Shared local neighbors"),
        Line2D([0], [0], marker="o", color=BLOCKER_COLOR, markerfacecolor="none", markersize=8, linewidth=0, markeredgewidth=1.8, label="Before retrieval blockers"),
        Line2D([0, 1], [0, 0], linestyle="--", color=PAIR_LINE_COLOR, linewidth=1.3, label="Paired image-text anchor"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])

    ensure_dir(out_prefix.parent)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "dataset": args.dataset,
        "model": model_name,
        "device": device,
        "style": "shared_local_projection_minimal",
        "k": k,
        "selection": selection_info,
        "method_info": method_info,
        "config": {
            "text_variant": args.text_variant,
            "paragraph_sentences": int(args.paragraph_sentences),
            "max_test_images": None if test_cap is None else int(test_cap),
            "effective_lora_state": effective_lora_state or None,
            "effective_lora_mix": float(effective_lora_mix),
            "neighbor_chunk": int(args.neighbor_chunk),
            "projection": "shared_pca_over_full_before_after",
        },
        "anchor": {
            "sample_index": int(selected),
            "image_path": test_ds.items[selected][0],
            "text": paired_caption(test_bundle, selected),
        },
        "before": before_spec,
        "after": after_spec,
        "blockers": blocker_spec,
    }
    save_json(out_prefix.with_suffix(".json"), metadata)

    print(f"[Saved] {out_prefix.with_suffix('.png')}")
    print(f"[Saved] {out_prefix.with_suffix('.pdf')}")
    print(f"[Saved] {out_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
