#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal shared-projection figure for Flickr30K:
- embeddings only
- overlap only
- no sidebars, no task metrics, no pairwise connector clutter

Outputs:
  <out_prefix>.png
  <out_prefix>.pdf
  <out_prefix>.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.decomposition import PCA

import our_code_final as fs


IMAGE_COLOR = "#2563eb"
TEXT_COLOR = "#f97316"
OVERLAP_COLOR = "#16a34a"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def bundle_to_cpu_dict(bundle: fs.RetrievalFeatureBundle) -> Dict[str, Any]:
    return {
        "image_feats": bundle.image_feats.detach().cpu(),
        "text_feats": bundle.text_feats.detach().cpu(),
        "cap2img": bundle.cap2img.detach().cpu(),
        "pair_map": bundle.pair_map.detach().cpu(),
        "cap_indices": bundle.cap_indices,
        "image_captions": bundle.image_captions,
        "flat_captions": bundle.flat_captions,
    }


def bundle_from_cpu_dict(payload: Dict[str, Any], device: str) -> fs.RetrievalFeatureBundle:
    image_feats = payload["image_feats"].to(device)
    text_feats = payload["text_feats"].to(device)
    cap2img = payload["cap2img"].to(device)
    pair_map = payload["pair_map"].to(device)
    return fs.RetrievalFeatureBundle(
        image_feats=image_feats,
        text_feats=text_feats,
        cap2img=cap2img,
        pair_map=pair_map,
        paired_text=text_feats[pair_map],
        cap_indices=payload["cap_indices"],
        image_captions=payload["image_captions"],
        flat_captions=payload["flat_captions"],
    )


def safe_model_tag(model_name: str) -> str:
    return fs.safe_filename(model_name.replace(":", "_"))


def build_model(args, device: str) -> Tuple[str, fs.VLBackbone]:
    if args.model_key == "clip":
        return f"clip:{args.clip_model}:openai", fs.CLIPWrapper(args.clip_model, device=device)
    if args.model_key == "openclip":
        return (
            f"open_clip:{args.openclip_model}:{args.openclip_pretrained}",
            fs.OpenCLIPWrapper(args.openclip_model, args.openclip_pretrained, device=device),
        )
    if args.model_key == "siglip":
        return f"siglip:{args.siglip_name}", fs.SigLIPWrapper(args.siglip_name, device=device)
    raise ValueError(f"Unsupported model_key: {args.model_key}")


def flickr_dataset(data_root: str, split: str, karpathy_json: Optional[str]) -> Any:
    if karpathy_json:
        kjson = Path(karpathy_json)
    else:
        kjson = fs.ensure_karpathy_json(data_root, "flickr30k")
    roots = [
        str(Path(data_root) / "flickr30k" / "flickr30k-images"),
        str(Path(data_root) / "flickr30k" / "images"),
        str(Path(data_root) / "flickr30k"),
    ]
    return fs.KarpathyRetrievalDataset(str(kjson), roots, split=split, max_images=None)


def get_retrieval_bundle(
    model: fs.VLBackbone,
    dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    text_variant: str,
    paragraph_sentences: int,
    cache_path: Path,
) -> fs.RetrievalFeatureBundle:
    if cache_path.exists():
        return bundle_from_cpu_dict(torch.load(cache_path, map_location="cpu"), device=device)
    bundle = fs.encode_retrieval_features(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images=max_images,
        text_variant=text_variant,
        paragraph_sentences=paragraph_sentences,
    )
    ensure_dir(cache_path.parent)
    torch.save(bundle_to_cpu_dict(bundle), cache_path)
    return bundle


def build_fsalign_args(args, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        lora_state="",
        train_epochs=int(args.train_epochs),
        train_anchors=int(args.train_anchors),
        anchor_batch=int(args.anchor_batch),
        spectral_samples=int(args.spectral_samples),
        train_lr=float(args.train_lr),
        lambda_dbl=float(args.lambda_dbl),
        lambda_spec=float(args.lambda_spec),
        lambda_match=float(args.lambda_match),
        lambda_align=float(args.lambda_align),
        lambda_orth=float(args.lambda_orth),
        train_reg=float(args.train_reg),
        train_print_every=max(1, int(args.train_epochs)),
        align_temp=float(args.align_temp),
        align_samples=int(args.align_samples),
        pairwise_row_chunk=int(args.pairwise_row_chunk),
        pairwise_col_chunk=int(args.pairwise_col_chunk),
        pairwise_checkpoint=bool(args.pairwise_checkpoint),
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        save_lora=False,
        lora_mix=float(args.lora_mix),
        multi_caption=bool(args.multi_caption),
        caption_agg=str(args.caption_agg),
        text_variant=str(args.text_variant),
        paragraph_sentences=int(args.paragraph_sentences),
        structure_batch_size=int(args.structure_batch_size),
        noise_pair_rate=float(args.noise_pair_rate),
        noise_mix=float(args.noise_mix),
        dimension_mode="shared",
        dimension_offset_df=0.0,
        dimension_offset_ds=0.0,
        dimension_offset_dw=None,
        lambda_dim_offset=0.0,
        lambda_neighbor_compete=float(args.lambda_neighbor_compete),
        neighbor_compete_k=int(args.neighbor_compete_k),
        neighbor_compete_samples=int(args.neighbor_compete_samples),
        neighbor_compete_margin=float(args.neighbor_compete_margin),
        neighbor_compete_chunk=int(args.neighbor_compete_chunk),
        early_stop=False,
        val_split="internal",
        val_frac=0.1,
        patience=2,
        min_delta=0.0,
        df=float(args.df),
        ds=None if args.ds is None else float(args.ds),
        dw=float(args.dw),
        alpha=float(args.alpha),
        radii_min=float(args.radii_min),
        radii_max=float(args.radii_max),
        radii_count=int(args.radii_count),
        rho_list=str(args.rho_list),
        diffusion_min=float(args.diffusion_min),
        diffusion_max=float(args.diffusion_max),
        diffusion_count=int(args.diffusion_count),
    )


def load_or_train_lora_state(
    model: fs.VLBackbone,
    args,
    device: str,
    cache_dir: Path,
    model_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if args.lora_state:
        state = torch.load(args.lora_state, map_location="cpu")
        return state, {"mode": "lora_state", "path": str(args.lora_state)}

    train_ds = flickr_dataset(args.data_root, split="train", karpathy_json=args.karpathy_json or None)
    train_cap = None if args.max_train_images <= 0 else int(args.max_train_images)
    cache_stub = f"flickr30k_{safe_model_tag(model_name)}_{args.text_variant}_p{args.paragraph_sentences}"
    train_bundle = get_retrieval_bundle(
        model,
        train_ds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=train_cap,
        text_variant=args.text_variant,
        paragraph_sentences=args.paragraph_sentences,
        cache_path=cache_dir / f"{cache_stub}_train_{train_cap or 'all'}.pt",
    )
    fsalign_args = build_fsalign_args(args, device)
    radii = fs.logspace_scales(fsalign_args.radii_min, fsalign_args.radii_max, fsalign_args.radii_count)
    rho_list = [float(x) for x in fsalign_args.rho_list.split(",") if x.strip()]
    diffusion_scales = fs.logspace_scales(
        fsalign_args.diffusion_min,
        fsalign_args.diffusion_max,
        fsalign_args.diffusion_count,
    )
    state, history = fs.train_lora_postprocess(
        train_bundle.image_feats,
        train_bundle.paired_text,
        radii,
        rho_list,
        diffusion_scales,
        fsalign_args,
        caption_pool=(train_bundle.text_feats, train_bundle.cap_indices),
    )
    return state, {"mode": "train", "history": history}


@torch.no_grad()
def apply_method(
    bundle: fs.RetrievalFeatureBundle,
    lora_state: Dict[str, Any],
    lora_mix: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    image_feats = fs.apply_lora_state(bundle.image_feats, layer_img, lora_mix)
    text_feats = fs.apply_lora_state(bundle.text_feats, layer_txt, lora_mix)
    return image_feats, text_feats


def compute_global_overlap(
    bundle: fs.RetrievalFeatureBundle,
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    nas_k: int,
    nas_max_items: int,
) -> float:
    paired_text = text_feats[bundle.pair_map]
    return float(fs.nas_k(image_feats, paired_text, k=nas_k, max_items=nas_max_items))


def choose_sample_indices(total: int, sample_size: int, seed: int, anchor_index: Optional[int]) -> np.ndarray:
    if total <= 0:
        raise ValueError("No test samples available.")
    if sample_size <= 0 or sample_size >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if anchor_index is None:
        return np.sort(rng.choice(total, size=sample_size, replace=False)).astype(np.int64)

    if anchor_index < 0 or anchor_index >= total:
        raise ValueError(f"anchor_index must be in [0, {total - 1}]")
    remaining = np.array([idx for idx in range(total) if idx != anchor_index], dtype=np.int64)
    picked = rng.choice(remaining, size=sample_size - 1, replace=False)
    return np.sort(np.concatenate([[anchor_index], picked])).astype(np.int64)


def overlap_scores_and_neighbors(image_feats: torch.Tensor, paired_text: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_cpu = image_feats.detach().cpu()
    text_cpu = paired_text.detach().cpu()
    img_neighbors = fs.same_modality_topk_neighbors(image_cpu, k, device="cpu", chunk=1024).cpu().numpy()
    txt_neighbors = fs.same_modality_topk_neighbors(text_cpu, k, device="cpu", chunk=1024).cpu().numpy()
    if img_neighbors.shape[1] == 0:
        zeros = np.zeros(image_cpu.shape[0], dtype=np.float32)
        return zeros, img_neighbors, txt_neighbors

    overlap = np.zeros(image_cpu.shape[0], dtype=np.float32)
    for idx in range(image_cpu.shape[0]):
        overlap[idx] = float(len(set(img_neighbors[idx].tolist()) & set(txt_neighbors[idx].tolist()))) / float(img_neighbors.shape[1])
    return overlap, img_neighbors, txt_neighbors


def choose_anchor_index(
    before_img: torch.Tensor,
    before_txt: torch.Tensor,
    after_img: torch.Tensor,
    after_txt: torch.Tensor,
    local_k: int,
    forced_index: Optional[int],
) -> Tuple[int, Dict[str, Any]]:
    before_overlap, before_img_nn, before_txt_nn = overlap_scores_and_neighbors(before_img, before_txt, local_k)
    after_overlap, after_img_nn, after_txt_nn = overlap_scores_and_neighbors(after_img, after_txt, local_k)

    if forced_index is not None:
        idx = int(forced_index)
    else:
        improvement = after_overlap - before_overlap
        idx = int(np.argmax(improvement))

    return idx, {
        "before_overlap": float(before_overlap[idx]),
        "after_overlap": float(after_overlap[idx]),
        "improvement": float(after_overlap[idx] - before_overlap[idx]),
        "before_img_neighbors": before_img_nn[idx].tolist(),
        "before_txt_neighbors": before_txt_nn[idx].tolist(),
        "after_img_neighbors": after_img_nn[idx].tolist(),
        "after_txt_neighbors": after_txt_nn[idx].tolist(),
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
    x_pad = 0.10 * x_span
    y_pad = 0.10 * y_span
    return (xs.min() - x_pad, xs.max() + x_pad), (ys.min() - y_pad, ys.max() + y_pad)


def local_limits(
    before_img_xy: np.ndarray,
    before_txt_xy: np.ndarray,
    after_img_xy: np.ndarray,
    after_txt_xy: np.ndarray,
    neighbor_spec: Dict[str, Any],
    anchor_idx: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ids = {anchor_idx}
    ids.update(neighbor_spec["before_img_neighbors"])
    ids.update(neighbor_spec["before_txt_neighbors"])
    ids.update(neighbor_spec["after_img_neighbors"])
    ids.update(neighbor_spec["after_txt_neighbors"])
    idx = sorted(ids)
    return panel_limits(before_img_xy[idx], before_txt_xy[idx], after_img_xy[idx], after_txt_xy[idx])


def overlap_label(name: str, value: float) -> str:
    return f"{name} = {value:.2f}"


def plot_embedding_panel(
    ax,
    img_xy: np.ndarray,
    txt_xy: np.ndarray,
    title: str,
    overlap_text: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    ax.scatter(img_xy[:, 0], img_xy[:, 1], s=26, marker="^", color=IMAGE_COLOR, alpha=0.72, edgecolors="none")
    ax.scatter(txt_xy[:, 0], txt_xy[:, 1], s=24, marker="o", color=TEXT_COLOR, alpha=0.62, edgecolors="none")
    ax.text(
        0.02,
        0.98,
        overlap_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d4d4d8", "alpha": 0.92},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def plot_local_panel(
    ax,
    img_xy: np.ndarray,
    txt_xy: np.ndarray,
    anchor_idx: int,
    img_neighbors: Sequence[int],
    txt_neighbors: Sequence[int],
    title: str,
    overlap_text: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    overlap_ids = sorted(set(img_neighbors) & set(txt_neighbors))

    ax.scatter(img_xy[:, 0], img_xy[:, 1], s=18, marker="^", color=IMAGE_COLOR, alpha=0.08, edgecolors="none")
    ax.scatter(txt_xy[:, 0], txt_xy[:, 1], s=18, marker="o", color=TEXT_COLOR, alpha=0.08, edgecolors="none")

    if img_neighbors:
        pts = img_xy[list(img_neighbors)]
        ax.scatter(pts[:, 0], pts[:, 1], s=58, marker="^", color=IMAGE_COLOR, alpha=0.92, edgecolors="white", linewidth=0.4)
    if txt_neighbors:
        pts = txt_xy[list(txt_neighbors)]
        ax.scatter(pts[:, 0], pts[:, 1], s=54, marker="o", color=TEXT_COLOR, alpha=0.88, edgecolors="white", linewidth=0.4)

    if overlap_ids:
        ax.scatter(img_xy[overlap_ids, 0], img_xy[overlap_ids, 1], s=132, marker="o", facecolors="none", edgecolors=OVERLAP_COLOR, linewidths=1.5)
        ax.scatter(txt_xy[overlap_ids, 0], txt_xy[overlap_ids, 1], s=124, marker="o", facecolors="none", edgecolors=OVERLAP_COLOR, linewidths=1.5)

    ax.scatter([img_xy[anchor_idx, 0]], [img_xy[anchor_idx, 1]], s=180, marker="^", color=IMAGE_COLOR, edgecolors="black", linewidth=1.0)
    ax.scatter([txt_xy[anchor_idx, 0]], [txt_xy[anchor_idx, 1]], s=170, marker="o", color=TEXT_COLOR, edgecolors="black", linewidth=1.0)
    ax.text(
        0.02,
        0.98,
        overlap_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d4d4d8", "alpha": 0.92},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def make_figure(
    before_img_xy: np.ndarray,
    before_txt_xy: np.ndarray,
    after_img_xy: np.ndarray,
    after_txt_xy: np.ndarray,
    anchor_idx: int,
    neighbor_spec: Dict[str, Any],
    nas_before: float,
    nas_after: float,
    nas_k: int,
    local_k: int,
    out_prefix: Path,
) -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))

    global_xlim, global_ylim = panel_limits(before_img_xy, before_txt_xy, after_img_xy, after_txt_xy)
    local_xlim, local_ylim = local_limits(before_img_xy, before_txt_xy, after_img_xy, after_txt_xy, neighbor_spec, anchor_idx)

    plot_embedding_panel(
        axes[0, 0],
        before_img_xy,
        before_txt_xy,
        "Before FSAlign",
        overlap_label(f"NAS@{nas_k}", nas_before),
        global_xlim,
        global_ylim,
    )
    plot_embedding_panel(
        axes[0, 1],
        after_img_xy,
        after_txt_xy,
        "After FSAlign",
        overlap_label(f"NAS@{nas_k}", nas_after),
        global_xlim,
        global_ylim,
    )
    plot_local_panel(
        axes[1, 0],
        before_img_xy,
        before_txt_xy,
        anchor_idx,
        neighbor_spec["before_img_neighbors"],
        neighbor_spec["before_txt_neighbors"],
        f"Anchor Neighborhood Before (k={local_k})",
        overlap_label(f"Local overlap@{local_k}", neighbor_spec["before_overlap"]),
        local_xlim,
        local_ylim,
    )
    plot_local_panel(
        axes[1, 1],
        after_img_xy,
        after_txt_xy,
        anchor_idx,
        neighbor_spec["after_img_neighbors"],
        neighbor_spec["after_txt_neighbors"],
        f"Anchor Neighborhood After (k={local_k})",
        overlap_label(f"Local overlap@{local_k}", neighbor_spec["after_overlap"]),
        local_xlim,
        local_ylim,
    )

    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=IMAGE_COLOR, markeredgecolor="none", markersize=8, label="Image embedding"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEXT_COLOR, markeredgecolor="none", markersize=8, label="Text embedding"),
        Line2D([0], [0], marker="o", color=OVERLAP_COLOR, markerfacecolor="none", markersize=8, linewidth=0, markeredgewidth=1.5, label="Shared neighbors"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Flickr30K embeddings under one shared PCA projection", y=0.995, fontsize=15)
    fig.text(
        0.5,
        0.02,
        "One PCA is fit jointly on before/after image-text embeddings so every panel is directly comparable.",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])

    ensure_dir(out_prefix.parent)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    parser.add_argument("--karpathy-json", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="/work/was598/modilty_gap/plot/cache")
    parser.add_argument("--out-prefix", type=str, default="/work/was598/modilty_gap/plot/fk30k_shared_projection_main")
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

    parser.add_argument("--figure-samples", type=int, default=400)
    parser.add_argument("--pair-line-count", type=int, default=36)
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--nas-k", type=int, default=10)
    parser.add_argument("--nas-max-items", type=int, default=5000)
    parser.add_argument("--anchor-index", type=int, default=-1)

    parser.add_argument("--lora-state", type=str, default="")
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
    out_prefix = Path(args.out_prefix)
    cache_dir = Path(args.cache_dir)

    model_name, model = build_model(args, device)
    test_ds = flickr_dataset(args.data_root, split="test", karpathy_json=args.karpathy_json or None)
    test_cap = None if args.max_test_images <= 0 else int(args.max_test_images)
    cache_stub = f"flickr30k_{safe_model_tag(model_name)}_{args.text_variant}_p{args.paragraph_sentences}"
    test_bundle = get_retrieval_bundle(
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

    lora_state, method_info = load_or_train_lora_state(model, args, device, cache_dir, model_name)
    ours_image_feats, ours_text_feats = apply_method(test_bundle, lora_state, args.lora_mix, device)

    nas_before = compute_global_overlap(
        test_bundle,
        test_bundle.image_feats,
        test_bundle.text_feats,
        nas_k=args.nas_k,
        nas_max_items=args.nas_max_items,
    )
    nas_after = compute_global_overlap(
        test_bundle,
        ours_image_feats,
        ours_text_feats,
        nas_k=args.nas_k,
        nas_max_items=args.nas_max_items,
    )

    forced_anchor = None if args.anchor_index < 0 else int(args.anchor_index)
    sample_indices = choose_sample_indices(
        total=int(test_bundle.image_feats.size(0)),
        sample_size=int(args.figure_samples),
        seed=int(args.seed),
        anchor_index=forced_anchor,
    )

    sample_t = torch.tensor(sample_indices.tolist(), device=test_bundle.image_feats.device, dtype=torch.long)
    before_img = test_bundle.image_feats[sample_t]
    before_txt = test_bundle.text_feats[test_bundle.pair_map[sample_t]]
    after_img = ours_image_feats[sample_t]
    after_txt = ours_text_feats[test_bundle.pair_map[sample_t]]

    anchor_local_idx = None
    if forced_anchor is not None:
        anchor_local_idx = int(np.where(sample_indices == forced_anchor)[0][0])
    anchor_local_idx, anchor_info = choose_anchor_index(
        before_img,
        before_txt,
        after_img,
        after_txt,
        local_k=args.local_k,
        forced_index=anchor_local_idx,
    )

    before_img_xy, before_txt_xy, after_img_xy, after_txt_xy = shared_projection(
        before_img.detach().cpu().numpy(),
        before_txt.detach().cpu().numpy(),
        after_img.detach().cpu().numpy(),
        after_txt.detach().cpu().numpy(),
    )

    make_figure(
        before_img_xy=before_img_xy,
        before_txt_xy=before_txt_xy,
        after_img_xy=after_img_xy,
        after_txt_xy=after_txt_xy,
        anchor_idx=anchor_local_idx,
        neighbor_spec=anchor_info,
        nas_before=nas_before,
        nas_after=nas_after,
        nas_k=args.nas_k,
        local_k=args.local_k,
        out_prefix=out_prefix,
    )

    report = {
        "model": model_name,
        "device": device,
        "lora_mix": float(args.lora_mix),
        "method_info": method_info,
        "config": {
            "figure_samples": int(sample_indices.shape[0]),
            "local_k": int(args.local_k),
            "nas_k": int(args.nas_k),
            "text_variant": args.text_variant,
            "paragraph_sentences": int(args.paragraph_sentences),
            "max_test_images": None if test_cap is None else int(test_cap),
            "lora_state": args.lora_state,
        },
        "sample_indices": sample_indices.tolist(),
        "global_overlap": {
            f"NAS@{args.nas_k}_before": float(nas_before),
            f"NAS@{args.nas_k}_after": float(nas_after),
            "delta": float(nas_after - nas_before),
        },
        "anchor": {
            "global_test_index": int(sample_indices[anchor_local_idx]),
            "sample_index": int(anchor_local_idx),
            **anchor_info,
        },
    }
    save_json(out_prefix.with_suffix(".json"), report)

    print(f"[Saved] {out_prefix.with_suffix('.png')}")
    print(f"[Saved] {out_prefix.with_suffix('.pdf')}")
    print(f"[Saved] {out_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
