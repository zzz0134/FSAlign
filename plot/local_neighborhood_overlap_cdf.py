#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset-level CDF of local neighborhood overlap (LNO@k).

For each paired image-text sample i, we compute

    LNO@k(i) = |N_k^I(i) ∩ N_k^T(i)| / k

where N_k^I(i) are the top-k nearest image neighbors of x_i, and N_k^T(i)
are the top-k nearest text neighbors of y_i, both excluding self. Because we
work with one paired text embedding per image, the image-neighbor index set is
already aligned with the text index set, so the pairing map P is the identity
on the paired test set.

Outputs:
  <out_prefix>.png
  <out_prefix>.pdf
  <out_prefix>.json
  <out_prefix>.npz
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import our_code_final as fs


BASELINE_COLOR = "#6b7280"
OURS_COLOR = "#c2410c"
RECOMMENDED_LNO_PRESETS = {
    ("flickr30k", "clip", "ViT-B-32", "short", 3): {
        "name": "flickr30k_clip_vitb32_strong_gap",
        "lora_state": REPO_ROOT / "results" / "our_final_train_1.24" / "clip_ViT-B-32_openai_flickr30k_karpathy_test_lora_state.pt",
        "lora_mix": 0.6,
        "notes": "Recommended preset for the paper-style Flickr30k CLIP LNO figure.",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def save_embedding_snapshot(
    path: Path,
    baseline_image_feats: torch.Tensor,
    baseline_paired_text: torch.Tensor,
    ours_image_feats: torch.Tensor,
    ours_paired_text: torch.Tensor,
    pair_map: torch.Tensor,
) -> None:
    ensure_dir(path.parent)
    n = int(baseline_image_feats.size(0))
    np.savez_compressed(
        path,
        baseline_image=baseline_image_feats.detach().cpu().numpy().astype(np.float32),
        baseline_text=baseline_paired_text.detach().cpu().numpy().astype(np.float32),
        ours_image=ours_image_feats.detach().cpu().numpy().astype(np.float32),
        ours_text=ours_paired_text.detach().cpu().numpy().astype(np.float32),
        pair_map=pair_map.detach().cpu().numpy().astype(np.int64),
        sample_index=np.arange(n, dtype=np.int64),
    )



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



def resolve_recommended_preset(args) -> Optional[Dict[str, Any]]:
    if args.disable_recommended_preset or args.lora_state:
        return None
    if args.model_key == "clip":
        model_tag = args.clip_model
    elif args.model_key == "openclip":
        model_tag = f"{args.openclip_model}:{args.openclip_pretrained}"
    elif args.model_key == "siglip":
        model_tag = args.siglip_name
    else:
        return None
    preset = RECOMMENDED_LNO_PRESETS.get(
        (args.dataset, args.model_key, model_tag, args.text_variant, int(args.paragraph_sentences))
    )
    if not preset:
        return None
    preset_path = Path(preset["lora_state"])
    if not preset_path.exists():
        return None
    resolved = dict(preset)
    resolved["lora_state"] = str(preset_path)
    return resolved



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



def get_dataset(data_root: str, dataset_name: str, split: str):
    if dataset_name == "flickr30k":
        kjson = fs.ensure_karpathy_json(data_root, "flickr30k")
        roots = [
            str(Path(data_root) / "flickr30k" / "flickr30k-images"),
            str(Path(data_root) / "flickr30k" / "images"),
            str(Path(data_root) / "flickr30k"),
        ]
    elif dataset_name == "mscoco":
        kjson = fs.ensure_karpathy_json(data_root, "coco")
        roots = [
            str(Path(data_root) / "mscoco2014" / "train2014"),
            str(Path(data_root) / "mscoco2014" / "val2014"),
            str(Path(data_root) / "coco2014" / "train2014"),
            str(Path(data_root) / "coco2014" / "val2014"),
            str(Path(data_root) / "coco" / "train2014"),
            str(Path(data_root) / "coco" / "val2014"),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
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
    dataset_name: str,
    device: str,
    cache_dir: Path,
    model_name: str,
    effective_lora_state: str,
    preset_info: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if effective_lora_state:
        state = torch.load(effective_lora_state, map_location="cpu")
        method_info: Dict[str, Any] = {"mode": "lora_state", "path": str(effective_lora_state)}
        if preset_info:
            method_info["preset"] = {
                "name": str(preset_info["name"]),
                "notes": str(preset_info["notes"]),
                "lora_mix": float(preset_info["lora_mix"]),
            }
        return state, method_info

    train_ds = get_dataset(args.data_root, dataset_name, split="train")
    train_cap = None if args.max_train_images <= 0 else int(args.max_train_images)
    cache_stub = f"{dataset_name}_{safe_model_tag(model_name)}_{args.text_variant}_p{args.paragraph_sentences}"
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
def apply_method(bundle: fs.RetrievalFeatureBundle, lora_state: Dict[str, Any], lora_mix: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    image_feats = fs.apply_lora_state(bundle.image_feats, layer_img, lora_mix)
    text_feats = fs.apply_lora_state(bundle.text_feats, layer_txt, lora_mix)
    return image_feats, text_feats



def compute_lno_distributions(
    image_feats: torch.Tensor,
    paired_text: torch.Tensor,
    ks: List[int],
    neighbor_chunk: int,
) -> Dict[int, np.ndarray]:
    clean_ks = sorted({int(k) for k in ks if int(k) > 0})
    max_k = max(clean_ks)
    image_cpu = image_feats.detach().cpu()
    text_cpu = paired_text.detach().cpu()
    img_neighbors = fs.same_modality_topk_neighbors(image_cpu, max_k, device="cpu", chunk=neighbor_chunk).cpu().numpy()
    txt_neighbors = fs.same_modality_topk_neighbors(text_cpu, max_k, device="cpu", chunk=neighbor_chunk).cpu().numpy()

    distributions: Dict[int, np.ndarray] = {}
    for k in clean_ks:
        img_k = img_neighbors[:, :k]
        txt_k = txt_neighbors[:, :k]
        overlap = (img_k[:, :, None] == txt_k[:, None, :]).any(axis=2).sum(axis=1).astype(np.float32) / float(k)
        distributions[k] = overlap
    return distributions



def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "count": int(values.shape[0]),
    }



def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(values)
    ys = np.arange(1, xs.shape[0] + 1, dtype=np.float64) / float(xs.shape[0])
    return xs, ys



def stats_box(base_stats: Dict[str, float], ours_stats: Dict[str, float]) -> str:
    mean_gain = ours_stats["mean"] - base_stats["mean"]
    rel_gain = 100.0 * mean_gain / max(base_stats["mean"], 1e-8)
    return (
        "mean / med / q25\n"
        f"CLIP   {base_stats['mean']:.3f} / {base_stats['median']:.3f} / {base_stats['q25']:.3f}\n"
        f"FSA    {ours_stats['mean']:.3f} / {ours_stats['median']:.3f} / {ours_stats['q25']:.3f}\n"
        f"Gain   {mean_gain:+.3f} ({rel_gain:+.0f}%)"
    )



def plot_cdf(distributions: Dict[str, Dict[int, np.ndarray]], summaries: Dict[str, Dict[int, Dict[str, float]]], ks: List[int], dataset_name: str, out_prefix: Path) -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10.5,
    })

    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.1), sharex=True, sharey=True)
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        base_x, base_y = ecdf(distributions["baseline"][k])
        ours_x, ours_y = ecdf(distributions["ours"][k])
        ax.step(base_x, base_y, where="post", color=BASELINE_COLOR, linewidth=2.1, label="CLIP")
        ax.step(ours_x, ours_y, where="post", color=OURS_COLOR, linewidth=2.1, label="FSAlign")
        ax.set_title(f"k = {k}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Local neighborhood overlap")
        ax.grid(True, alpha=0.20)
        ax.text(
            0.03,
            0.97,
            stats_box(summaries["baseline"][k], summaries["ours"][k]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.6,
            family="DejaVu Sans Mono",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d4d4d8", "alpha": 0.92},
        )

    axes[0].set_ylabel("Fraction of test pairs")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"CDF of local neighborhood overlap on {dataset_name}", y=1.04, fontsize=14)
    fig.tight_layout()
    ensure_dir(out_prefix.parent)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    parser.add_argument("--dataset", type=str, default="flickr30k", choices=["flickr30k", "mscoco"])
    parser.add_argument("--cache-dir", type=str, default="/work/was598/modilty_gap/plot/cache")
    parser.add_argument("--out-prefix", type=str, default="/work/was598/modilty_gap/plot/flickr30k_lno_cdf")
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

    parser.add_argument("--ks", type=str, default="10,50,100")
    parser.add_argument("--neighbor-chunk", type=int, default=1024)

    parser.add_argument("--lora-state", type=str, default="")
    parser.add_argument("--disable-recommended-preset", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--embedding-out", type=str, default="")
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
    embedding_out = Path(args.embedding_out) if args.embedding_out else Path(f"{out_prefix}_embeddings.npz")
    ks = [int(part.strip()) for part in args.ks.split(",") if part.strip()]
    preset_info = resolve_recommended_preset(args)
    effective_lora_state = str(args.lora_state)
    effective_lora_mix = float(args.lora_mix)
    if preset_info:
        effective_lora_state = str(preset_info["lora_state"])
        effective_lora_mix = float(preset_info["lora_mix"])
        print(
            f"[Preset] {preset_info['name']} -> {effective_lora_state} "
            f"(lora_mix={effective_lora_mix:.2f})"
        )

    model_name, model = build_model(args, device)
    test_ds = get_dataset(args.data_root, args.dataset, split="test")
    test_cap = None if args.max_test_images <= 0 else int(args.max_test_images)
    cache_stub = f"{args.dataset}_{safe_model_tag(model_name)}_{args.text_variant}_p{args.paragraph_sentences}"
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

    lora_state, method_info = load_or_train_lora_state(
        model,
        args,
        args.dataset,
        device,
        cache_dir,
        model_name,
        effective_lora_state,
        preset_info,
    )
    ours_image_feats, ours_text_feats = apply_method(test_bundle, lora_state, effective_lora_mix, device)

    baseline_paired = test_bundle.text_feats[test_bundle.pair_map]
    ours_paired = ours_text_feats[test_bundle.pair_map]

    distributions = {
        "baseline": compute_lno_distributions(test_bundle.image_feats, baseline_paired, ks, args.neighbor_chunk),
        "ours": compute_lno_distributions(ours_image_feats, ours_paired, ks, args.neighbor_chunk),
    }
    summaries = {
        "baseline": {k: summarize(distributions["baseline"][k]) for k in ks},
        "ours": {k: summarize(distributions["ours"][k]) for k in ks},
    }

    plot_cdf(distributions, summaries, ks, args.dataset, out_prefix)

    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        **{f"baseline_k{k}": distributions["baseline"][k] for k in ks},
        **{f"ours_k{k}": distributions["ours"][k] for k in ks},
    )

    if args.save_embeddings:
        save_embedding_snapshot(
            embedding_out,
            test_bundle.image_feats,
            baseline_paired,
            ours_image_feats,
            ours_paired,
            test_bundle.pair_map,
        )

    report = {
        "dataset": args.dataset,
        "model": model_name,
        "device": device,
        "ks": ks,
        "method_info": method_info,
        "config": {
            "max_test_images": None if test_cap is None else int(test_cap),
            "text_variant": args.text_variant,
            "paragraph_sentences": int(args.paragraph_sentences),
            "neighbor_chunk": int(args.neighbor_chunk),
            "lora_state": args.lora_state,
            "effective_lora_state": effective_lora_state or None,
            "lora_mix": float(args.lora_mix),
            "effective_lora_mix": float(effective_lora_mix),
            "disable_recommended_preset": bool(args.disable_recommended_preset),
            "save_embeddings": bool(args.save_embeddings),
            "embedding_out": str(embedding_out) if args.save_embeddings else None,
        },
        "summary": {
            "baseline": {str(k): summaries["baseline"][k] for k in ks},
            "ours": {str(k): summaries["ours"][k] for k in ks},
        },
    }
    save_json(out_prefix.with_suffix(".json"), report)

    print(f"[Saved] {out_prefix.with_suffix('.png')}")
    print(f"[Saved] {out_prefix.with_suffix('.pdf')}")
    print(f"[Saved] {out_prefix.with_suffix('.json')}")
    print(f"[Saved] {out_prefix.with_suffix('.npz')}")
    if args.save_embeddings:
        print(f"[Saved] {embedding_out}")


if __name__ == "__main__":
    main()
