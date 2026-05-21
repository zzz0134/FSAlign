#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

import our_code_final as fs


PROTOCOL_SCOPE = "method_specific_post_adaptation_on_shared_frozen_features"
TRAIN_TEXT_DEFINITION = "paired_text = train_bundle.text_feats[train_bundle.pair_map] (one caption per image)"
EVAL_TEXT_DEFINITION = "all test captions in test_bundle.text_feats"
FSALIGN_COMPLEXITY = "O(A N D + S M^2 D + L^2 D) per epoch"
LINEAR_BASELINE_COMPLEXITY = "O(N D)"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def benchmark_stage(device: str, fn):
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.time()
    out = fn()
    maybe_sync(device)
    peak = 0.0
    if str(device).startswith("cuda") and torch.cuda.is_available():
        peak = float(torch.cuda.max_memory_allocated(device))
    return out, {
        "fit_time_sec": float(time.time() - t0),
        "peak_cuda_memory_allocated_bytes": peak,
        "peak_cuda_memory_allocated_mb": peak / (1024.0 ** 2),
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_model(
    device: str,
    model_key: str,
    model_size: int,
    siglip_name: str,
    openclip_model: str,
    openclip_pretrained: str,
) -> Tuple[str, fs.VLBackbone]:
    if model_key == "clip":
        size_tag = "ViT-B-16" if model_size == 16 else "ViT-B-32"
        return f"clip:{size_tag}:openai", fs.CLIPWrapper(size_tag, device=device)
    if model_key == "openclip":
        return (
            f"open_clip:{openclip_model}:{openclip_pretrained}",
            fs.OpenCLIPWrapper(openclip_model, openclip_pretrained, device=device),
        )
    if model_key == "siglip":
        return f"siglip:{siglip_name}", fs.SigLIPWrapper(siglip_name, device=device)
    raise ValueError(f"Unsupported model_key: {model_key}")


def resolve_flickr_dataset_root(data_root: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    candidates: List[Path] = []
    if (root / "flickr30k").exists():
        candidates.append(root / "flickr30k")
    candidates.append(root)

    seen = set()
    ordered: List[Path] = []
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            seen.add(key)
            ordered.append(cand)

    for cand in ordered:
        if (cand / "karpathy" / "dataset_flickr30k.json").exists():
            return cand
        if (cand / "flickr30k-images").exists() or (cand / "images").exists():
            return cand

    return root / "flickr30k" if root.name != "flickr30k" else root


def flickr_roots(data_root: str) -> List[str]:
    root = resolve_flickr_dataset_root(data_root)
    return [
        str(root / "flickr30k-images"),
        str(root / "images"),
        str(root),
    ]


def flickr_train_test(data_root: str):
    dataset_root = resolve_flickr_dataset_root(data_root)
    kjson = dataset_root / "karpathy" / "dataset_flickr30k.json"
    if not kjson.exists():
        parent_root = dataset_root.parent if dataset_root.name == "flickr30k" else dataset_root
        kjson = fs.ensure_karpathy_json(str(parent_root), "flickr30k")
    roots = flickr_roots(str(dataset_root))
    train_ds = fs.KarpathyRetrievalDataset(str(kjson), roots, split="train", max_images=None)
    test_ds = fs.KarpathyRetrievalDataset(str(kjson), roots, split="test", max_images=None)
    return train_ds, test_ds


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
    torch.save(bundle_to_cpu_dict(bundle), cache_path)
    return bundle


def build_fsalign_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        lora_state="",
        train_epochs=30,
        train_anchors=512,
        anchor_batch=128,
        spectral_samples=512,
        train_lr=1e-3,
        lambda_dbl=1.0,
        lambda_spec=0.1,
        lambda_match=0.1,
        lambda_align=1.0,
        lambda_orth=0.1,
        train_reg=1e-3,
        train_print_every=30,
        align_temp=0.07,
        align_samples=2048,
        pairwise_row_chunk=0,
        pairwise_col_chunk=0,
        pairwise_checkpoint=False,
        lora_rank=8,
        lora_alpha=8.0,
        save_lora=False,
        lora_mix=0.3,
        multi_caption=True,
        caption_agg="random",
        text_variant="short",
        paragraph_sentences=3,
        structure_batch_size=0,
        noise_pair_rate=0.0,
        noise_mix=0.5,
        dimension_mode="shared",
        dimension_offset_df=0.0,
        dimension_offset_ds=0.0,
        dimension_offset_dw=None,
        lambda_dim_offset=0.0,
        early_stop=False,
        val_split="internal",
        val_frac=0.1,
        patience=2,
        min_delta=0.0,
        df=2.0,
        ds=None,
        dw=4.0,
        alpha=1.0,
        radii_min=0.05,
        radii_max=0.5,
        radii_count=6,
        rho_list="1.5,2.0,3.0",
        diffusion_min=0.01,
        diffusion_max=1.0,
        diffusion_count=6,
        lambda_neighbor_compete=0.0,
        neighbor_compete_k=10,
        neighbor_compete_margin=0.0,
        neighbor_compete_samples=1024,
        neighbor_compete_chunk=1024,
    )


def retrieval_metrics(bundle: fs.RetrievalFeatureBundle, image_feats: torch.Tensor, text_feats: torch.Tensor, device: str):
    return fs.retrieval_metrics_from_embeddings(
        bundle,
        device=device,
        nas_k_val=100,
        nas_max_items=min(5000, bundle.image_feats.size(0)),
        intra_samples=20000,
        image_feats=image_feats,
        text_feats=text_feats,
    )


def run_fsalign_paper(
    train_bundle: fs.RetrievalFeatureBundle,
    test_bundle: fs.RetrievalFeatureBundle,
    device: str,
) -> Dict[str, Any]:
    args = build_fsalign_args(device)
    radii = fs.logspace_scales(args.radii_min, args.radii_max, args.radii_count)
    rho_list = [float(x) for x in args.rho_list.split(",") if x.strip()]
    diffusion_scales = fs.logspace_scales(args.diffusion_min, args.diffusion_max, args.diffusion_count)
    lora_state, history = fs.train_lora_postprocess(
        train_bundle.image_feats,
        train_bundle.paired_text,
        radii,
        rho_list,
        diffusion_scales,
        args,
        caption_pool=(train_bundle.text_feats, train_bundle.cap_indices),
    )
    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    test_img = fs.apply_lora_state(test_bundle.image_feats.to(device), layer_img, args.lora_mix)
    test_txt = fs.apply_lora_state(test_bundle.text_feats.to(device), layer_txt, args.lora_mix)
    gap, i2t, t2i, extra = retrieval_metrics(test_bundle, test_img, test_txt, device)
    stats = history.get("train_stats", {})
    return {
        "method": "FSAlign",
        "variant": "paper_setting",
        "fit_scope": PROTOCOL_SCOPE,
        "shared_encoder_included": False,
        "train_images": int(train_bundle.image_feats.size(0)),
        "train_texts": int(train_bundle.paired_text.size(0)),
        "train_text_definition": TRAIN_TEXT_DEFINITION,
        "eval_text_definition": EVAL_TEXT_DEFINITION,
        "fit_time_sec": float(stats.get("train_time_sec", 0.0)),
        "fit_peak_memory_mb": float(stats.get("peak_cuda_memory_allocated_mb", 0.0)),
        "epochs": int(stats.get("complexity", {}).get("epochs", args.train_epochs)),
        "time_per_epoch_sec": float(stats.get("train_time_sec", 0.0)) / max(
            int(stats.get("complexity", {}).get("epochs", args.train_epochs)), 1
        ),
        "complexity_big_o": FSALIGN_COMPLEXITY,
        "complexity_detail": stats.get("complexity", {}),
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "config": vars(args).copy(),
    }


def run_iot_unified(
    train_bundle: fs.RetrievalFeatureBundle,
    test_bundle: fs.RetrievalFeatureBundle,
    device: str,
) -> Dict[str, Any]:
    eps = 1e-6

    def fit_stage():
        img_mean = train_bundle.image_feats.mean(dim=0)
        img_std = torch.sqrt(train_bundle.image_feats.var(dim=0, unbiased=False) + eps)
        txt_mean = train_bundle.paired_text.mean(dim=0)
        txt_std = torch.sqrt(train_bundle.paired_text.var(dim=0, unbiased=False) + eps)
        return img_mean, img_std, txt_mean, txt_std

    (img_mean, img_std, txt_mean, txt_std), stats = benchmark_stage(device, fit_stage)
    test_img = fs.l2norm((test_bundle.image_feats - img_mean) / (img_std + eps))
    test_txt = fs.l2norm((test_bundle.text_feats - txt_mean) / (txt_std + eps))
    gap, i2t, t2i, extra = retrieval_metrics(test_bundle, test_img, test_txt, device)
    feat_dim = int(train_bundle.image_feats.size(1))
    n_pairs = int(train_bundle.image_feats.size(0))
    return {
        "method": "IOT",
        "variant": "unified_paired_feature_fit",
        "fit_scope": PROTOCOL_SCOPE,
        "shared_encoder_included": False,
        "train_images": n_pairs,
        "train_texts": n_pairs,
        "train_text_definition": TRAIN_TEXT_DEFINITION,
        "eval_text_definition": EVAL_TEXT_DEFINITION,
        "fit_time_sec": float(stats["fit_time_sec"]),
        "fit_peak_memory_mb": float(stats["peak_cuda_memory_allocated_mb"]),
        "epochs": "",
        "time_per_epoch_sec": "",
        "complexity_big_o": LINEAR_BASELINE_COMPLEXITY,
        "complexity_detail": {
            "n_train": n_pairs,
            "feat_dim": feat_dim,
            "estimated_vector_ops": float(4 * n_pairs * feat_dim),
        },
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "config": {
            "eps": eps,
        },
    }


def run_gr_clip_unified(
    train_bundle: fs.RetrievalFeatureBundle,
    test_bundle: fs.RetrievalFeatureBundle,
    device: str,
) -> Dict[str, Any]:
    def fit_stage():
        img_mean = train_bundle.image_feats.mean(dim=0)
        txt_mean = train_bundle.paired_text.mean(dim=0)
        return img_mean, txt_mean

    (img_mean, txt_mean), stats = benchmark_stage(device, fit_stage)
    test_img = fs.l2norm(test_bundle.image_feats - img_mean)
    test_txt = fs.l2norm(test_bundle.text_feats - txt_mean)
    gap, i2t, t2i, extra = retrieval_metrics(test_bundle, test_img, test_txt, device)
    feat_dim = int(train_bundle.image_feats.size(1))
    n_pairs = int(train_bundle.image_feats.size(0))
    return {
        "method": "GR-CLIP",
        "variant": "unified_paired_feature_fit",
        "fit_scope": PROTOCOL_SCOPE,
        "shared_encoder_included": False,
        "train_images": n_pairs,
        "train_texts": n_pairs,
        "train_text_definition": TRAIN_TEXT_DEFINITION,
        "eval_text_definition": EVAL_TEXT_DEFINITION,
        "fit_time_sec": float(stats["fit_time_sec"]),
        "fit_peak_memory_mb": float(stats["peak_cuda_memory_allocated_mb"]),
        "epochs": "",
        "time_per_epoch_sec": "",
        "complexity_big_o": LINEAR_BASELINE_COMPLEXITY,
        "complexity_detail": {
            "n_train": n_pairs,
            "feat_dim": feat_dim,
            "estimated_vector_ops": float(2 * n_pairs * feat_dim),
        },
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "config": {},
    }


def comparison_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "method": record["method"],
        "variant": record["variant"],
        "fit_scope": record["fit_scope"],
        "shared_encoder_included": record["shared_encoder_included"],
        "train_images": record["train_images"],
        "train_texts": record["train_texts"],
        "train_text_definition": record["train_text_definition"],
        "eval_text_definition": record["eval_text_definition"],
        "I2T_R1": record["i2t"]["R@1"],
        "I2T_R5": record["i2t"]["R@5"],
        "I2T_R10": record["i2t"]["R@10"],
        "T2I_R1": record["t2i"]["R@1"],
        "T2I_R5": record["t2i"]["R@5"],
        "T2I_R10": record["t2i"]["R@10"],
        "centroid_distance": record["gap"]["centroid_distance"],
        "relative_modality_gap": record["gap"]["relative_modality_gap"],
        "NAS@100": record["gap"].get("NAS@100", 0.0),
        "CMAS": record["gap"]["CMAS"],
        "avg_words_per_text": record["extra"].get("avg_words_per_text", 0.0),
        "fit_time_sec": record["fit_time_sec"],
        "fit_peak_memory_mb": record["fit_peak_memory_mb"],
        "epochs": record["epochs"],
        "time_per_epoch_sec": record["time_per_epoch_sec"],
        "complexity_big_o": record["complexity_big_o"],
    }
    detail = record.get("complexity_detail", {})
    row["effective_train_size"] = detail.get("n_train", "")
    row["anchor_count"] = detail.get("anchor_count", "")
    row["spectral_count"] = detail.get("spectral_count", "")
    row["align_count"] = detail.get("align_count", "")
    return row


def write_summary(
    out_dir: Path,
    model_name: str,
    device: str,
    rows: List[Dict[str, Any]],
) -> None:
    fsalign = next(row for row in rows if row["method"] == "FSAlign")
    iot = next(row for row in rows if row["method"] == "IOT")
    gr = next(row for row in rows if row["method"] == "GR-CLIP")

    def ratio(a: float, b: float) -> float:
        if b == 0.0:
            return 0.0
        return a / b

    lines = [
        "# Unified Q2 Overhead Benchmark",
        "",
        f"Model: {model_name}",
        "Dataset: Flickr30k Karpathy",
        f"Device: {device}",
        "",
        "## Protocol",
        f"Scope: {PROTOCOL_SCOPE}.",
        "Shared frozen image/text encoding is excluded for all methods.",
        f"Train text definition: {TRAIN_TEXT_DEFINITION}.",
        f"Evaluation text definition: {EVAL_TEXT_DEFINITION}.",
        "",
        "## Results",
        (
            f"FSAlign (paper setting): {fsalign['fit_time_sec']:.2f}s over {fsalign['epochs']} epochs "
            f"({fsalign['time_per_epoch_sec']:.2f}s/epoch), {fsalign['fit_peak_memory_mb']:.2f} MB."
        ),
        f"IOT: {iot['fit_time_sec']:.4f}s, {iot['fit_peak_memory_mb']:.2f} MB.",
        f"GR-CLIP: {gr['fit_time_sec']:.4f}s, {gr['fit_peak_memory_mb']:.2f} MB.",
        "",
        "## Readout",
        (
            f"Under one shared feature boundary, FSAlign is {ratio(fsalign['fit_time_sec'], iot['fit_time_sec']):.1f}x "
            f"slower than IOT and {ratio(fsalign['fit_time_sec'], gr['fit_time_sec']):.1f}x slower than GR-CLIP "
            f"in adaptation time."
        ),
        (
            f"Peak memory differs by {fsalign['fit_peak_memory_mb'] - iot['fit_peak_memory_mb']:.2f} MB versus IOT "
            f"and {fsalign['fit_peak_memory_mb'] - gr['fit_peak_memory_mb']:.2f} MB versus GR-CLIP."
        ),
        f"FSAlign complexity: {FSALIGN_COMPLEXITY}. Baseline complexity: {LINEAR_BASELINE_COMPLEXITY}.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-key", type=str, default="siglip", choices=["clip", "openclip", "siglip"])
    parser.add_argument("--model-size", type=int, default=32, choices=[16, 32])
    parser.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument("--paragraph-sentences", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fs.seed_all(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    table_dir = out_dir / "tables"
    cache_dir = out_dir / "cache"
    for folder in [out_dir, raw_dir, table_dir, cache_dir]:
        ensure_dir(folder)

    model_name, model = build_model(
        device=device,
        model_key=args.model_key,
        model_size=args.model_size,
        siglip_name=args.siglip_name,
        openclip_model=args.openclip_model,
        openclip_pretrained=args.openclip_pretrained,
    )

    cache_tag = fs.safe_filename(model_name.replace(":", "_"))
    train_ds, test_ds = flickr_train_test(args.data_root)
    max_train = None if args.max_train_images <= 0 else args.max_train_images
    max_test = None if args.max_test_images <= 0 else args.max_test_images

    train_bundle = get_retrieval_bundle(
        model,
        train_ds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=max_train,
        text_variant="short",
        paragraph_sentences=args.paragraph_sentences,
        cache_path=cache_dir / f"flickr_train_short_{cache_tag}_{max_train or 'all'}.pt",
    )
    test_bundle = get_retrieval_bundle(
        model,
        test_ds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=max_test,
        text_variant="short",
        paragraph_sentences=args.paragraph_sentences,
        cache_path=cache_dir / f"flickr_test_short_{cache_tag}_{max_test or 'all'}.pt",
    )

    fsalign_record = run_fsalign_paper(train_bundle, test_bundle, device)
    save_json(raw_dir / "fsalign_paper_setting.json", fsalign_record)

    iot_record = run_iot_unified(train_bundle, test_bundle, device)
    save_json(raw_dir / "iot_unified.json", iot_record)

    gr_record = run_gr_clip_unified(train_bundle, test_bundle, device)
    save_json(raw_dir / "gr_clip_unified.json", gr_record)

    rows = [
        comparison_row(fsalign_record),
        comparison_row(iot_record),
        comparison_row(gr_record),
    ]
    save_csv(table_dir / "q2_unified_overhead_comparison.csv", rows)

    manifest = {
        "protocol_scope": PROTOCOL_SCOPE,
        "shared_encoder_included": False,
        "train_text_definition": TRAIN_TEXT_DEFINITION,
        "eval_text_definition": EVAL_TEXT_DEFINITION,
        "device": device,
        "model_name": model_name,
        "seed": args.seed,
        "artifacts": {
            "fsalign_raw": str(raw_dir / "fsalign_paper_setting.json"),
            "iot_raw": str(raw_dir / "iot_unified.json"),
            "gr_raw": str(raw_dir / "gr_clip_unified.json"),
            "comparison_csv": str(table_dir / "q2_unified_overhead_comparison.csv"),
        },
    }
    save_json(out_dir / "manifest.json", manifest)
    write_summary(out_dir, model_name, device, [fsalign_record, iot_record, gr_record])
    print(f"[Done] Unified Q2 benchmark saved to {out_dir}")


if __name__ == "__main__":
    main()
