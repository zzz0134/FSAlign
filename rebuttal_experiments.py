#!/usr/bin/env python3
import argparse
import gc
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.datasets as tvds

import our_code_final as fs

QUESTION_TITLES = {
    "q1": "Computational Complexity And Overhead",
    "q2": "FSAlign Versus Simpler Baselines",
    "q3": "Batch-Size Sensitivity",
    "q4": "Long-Paragraph Robustness",
    "q5": "Component Ablations",
    "q6": "Unimodal Representation Quality",
    "q7": "Relaxed Modality Dimensions",
    "q8": "Noisy Image-Text Pairs",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


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


def parse_int_list(spec: str) -> List[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def parse_float_list(spec: str) -> List[float]:
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def base_fsalign_args(device: str, cli_args) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        lora_state="",
        train_epochs=int(cli_args.train_epochs),
        train_anchors=int(cli_args.train_anchors),
        anchor_batch=int(cli_args.anchor_batch),
        spectral_samples=int(cli_args.spectral_samples),
        train_lr=float(cli_args.train_lr),
        lambda_dbl=float(cli_args.lambda_dbl),
        lambda_spec=float(cli_args.lambda_spec),
        lambda_match=float(cli_args.lambda_match),
        lambda_align=float(cli_args.lambda_align),
        lambda_orth=float(cli_args.lambda_orth),
        train_reg=float(cli_args.train_reg),
        train_print_every=max(int(cli_args.train_epochs), 1),
        align_temp=float(cli_args.align_temp),
        align_samples=int(cli_args.align_samples),
        pairwise_row_chunk=int(cli_args.pairwise_row_chunk),
        pairwise_col_chunk=int(cli_args.pairwise_col_chunk),
        pairwise_checkpoint=bool(cli_args.pairwise_checkpoint),
        lora_rank=int(cli_args.lora_rank),
        lora_alpha=float(cli_args.lora_alpha),
        save_lora=False,
        lora_mix=float(cli_args.lora_mix),
        multi_caption=bool(cli_args.multi_caption),
        caption_agg=str(cli_args.caption_agg),
        text_variant="short",
        paragraph_sentences=int(cli_args.paragraph_sentences),
        structure_batch_size=0,
        noise_pair_rate=0.0,
        noise_mix=float(cli_args.noise_mix),
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
        df=float(cli_args.df),
        ds=None if cli_args.ds is None else float(cli_args.ds),
        dw=float(cli_args.dw),
        alpha=float(cli_args.alpha),
        radii_min=float(cli_args.radii_min),
        radii_max=float(cli_args.radii_max),
        radii_count=int(cli_args.radii_count),
        rho_list=str(cli_args.rho_list),
        diffusion_min=float(cli_args.diffusion_min),
        diffusion_max=float(cli_args.diffusion_max),
        diffusion_count=int(cli_args.diffusion_count),
    )


def apply_overrides(ns: SimpleNamespace, **overrides: Any) -> SimpleNamespace:
    data = vars(ns).copy()
    data.update(overrides)
    return SimpleNamespace(**data)


def build_true_overhead_args(base_args: SimpleNamespace, train_bundle: fs.RetrievalFeatureBundle, cli_args) -> SimpleNamespace:
    full_n = int(train_bundle.image_feats.size(0))
    epochs = int(cli_args.true_overhead_epochs) if int(cli_args.true_overhead_epochs) > 0 else int(base_args.train_epochs)
    anchor_batch = int(cli_args.true_anchor_batch) if int(cli_args.true_anchor_batch) > 0 else int(base_args.anchor_batch)
    return apply_overrides(
        base_args,
        train_epochs=epochs,
        train_anchors=full_n,
        anchor_batch=anchor_batch,
        spectral_samples=full_n,
        align_samples=0,
        structure_batch_size=0,
        pairwise_row_chunk=int(cli_args.true_pairwise_row_chunk),
        pairwise_col_chunk=int(cli_args.true_pairwise_col_chunk),
        pairwise_checkpoint=bool(cli_args.true_pairwise_checkpoint),
    )


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


def flickr_train_test(data_root: str):
    kjson = fs.ensure_karpathy_json(data_root, "flickr30k")
    roots = [
        str(Path(data_root) / "flickr30k" / "flickr30k-images"),
        str(Path(data_root) / "flickr30k" / "images"),
        str(Path(data_root) / "flickr30k"),
    ]
    train_ds = fs.KarpathyRetrievalDataset(str(kjson), roots, split="train", max_images=None)
    test_ds = fs.KarpathyRetrievalDataset(str(kjson), roots, split="test", max_images=None)
    return train_ds, test_ds


def cifar100_test(data_root: str):
    return tvds.CIFAR100(root=str(Path(data_root) / "cifar100"), train=False, download=False, transform=None)


def cifar100_train(data_root: str):
    return tvds.CIFAR100(root=str(Path(data_root) / "cifar100"), train=True, download=False, transform=None)


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


def get_cifar_embeddings(
    model: fs.VLBackbone,
    data_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
    max_items: Optional[int],
    cache_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["image_feats"].to(device), payload["labels"].to(device), payload["weights"].to(device)
    dataset = cifar100_test(data_root)
    image_feats, labels = fs.encode_classification_images(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_items=max_items,
    )
    weights = fs.build_zeroshot_weights(model, dataset.classes, fs.CIFAR100_TEMPLATES, device)
    torch.save(
        {
            "image_feats": image_feats.detach().cpu(),
            "labels": labels.detach().cpu(),
            "weights": weights.detach().cpu(),
        },
        cache_path,
    )
    return image_feats, labels, weights


def get_cifar_image_embeddings(
    model: fs.VLBackbone,
    data_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
    train: bool,
    max_items: Optional[int],
    cache_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["image_feats"].to(device), payload["labels"].to(device)
    dataset = cifar100_train(data_root) if train else cifar100_test(data_root)
    image_feats, labels = fs.encode_classification_images(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_items=max_items,
    )
    torch.save(
        {
            "image_feats": image_feats.detach().cpu(),
            "labels": labels.detach().cpu(),
        },
        cache_path,
    )
    return image_feats, labels


def dtd_test(data_root: str):
    return tvds.DTD(root=str(Path(data_root) / "dtd"), split="test", download=False, transform=None)


def tinyimagenet_val(data_root: str):
    return fs.TinyImageNet200Val(data_root)


def get_zeroshot_embeddings(
    model: fs.VLBackbone,
    dataset,
    classnames: List[str],
    templates: List[str],
    device: str,
    batch_size: int,
    num_workers: int,
    max_items: Optional[int],
    cache_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["image_feats"].to(device), payload["labels"].to(device), payload["weights"].to(device)
    image_feats, labels = fs.encode_classification_images(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_items=max_items,
    )
    weights = fs.build_zeroshot_weights(model, classnames, templates, device)
    torch.save(
        {
            "image_feats": image_feats.detach().cpu(),
            "labels": labels.detach().cpu(),
            "weights": weights.detach().cpu(),
        },
        cache_path,
    )
    return image_feats, labels, weights


def evaluate_zeroshot_suite(
    lora_state: Dict[str, Any],
    suites: Dict[str, Dict[str, torch.Tensor]],
    device: str,
    lora_mix: float,
) -> Dict[str, Dict[str, float]]:
    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    out: Dict[str, Dict[str, float]] = {}
    for name, payload in suites.items():
        image_feats = fs.apply_lora_state(payload["image_feats"], layer_img, lora_mix)
        weights = fs.apply_lora_state(payload["weights"], layer_txt, lora_mix)
        out[name] = classification_metrics(image_feats, weights, payload["labels"])
    return out


def build_q7_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = retrieval_row(record)
    final_dims = record.get("train_stats", {}).get("final_dimensions", {})
    row["dimension_mode"] = final_dims.get("mode", record.get("config", {}).get("dimension_mode", "shared"))
    for key in ["df_shared", "ds_shared", "df_img", "df_txt", "ds_img", "ds_txt", "dw_img", "dw_txt", "delta_f", "delta_s"]:
        row[key] = final_dims.get(key, 0.0)
    zeroshot = record.get("zeroshot", {})
    for dataset_name, prefix in [("cifar100", "cifar100"), ("dtd", "dtd"), ("tiny_imagenet_200", "tiny_imagenet_200")]:
        metrics = zeroshot.get(dataset_name, {})
        row[f"{prefix}_top1"] = metrics.get("top1", 0.0)
        row[f"{prefix}_top5"] = metrics.get("top5", 0.0)
    return row


def benchmark_start(device: str) -> float:
    maybe_sync(device)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    return time.time()


def benchmark_end(device: str, t0: float) -> Dict[str, float]:
    maybe_sync(device)
    peak = 0.0
    if str(device).startswith("cuda") and torch.cuda.is_available():
        peak = float(torch.cuda.max_memory_allocated(device))
    return {
        "fit_time_sec": float(time.time() - t0),
        "peak_cuda_memory_allocated_bytes": peak,
        "peak_cuda_memory_allocated_mb": peak / (1024.0 ** 2),
    }


def iot_adapt(train_bundle: fs.RetrievalFeatureBundle, test_bundle: fs.RetrievalFeatureBundle, device: str) -> Dict[str, Any]:
    eps = 1e-6
    t0 = benchmark_start(device)
    img_mean = train_bundle.image_feats.mean(dim=0)
    img_std = torch.sqrt(train_bundle.image_feats.var(dim=0, unbiased=False) + eps)
    txt_mean = train_bundle.text_feats.mean(dim=0)
    txt_std = torch.sqrt(train_bundle.text_feats.var(dim=0, unbiased=False) + eps)
    stats = benchmark_end(device, t0)
    test_img = fs.l2norm((test_bundle.image_feats - img_mean) / (img_std + eps))
    test_txt = fs.l2norm((test_bundle.text_feats - txt_mean) / (txt_std + eps))
    complexity = {
        "big_o": "O((N_img + N_txt) * D)",
        "estimated_vector_ops": float((train_bundle.image_feats.size(0) + train_bundle.text_feats.size(0)) * train_bundle.image_feats.size(1)),
        "n_images": int(train_bundle.image_feats.size(0)),
        "n_texts": int(train_bundle.text_feats.size(0)),
        "feat_dim": int(train_bundle.image_feats.size(1)),
    }
    return {"image_feats": test_img, "text_feats": test_txt, "train_stats": stats, "complexity": complexity}


def gr_adapt(train_bundle: fs.RetrievalFeatureBundle, test_bundle: fs.RetrievalFeatureBundle, device: str) -> Dict[str, Any]:
    t0 = benchmark_start(device)
    mu_img = train_bundle.image_feats.mean(dim=0)
    mu_txt = train_bundle.paired_text.mean(dim=0)
    stats = benchmark_end(device, t0)
    test_img = fs.l2norm(test_bundle.image_feats - mu_img)
    test_txt = fs.l2norm(test_bundle.text_feats - mu_txt)
    complexity = {
        "big_o": "O((N_img + N_txt) * D)",
        "estimated_vector_ops": float((train_bundle.image_feats.size(0) + train_bundle.paired_text.size(0)) * train_bundle.image_feats.size(1)),
        "n_images": int(train_bundle.image_feats.size(0)),
        "n_texts": int(train_bundle.paired_text.size(0)),
        "feat_dim": int(train_bundle.image_feats.size(1)),
    }
    return {"image_feats": test_img, "text_feats": test_txt, "train_stats": stats, "complexity": complexity}


def run_fsalign_record(
    label: str,
    train_bundle: fs.RetrievalFeatureBundle,
    test_bundle: fs.RetrievalFeatureBundle,
    args: SimpleNamespace,
    device: str,
    save_lora_path: Optional[Path] = None,
    overhead_mode: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    if save_lora_path is not None:
        ensure_dir(save_lora_path.parent)
        torch.save(lora_state, save_lora_path)
    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    test_img = fs.apply_lora_state(test_bundle.image_feats.to(device), layer_img, args.lora_mix)
    test_txt = fs.apply_lora_state(test_bundle.text_feats.to(device), layer_txt, args.lora_mix)
    gap, i2t, t2i, extra = fs.retrieval_metrics_from_embeddings(
        test_bundle,
        device=device,
        nas_k_val=100,
        nas_max_items=min(5000, test_bundle.image_feats.size(0)),
        intra_samples=20000,
        image_feats=test_img,
        text_feats=test_txt,
    )
    record = {
        "label": label,
        "method": "FSAlign",
        "config": vars(args).copy(),
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "train_stats": history.get("train_stats", {}),
        "history": history,
        "overhead_mode": overhead_mode,
    }
    return record, lora_state


def make_baseline_record(
    label: str,
    method: str,
    bundle: fs.RetrievalFeatureBundle,
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    stats: Dict[str, Any],
    complexity: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    gap, i2t, t2i, extra = fs.retrieval_metrics_from_embeddings(
        bundle,
        device=device,
        nas_k_val=100,
        nas_max_items=min(5000, bundle.image_feats.size(0)),
        intra_samples=20000,
        image_feats=image_feats,
        text_feats=text_feats,
    )
    return {
        "label": label,
        "method": method,
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "train_stats": stats,
        "complexity": complexity,
    }


def retrieval_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "label": record["label"],
        "method": record["method"],
        "I2T_R1": record["i2t"]["R@1"],
        "I2T_R5": record["i2t"]["R@5"],
        "I2T_R10": record["i2t"]["R@10"],
        "T2I_R1": record["t2i"]["R@1"],
        "T2I_R5": record["t2i"]["R@5"],
        "T2I_R10": record["t2i"]["R@10"],
        "centroid_distance": record["gap"]["centroid_distance"],
        "CD": record["gap"]["centroid_distance"],
        "relative_modality_gap": record["gap"]["relative_modality_gap"],
        "NAS@100": record["gap"].get("NAS@100", 0.0),
        "CMAS": record["gap"]["CMAS"],
        "avg_words_per_text": record["extra"].get("avg_words_per_text", 0.0),
    }
    stats = record.get("train_stats", {})
    config = record.get("config", {})
    complexity = record.get("complexity", stats.get("complexity", {}))
    row["train_time_sec"] = stats.get("train_time_sec", stats.get("fit_time_sec", 0.0))
    row["peak_memory_mb"] = stats.get("peak_cuda_memory_allocated_mb", 0.0)
    row["epochs"] = complexity.get("epochs", config.get("train_epochs", ""))
    row["time_per_epoch_sec"] = (row["train_time_sec"] / max(int(row["epochs"]), 1)) if row["epochs"] != "" else ""
    row["overhead_mode"] = record.get("overhead_mode", "")
    row["effective_train_size"] = complexity.get("n_train", "")
    row["anchor_count"] = complexity.get("anchor_count", "")
    row["spectral_count"] = complexity.get("spectral_count", "")
    row["align_count"] = complexity.get("align_count", "")
    row["pairwise_row_chunk"] = config.get("pairwise_row_chunk", stats.get("pairwise_row_chunk", ""))
    row["pairwise_col_chunk"] = config.get("pairwise_col_chunk", stats.get("pairwise_col_chunk", ""))
    return row


def overhead_snapshot(record: Dict[str, Any]) -> Dict[str, Any]:
    stats = record.get("train_stats", {})
    return {
        "label": record.get("label", ""),
        "overhead_mode": record.get("overhead_mode", ""),
        "train_time_sec": stats.get("train_time_sec", 0.0),
        "peak_memory_mb": stats.get("peak_cuda_memory_allocated_mb", 0.0),
        "epochs": stats.get("complexity", {}).get("epochs", record.get("config", {}).get("train_epochs", 0)),
        "time_per_epoch_sec": (stats.get("train_time_sec", 0.0) / max(int(stats.get("complexity", {}).get("epochs", record.get("config", {}).get("train_epochs", 1))), 1)),
        "complexity": stats.get("complexity", {}),
        "pairwise_row_chunk": stats.get("pairwise_row_chunk", 0),
        "pairwise_col_chunk": stats.get("pairwise_col_chunk", 0),
        "config": record.get("config", {}),
    }


def classification_metrics(image_feats: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits = image_feats @ weights.t()
    top1 = torch.argmax(logits, dim=1)
    correct1 = float((top1 == labels).sum().item())
    top5 = torch.topk(logits, k=5, dim=1).indices
    correct5 = 0.0
    for idx in range(labels.size(0)):
        if int(labels[idx].item()) in top5[idx].tolist():
            correct5 += 1.0
    total = float(labels.size(0))
    return {"top1": 100.0 * correct1 / total, "top5": 100.0 * correct5 / total, "n": total}


def image_only_prototype_metrics(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
) -> Dict[str, float]:
    num_classes = int(torch.max(train_labels).item()) + 1
    feat_dim = int(train_feats.size(1))
    proto_sums = torch.zeros(num_classes, feat_dim, device=train_feats.device, dtype=train_feats.dtype)
    proto_sums.index_add_(0, train_labels, train_feats)
    counts = torch.bincount(train_labels, minlength=num_classes).to(train_feats.device).clamp_min(1)
    prototypes = fs.l2norm(proto_sums / counts.unsqueeze(1))
    metrics = classification_metrics(test_feats, prototypes, test_labels)
    metrics["n_train"] = float(train_feats.size(0))
    metrics["n_classes"] = float(num_classes)
    return metrics


def run_unimodal_quality(
    lora_state: Dict[str, Any],
    retrieval_test_bundle: fs.RetrievalFeatureBundle,
    cifar_train_x: torch.Tensor,
    cifar_train_labels: torch.Tensor,
    cifar_test_x: torch.Tensor,
    cifar_test_labels: torch.Tensor,
    device: str,
    lora_mix: float,
) -> Dict[str, Any]:
    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    base_cls = image_only_prototype_metrics(cifar_train_x, cifar_train_labels, cifar_test_x, cifar_test_labels)
    fs_cifar_train_x = fs.apply_lora_state(cifar_train_x, layer_img, lora_mix)
    fs_cifar_test_x = fs.apply_lora_state(cifar_test_x, layer_img, lora_mix)
    fs_cls = image_only_prototype_metrics(
        fs_cifar_train_x,
        cifar_train_labels,
        fs_cifar_test_x,
        cifar_test_labels,
    )
    base_img_ret = fs.grouped_retrieval_metrics(cifar_test_x, cifar_test_labels, device)
    fs_img_ret = fs.grouped_retrieval_metrics(fs_cifar_test_x, cifar_test_labels, device)
    base_txt_ret = fs.grouped_retrieval_metrics(retrieval_test_bundle.text_feats, retrieval_test_bundle.cap2img, device)
    fs_txt_ret = fs.grouped_retrieval_metrics(
        fs.apply_lora_state(retrieval_test_bundle.text_feats, layer_txt, lora_mix),
        retrieval_test_bundle.cap2img,
        device,
    )
    return {
        "classification": {
            "protocol": "image_only_class_prototypes",
            "baseline": base_cls,
            "fsalign": fs_cls,
        },
        "image_to_image": {"baseline": base_img_ret, "fsalign": fs_img_ret},
        "text_to_text": {"baseline": base_txt_ret, "fsalign": fs_txt_ret},
    }


def write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = ["# Rebuttal Supplementary Experiments", ""]
    lines.append(f"Model: {payload['model_name']}")
    lines.append(f"Device: {payload['device']}")
    lines.append("")

    if "q1" in payload:
        q1 = payload["q1"]
        practical = q1.get("practical_recipe")
        true_full = q1.get("true_full_cost")
        lines.append(f"## Q1 {QUESTION_TITLES['q1']}")
        lines.append("FSAlign complexity is summarized as O(A N D + S M^2 D + L^2 D) per epoch.")
        if practical is not None:
            lines.append(
                f"Practical recipe overhead: {practical['train_time_sec']:.2f}s over {practical['epochs']} epochs "
                f"({practical['time_per_epoch_sec']:.2f}s/epoch) and {practical['peak_memory_mb']:.2f} MB."
            )
        if true_full is not None:
            lines.append(
                f"True full-cost overhead: {true_full['train_time_sec']:.2f}s over {true_full['epochs']} epochs "
                f"({true_full['time_per_epoch_sec']:.2f}s/epoch) and {true_full['peak_memory_mb']:.2f} MB."
            )
        lines.append("")

    if "q2_rows" in payload:
        rows = payload["q2_rows"]
        lines.append(f"## Q2 {QUESTION_TITLES['q2']}")
        practical = next((row for row in rows if row.get("overhead_mode") == "practical_recipe"), None)
        true_full = next((row for row in rows if row.get("overhead_mode") == "true_full_cost"), None)
        baselines = [row for row in rows if row["method"] in {"IOT", "GR-CLIP"}]
        if practical is not None:
            lines.append(
                f"FSAlign practical recipe: {practical['train_time_sec']:.2f}s over {practical['epochs']} epochs "
                f"({practical['time_per_epoch_sec']:.2f}s/epoch), {practical['peak_memory_mb']:.2f} MB."
            )
        if true_full is not None:
            lines.append(
                f"FSAlign true full cost: {true_full['train_time_sec']:.2f}s over {true_full['epochs']} epochs "
                f"({true_full['time_per_epoch_sec']:.2f}s/epoch), {true_full['peak_memory_mb']:.2f} MB."
            )
        if baselines:
            fastest = min(baselines, key=lambda row: row["train_time_sec"])
            lines.append(f"Fastest closed-form baseline: {fastest['method']} ({fastest['train_time_sec']:.2f}s).")
        lines.append("")

    if "q3_rows" in payload:
        rows = payload["q3_rows"]
        best = max(rows, key=lambda row: row["I2T_R1"])
        worst = min(rows, key=lambda row: row["I2T_R1"])
        lines.append(f"## Q3 {QUESTION_TITLES['q3']}")
        lines.append(
            f"Best I2T R@1: {best['I2T_R1']:.2f} at structure batch {best['structure_batch_size']}. "
            f"Worst I2T R@1: {worst['I2T_R1']:.2f} at structure batch {worst['structure_batch_size']}."
        )
        lines.append("")

    if "q4_rows" in payload:
        short_row, para_row = payload["q4_rows"]
        lines.append(f"## Q4 {QUESTION_TITLES['q4']}")
        lines.append(
            f"Average words per text changed from {short_row['avg_words_per_text']:.2f} to {para_row['avg_words_per_text']:.2f}. "
            f"I2T R@1 changed from {short_row['I2T_R1']:.2f} to {para_row['I2T_R1']:.2f}."
        )
        lines.append("")

    if "q5_rows" in payload:
        best = max(payload["q5_rows"], key=lambda row: row["I2T_R1"])
        lines.append(f"## Q5 {QUESTION_TITLES['q5']}")
        lines.append(f"Best ablation by I2T R@1: {best['label']} at {best['I2T_R1']:.2f}.")
        lines.append("")

    if "q6" in payload:
        q6 = payload["q6"]
        lines.append(f"## Q6 {QUESTION_TITLES['q6']}")
        lines.append(
            f"CIFAR100 image-only top-1 baseline {q6['classification']['baseline']['top1']:.2f}, "
            f"FSAlign {q6['classification']['fsalign']['top1']:.2f}. "
            f"Text-to-text R@1 baseline {q6['text_to_text']['baseline']['R@1']:.2f}, "
            f"FSAlign {q6['text_to_text']['fsalign']['R@1']:.2f}."
        )
        lines.append("")

    if "q7_rows" in payload:
        lines.append(f"## Q7 {QUESTION_TITLES['q7']}")
        lines.append("Shared, learned-offset, and separate-dimensions settings, plus retrieval and zero-shot metrics, are saved in tables/q7_dimension_relaxation.csv.")
        lines.append("")

    if "q8_rows" in payload:
        zero = payload["q8_rows"][0]
        worst = min(payload["q8_rows"], key=lambda row: row["I2T_R1"])
        lines.append(f"## Q8 {QUESTION_TITLES['q8']}")
        lines.append(
            f"Clean reference I2T R@1: {zero['I2T_R1']:.2f}. "
            f"Worst noisy setting: {worst['I2T_R1']:.2f} at noise rate {worst['noise_pair_rate']:.2f}."
        )
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-key", type=str, choices=["clip", "openclip", "siglip"], default="clip")
    parser.add_argument("--model-size", type=int, choices=[16, 32], default=32)
    parser.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument("--max-cifar-items", type=int, default=0)
    parser.add_argument("--questions", type=str, default="q1,q2,q3,q4,q5,q6,q7,q8")
    parser.add_argument("--train-epochs", type=int, default=30)
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
    parser.add_argument("--paragraph-sentences", type=int, default=3)
    parser.add_argument("--noise-mix", type=float, default=0.5)
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
    parser.add_argument("--q3-structure-batches", type=str, default="64,128,256,512,1024,0")
    parser.add_argument("--q8-noise-rates", type=str, default="0.0,0.1,0.25,0.5")
    parser.add_argument("--q7-fixed-df-offset", type=float, default=0.1)
    parser.add_argument("--q7-fixed-ds-offset", type=float, default=0.1)
    parser.add_argument("--q7-fixed-dw-offset", type=float, default=None)
    parser.add_argument("--q7-lambda-dim-offset", "--q7-lambda-delta", dest="q7_lambda_dim_offset", type=float, default=0.01)
    parser.add_argument("--true-overhead", action="store_true")
    parser.add_argument("--true-overhead-epochs", type=int, default=0)
    parser.add_argument("--true-anchor-batch", type=int, default=0)
    parser.add_argument("--true-pairwise-row-chunk", type=int, default=1024)
    parser.add_argument("--true-pairwise-col-chunk", type=int, default=1024)
    parser.add_argument("--true-pairwise-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fs.seed_all(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    table_dir = out_dir / "tables"
    cache_dir = out_dir / "cache"
    ckpt_dir = out_dir / "artifacts"
    for folder in [out_dir, raw_dir, table_dir, cache_dir, ckpt_dir]:
        ensure_dir(folder)

    selected = {item.strip() for item in args.questions.split(",") if item.strip()}
    max_train = None if args.max_train_images <= 0 else args.max_train_images
    max_test = None if args.max_test_images <= 0 else args.max_test_images
    max_cifar = None if args.max_cifar_items <= 0 else args.max_cifar_items

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
    short_train = get_retrieval_bundle(
        model,
        train_ds,
        device,
        args.batch_size,
        args.num_workers,
        max_train,
        "short",
        args.paragraph_sentences,
        cache_dir / f"flickr_train_short_{cache_tag}_{max_train or 'all'}.pt",
    )
    short_test = get_retrieval_bundle(
        model,
        test_ds,
        device,
        args.batch_size,
        args.num_workers,
        max_test,
        "short",
        args.paragraph_sentences,
        cache_dir / f"flickr_test_short_{cache_tag}_{max_test or 'all'}.pt",
    )

    base_args = base_fsalign_args(device, args)
    payload: Dict[str, Any] = {"model_name": model_name, "device": device}

    full_record = None
    full_state = None
    if selected & {"q1", "q2", "q4", "q5", "q6", "q7", "q8"}:
        full_record, full_state = run_fsalign_record(
            "fsalign_full",
            short_train,
            short_test,
            base_args,
            device,
            save_lora_path=ckpt_dir / "fsalign_full_lora.pt",
            overhead_mode="practical_recipe",
        )
        save_json(raw_dir / "fsalign_full.json", full_record)

    true_overhead_record = None
    if args.true_overhead and selected & {"q1", "q2"}:
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        true_args = build_true_overhead_args(base_args, short_train, args)
        true_overhead_record, _ = run_fsalign_record(
            "fsalign_true_full_cost",
            short_train,
            short_test,
            true_args,
            device,
            overhead_mode="true_full_cost",
        )
        save_json(raw_dir / "fsalign_true_full_cost.json", true_overhead_record)

    if "q1" in selected and full_record is not None:
        payload["q1"] = {"practical_recipe": overhead_snapshot(full_record)}
        if true_overhead_record is not None:
            payload["q1"]["true_full_cost"] = overhead_snapshot(true_overhead_record)
            save_json(raw_dir / "q1_true_full_cost.json", payload["q1"]["true_full_cost"])
        save_json(raw_dir / "q1_complexity.json", payload["q1"])

    if "q2" in selected and full_record is not None:
        iot = iot_adapt(short_train, short_test, device)
        iot_record = make_baseline_record("iot", "IOT", short_test, iot["image_feats"], iot["text_feats"], iot["train_stats"], iot["complexity"], device)
        gr = gr_adapt(short_train, short_test, device)
        gr_record = make_baseline_record("gr_clip", "GR-CLIP", short_test, gr["image_feats"], gr["text_feats"], gr["train_stats"], gr["complexity"], device)
        save_json(raw_dir / "q2_iot.json", iot_record)
        save_json(raw_dir / "q2_gr_clip.json", gr_record)
        rows = [retrieval_row(full_record)]
        if true_overhead_record is not None:
            rows.append(retrieval_row(true_overhead_record))
        rows.extend([retrieval_row(iot_record), retrieval_row(gr_record)])
        for row in rows:
            if row["method"] == "FSAlign":
                row["complexity_big_o"] = "O(A N D + S M^2 D + L^2 D) per epoch"
            elif row["method"] == "IOT":
                row["complexity_big_o"] = iot_record["complexity"]["big_o"]
            elif row["method"] == "GR-CLIP":
                row["complexity_big_o"] = gr_record["complexity"]["big_o"]
        save_csv(table_dir / "q2_method_overhead.csv", rows)
        payload["q2_rows"] = rows

    if "q3" in selected:
        q3_rows: List[Dict[str, Any]] = []
        for batch_size in parse_int_list(args.q3_structure_batches):
            run_args = apply_overrides(base_args, structure_batch_size=batch_size)
            label = f"q3_batch_{batch_size or 'full'}"
            record, _ = run_fsalign_record(label, short_train, short_test, run_args, device)
            save_json(raw_dir / f"{label}.json", record)
            row = retrieval_row(record)
            row["structure_batch_size"] = batch_size if batch_size > 0 else int(short_train.image_feats.size(0))
            q3_rows.append(row)
        save_csv(table_dir / "q3_batch_sensitivity.csv", q3_rows)
        payload["q3_rows"] = q3_rows

    if "q4" in selected and full_record is not None:
        para_train = get_retrieval_bundle(
            model,
            train_ds,
            device,
            args.batch_size,
            args.num_workers,
            max_train,
            "paragraph",
            args.paragraph_sentences,
            cache_dir / f"flickr_train_paragraph_{cache_tag}_{max_train or 'all'}.pt",
        )
        para_test = get_retrieval_bundle(
            model,
            test_ds,
            device,
            args.batch_size,
            args.num_workers,
            max_test,
            "paragraph",
            args.paragraph_sentences,
            cache_dir / f"flickr_test_paragraph_{cache_tag}_{max_test or 'all'}.pt",
        )
        para_args = apply_overrides(base_args, text_variant="paragraph", paragraph_sentences=args.paragraph_sentences)
        para_record, _ = run_fsalign_record("q4_paragraph", para_train, para_test, para_args, device)
        save_json(raw_dir / "q4_paragraph.json", para_record)
        rows = [retrieval_row(full_record), retrieval_row(para_record)]
        rows[0]["label"] = "short_text"
        rows[1]["label"] = "paragraph_text"
        save_csv(table_dir / "q4_long_text.csv", rows)
        payload["q4_rows"] = rows

    if "q5" in selected:
        ablations = [
            ("full", {}),
            ("no_fractal_regularization", {"lambda_dbl": 0.0}),
            ("no_spectral_descriptor", {"lambda_spec": 0.0}),
            ("no_zeta_matching", {"lambda_match": 0.0}),
            ("no_matching_loss", {"lambda_align": 0.0}),
        ]
        rows: List[Dict[str, Any]] = []
        for name, overrides in ablations:
            run_args = apply_overrides(base_args, **overrides)
            record, _ = run_fsalign_record(f"q5_{name}", short_train, short_test, run_args, device)
            save_json(raw_dir / f"q5_{name}.json", record)
            row = retrieval_row(record)
            row["label"] = name
            row.update(overrides)
            rows.append(row)
        save_csv(table_dir / "q5_ablations.csv", rows)
        payload["q5_rows"] = rows

    if "q6" in selected and full_state is not None:
        cifar_train_x, cifar_train_labels = get_cifar_image_embeddings(
            model,
            args.data_root,
            device,
            args.batch_size,
            args.num_workers,
            True,
            max_cifar,
            cache_dir / f"cifar100_train_images_{cache_tag}_{max_cifar or 'all'}.pt",
        )
        cifar_test_x, cifar_test_labels = get_cifar_image_embeddings(
            model,
            args.data_root,
            device,
            args.batch_size,
            args.num_workers,
            False,
            max_cifar,
            cache_dir / f"cifar100_test_images_{cache_tag}_{max_cifar or 'all'}.pt",
        )
        q6 = run_unimodal_quality(
            full_state,
            short_test,
            cifar_train_x,
            cifar_train_labels,
            cifar_test_x,
            cifar_test_labels,
            device,
            args.lora_mix,
        )
        save_json(raw_dir / "q6_unimodal_quality.json", q6)
        payload["q6"] = q6
        save_csv(
            table_dir / "q6_unimodal_quality.csv",
            [
                {"metric": "cifar100_image_only_top1", "baseline": q6["classification"]["baseline"]["top1"], "fsalign": q6["classification"]["fsalign"]["top1"]},
                {"metric": "cifar100_image_only_top5", "baseline": q6["classification"]["baseline"]["top5"], "fsalign": q6["classification"]["fsalign"]["top5"]},
                {"metric": "image_to_image_R@1", "baseline": q6["image_to_image"]["baseline"]["R@1"], "fsalign": q6["image_to_image"]["fsalign"]["R@1"]},
                {"metric": "text_to_text_R@1", "baseline": q6["text_to_text"]["baseline"]["R@1"], "fsalign": q6["text_to_text"]["fsalign"]["R@1"]},
            ],
        )

    if "q7" in selected:
        q7_ds_offset = args.q7_fixed_ds_offset if args.q7_fixed_ds_offset is not None else (args.q7_fixed_dw_offset or 0.0)
        cifar_x, cifar_labels, cifar_weights = get_cifar_embeddings(
            model,
            args.data_root,
            device,
            args.batch_size,
            args.num_workers,
            max_cifar,
            cache_dir / f"cifar100_{cache_tag}_{max_cifar or 'all'}.pt",
        )
        dtd_ds = dtd_test(args.data_root)
        dtd_x, dtd_labels, dtd_weights = get_zeroshot_embeddings(
            model,
            dtd_ds,
            dtd_ds.classes,
            fs.DTD_TEMPLATES,
            device,
            args.batch_size,
            args.num_workers,
            None,
            cache_dir / f"dtd_{cache_tag}.pt",
        )
        tiny_ds = tinyimagenet_val(args.data_root)
        tiny_x, tiny_labels, tiny_weights = get_zeroshot_embeddings(
            model,
            tiny_ds,
            tiny_ds.classnames,
            fs.CIFAR100_TEMPLATES,
            device,
            args.batch_size,
            args.num_workers,
            None,
            cache_dir / f"tiny_imagenet_200_{cache_tag}.pt",
        )
        zeroshot_suites = {
            "cifar100": {"image_feats": cifar_x, "labels": cifar_labels, "weights": cifar_weights},
            "dtd": {"image_feats": dtd_x, "labels": dtd_labels, "weights": dtd_weights},
            "tiny_imagenet_200": {"image_feats": tiny_x, "labels": tiny_labels, "weights": tiny_weights},
        }
        rows: List[Dict[str, Any]] = []
        if full_record is not None and full_state is not None:
            shared_record = dict(full_record)
            shared_record["label"] = "q7_shared"
            shared_record["zeroshot"] = evaluate_zeroshot_suite(full_state, zeroshot_suites, device, args.lora_mix)
            save_json(raw_dir / "q7_shared.json", shared_record)
            rows.append(build_q7_row(shared_record))
        settings = [
            ("learned_offset", {"dimension_mode": "learned_offset", "dimension_offset_df": args.q7_fixed_df_offset, "dimension_offset_ds": q7_ds_offset, "lambda_dim_offset": args.q7_lambda_dim_offset}),
            ("separate", {"dimension_mode": "separate", "dimension_offset_df": args.q7_fixed_df_offset, "dimension_offset_ds": q7_ds_offset}),
        ]
        for name, overrides in settings:
            fs.seed_all(args.seed)
            run_args = apply_overrides(base_args, **overrides)
            record, lora_state = run_fsalign_record(f"q7_{name}", short_train, short_test, run_args, device)
            record["zeroshot"] = evaluate_zeroshot_suite(lora_state, zeroshot_suites, device, args.lora_mix)
            save_json(raw_dir / f"q7_{name}.json", record)
            rows.append(build_q7_row(record))
        save_csv(table_dir / "q7_dimension_relaxation.csv", rows)
        payload["q7_rows"] = rows

    if "q8" in selected:
        rows: List[Dict[str, Any]] = []
        for rate in parse_float_list(args.q8_noise_rates):
            run_args = apply_overrides(base_args, noise_pair_rate=rate, noise_mix=args.noise_mix)
            record, lora_state = run_fsalign_record(f"q8_noise_{rate}", short_train, short_test, run_args, device)
            save_json(raw_dir / f"q8_noise_{rate}.json", record)
            _, layer_txt = fs.build_lora_layers(lora_state, device)
            transformed_text = fs.apply_lora_state(short_test.text_feats, layer_txt, args.lora_mix)
            txt_metrics = fs.grouped_retrieval_metrics(transformed_text, short_test.cap2img, device)
            row = retrieval_row(record)
            row["noise_pair_rate"] = rate
            row["text_to_text_R@1"] = txt_metrics["R@1"]
            row["text_to_text_R@5"] = txt_metrics["R@5"]
            row["text_to_text_R@10"] = txt_metrics["R@10"]
            rows.append(row)
        save_csv(table_dir / "q8_noisy_pairs.csv", rows)
        payload["q8_rows"] = rows

    save_json(out_dir / "manifest.json", payload)
    write_summary(out_dir, payload)
    print(f"[Done] Rebuttal outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
