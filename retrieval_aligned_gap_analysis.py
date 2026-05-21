#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import our_code_final as fs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def coco_train_test(data_root: str):
    kjson = fs.ensure_karpathy_json(data_root, "coco")
    roots = [
        str(Path(data_root) / "mscoco2014" / "train2014"),
        str(Path(data_root) / "mscoco2014" / "val2014"),
        str(Path(data_root) / "coco2014" / "train2014"),
        str(Path(data_root) / "coco2014" / "val2014"),
        str(Path(data_root) / "coco" / "train2014"),
        str(Path(data_root) / "coco" / "val2014"),
    ]
    train_ds = fs.KarpathyRetrievalDataset(str(kjson), roots, split="train", max_images=None)
    test_ds = fs.KarpathyRetrievalDataset(str(kjson), roots, split="test", max_images=None)
    return train_ds, test_ds


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
        text_variant=str(cli_args.text_variant),
        paragraph_sentences=int(cli_args.paragraph_sentences),
        structure_batch_size=int(cli_args.structure_batch_size),
        noise_pair_rate=float(cli_args.noise_pair_rate),
        noise_mix=float(cli_args.noise_mix),
        dimension_mode="shared",
        dimension_offset_df=0.0,
        dimension_offset_ds=0.0,
        dimension_offset_dw=None,
        lambda_dim_offset=0.0,
        lambda_neighbor_compete=float(cli_args.lambda_neighbor_compete),
        neighbor_compete_k=int(cli_args.neighbor_compete_k),
        neighbor_compete_samples=int(cli_args.neighbor_compete_samples),
        neighbor_compete_margin=float(cli_args.neighbor_compete_margin),
        neighbor_compete_chunk=int(cli_args.neighbor_compete_chunk),
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


def analysis_to_jsonable(analysis: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"metadata": analysis["metadata"]}
    for direction in ["i2t", "t2i"]:
        payload[direction] = {
            "aggregated": analysis[direction]["aggregated"],
            "per_sample": {},
        }
        for k, per_sample in analysis[direction]["per_sample"].items():
            payload[direction]["per_sample"][str(k)] = {
                key: value.tolist() if torch.is_tensor(value) else value
                for key, value in per_sample.items()
            }
    return payload


def per_sample_rows(
    dataset_name: str,
    method_name: str,
    direction: str,
    payloads: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for requested_k in sorted((int(k) for k in payloads.keys())):
        payload = payloads[str(requested_k)]
        sample_index = payload["sample_index"]
        gt_score = payload["gt_score"]
        hardest_neighbor_index = payload["hardest_neighbor_index"]
        hardest_neighbor_score = payload["hardest_neighbor_score"]
        oor = payload["oor"]
        hit = payload["hit"]
        margin = payload["margin"]
        for idx, gt, hard_idx, hard_score, oor_val, hit_val, margin_val in zip(
            sample_index,
            gt_score,
            hardest_neighbor_index,
            hardest_neighbor_score,
            oor,
            hit,
            margin,
        ):
            rows.append({
                "dataset": dataset_name,
                "method": method_name,
                "direction": direction,
                "k": int(payload["k"]),
                "effective_k": int(payload["effective_k"]),
                "sample_index": int(idx),
                "gt_score": float(gt),
                "hardest_neighbor_index": int(hard_idx),
                "hardest_neighbor_score": float(hard_score),
                "oor": float(oor_val),
                "hit": float(hit_val),
                "margin": float(margin_val),
            })
    return rows


def aggregated_rows(
    dataset_name: str,
    method_name: str,
    model_name: str,
    retrieval_record: Dict[str, Any],
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for direction in ["i2t", "t2i"]:
        for agg in analysis[direction]["aggregated"]:
            rows.append({
                "dataset": dataset_name,
                "method": method_name,
                "model": model_name,
                "direction": direction,
                "k": int(agg["k"]),
                "effective_k": int(agg["effective_k"]),
                "oor": float(agg["oor"]),
                "hit": float(agg["hit"]),
                "margin": float(agg["margin"]),
                "I2T_R1": float(retrieval_record["i2t"]["R@1"]),
                "I2T_R5": float(retrieval_record["i2t"]["R@5"]),
                "I2T_R10": float(retrieval_record["i2t"]["R@10"]),
                "T2I_R1": float(retrieval_record["t2i"]["R@1"]),
                "T2I_R5": float(retrieval_record["t2i"]["R@5"]),
                "T2I_R10": float(retrieval_record["t2i"]["R@10"]),
                "centroid_distance": float(retrieval_record["gap"]["centroid_distance"]),
                "relative_modality_gap": float(retrieval_record["gap"]["relative_modality_gap"]),
                "NAS": float(retrieval_record["gap"].get("NAS@100", retrieval_record["gap"].get("NAS@10", 0.0))),
                "CMAS": float(retrieval_record["gap"]["CMAS"]),
                "num_pairs": int(analysis["metadata"]["num_pairs"]),
            })
    return rows


def aggregated_lookup(record: Dict[str, Any], direction: str) -> Dict[int, Dict[str, float]]:
    return {int(item["k"]): item for item in record["analysis"][direction]["aggregated"]}


def plot_direction(
    records: Dict[Tuple[str, str], Dict[str, Any]],
    datasets: List[str],
    ks: List[int],
    direction: str,
    out_path: Path,
) -> None:
    metric_specs = [("oor", "OOR"), ("hit", "Hit"), ("margin", "Margin")]
    dataset_titles = {"flickr30k": "Flickr30k", "mscoco": "MSCOCO"}
    method_specs = [("baseline", "Baseline", "#6b7280"), ("ours", "Ours", "#c2410c")]

    fig, axes = plt.subplots(len(datasets), len(metric_specs), figsize=(13, 4.2 * len(datasets)), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, (metric_key, metric_title) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            for method_name, label, color in method_specs:
                record = records[(dataset_name, method_name)]
                lookup = aggregated_lookup(record, direction)
                y = [float(lookup[k][metric_key]) for k in ks if k in lookup]
                x = [k for k in ks if k in lookup]
                ax.plot(x, y, marker="o", linewidth=2.0, markersize=5, color=color, label=label)
            if row_idx == 0:
                ax.set_title(metric_title)
            if col_idx == 0:
                ax.set_ylabel(dataset_titles.get(dataset_name, dataset_name))
            ax.set_xlabel("k")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(method_specs), frameon=False)
    fig.suptitle(f"Retrieval-aligned modality-gap analysis: {direction.upper()}")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def analyze_method(
    dataset_name: str,
    method_name: str,
    model_name: str,
    train_bundle: fs.RetrievalFeatureBundle,
    test_bundle: fs.RetrievalFeatureBundle,
    fsalign_args: SimpleNamespace,
    ks: List[int],
    device: str,
    cli_args,
) -> Dict[str, Any]:
    if method_name == "baseline":
        test_img = test_bundle.image_feats
        test_text = test_bundle.text_feats
        method_info: Dict[str, Any] = {"mode": "baseline"}
    elif method_name == "ours":
        radii = fs.logspace_scales(fsalign_args.radii_min, fsalign_args.radii_max, fsalign_args.radii_count)
        rho_list = [float(x) for x in fsalign_args.rho_list.split(",") if x.strip()]
        diffusion_scales = fs.logspace_scales(
            fsalign_args.diffusion_min,
            fsalign_args.diffusion_max,
            fsalign_args.diffusion_count,
        )
        lora_state, history = fs.train_lora_postprocess(
            train_bundle.image_feats,
            train_bundle.paired_text,
            radii,
            rho_list,
            diffusion_scales,
            fsalign_args,
            caption_pool=(train_bundle.text_feats, train_bundle.cap_indices),
        )
        layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
        test_img = fs.apply_lora_state(test_bundle.image_feats, layer_img, fsalign_args.lora_mix)
        test_text = fs.apply_lora_state(test_bundle.text_feats, layer_txt, fsalign_args.lora_mix)
        method_info = {
            "mode": "ours",
            "train_stats": history.get("train_stats", {}),
            "lora_mix": float(fsalign_args.lora_mix),
        }
    else:
        raise ValueError(f"Unsupported method_name: {method_name}")

    gap, i2t, t2i, extra = fs.retrieval_metrics_from_embeddings(
        test_bundle,
        device=device,
        nas_k_val=int(cli_args.nas_k),
        nas_max_items=int(cli_args.nas_max_items),
        intra_samples=int(cli_args.intra_samples),
        image_feats=test_img,
        text_feats=test_text,
    )
    paired_text = test_text[test_bundle.pair_map]
    analysis = fs.retrieval_aligned_neighbor_outrank_analysis(
        test_img,
        paired_text,
        ks,
        device=device,
        neighbor_chunk=int(cli_args.neighbor_chunk),
        score_chunk=int(cli_args.analysis_chunk),
    )
    return {
        "dataset": dataset_name,
        "method": method_name,
        "model": model_name,
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "analysis": analysis_to_jsonable(analysis),
        "method_info": method_info,
    }


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
    parser.add_argument("--datasets", type=str, default="flickr30k,mscoco")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument("--text-variant", type=str, default="short", choices=["short", "paragraph"])
    parser.add_argument("--paragraph-sentences", type=int, default=3)
    parser.add_argument("--ks", type=str, default="1,5,10,20,50,100")
    parser.add_argument("--neighbor-chunk", type=int, default=1024)
    parser.add_argument("--analysis-chunk", type=int, default=1024)
    parser.add_argument("--nas-k", type=int, default=10)
    parser.add_argument("--nas-max-items", type=int, default=5000)
    parser.add_argument("--intra-samples", type=int, default=20000)
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--train-anchors", type=int, default=512)
    parser.add_argument("--anchor-batch", type=int, default=128)
    parser.add_argument("--spectral-samples", type=int, default=512)
    parser.add_argument("--train-lr", type=float, default=5e-4)
    parser.add_argument("--lambda-dbl", type=float, default=0.0)
    parser.add_argument("--lambda-spec", type=float, default=0.0)
    parser.add_argument("--lambda-match", type=float, default=0.0)
    parser.add_argument("--lambda-align", type=float, default=0.0)
    parser.add_argument("--lambda-orth", type=float, default=0.0)
    parser.add_argument("--train-reg", type=float, default=0.01)
    parser.add_argument("--align-temp", type=float, default=0.07)
    parser.add_argument("--align-samples", type=int, default=0)
    parser.add_argument("--pairwise-row-chunk", type=int, default=0)
    parser.add_argument("--pairwise-col-chunk", type=int, default=0)
    parser.add_argument("--pairwise-checkpoint", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-mix", type=float, default=0.1)
    parser.add_argument("--multi-caption", action="store_true")
    parser.add_argument("--caption-agg", type=str, default="random", choices=["random", "mean"])
    parser.add_argument("--structure-batch-size", type=int, default=0)
    parser.add_argument("--noise-pair-rate", type=float, default=0.0)
    parser.add_argument("--noise-mix", type=float, default=0.5)
    parser.add_argument("--lambda-neighbor-compete", type=float, default=5.0)
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
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    cache_dir = out_dir / "cache"
    for path in [out_dir, raw_dir, table_dir, plot_dir, cache_dir]:
        ensure_dir(path)

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    ks = parse_int_list(args.ks)
    max_train = None if int(args.max_train_images) <= 0 else int(args.max_train_images)
    max_test = None if int(args.max_test_images) <= 0 else int(args.max_test_images)

    model_name, model = build_model(
        device,
        args.model_key,
        args.model_size,
        args.siglip_name,
        args.openclip_model,
        args.openclip_pretrained,
    )
    cache_tag = fs.safe_filename(model_name)
    fsalign_args = base_fsalign_args(device, args)

    dataset_builders = {
        "flickr30k": flickr_train_test,
        "mscoco": coco_train_test,
    }
    dataset_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    aggregated: List[Dict[str, Any]] = []
    per_sample_all: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        if dataset_name not in dataset_builders:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        train_ds, test_ds = dataset_builders[dataset_name](args.data_root)
        train_bundle = get_retrieval_bundle(
            model,
            train_ds,
            device,
            args.batch_size,
            args.num_workers,
            max_train,
            args.text_variant,
            args.paragraph_sentences,
            cache_dir / f"{dataset_name}_train_{args.text_variant}_{cache_tag}_{max_train or 'all'}.pt",
        )
        test_bundle = get_retrieval_bundle(
            model,
            test_ds,
            device,
            args.batch_size,
            args.num_workers,
            max_test,
            args.text_variant,
            args.paragraph_sentences,
            cache_dir / f"{dataset_name}_test_{args.text_variant}_{cache_tag}_{max_test or 'all'}.pt",
        )
        for method_name in ["baseline", "ours"]:
            print(f"[Analysis] dataset={dataset_name} method={method_name}")
            record = analyze_method(
                dataset_name,
                method_name,
                model_name,
                train_bundle,
                test_bundle,
                fsalign_args,
                ks,
                device,
                args,
            )
            dataset_records[(dataset_name, method_name)] = record
            aggregated.extend(aggregated_rows(dataset_name, method_name, model_name, record, record["analysis"]))
            method_rows: List[Dict[str, Any]] = []
            for direction in ["i2t", "t2i"]:
                per_sample_rows_part = per_sample_rows(
                    dataset_name,
                    method_name,
                    direction,
                    record["analysis"][direction]["per_sample"],
                )
                method_rows.extend(per_sample_rows_part)
                per_sample_all.extend(per_sample_rows_part)
            save_json(raw_dir / f"{dataset_name}_{method_name}_analysis.json", record)
            save_csv(raw_dir / f"{dataset_name}_{method_name}_per_sample.csv", method_rows)

    save_csv(table_dir / "retrieval_aligned_gap_summary.csv", aggregated)
    save_csv(table_dir / "retrieval_aligned_gap_per_sample.csv", per_sample_all)
    save_json(
        table_dir / "retrieval_aligned_gap_summary.json",
        {
            "model": model_name,
            "datasets": datasets,
            "ks": ks,
            "records": {
                f"{dataset}_{method}": {
                    "dataset": record["dataset"],
                    "method": record["method"],
                    "model": record["model"],
                    "gap": record["gap"],
                    "i2t": record["i2t"],
                    "t2i": record["t2i"],
                    "extra": record["extra"],
                    "analysis": {
                        "metadata": record["analysis"]["metadata"],
                        "i2t": {"aggregated": record["analysis"]["i2t"]["aggregated"]},
                        "t2i": {"aggregated": record["analysis"]["t2i"]["aggregated"]},
                    },
                    "method_info": record["method_info"],
                }
                for (dataset, method), record in dataset_records.items()
            },
        },
    )
    plot_direction(dataset_records, datasets, ks, "i2t", plot_dir / "retrieval_aligned_gap_i2t.png")
    plot_direction(dataset_records, datasets, ks, "t2i", plot_dir / "retrieval_aligned_gap_t2i.png")
    print(f"[Done] Retrieval-aligned gap outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
