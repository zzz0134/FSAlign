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

import I0T_full as iot_impl
import GR_CLIP as gr_impl


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
        "time_sec": float(time.time() - t0),
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


def make_model_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        models=args.model_key,
        siglip_name=args.siglip_name,
        openclip_model=args.openclip_model,
        openclip_pretrained=args.openclip_pretrained,
    )


def flickr_roots(data_root: str) -> List[str]:
    root = Path(data_root)
    return [
        str(root / "flickr30k" / "flickr30k-images"),
        str(root / "flickr30k" / "images"),
        str(root / "flickr30k"),
    ]


def count_captions(items: List[Tuple[str, List[str]]], limit: Optional[int]) -> int:
    use_items = items if limit is None else items[:limit]
    return int(sum(len(caps) for _, caps in use_items))


def load_fsalign_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("method") != "FSAlign":
            continue
        out.append(
            {
                "method": "FSAlign",
                "variant": row.get("overhead_mode", row.get("label", "")),
                "fit_scope": "cached_embedding_training_only",
                "fit_samples": row.get("effective_train_size", ""),
                "fit_texts": "",
                "I2T_R1": float(row["I2T_R1"]),
                "I2T_R5": float(row["I2T_R5"]),
                "I2T_R10": float(row["I2T_R10"]),
                "T2I_R1": float(row["T2I_R1"]),
                "T2I_R5": float(row["T2I_R5"]),
                "T2I_R10": float(row["T2I_R10"]),
                "fit_time_sec": float(row["train_time_sec"]),
                "fit_peak_memory_mb": float(row["peak_memory_mb"]),
                "eval_time_sec": "",
                "eval_peak_memory_mb": "",
                "total_time_sec": float(row["train_time_sec"]),
                "complexity_big_o": row["complexity_big_o"],
                "notes": "Reference FSAlign row from rebuttal_true_overhead results.",
            }
        )
    return out


def evaluate_iot(
    data_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
    nas_k: int,
    nas_max_items: int,
    intra_samples: int,
    model_args: SimpleNamespace,
    fit_max_flickr: int,
) -> Dict[str, Any]:
    model_name, base_model = iot_impl.make_models(model_args, device=device)[0]
    kjson = iot_impl.ensure_karpathy_json(data_root, "flickr30k")
    roots = flickr_roots(data_root)
    train_ds = iot_impl.KarpathyRetrievalDataset(str(kjson), roots, split="train", max_images=None)
    test_ds = iot_impl.KarpathyRetrievalDataset(str(kjson), roots, split="test", max_images=None)

    fit_limit = None if fit_max_flickr <= 0 else min(int(fit_max_flickr), len(train_ds))
    fit_images = len(train_ds) if fit_limit is None else fit_limit
    fit_captions = count_captions(train_ds.items, fit_limit)

    stdizer = iot_impl.I0TStandardizer(dim=base_model.dim, device=device)

    def fit_stage():
        stdizer.fit_retrieval_from_backbone(
            backbone=base_model,
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            max_images=fit_limit,
        )

    _, fit_stats = benchmark_stage(device, fit_stage)
    model_iot = iot_impl.I0TBackbone(base=base_model, stdizer=stdizer, device=device)

    def eval_stage():
        return iot_impl.retrieval_eval(
            model_iot,
            test_ds,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_images=None,
            nas_k_val=nas_k,
            nas_max_items=nas_max_items,
            intra_samples=intra_samples,
        )

    eval_out, eval_stats = benchmark_stage(device, eval_stage)
    gap, i2t, t2i, extra = eval_out

    return {
        "method": "IOT",
        "model_name": model_name,
        "fit_scope": "official_flickr_fit",
        "fit_images": fit_images,
        "fit_captions": fit_captions,
        "fit_time_sec": fit_stats["time_sec"],
        "fit_peak_memory_mb": fit_stats["peak_cuda_memory_allocated_mb"],
        "eval_time_sec": eval_stats["time_sec"],
        "eval_peak_memory_mb": eval_stats["peak_cuda_memory_allocated_mb"],
        "total_time_sec": fit_stats["time_sec"] + eval_stats["time_sec"],
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "complexity": {
            "big_o": "O(N_fit_img * Enc_img + N_fit_cap * Enc_txt + (N_fit_img + N_fit_cap) * D)",
            "fit_images": fit_images,
            "fit_captions": fit_captions,
            "feat_dim": int(base_model.dim),
        },
        "notes": "Full Flickr30k fit path from I0T_full.py with official default fit_max_flickr unless overridden.",
    }


def evaluate_gr_clip(
    data_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
    nas_k: int,
    nas_max_items: int,
    intra_samples: int,
    model_args: SimpleNamespace,
    calib_n: int,
    calib_batch: int,
) -> Dict[str, Any]:
    model_name, base_model = gr_impl.make_models(model_args, device=device)[0]
    kjson = gr_impl.ensure_karpathy_json(data_root, "flickr30k")
    roots = flickr_roots(data_root)
    train_ds = gr_impl.KarpathyRetrievalDataset(str(kjson), roots, split="train", max_images=None)
    test_ds = gr_impl.KarpathyRetrievalDataset(str(kjson), roots, split="test", max_images=None)

    calib_count = min(int(calib_n), len(train_ds)) if calib_n > 0 else len(train_ds)
    calibrator = gr_impl.GRCalibrator(dim=base_model.dim, device=device)

    def calib_stage():
        imgs, caps = gr_impl.sample_karpathy_calib_texts_and_images(train_ds, n=calib_count, seed=42)
        mu_img = gr_impl.compute_mean_image(base_model, imgs, device=device, batch_size=calib_batch)
        mu_txt = gr_impl.compute_mean_text(base_model, caps, device=device, batch_size=calib_batch)
        calibrator.mu_img = mu_img
        calibrator.mu_txt_q = mu_txt
        calibrator.mu_txt_d = mu_txt
        calibrator.ready = True

    _, calib_stats = benchmark_stage(device, calib_stage)
    model_gr = gr_impl.GRWrappedBackbone(base_model, calibrator)

    def eval_stage():
        return gr_impl.retrieval_eval_gr(
            model_gr,
            test_ds,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_images=None,
            nas_k_val=nas_k,
            nas_max_items=nas_max_items,
            intra_samples=intra_samples,
        )

    eval_out, eval_stats = benchmark_stage(device, eval_stage)
    gap, i2t, t2i, extra = eval_out

    return {
        "method": "GR-CLIP",
        "model_name": model_name,
        "fit_scope": "official_flickr_calibration",
        "fit_images": calib_count,
        "fit_captions": calib_count,
        "fit_time_sec": calib_stats["time_sec"],
        "fit_peak_memory_mb": calib_stats["peak_cuda_memory_allocated_mb"],
        "eval_time_sec": eval_stats["time_sec"],
        "eval_peak_memory_mb": eval_stats["peak_cuda_memory_allocated_mb"],
        "total_time_sec": calib_stats["time_sec"] + eval_stats["time_sec"],
        "gap": gap,
        "i2t": i2t,
        "t2i": t2i,
        "extra": extra,
        "complexity": {
            "big_o": "O(N_calib * Enc_img + N_calib * Enc_txt + N_calib * D)",
            "fit_images": calib_count,
            "fit_captions": calib_count,
            "feat_dim": int(base_model.dim),
        },
        "notes": "Full Flickr30k calibration path from GR_CLIP.py with official default calib_n unless overridden.",
    }


def comparison_row(record: Dict[str, Any]) -> Dict[str, Any]:
    if record["method"] == "FSAlign":
        return record
    return {
        "method": record["method"],
        "variant": "paper_faithful",
        "fit_scope": record["fit_scope"],
        "fit_samples": record["fit_images"],
        "fit_texts": record["fit_captions"],
        "I2T_R1": record["i2t"]["R@1"],
        "I2T_R5": record["i2t"]["R@5"],
        "I2T_R10": record["i2t"]["R@10"],
        "T2I_R1": record["t2i"]["R@1"],
        "T2I_R5": record["t2i"]["R@5"],
        "T2I_R10": record["t2i"]["R@10"],
        "fit_time_sec": record["fit_time_sec"],
        "fit_peak_memory_mb": record["fit_peak_memory_mb"],
        "eval_time_sec": record["eval_time_sec"],
        "eval_peak_memory_mb": record["eval_peak_memory_mb"],
        "total_time_sec": record["total_time_sec"],
        "complexity_big_o": record["complexity"]["big_o"],
        "notes": record["notes"],
    }


def write_summary(out_dir: Path, fsalign_rows: List[Dict[str, Any]], iot_record: Dict[str, Any], gr_record: Dict[str, Any]) -> None:
    practical = next(row for row in fsalign_rows if row["variant"] == "practical_recipe")
    true_full = next(row for row in fsalign_rows if row["variant"] == "true_full_cost")
    lines = [
        "# Q2 Paper-Faithful Baseline Timing",
        "",
        "Model: siglip:google/siglip-base-patch16-224",
        "Dataset: Flickr30k Karpathy",
        "",
        "## FSAlign",
        f"Practical recipe training-only time: {practical['fit_time_sec']:.2f}s, peak GPU memory {practical['fit_peak_memory_mb']:.2f} MB.",
        f"True full-structure 1-epoch time: {true_full['fit_time_sec']:.2f}s, peak GPU memory {true_full['fit_peak_memory_mb']:.2f} MB.",
        "",
        "## I0T",
        f"Official fit path: {iot_record['fit_time_sec']:.2f}s over {iot_record['fit_images']} train images and {iot_record['fit_captions']} captions.",
        f"Fit peak GPU memory: {iot_record['fit_peak_memory_mb']:.2f} MB. Test eval time: {iot_record['eval_time_sec']:.2f}s.",
        "",
        "## GR-CLIP",
        f"Official calibration path: {gr_record['fit_time_sec']:.2f}s over {gr_record['fit_images']} calibration images and {gr_record['fit_captions']} texts.",
        f"Calibration peak GPU memory: {gr_record['fit_peak_memory_mb']:.2f} MB. Test eval time: {gr_record['eval_time_sec']:.2f}s.",
        "",
        "I0T and GR-CLIP numbers here use the full Flickr30k fit/calibration code paths from the baseline scripts, not cached-embedding closed-form approximations.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-key", type=str, default="siglip", choices=["siglip", "clip", "openclip"])
    parser.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--nas-k", type=int, default=100)
    parser.add_argument("--nas-max-items", type=int, default=5000)
    parser.add_argument("--intra-samples", type=int, default=20000)
    parser.add_argument("--iot-fit-max-flickr", type=int, default=20000)
    parser.add_argument("--gr-calib-n", type=int, default=10000)
    parser.add_argument("--gr-calib-batch", type=int, default=128)
    parser.add_argument(
        "--fsalign-q2-csv",
        type=str,
        default="/work/was598/modilty_gap/results/rebuttal_true_overhead_siglip_20260325_v3/tables/q2_method_overhead.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    table_dir = out_dir / "tables"
    ensure_dir(raw_dir)
    ensure_dir(table_dir)

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model_args = make_model_args(args)

    iot_impl.seed_all(42)
    gr_impl.seed_all(42)

    fsalign_rows = load_fsalign_rows(Path(args.fsalign_q2_csv))
    iot_record = evaluate_iot(
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nas_k=args.nas_k,
        nas_max_items=args.nas_max_items,
        intra_samples=args.intra_samples,
        model_args=model_args,
        fit_max_flickr=args.iot_fit_max_flickr,
    )
    gr_record = evaluate_gr_clip(
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nas_k=args.nas_k,
        nas_max_items=args.nas_max_items,
        intra_samples=args.intra_samples,
        model_args=model_args,
        calib_n=args.gr_calib_n,
        calib_batch=args.gr_calib_batch,
    )

    save_json(raw_dir / "iot_paper_faithful.json", iot_record)
    save_json(raw_dir / "gr_clip_paper_faithful.json", gr_record)

    rows = fsalign_rows + [comparison_row(iot_record), comparison_row(gr_record)]
    save_csv(table_dir / "q2_paper_faithful_comparison.csv", rows)

    manifest = {
        "device": device,
        "model_key": args.model_key,
        "siglip_name": args.siglip_name,
        "iot_fit_max_flickr": args.iot_fit_max_flickr,
        "gr_calib_n": args.gr_calib_n,
        "fsalign_q2_csv": args.fsalign_q2_csv,
        "artifacts": {
            "iot_raw": str(raw_dir / "iot_paper_faithful.json"),
            "gr_raw": str(raw_dir / "gr_clip_paper_faithful.json"),
            "comparison_csv": str(table_dir / "q2_paper_faithful_comparison.csv"),
        },
    }
    save_json(out_dir / "manifest.json", manifest)
    write_summary(out_dir, fsalign_rows, iot_record, gr_record)
    print(f"[Done] Paper-faithful Q2 outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
