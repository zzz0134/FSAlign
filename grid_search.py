#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import subprocess
from pathlib import Path


def parse_list(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--models", type=str, default="clip,siglip,openclip")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-coco", type=int, default=5000)
    ap.add_argument("--max-flickr", type=int, default=5000)
    ap.add_argument("--max-cls", type=int, default=10000)
    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)

    ap.add_argument("--train-epochs", type=int, default=5)
    ap.add_argument("--train-anchors", type=int, default=512)
    ap.add_argument("--spectral-samples", type=int, default=512)
    ap.add_argument("--anchor-batch", type=int, default=128)

    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument("--multi-caption", action="store_true")
    ap.add_argument("--caption-agg", type=str, default="mean", choices=["random", "mean"])

    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--val-split", type=str, default="karpathy", choices=["internal", "karpathy"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-delta", type=float, default=0.0)

    ap.add_argument("--lora-mix-grid", type=str, default="0.2,0.4,0.6")
    ap.add_argument("--train-lr-grid", type=str, default="5e-4,1e-3")
    ap.add_argument("--lambda-align-grid", type=str, default="1.0,2.0")
    ap.add_argument("--lambda-dbl-grid", type=str, default="0.5,1.0")
    ap.add_argument("--lambda-spec-grid", type=str, default="0.05,0.1")
    ap.add_argument("--lambda-match-grid", type=str, default="0.05,0.1")
    ap.add_argument("--lambda-orth-grid", type=str, default="0.0,0.1")
    ap.add_argument("--train-reg-grid", type=str, default="1e-4,1e-3")
    ap.add_argument("--align-temp-grid", type=str, default="0.07")

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)

    args = ap.parse_args()

    grids = {
        "lora_mix": parse_list(args.lora_mix_grid),
        "train_lr": parse_list(args.train_lr_grid),
        "lambda_align": parse_list(args.lambda_align_grid),
        "lambda_dbl": parse_list(args.lambda_dbl_grid),
        "lambda_spec": parse_list(args.lambda_spec_grid),
        "lambda_match": parse_list(args.lambda_match_grid),
        "lambda_orth": parse_list(args.lambda_orth_grid),
        "train_reg": parse_list(args.train_reg_grid),
        "align_temp": parse_list(args.align_temp_grid),
    }

    # Ensure each grid has at least one value
    for k, v in grids.items():
        if not v:
            grids[k] = ["0"]

    keys = list(grids.keys())
    combos = list(itertools.product(*(grids[k] for k in keys)))
    if args.max_runs > 0:
        combos = combos[:args.max_runs]

    script = str(Path(__file__).parent / "our_code_final.py")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for vals in combos:
        cfg = dict(zip(keys, vals))
        tag = "_".join([f"{k}{cfg[k]}" for k in keys])
        out_dir = out_root / safe_name(tag)

        cmd = [
            "python", script,
            "--data-root", args.data_root,
            "--out-dir", str(out_dir),
            "--models", args.models,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--max-coco", str(args.max_coco),
            "--max-flickr", str(args.max_flickr),
            "--max-cls", str(args.max_cls),
            "--nas-k", str(args.nas_k),
            "--nas-max-items", str(args.nas_max_items),
            "--intra-samples", str(args.intra_samples),
            "--train-epochs", str(args.train_epochs),
            "--train-anchors", str(args.train_anchors),
            "--spectral-samples", str(args.spectral_samples),
            "--anchor-batch", str(args.anchor_batch),
            "--lora-rank", str(args.lora_rank),
            "--lora-alpha", str(args.lora_alpha),
            "--lora-mix", cfg["lora_mix"],
            "--train-lr", cfg["train_lr"],
            "--lambda-align", cfg["lambda_align"],
            "--lambda-dbl", cfg["lambda_dbl"],
            "--lambda-spec", cfg["lambda_spec"],
            "--lambda-match", cfg["lambda_match"],
            "--lambda-orth", cfg["lambda_orth"],
            "--train-reg", cfg["train_reg"],
            "--align-temp", cfg["align_temp"],
        ]

        if args.multi_caption:
            cmd.append("--multi-caption")
            cmd.extend(["--caption-agg", args.caption_agg])
        if args.early_stop:
            cmd.append("--early-stop")
            cmd.extend(["--val-split", args.val_split])
            cmd.extend(["--val-frac", str(args.val_frac)])
            cmd.extend(["--patience", str(args.patience)])
            cmd.extend(["--min-delta", str(args.min_delta)])

        print("RUN:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
