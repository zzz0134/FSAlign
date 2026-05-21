#!/usr/bin/env python3
import argparse
import copy
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import our_code_final as fs
from vqav2_eval import (
    VQAv2ClassificationDataset,
    build_vqav2_answer_vocab,
    collate_vqa,
    sparse_vqa_targets_to_dense,
    sparse_vqa_targets_to_embeddings,
    vqa_topk_scores,
)


PROTOCOL_NAME = "unified_supervised_vqa_classification"
TRAIN_PROTOCOL = "shared_frozen_query_answer_embeddings + sparse_soft_targets -> method_transform -> shared_linear_head_bce"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def maybe_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def peak_memory_mb(device: str) -> float:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
    return 0.0


def timed_stage(device: str, fn):
    reset_peak_memory(device)
    t0 = time.time()
    out = fn()
    maybe_sync(device)
    return out, {
        "time_sec": float(time.time() - t0),
        "peak_cuda_memory_allocated_mb": peak_memory_mb(device),
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


@torch.no_grad()
def encode_train_pairs_payload(
    model: fs.VLBackbone,
    dataset,
    batch_size: int,
    num_workers: int,
    question_template: str,
    answer_template: str,
    fusion_mode: str,
    max_items: Optional[int],
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_vqa,
    )

    query_chunks: List[torch.Tensor] = []
    answer_fallback_chunks: List[torch.Tensor] = []
    target_indices: List[List[int]] = []
    target_scores: List[List[float]] = []
    gt_answers: List[List[str]] = []
    total = 0

    for pil_images, questions, labels, canonical_answers, answers, _, batch_target_indices, batch_target_scores in loader:
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            questions = questions[:keep]
            canonical_answers = canonical_answers[:keep]
            answers = answers[:keep]
            batch_target_indices = batch_target_indices[:keep]
            batch_target_scores = batch_target_scores[:keep]
            b = keep

        query = fs.encode_vqa_query_batch(model, pil_images, questions, question_template, fusion_mode)
        answer_fallback = fs.encode_vqa_answer_batch(model, canonical_answers, answer_template)
        query_chunks.append(query.detach().cpu())
        answer_fallback_chunks.append(answer_fallback.detach().cpu())
        target_indices.extend([list(x) for x in batch_target_indices])
        target_scores.extend([list(map(float, x)) for x in batch_target_scores])
        gt_answers.extend([list(x) for x in answers])
        total += b

    if not query_chunks:
        raise ValueError("No VQAv2 train pairs were encoded.")

    return {
        "query_feats": torch.cat(query_chunks, dim=0),
        "answer_fallback_feats": torch.cat(answer_fallback_chunks, dim=0),
        "target_indices": target_indices,
        "target_scores": target_scores,
        "gt_answers": gt_answers,
        "n_items": total,
    }


@torch.no_grad()
def encode_val_queries_payload(
    model: fs.VLBackbone,
    dataset,
    batch_size: int,
    num_workers: int,
    question_template: str,
    fusion_mode: str,
    max_items: Optional[int],
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_vqa,
    )

    query_chunks: List[torch.Tensor] = []
    gt_answers: List[List[str]] = []
    total = 0

    for pil_images, questions, labels, canonical_answers, answers, _, _, _ in loader:
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            questions = questions[:keep]
            answers = answers[:keep]
            b = keep

        query = fs.encode_vqa_query_batch(model, pil_images, questions, question_template, fusion_mode)
        query_chunks.append(query.detach().cpu())
        gt_answers.extend([list(x) for x in answers])
        total += b

    if not query_chunks:
        raise ValueError("No VQAv2 val queries were encoded.")

    return {
        "query_feats": torch.cat(query_chunks, dim=0),
        "gt_answers": gt_answers,
        "n_items": total,
    }


@torch.no_grad()
def encode_answer_vocab_payload(
    model: fs.VLBackbone,
    answer_vocab: List[str],
    answer_template: str,
    device: str,
) -> Dict[str, Any]:
    feats = fs.build_vqa_answer_weights(model, answer_vocab, answer_template, device)
    return {
        "answer_vocab": answer_vocab,
        "answer_feats": feats.detach().cpu(),
        "n_answers": len(answer_vocab),
    }


def load_or_create_cache(cache_path: Path, builder) -> Dict[str, Any]:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")
    payload = builder()
    ensure_dir(cache_path.parent)
    torch.save(payload, cache_path)
    return payload


def split_train_indices(n: int, val_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n <= 1 or val_frac <= 0.0:
        return torch.arange(n, dtype=torch.long), torch.empty(0, dtype=torch.long)
    n_val = int(round(float(n) * float(val_frac)))
    n_val = max(1, min(n - 1, n_val))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    return perm[n_val:], perm[:n_val]


@torch.no_grad()
def apply_in_chunks(
    feats: torch.Tensor,
    device: str,
    chunk_size: int,
    fn,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    for s in range(0, feats.size(0), chunk_size):
        e = min(feats.size(0), s + chunk_size)
        x = feats[s:e].to(device)
        y = fn(x)
        outs.append(y.detach().cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty_like(feats)


class LinearVQAHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@torch.no_grad()
def evaluate_head(
    model: nn.Module,
    feats: torch.Tensor,
    gt_answers: Sequence[Sequence[str]],
    answer_vocab: Sequence[str],
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    total = 0
    score_top1 = 0.0
    score_top5 = 0.0
    k_eval = max(1, min(5, len(answer_vocab)))

    for s in range(0, feats.size(0), batch_size):
        e = min(feats.size(0), s + batch_size)
        x = feats[s:e].to(device)
        logits = model(x)
        topk = torch.topk(logits, k=k_eval, dim=1).indices
        batch_gt = gt_answers[s:e]
        s1, s5 = vqa_topk_scores(topk, answer_vocab, batch_gt)
        score_top1 += s1
        score_top5 += s5
        total += (e - s)

    if total == 0:
        return {"vqa_acc": 0.0, "vqa_acc_top5": 0.0, "n": 0.0}
    return {
        "vqa_acc": 100.0 * score_top1 / float(total),
        "vqa_acc_top5": 100.0 * score_top5 / float(total),
        "n": float(total),
    }


def train_shared_head(
    train_feats: torch.Tensor,
    train_target_indices: Sequence[Sequence[int]],
    train_target_scores: Sequence[Sequence[float]],
    train_gt_answers: Sequence[Sequence[str]],
    answer_init: torch.Tensor,
    answer_vocab: Sequence[str],
    device: str,
    head_epochs: int,
    head_batch_size: int,
    head_eval_batch_size: int,
    head_lr: float,
    head_weight_decay: float,
    head_val_frac: float,
    head_patience: int,
    seed: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_idx_cpu, dev_idx_cpu = split_train_indices(train_feats.size(0), head_val_frac, seed)
    if train_idx_cpu.numel() == 0:
        raise ValueError("No train items available for head training.")

    x_all = train_feats.to(device)
    train_idx = train_idx_cpu.to(device)
    dev_gt_answers = [list(train_gt_answers[int(i)]) for i in dev_idx_cpu.tolist()]

    model = LinearVQAHead(train_feats.size(1), answer_init.size(0)).to(device)
    with torch.no_grad():
        model.linear.weight.copy_(answer_init.to(device))
        model.linear.bias.zero_()

    optimizer = torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=head_weight_decay)

    reset_peak_memory(device)
    t0 = time.time()
    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, head_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        loss_sum = 0.0
        count = 0
        for s in range(0, perm.numel(), head_batch_size):
            idx = perm[s:s + head_batch_size]
            idx_cpu = idx.detach().cpu().tolist()
            batch_target = sparse_vqa_targets_to_dense(
                [train_target_indices[i] for i in idx_cpu],
                [train_target_scores[i] for i in idx_cpu],
                num_answers=answer_init.size(0),
                device=torch.device(device),
                dtype=x_all.dtype,
            )
            logits = model(x_all[idx])
            loss = F.binary_cross_entropy_with_logits(logits, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(idx.numel())
            count += int(idx.numel())

        epoch_rec: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(loss_sum / float(max(count, 1))),
        }

        if dev_idx_cpu.numel() > 0:
            dev_acc = evaluate_head(
                model,
                train_feats.index_select(0, dev_idx_cpu),
                dev_gt_answers,
                answer_vocab,
                device,
                head_eval_batch_size,
            )
            epoch_rec["dev_vqa_acc"] = float(dev_acc["vqa_acc"])
            epoch_rec["dev_vqa_acc_top5"] = float(dev_acc["vqa_acc_top5"])
            improved = dev_acc["vqa_acc"] > best_score
            if improved:
                best_score = float(dev_acc["vqa_acc"])
                best_epoch = epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= head_patience:
                    history.append(epoch_rec)
                    break
        else:
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        history.append(epoch_rec)

    maybe_sync(device)
    train_time_sec = float(time.time() - t0)
    peak_mb = peak_memory_mb(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    meta = {
        "train_time_sec": train_time_sec,
        "peak_cuda_memory_allocated_mb": peak_mb,
        "best_epoch": int(best_epoch),
        "best_dev_vqa_acc": float(best_score if best_score >= 0.0 else 0.0),
        "head_epochs_requested": int(head_epochs),
        "head_epochs_completed": int(len(history)),
        "head_batch_size": int(head_batch_size),
        "head_eval_batch_size": int(head_eval_batch_size),
        "head_lr": float(head_lr),
        "head_weight_decay": float(head_weight_decay),
        "head_val_frac": float(head_val_frac),
        "head_patience": int(head_patience),
        "train_split_size": int(train_idx_cpu.numel()),
        "dev_split_size": int(dev_idx_cpu.numel()),
        "history": history,
    }
    return model, meta


def build_fsalign_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        lora_state="",
        train_epochs=30,
        train_anchors=1000,
        anchor_batch=128,
        spectral_samples=1000,
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
        multi_caption=False,
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
        ds=1.0,
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


def fit_transform_fsalign(
    train_query_fit: torch.Tensor,
    train_answer_fit: torch.Tensor,
    train_query_all: torch.Tensor,
    val_query: torch.Tensor,
    answer_vocab_feats: torch.Tensor,
    device: str,
    out_dir: Path,
    feature_chunk_size: int,
) -> Dict[str, Any]:
    args = build_fsalign_args(device)
    _, _, method_info, lora_state = fs.maybe_postprocess(
        train_query_fit.to(device),
        train_answer_fit.to(device),
        args,
        tag="vqav2_unified_supervised_fsalign",
        out_dir=out_dir,
    )
    if lora_state is None:
        raise RuntimeError("FSAlign failed to produce a LoRA state.")
    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    train_query_tx = apply_in_chunks(
        train_query_all, device, feature_chunk_size,
        lambda x: fs.apply_lora_state(x, layer_img, args.lora_mix),
    )
    val_query_tx = apply_in_chunks(
        val_query, device, feature_chunk_size,
        lambda x: fs.apply_lora_state(x, layer_img, args.lora_mix),
    )
    answer_vocab_tx = apply_in_chunks(
        answer_vocab_feats, device, feature_chunk_size,
        lambda x: fs.apply_lora_state(x, layer_txt, args.lora_mix),
    )
    train_stats = dict(method_info.get("train_stats", {}))
    return {
        "method": "FSAlign",
        "train_query_feats": train_query_tx,
        "val_query_feats": val_query_tx,
        "answer_init_feats": answer_vocab_tx,
        "fit_time_sec": float(train_stats.get("train_time_sec", 0.0)),
        "fit_peak_memory_mb": float(train_stats.get("peak_cuda_memory_allocated_mb", 0.0)),
        "fit_details": method_info,
    }


def fit_transform_iot(
    train_query_fit: torch.Tensor,
    train_answer_fit: torch.Tensor,
    train_query_all: torch.Tensor,
    val_query: torch.Tensor,
    answer_vocab_feats: torch.Tensor,
    device: str,
    feature_chunk_size: int,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    def fit_stage():
        q = train_query_fit.to(device)
        a = train_answer_fit.to(device)
        q_mean = q.mean(dim=0)
        q_std = torch.sqrt(q.var(dim=0, unbiased=False) + eps)
        a_mean = a.mean(dim=0)
        a_std = torch.sqrt(a.var(dim=0, unbiased=False) + eps)
        return q_mean, q_std, a_mean, a_std

    (q_mean, q_std, a_mean, a_std), stats = timed_stage(device, fit_stage)

    def tx_q(x: torch.Tensor) -> torch.Tensor:
        return fs.l2norm((x - q_mean) / (q_std + eps))

    def tx_a(x: torch.Tensor) -> torch.Tensor:
        return fs.l2norm((x - a_mean) / (a_std + eps))

    return {
        "method": "IOT",
        "train_query_feats": apply_in_chunks(train_query_all, device, feature_chunk_size, tx_q),
        "val_query_feats": apply_in_chunks(val_query, device, feature_chunk_size, tx_q),
        "answer_init_feats": apply_in_chunks(answer_vocab_feats, device, feature_chunk_size, tx_a),
        "fit_time_sec": float(stats["time_sec"]),
        "fit_peak_memory_mb": float(stats["peak_cuda_memory_allocated_mb"]),
        "fit_details": {
            "eps": float(eps),
            "fit_train_size": int(train_query_fit.size(0)),
            "feat_dim": int(train_query_fit.size(1)),
        },
    }


def fit_transform_grclip(
    train_query_fit: torch.Tensor,
    train_answer_fit: torch.Tensor,
    train_query_all: torch.Tensor,
    val_query: torch.Tensor,
    answer_vocab_feats: torch.Tensor,
    device: str,
    feature_chunk_size: int,
) -> Dict[str, Any]:
    def fit_stage():
        q = train_query_fit.to(device)
        a = train_answer_fit.to(device)
        q_mean = q.mean(dim=0)
        a_mean = a.mean(dim=0)
        return q_mean, a_mean

    (q_mean, a_mean), stats = timed_stage(device, fit_stage)

    def tx_q(x: torch.Tensor) -> torch.Tensor:
        return fs.l2norm(x - q_mean)

    def tx_a(x: torch.Tensor) -> torch.Tensor:
        return fs.l2norm(x - a_mean)

    return {
        "method": "GR-CLIP",
        "train_query_feats": apply_in_chunks(train_query_all, device, feature_chunk_size, tx_q),
        "val_query_feats": apply_in_chunks(val_query, device, feature_chunk_size, tx_q),
        "answer_init_feats": apply_in_chunks(answer_vocab_feats, device, feature_chunk_size, tx_a),
        "fit_time_sec": float(stats["time_sec"]),
        "fit_peak_memory_mb": float(stats["peak_cuda_memory_allocated_mb"]),
        "fit_details": {
            "fit_train_size": int(train_query_fit.size(0)),
            "feat_dim": int(train_query_fit.size(1)),
        },
    }


def comparison_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": record["method"],
        "vqa_acc": record["acc"]["vqa_acc"],
        "vqa_acc_top5": record["acc"]["vqa_acc_top5"],
        "fit_time_sec": record["fit_time_sec"],
        "fit_peak_memory_mb": record["fit_peak_memory_mb"],
        "head_train_time_sec": record["head"]["train_time_sec"],
        "head_peak_memory_mb": record["head"]["peak_cuda_memory_allocated_mb"],
        "eval_time_sec": record["eval_time_sec"],
        "total_time_sec": record["total_time_sec"],
        "train_items_total": record["dataset"]["train_items_total"],
        "train_items_fit": record["dataset"]["train_items_fit"],
        "train_items_head": record["dataset"]["train_items_head"],
        "train_items_dev": record["dataset"]["train_items_dev"],
        "val_items": record["dataset"]["val_items"],
        "answer_vocab_size": record["dataset"]["answer_vocab_size"],
        "question_template": record["dataset"]["question_template"],
        "answer_template": record["dataset"]["answer_template"],
        "fusion_mode": record["dataset"]["fusion_mode"],
    }


def write_summary(out_dir: Path, model_name: str, rows: List[Dict[str, Any]], dataset_meta: Dict[str, Any]) -> None:
    lines = [
        "# Unified Supervised VQAv2 Benchmark",
        "",
        f"Model: {model_name}",
        f"Protocol: {PROTOCOL_NAME}",
        f"Training path: {TRAIN_PROTOCOL}",
        "",
        "## Dataset",
        f"Train labeled items: {dataset_meta['train_items_total']}",
        f"Fit split items: {dataset_meta['train_items_fit']}",
        f"Head dev items: {dataset_meta['train_items_dev']}",
        f"Official val items: {dataset_meta['val_items']}",
        f"Answer vocab size: {dataset_meta['answer_vocab_size']}",
        f"Val answer coverage: {dataset_meta['val_answer_coverage_pct']:.2f}%",
        "",
        "## Results",
    ]
    for row in rows:
        lines.append(
            f"{row['method']}: VQA Acc {row['vqa_acc']:.2f}, Top-5 {row['vqa_acc_top5']:.2f}, "
            f"fit {row['fit_time_sec']:.2f}s, head {row['head_train_time_sec']:.2f}s, eval {row['eval_time_sec']:.2f}s."
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqav2-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-key", type=str, default="clip", choices=["clip", "openclip", "siglip"])
    parser.add_argument("--model-size", type=int, default=32, choices=[16, 32])
    parser.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-items", type=int, default=20000)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--topk-answers", type=int, default=3129)
    parser.add_argument("--question-template", type=str, default="Question: {q}")
    parser.add_argument("--answer-template", type=str, default="Answer: {a}.")
    parser.add_argument("--fusion-mode", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--head-batch-size", type=int, default=1024)
    parser.add_argument("--head-eval-batch-size", type=int, default=4096)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-val-frac", type=float, default=0.1)
    parser.add_argument("--head-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fs.seed_all(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    vqa_root = Path(args.vqav2_root).expanduser().resolve()
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

    answer_vocab, answer_to_idx = build_vqav2_answer_vocab(str(vqa_root), args.topk_answers)
    max_train_items = None if args.max_train_items <= 0 else args.max_train_items
    max_val_items = None if args.max_val_items <= 0 else args.max_val_items
    train_dataset = VQAv2ClassificationDataset(
        str(vqa_root),
        split="train",
        answer_to_idx=answer_to_idx,
        drop_oov=True,
        max_items=max_train_items,
    )
    val_dataset = VQAv2ClassificationDataset(
        str(vqa_root),
        split="val",
        answer_to_idx=answer_to_idx,
        drop_oov=False,
        max_items=max_val_items,
    )

    cache_tag = sha1_text(
        json.dumps(
            {
                "model": model_name,
                "train_items": max_train_items,
                "val_items": max_val_items,
                "answers": args.topk_answers,
                "q_template": args.question_template,
                "a_template": args.answer_template,
                "fusion": args.fusion_mode,
                "target_version": "soft_v1",
            },
            sort_keys=True,
        )
    )

    train_cache = load_or_create_cache(
        cache_dir / f"vqav2_train_pairs_{cache_tag}.pt",
        lambda: encode_train_pairs_payload(
            model,
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            question_template=args.question_template,
            answer_template=args.answer_template,
            fusion_mode=args.fusion_mode,
            max_items=max_train_items,
        ),
    )
    val_cache = load_or_create_cache(
        cache_dir / f"vqav2_val_queries_{cache_tag}.pt",
        lambda: encode_val_queries_payload(
            model,
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            question_template=args.question_template,
            fusion_mode=args.fusion_mode,
            max_items=max_val_items,
        ),
    )
    answer_cache = load_or_create_cache(
        cache_dir / f"vqav2_answer_vocab_{cache_tag}.pt",
        lambda: encode_answer_vocab_payload(
            model,
            answer_vocab,
            args.answer_template,
            device,
        ),
    )

    train_query_all = train_cache["query_feats"].float()
    train_answer_fallback_all = train_cache["answer_fallback_feats"].float()
    train_target_indices = [list(map(int, x)) for x in train_cache["target_indices"]]
    train_target_scores = [list(map(float, x)) for x in train_cache["target_scores"]]
    train_gt_answers = [list(x) for x in train_cache["gt_answers"]]
    val_query_all = val_cache["query_feats"].float()
    val_gt_answers = [list(x) for x in val_cache["gt_answers"]]
    answer_vocab_feats = answer_cache["answer_feats"].float()
    train_answer_all = sparse_vqa_targets_to_embeddings(
        answer_vocab_feats,
        train_target_indices,
        train_target_scores,
        train_answer_fallback_all,
    ).cpu()

    fit_idx_cpu, dev_idx_cpu = split_train_indices(train_query_all.size(0), args.head_val_frac, args.seed)
    train_query_fit = train_query_all.index_select(0, fit_idx_cpu)
    train_answer_fit = train_answer_all.index_select(0, fit_idx_cpu)

    shared_dataset_meta = {
        "train_items_total": int(train_query_all.size(0)),
        "train_items_fit": int(fit_idx_cpu.numel()),
        "train_items_head": int(fit_idx_cpu.numel()),
        "train_items_dev": int(dev_idx_cpu.numel()),
        "val_items": int(val_query_all.size(0)),
        "answer_vocab_size": int(len(answer_vocab)),
        "val_answer_coverage_pct": 100.0 * float(getattr(val_dataset, "answer_coverage", 0.0)),
        "val_answer_mass_coverage_pct": 100.0 * float(getattr(val_dataset, "answer_mass_coverage", 0.0)),
        "question_template": args.question_template,
        "answer_template": args.answer_template,
        "fusion_mode": args.fusion_mode,
    }

    transformed_records = [
        fit_transform_fsalign(
            train_query_fit,
            train_answer_fit,
            train_query_all,
            val_query_all,
            answer_vocab_feats,
            device,
            out_dir,
            args.feature_chunk_size,
        ),
        fit_transform_iot(
            train_query_fit,
            train_answer_fit,
            train_query_all,
            val_query_all,
            answer_vocab_feats,
            device,
            args.feature_chunk_size,
        ),
        fit_transform_grclip(
            train_query_fit,
            train_answer_fit,
            train_query_all,
            val_query_all,
            answer_vocab_feats,
            device,
            args.feature_chunk_size,
        ),
    ]

    final_records: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for tx in transformed_records:
        head_model, head_meta = train_shared_head(
            tx["train_query_feats"],
            train_target_indices,
            train_target_scores,
            train_gt_answers,
            tx["answer_init_feats"],
            answer_vocab,
            device,
            head_epochs=args.head_epochs,
            head_batch_size=args.head_batch_size,
            head_eval_batch_size=args.head_eval_batch_size,
            head_lr=args.head_lr,
            head_weight_decay=args.head_weight_decay,
            head_val_frac=args.head_val_frac,
            head_patience=args.head_patience,
            seed=args.seed,
        )
        eval_acc, eval_stats = timed_stage(
            device,
            lambda: evaluate_head(
                head_model,
                tx["val_query_feats"],
                val_gt_answers,
                answer_vocab,
                device,
                args.head_eval_batch_size,
            ),
        )
        record = {
            "method": tx["method"],
            "protocol": PROTOCOL_NAME,
            "train_protocol": TRAIN_PROTOCOL,
            "model": model_name,
            "dataset": shared_dataset_meta,
            "acc": eval_acc,
            "fit_time_sec": float(tx["fit_time_sec"]),
            "fit_peak_memory_mb": float(tx["fit_peak_memory_mb"]),
            "fit_details": tx["fit_details"],
            "head": head_meta,
            "eval_time_sec": float(eval_stats["time_sec"]),
            "total_time_sec": float(tx["fit_time_sec"] + head_meta["train_time_sec"] + eval_stats["time_sec"]),
        }
        save_json(raw_dir / f"{fs.safe_filename(tx['method'].lower().replace('-', '_'))}.json", record)
        final_records.append(record)
        comparison_rows.append(comparison_row(record))

    save_csv(table_dir / "vqav2_unified_supervised_comparison.csv", comparison_rows)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "train_protocol": TRAIN_PROTOCOL,
        "model_name": model_name,
        "vqav2_root": str(vqa_root),
        "seed": args.seed,
        "artifacts": {
            "comparison_csv": str(table_dir / "vqav2_unified_supervised_comparison.csv"),
            "summary": str(out_dir / "summary.md"),
            "raw_dir": str(raw_dir),
        },
    }
    save_json(out_dir / "manifest.json", manifest)
    write_summary(out_dir, model_name, comparison_rows, shared_dataset_meta)
    print(f"[Done] Unified supervised VQAv2 benchmark saved to {out_dir}")


if __name__ == "__main__":
    main()
