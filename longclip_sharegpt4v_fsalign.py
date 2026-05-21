#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import our_code_final as fs

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SHAREGPT4V_PT_JSON = "share-captioner_coco_lcs_sam_1246k_1107.json"
EVAL_COUNT = 1000
LONGCLIP_CONTEXT = 248

# Keep the same FSAlign defaults used by the main entrypoint and only override
# the settings required by this experiment and to keep alignment memory bounded.
FSALIGN_OVERRIDES = {
    "train_epochs": 1,
    "align_samples": 2048,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_tqdm(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def normalize_caption(text: str) -> str:
    return fs.normalize_text(str(text).replace("\n", " "))


def extract_sharegpt4v_caption(record: Dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("Expected ShareGPT4V record with at least two conversation turns.")
    caption = normalize_caption(conversations[1].get("value", ""))
    if not caption:
        raise ValueError("Encountered empty ShareGPT4V caption.")
    return caption


class ShareGPT4VLongCaptionDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], root: Path):
        super().__init__()
        self.root = root
        self.items: List[Tuple[str, str]] = []
        missing = 0
        for record in records:
            rel_image = str(record.get("image", "")).strip()
            if not rel_image:
                continue
            image_path = self.root / rel_image
            if not image_path.exists():
                missing += 1
                continue
            caption = extract_sharegpt4v_caption(record)
            self.items.append((str(image_path), caption))
        if not self.items:
            raise RuntimeError(
                f"No valid ShareGPT4V pairs found under root={self.root}. "
                f"Expected image paths like 'coco/train2017/...'. Missing paths={missing}."
            )
        print(f"[ShareGPT4V] kept={len(self.items)} missing_paths={missing} root={self.root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, caption = self.items[index]
        image = Image.open(image_path).convert("RGB")
        return image, [caption]


class LongCLIPWrapper(fs.VLBackbone):
    def __init__(self, repo_root: Path, checkpoint_path: Path, device: str, context_length: int = LONGCLIP_CONTEXT):
        super().__init__(device=device)
        self.repo_root = repo_root
        self.checkpoint_path = checkpoint_path
        self.context_length = int(context_length)
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Long-CLIP repo not found: {self.repo_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"LongCLIP checkpoint not found: {self.checkpoint_path}")
        sys.path.insert(0, str(self.repo_root))
        self.longclip = importlib.import_module("model.longclip")
        model, preprocess = self.longclip.load(str(self.checkpoint_path), device=device)
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.preprocess = preprocess
        with torch.no_grad():
            dummy = self.preprocess(Image.new("RGB", (224, 224), color=(127, 127, 127))).unsqueeze(0).to(device)
            self._dim = int(self.model.encode_image(dummy).shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in pil_images], dim=0).to(self.device, non_blocking=True)
        feats = self.model.encode_image(batch).float()
        return fs.l2norm(feats)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self.longclip.tokenize(texts, context_length=self.context_length, truncate=True).to(self.device)
        feats = self.model.encode_text(tokens).float()
        return fs.l2norm(feats)


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


@torch.no_grad()
def encode_retrieval_features_with_progress(
    model: fs.VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    text_batch_size: int,
    desc: str,
) -> fs.RetrievalFeatureBundle:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=fs.collate_retrieval,
    )

    image_feats_chunks: List[torch.Tensor] = []
    image_captions: List[List[str]] = []
    flat_captions: List[str] = []
    cap2img: List[int] = []

    n_images = 0
    for pil_images, caps_list in maybe_tqdm(loader, enabled=True, desc=f"{desc} images", unit="batch"):
        feats = model.encode_images(pil_images)
        image_feats_chunks.append(feats)
        for i, captions in enumerate(caps_list):
            normalized_caps = [normalize_caption(text) for text in captions if normalize_caption(text)]
            if len(normalized_caps) != 1:
                raise ValueError("This experiment expects exactly one long caption per image.")
            image_captions.append(normalized_caps)
            flat_captions.append(normalized_caps[0])
            cap2img.append(n_images + i)
        n_images += len(pil_images)

    image_feats = torch.cat(image_feats_chunks, dim=0)
    text_feats_chunks: List[torch.Tensor] = []
    for start in maybe_tqdm(range(0, len(flat_captions), text_batch_size), enabled=True, desc=f"{desc} texts", unit="batch"):
        text_feats_chunks.append(model.encode_texts(flat_captions[start:start + text_batch_size]))
    text_feats = torch.cat(text_feats_chunks, dim=0)

    pair_map = torch.arange(image_feats.size(0), device=device, dtype=torch.long)
    cap2img_t = torch.tensor(cap2img, device=device, dtype=torch.long)
    cap_indices = [[i] for i in range(image_feats.size(0))]
    return fs.RetrievalFeatureBundle(
        image_feats=image_feats,
        text_feats=text_feats,
        cap2img=cap2img_t,
        pair_map=pair_map,
        paired_text=text_feats[pair_map],
        cap_indices=cap_indices,
        image_captions=image_captions,
        flat_captions=flat_captions,
    )


def get_bundle(
    model: fs.VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    text_batch_size: int,
    cache_path: Path,
    desc: str,
) -> fs.RetrievalFeatureBundle:
    if cache_path.exists():
        print(f"[Cache] loading {cache_path}")
        return bundle_from_cpu_dict(torch.load(cache_path, map_location="cpu"), device=device)
    bundle = encode_retrieval_features_with_progress(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        text_batch_size=text_batch_size,
        desc=desc,
    )
    ensure_dir(cache_path.parent)
    torch.save(bundle_to_cpu_dict(bundle), cache_path)
    print(f"[Cache] saved {cache_path}")
    return bundle


def load_sharegpt4v_annotation(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ShareGPT4V annotation not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < EVAL_COUNT:
        raise RuntimeError(f"Expected a list with at least {EVAL_COUNT} ShareGPT4V records in {path}")
    return records


def split_longclip_sharegpt4v(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # This follows the official Long-CLIP ShareGPT4V loader:
    # the first 1k rows are used for ShareGPT4V-1k evaluation, and training uses the remainder.
    eval_records = records[:EVAL_COUNT]
    train_records = [record for record in records[EVAL_COUNT:] if str(record.get("image", "")).startswith("coco/")]
    if not train_records:
        raise RuntimeError("No COCO rows found after the ShareGPT4V-1k evaluation prefix.")
    return train_records, eval_records


def build_fsalign_args(device: str) -> SimpleNamespace:
    config = fs.get_default_args()
    config.update(FSALIGN_OVERRIDES)
    config.update(
        {
            "device": device,
            "lora_state": "",
            "save_lora": False,
        }
    )
    return SimpleNamespace(**config)


def evaluate_retrieval(
    bundle: fs.RetrievalFeatureBundle,
    device: str,
    image_feats: Optional[torch.Tensor] = None,
    text_feats: Optional[torch.Tensor] = None,
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
    return {"gap": gap, "i2t": i2t, "t2i": t2i, "extra": extra}


def run_fsalign(
    train_bundle: fs.RetrievalFeatureBundle,
    eval_bundle: fs.RetrievalFeatureBundle,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = build_fsalign_args(device)
    radii = fs.logspace_scales(args.radii_min, args.radii_max, args.radii_count)
    rho_list = [float(value) for value in args.rho_list.split(",") if value.strip()]
    diffusion_scales = fs.logspace_scales(args.diffusion_min, args.diffusion_max, args.diffusion_count)

    train_bar = tqdm(total=1, desc="FSAlign train", unit="run") if tqdm is not None else None
    try:
        lora_state, history = fs.train_lora_postprocess(
            train_bundle.image_feats,
            train_bundle.paired_text,
            radii,
            rho_list,
            diffusion_scales,
            args,
            caption_pool=(train_bundle.text_feats, train_bundle.cap_indices),
        )
        if train_bar is not None:
            train_bar.update(1)
    finally:
        if train_bar is not None:
            train_bar.close()

    layer_img, layer_txt = fs.build_lora_layers(lora_state, device)
    eval_bar = tqdm(total=2, desc="FSAlign eval", unit="step") if tqdm is not None else None
    try:
        eval_img = fs.apply_lora_state(eval_bundle.image_feats.to(device), layer_img, args.lora_mix)
        if eval_bar is not None:
            eval_bar.update(1)
        eval_txt = fs.apply_lora_state(eval_bundle.text_feats.to(device), layer_txt, args.lora_mix)
        if eval_bar is not None:
            eval_bar.update(1)
    finally:
        if eval_bar is not None:
            eval_bar.close()

    metrics = evaluate_retrieval(eval_bundle, device=device, image_feats=eval_img, text_feats=eval_txt)
    record = {
        "method": "LongCLIP-L + FSAlign",
        "config": vars(args).copy(),
        "gap": metrics["gap"],
        "i2t": metrics["i2t"],
        "t2i": metrics["t2i"],
        "extra": metrics["extra"],
        "train_stats": history.get("train_stats", {}),
        "history": history,
    }
    return record, lora_state


def render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["Method", "I2T R@1", "I2T R@5", "I2T R@10", "T2I R@1", "T2I R@5", "T2I R@10"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["Method"],
            f"{row['I2T R@1']:.2f}",
            f"{row['I2T R@5']:.2f}",
            f"{row['I2T R@10']:.2f}",
            f"{row['T2I R@1']:.2f}",
            f"{row['T2I R@5']:.2f}",
            f"{row['T2I R@10']:.2f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def to_row(method: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Method": method,
        "I2T R@1": float(record["i2t"]["R@1"]),
        "I2T R@5": float(record["i2t"]["R@5"]),
        "I2T R@10": float(record["i2t"]["R@10"]),
        "T2I R@1": float(record["t2i"]["R@1"]),
        "T2I R@5": float(record["t2i"]["R@5"]),
        "T2I R@10": float(record["t2i"]["R@10"]),
    }


def average_recall(record: Dict[str, Any]) -> float:
    metrics = [
        record["i2t"]["R@1"],
        record["i2t"]["R@5"],
        record["i2t"]["R@10"],
        record["t2i"]["R@1"],
        record["t2i"]["R@5"],
        record["t2i"]["R@10"],
    ]
    return float(sum(metrics) / len(metrics))


def build_summary(baseline: Dict[str, Any], fsalign: Dict[str, Any]) -> str:
    metric_names = [
        ("I2T R@1", baseline["i2t"]["R@1"], fsalign["i2t"]["R@1"]),
        ("I2T R@5", baseline["i2t"]["R@5"], fsalign["i2t"]["R@5"]),
        ("I2T R@10", baseline["i2t"]["R@10"], fsalign["i2t"]["R@10"]),
        ("T2I R@1", baseline["t2i"]["R@1"], fsalign["t2i"]["R@1"]),
        ("T2I R@5", baseline["t2i"]["R@5"], fsalign["t2i"]["R@5"]),
        ("T2I R@10", baseline["t2i"]["R@10"], fsalign["t2i"]["R@10"]),
    ]
    improved = sum(1 for _, base, aligned in metric_names if aligned > base)
    avg_delta = average_recall(fsalign) - average_recall(baseline)
    if improved >= 4 and avg_delta > 0:
        return (
            f"Compared with frozen LongCLIP-L, FSAlign improves {improved} of 6 retrieval metrics on ShareGPT4V-1k "
            f"and raises the six-metric average recall by {avg_delta:.2f} points. "
            "FSAlign remains effective beyond short captions and fixed prompt-style text, and its gain persists under a long-caption setting."
        )
    return (
        f"Compared with frozen LongCLIP-L, FSAlign improves {improved} of 6 retrieval metrics on ShareGPT4V-1k "
        f"with an average recall delta of {avg_delta:.2f} points, so this run does not show a clear long-caption retrieval gain over the frozen backbone."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LongCLIP-L + FSAlign long-caption retrieval experiment on ShareGPT4V")
    parser.add_argument("--sharegpt4v-root", type=str, required=True, help="Root directory containing ShareGPT4V annotations and image subfolders.")
    parser.add_argument("--longclip-repo", type=str, required=True, help="Path to a local clone of the official Long-CLIP repository.")
    parser.add_argument("--longclip-checkpoint", type=str, required=True, help="Path to the official LongCLIP-L checkpoint.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--annotation-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fs.seed_all(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    sharegpt4v_root = Path(args.sharegpt4v_root)
    if not sharegpt4v_root.exists():
        raise FileNotFoundError(f"ShareGPT4V root not found: {sharegpt4v_root}")
    annotation_path = Path(args.annotation_json) if args.annotation_json else sharegpt4v_root / SHAREGPT4V_PT_JSON

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    raw_dir = out_dir / "raw"
    ensure_dir(out_dir)
    ensure_dir(cache_dir)
    ensure_dir(raw_dir)

    print("[Setup] Loading ShareGPT4V annotations")
    records = load_sharegpt4v_annotation(annotation_path)
    train_records, eval_records = split_longclip_sharegpt4v(records)
    print(f"[Setup] annotation={annotation_path}")
    print(f"[Setup] ShareGPT4V-1k eval rows={len(eval_records)}")
    print(f"[Setup] ShareGPT4V COCO-only train rows={len(train_records)}")

    train_dataset = ShareGPT4VLongCaptionDataset(train_records, sharegpt4v_root)
    eval_dataset = ShareGPT4VLongCaptionDataset(eval_records, sharegpt4v_root)

    print("[Setup] Loading LongCLIP-L")
    model = LongCLIPWrapper(
        repo_root=Path(args.longclip_repo),
        checkpoint_path=Path(args.longclip_checkpoint),
        device=device,
        context_length=LONGCLIP_CONTEXT,
    )
    model_name = "LongCLIP-L"
    cache_tag = fs.safe_filename(model_name)

    train_bundle = get_bundle(
        model=model,
        dataset=train_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_batch_size=args.text_batch_size,
        cache_path=cache_dir / f"sharegpt4v_coco_train_{cache_tag}.pt",
        desc="Train",
    )
    eval_bundle = get_bundle(
        model=model,
        dataset=eval_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_batch_size=args.text_batch_size,
        cache_path=cache_dir / f"sharegpt4v_1k_eval_{cache_tag}.pt",
        desc="Eval",
    )

    baseline_bar = tqdm(total=1, desc="Baseline eval", unit="run") if tqdm is not None else None
    try:
        baseline_metrics = evaluate_retrieval(eval_bundle, device=device)
        baseline_record = {"method": model_name, **baseline_metrics}
        if baseline_bar is not None:
            baseline_bar.update(1)
    finally:
        if baseline_bar is not None:
            baseline_bar.close()

    fsalign_record, lora_state = run_fsalign(train_bundle, eval_bundle, device=device)
    torch.save(lora_state, raw_dir / "longclip_l_fsalign_state.pt")

    payload = {
        "backbone": "LongCLIP-L",
        "train_data": "ShareGPT4V COCO subset",
        "eval_data": "ShareGPT4V-1k",
        "baseline": {
            "I2T_R1": float(baseline_record["i2t"]["R@1"]),
            "I2T_R5": float(baseline_record["i2t"]["R@5"]),
            "I2T_R10": float(baseline_record["i2t"]["R@10"]),
            "T2I_R1": float(baseline_record["t2i"]["R@1"]),
            "T2I_R5": float(baseline_record["t2i"]["R@5"]),
            "T2I_R10": float(baseline_record["t2i"]["R@10"]),
        },
        "fsalign": {
            "I2T_R1": float(fsalign_record["i2t"]["R@1"]),
            "I2T_R5": float(fsalign_record["i2t"]["R@5"]),
            "I2T_R10": float(fsalign_record["i2t"]["R@10"]),
            "T2I_R1": float(fsalign_record["t2i"]["R@1"]),
            "T2I_R5": float(fsalign_record["t2i"]["R@5"]),
            "T2I_R10": float(fsalign_record["t2i"]["R@10"]),
        },
    }
    json_path = raw_dir / "longclip_l_sharegpt4v_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [
        to_row("LongCLIP-L", baseline_record),
        to_row("LongCLIP-L + FSAlign", fsalign_record),
    ]
    table = render_markdown_table(rows)
    summary = build_summary(baseline_record, fsalign_record)

    print("\n" + table)
    print(f"\nJSON path: {json_path}")
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
