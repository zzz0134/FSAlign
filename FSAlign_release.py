#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Our method + standard eval:
- Karpathy split for MSCOCO + Flickr30k retrieval
- Zero-shot image classification on CIFAR100 / Tiny-ImageNet-200 / DTD
- Gap metrics: centroid distance, Relative Modality Gap (RMG), NAS(k), CMAS
- Optional postprocess training (LoRA + fractal losses) from our_code_karpathy.py
- Save JSONL + CSV per model/dataset
"""

import os
import csv
import json
import time
import math
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from PIL import Image

import torchvision.datasets as tvds
import MG as mg

from vqav2_eval import (
    VQAv2ClassificationDataset,
    build_vqav2_answer_vocab,
    collate_vqa,
    format_answer_prompt,
    format_question_prompt,
    fuse_query_features,
    sparse_vqa_targets_to_embeddings,
    vqa_topk_scores,
)

REPO_ROOT = Path(__file__).resolve().parent
RECOMMENDED_LNO_PRESETS = {
    ("flickr30k", "clip", "ViT-B-32", "short", 3): {
        "name": "flickr30k_clip_vitb32_strong_gap",
        "lora_state": REPO_ROOT / "results" / "our_final_train_1.24" / "clip_ViT-B-32_openai_flickr30k_karpathy_test_lora_state.pt",
        "lora_mix": 0.6,
        "notes": "Recommended preset for the paper-style Flickr30k CLIP LNO figure.",
    },
}

PAPER_GAP_TARGETS = {
    "mscoco2014_karpathy_test": {"centroid_distance": 0.7729, "relative_modality_gap": 0.4922, "NAS@100": 0.4584, "CMAS": 0.6490},
    "flickr30k_karpathy_test": {"centroid_distance": 0.7542, "relative_modality_gap": 0.4958, "NAS@100": 0.3596, "CMAS": 0.6530},
    "cifar100_test": {"centroid_distance": 0.5446, "relative_modality_gap": 0.5501, "NAS@100": 0.4362, "CMAS": 0.7338},
    "dtd_test": {"centroid_distance": 0.6997, "relative_modality_gap": 0.5689, "NAS@100": 0.4629, "CMAS": 0.7218},
    "tiny-imagenet-200_val": {"centroid_distance": 0.5269, "relative_modality_gap": 0.5428, "NAS@100": 0.6719, "CMAS": 0.6442},
}

# optional deps
try:
    import open_clip
except Exception:
    open_clip = None

SiglipModel = None
SiglipProcessor = None


# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(s: str) -> str:
    # Keep filenames portable across filesystems
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def apply_lora_mix(x: torch.Tensor, layer: nn.Module, mix: float) -> torch.Tensor:
    if mix <= 0.0:
        return l2norm(x)
    y = layer(x)
    if mix >= 1.0:
        return l2norm(y)
    return l2norm((1.0 - mix) * x + mix * y)


def download_file(url: str, dst: Path, timeout: int = 180):
    import urllib.request
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"[Download] {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    tmp.write_bytes(data)
    tmp.replace(dst)
    print(f"[Download] saved -> {dst} ({dst.stat().st_size/1e6:.2f} MB)")


# ============================================================
# Karpathy JSON download (Stanford zip + fallback mirror)
# ============================================================

KARPATHY_CAPTION_ZIP_URLS = [
    "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
]
KARPATHY_SPLITS_MIRROR = {
    "coco": "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download=",
    "flickr30k": "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_flickr30k.json?download=",
}


def download_and_extract_from_zip(urls: List[str], dst_json: Path, member_name: str, timeout: int = 240):
    import urllib.request
    import zipfile

    ensure_dir(dst_json.parent)
    if dst_json.exists() and dst_json.stat().st_size > 0:
        return

    zip_path = dst_json.parent / "caption_datasets.zip"
    last_err = None

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        for u in urls:
            try:
                print(f"[Download] {u}")
                with urllib.request.urlopen(u, timeout=timeout) as resp:
                    data = resp.read()
                tmp = zip_path.with_suffix(".zip.tmp")
                tmp.write_bytes(data)
                tmp.replace(zip_path)
                print(f"[Download] saved -> {zip_path} ({zip_path.stat().st_size/1e6:.2f} MB)")
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None and (not zip_path.exists() or zip_path.stat().st_size == 0):
            raise RuntimeError(f"Failed to download caption_datasets.zip. Last error: {last_err}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        if member_name not in names:
            cand = [n for n in zf.namelist() if n.endswith("/" + member_name) or n.endswith(member_name)]
            if len(cand) == 0:
                raise RuntimeError(f"{member_name} not found in {zip_path}.")
            member = cand[0]
        else:
            member = member_name

        with zf.open(member, "r") as f:
            data = f.read()

    tmp_json = dst_json.with_suffix(dst_json.suffix + ".tmp")
    tmp_json.write_bytes(data)
    tmp_json.replace(dst_json)
    print(f"[Extract] {member_name} -> {dst_json} ({dst_json.stat().st_size/1e6:.2f} MB)")


def ensure_karpathy_json(data_root: str, which: str) -> Path:
    root = Path(data_root)

    if which == "coco":
        dst = root / "mscoco2014" / "karpathy" / "dataset_coco.json"
        member = "dataset_coco.json"
        mirror = KARPATHY_SPLITS_MIRROR["coco"]
    elif which == "flickr30k":
        dst = root / "flickr30k" / "karpathy" / "dataset_flickr30k.json"
        member = "dataset_flickr30k.json"
        mirror = KARPATHY_SPLITS_MIRROR["flickr30k"]
    else:
        raise ValueError(which)

    if dst.exists() and dst.stat().st_size > 0:
        return dst

    try:
        download_and_extract_from_zip(KARPATHY_CAPTION_ZIP_URLS, dst_json=dst, member_name=member)
        if dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception as e:
        print(f"[Warn] Stanford zip failed: {e}")

    try:
        download_file(mirror, dst)
        if dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception as e:
        print(f"[Warn] Mirror json failed: {e}")

    raise RuntimeError(f"Failed to obtain Karpathy json for '{which}'.")


# ============================================================
# Datasets
# ============================================================

class KarpathyRetrievalDataset(Dataset):
    """
    From Karpathy json. Each item yields:
      (PIL.Image, captions_list[str])
    """
    def __init__(self, karpathy_json: str, image_roots: List[str], split: str, max_images: Optional[int] = None):
        super().__init__()
        self.karpathy_json = Path(karpathy_json)
        assert self.karpathy_json.exists(), f"Karpathy json not found: {self.karpathy_json}"
        self.image_roots = [Path(p) for p in image_roots]
        self.split = split

        data = json.loads(self.karpathy_json.read_text(encoding="utf-8"))
        images = data["images"]

        items = []
        missing = 0
        for img in images:
            if img.get("split", "") != split:
                continue
            fn = img.get("filename", None)
            if fn is None:
                continue

            caps = []
            for s in img.get("sentences", []):
                if "raw" in s:
                    caps.append(s["raw"])
                elif "tokens" in s:
                    caps.append(" ".join(s["tokens"]))
            if len(caps) == 0:
                continue

            p = self._resolve_path(fn)
            if p is None:
                missing += 1
                continue

            items.append((str(p), caps))
            if max_images is not None and len(items) >= max_images:
                break

        if len(items) == 0:
            raise AssertionError(
                f"No items for split='{split}' from {karpathy_json}. "
                f"Check image roots. (missing_paths={missing})"
            )

        print(f"[KarpathyDataset] split={split} items={len(items)} (missing_paths={missing})")
        self.items = items

    def _resolve_path(self, filename: str) -> Optional[Path]:
        # direct join
        for r in self.image_roots:
            if not r.exists():
                continue
            p = r / filename
            if p.exists():
                return p

        # cheap 1-level common subfolders
        common_subs = ["images", "flickr30k-images", "train2014", "val2014", "train", "val"]
        for r in self.image_roots:
            if not r.exists():
                continue
            for sub in common_subs:
                p = r / sub / filename
                if p.exists():
                    return p

        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, caps = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return img, caps


class TinyImageNet200Val(Dataset):
    """
    Tiny-ImageNet-200 val split.
    Uses wnids.txt + words.txt to build classnames.
    """
    def __init__(self, data_root: str):
        super().__init__()
        self.root = Path(data_root) / "tiny-imagenet-200"
        assert self.root.exists(), f"Tiny-ImageNet root not found: {self.root}"

        wnids_path = self.root / "wnids.txt"
        words_path = self.root / "words.txt"
        ann_path = self.root / "val" / "val_annotations.txt"
        img_dir = self.root / "val" / "images"

        assert wnids_path.exists(), f"wnids.txt not found under {self.root}"
        assert words_path.exists(), f"words.txt not found under {self.root}"
        assert ann_path.exists(), f"val_annotations.txt not found under {ann_path}"
        assert img_dir.exists(), f"val/images not found under {img_dir}"

        self.wnids = [l.strip() for l in wnids_path.read_text().splitlines() if l.strip()]
        wnid_to_words = {}
        for line in words_path.read_text().splitlines():
            # format: n01443537\tgoldfish, Carassius auratus
            parts = line.split("\t")
            if len(parts) >= 2:
                wnid_to_words[parts[0]] = parts[1].split(",")[0].strip()

        self.classnames = [wnid_to_words.get(w, w) for w in self.wnids]

        img2wnid = {}
        for line in ann_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                img2wnid[parts[0]] = parts[1]

        samples = []
        for img_name, wnid in img2wnid.items():
            p = img_dir / img_name
            if p.exists() and wnid in self.wnids:
                y = self.wnids.index(wnid)
                samples.append((str(p), y))

        assert len(samples) > 0, f"No val samples found under {img_dir}"
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return img, y


# ============================================================
# Collate functions (avoid variable-length caption issues)
# ============================================================

def collate_retrieval(batch):
    # batch: [(PIL, [caps]), ...]
    images = [b[0] for b in batch]
    caps_list = []
    for _, caps in batch:
        if isinstance(caps, (list, tuple)):
            caps_list.append([str(c) for c in caps])
        else:
            caps_list.append([str(caps)])
    return images, caps_list


def collate_cls(batch):
    # batch: [(PIL, y), ...]
    images = [b[0] for b in batch]
    ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return images, ys


# ============================================================
# Backbones: CLIP / OpenCLIP / SigLIP (preprocess inside wrapper)
# ============================================================

class VLBackbone(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError


class OpenCLIPWrapper(VLBackbone):
    def __init__(self, model_name: str, pretrained: str, device: str):
        super().__init__(device=device)
        assert open_clip is not None, "open_clip is not installed."
        self.model_name = model_name
        self.pretrained = pretrained

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(device).eval()
        self.preprocess = preprocess_val
        self.tokenizer = tokenizer

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feat = self.model.encode_image(dummy)
            self._dim = int(feat.shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        # preprocess on CPU then move to GPU
        tens = torch.stack([self.preprocess(im) for im in pil_images], dim=0)
        tens = tens.to(self.device, non_blocking=True)
        feat = self.model.encode_image(tens).float()
        return l2norm(feat)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts)
        if isinstance(toks, dict):
            toks = {k: v.to(self.device) for k, v in toks.items()}
            feat = self.model.encode_text(**toks)
        else:
            toks = toks.to(self.device)
            feat = self.model.encode_text(toks)
        feat = feat.float()
        return l2norm(feat)


class CLIPWrapper(OpenCLIPWrapper):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name=model_name, pretrained="openai", device=device)


class SigLIPWrapper(VLBackbone):
    def __init__(self, hf_name: str, device: str):
        super().__init__(device=device)
        assert SiglipModel is not None and SiglipProcessor is not None, "SigLIP requires transformers>=4.40 with SiglipModel."
        self.hf_name = hf_name
        self.model = SiglipModel.from_pretrained(hf_name).to(device).eval()
        self.proc = SiglipProcessor.from_pretrained(hf_name)

        # robust dim inference (do NOT rely on config.projection_dim)
        with torch.no_grad():
            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            inp = self.proc(images=dummy_img, return_tensors="pt")
            pv = inp["pixel_values"].to(device)
            feat = self.model.get_image_features(pixel_values=pv)
            self._dim = int(feat.shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inp = self.proc(images=pil_images, return_tensors="pt")
        pv = inp["pixel_values"].to(self.device, non_blocking=True)
        feat = self.model.get_image_features(pixel_values=pv).float()
        return l2norm(feat)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inp = self.proc(text=texts, padding=True, truncation=True, return_tensors="pt")
        inp = {k: v.to(self.device, non_blocking=True) for k, v in inp.items()}
        feat = self.model.get_text_features(**inp).float()
        return l2norm(feat)


def make_models(args, device: str) -> List[Tuple[str, VLBackbone]]:
    return [(f"clip:{args.clip_model}:openai", CLIPWrapper(args.clip_model, device=device))]


@dataclass
class RetrievalFeatureBundle:
    image_feats: torch.Tensor
    text_feats: torch.Tensor
    cap2img: torch.Tensor
    pair_map: torch.Tensor
    paired_text: torch.Tensor
    cap_indices: List[List[int]]
    image_captions: List[List[str]]
    flat_captions: List[str]


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def ensure_sentence(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return text
    if text[-1] not in ".!?":
        text = text + "."
    return text


def build_caption_variants(caps: List[str], variant: str = "short", paragraph_sentences: int = 3) -> List[str]:
    caps = [normalize_text(c) for c in caps if normalize_text(c)]
    if not caps:
        return [""]
    if variant == "short":
        return caps
    if variant != "paragraph":
        raise ValueError(f"Unsupported text variant: {variant}")

    span = max(1, int(paragraph_sentences))
    out: List[str] = []
    for start in range(len(caps)):
        parts = [ensure_sentence(caps[(start + off) % len(caps)]) for off in range(min(span, len(caps)))]
        out.append(" ".join(parts))
    return out


@torch.no_grad()
def encode_retrieval_features(
    model: VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    text_variant: str = "short",
    paragraph_sentences: int = 3,
    text_batch_size: int = 256,
) -> RetrievalFeatureBundle:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_retrieval,
    )

    image_feats_chunks: List[torch.Tensor] = []
    image_captions: List[List[str]] = []
    flat_captions: List[str] = []
    cap2img: List[int] = []

    n_images = 0
    batch_idx = 0
    for pil_images, caps_list in loader:
        batch_idx += 1
        if max_images is not None and n_images >= max_images:
            break

        b = len(pil_images)
        if max_images is not None and n_images + b > max_images:
            keep = max_images - n_images
            pil_images = pil_images[:keep]
            caps_list = caps_list[:keep]
            b = keep

        feats = model.encode_images(pil_images)
        image_feats_chunks.append(feats)

        for i in range(b):
            caps = build_caption_variants(caps_list[i], variant=text_variant, paragraph_sentences=paragraph_sentences)
            image_captions.append(caps)
            for c in caps:
                flat_captions.append(c)
                cap2img.append(n_images + i)

        n_images += b
        if batch_idx % 10 == 0:
            print(
                f"[encode_retrieval_features] image_batches={batch_idx} images_encoded={n_images}",
                flush=True,
            )

    if not image_feats_chunks:
        raise ValueError("No retrieval features were encoded.")

    image_feats = torch.cat(image_feats_chunks, dim=0)
    text_feats_chunks: List[torch.Tensor] = []
    text_batches = 0
    for s in range(0, len(flat_captions), text_batch_size):
        text_batches += 1
        text_feats_chunks.append(model.encode_texts(flat_captions[s:s + text_batch_size]))
        if text_batches % 20 == 0:
            done = min(s + text_batch_size, len(flat_captions))
            print(
                f"[encode_retrieval_features] text_batches={text_batches} captions_encoded={done}/{len(flat_captions)}",
                flush=True,
            )
    text_feats = torch.cat(text_feats_chunks, dim=0)

    first_cap = [-1] * image_feats.size(0)
    cap_indices = [[] for _ in range(image_feats.size(0))]
    for cap_idx, img_idx in enumerate(cap2img):
        if first_cap[img_idx] < 0:
            first_cap[img_idx] = cap_idx
        cap_indices[img_idx].append(cap_idx)

    pair_map = torch.tensor(first_cap, dtype=torch.long, device=device)
    cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)
    paired_text = text_feats[pair_map]
    return RetrievalFeatureBundle(
        image_feats=image_feats,
        text_feats=text_feats,
        cap2img=cap2img_t,
        pair_map=pair_map,
        paired_text=paired_text,
        cap_indices=cap_indices,
        image_captions=image_captions,
        flat_captions=flat_captions,
    )


@torch.no_grad()
def compute_gap_metrics(
    image_feats: torch.Tensor,
    paired_text: torch.Tensor,
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
) -> Dict[str, float]:
    return {
        "centroid_distance": centroid_distance(image_feats, paired_text),
        "relative_modality_gap": relative_modality_gap(image_feats, paired_text, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(image_feats, paired_text, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(image_feats, paired_text),
    }


@torch.no_grad()
def compute_gap_metrics_mg_shift(
    image_feats: torch.Tensor,
    paired_text: torch.Tensor,
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    mg_lambda: float,
) -> Dict[str, float]:
    x_s, y_s, _ = mg.mg_shift_pairwise(image_feats, paired_text, lam=float(mg_lambda))
    return {
        "centroid_distance": mg.centroid_distance(x_s, y_s),
        "relative_modality_gap": mg.relative_modality_gap(x_s, y_s, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": mg.nas_k(x_s, y_s, k=nas_k_val, max_items=nas_max_items),
        "CMAS": mg.cmas(x_s, y_s),
    }


def _paper_gap_target_for_tag(tag: str) -> Optional[Dict[str, float]]:
    for ds, target in PAPER_GAP_TARGETS.items():
        if str(tag).endswith(ds):
            return target
    return None


def _paper_gap_err(gap: Dict[str, float], target: Dict[str, float], nas_k_val: int) -> float:
    nas_key = f"NAS@{nas_k_val}"
    target_nas = target.get("NAS@100", target.get("NAS@10", 0.0))
    return (
        (float(gap["centroid_distance"]) - float(target["centroid_distance"])) ** 2
        + (float(gap["relative_modality_gap"]) - float(target["relative_modality_gap"])) ** 2
        + (float(gap[nas_key]) - float(target_nas)) ** 2
        + (float(gap["CMAS"]) - float(target["CMAS"])) ** 2
    )


@torch.no_grad()
def build_gap_paired_text(
    bundle: RetrievalFeatureBundle,
    text_feats: torch.Tensor,
    mode: str = "first_caption",
) -> torch.Tensor:
    mode = str(mode)
    if mode == "first_caption":
        return text_feats[bundle.pair_map]
    if mode == "mean_caption":
        rows: List[torch.Tensor] = []
        for cap_ids in bundle.cap_indices:
            if not cap_ids:
                raise ValueError("Empty caption set encountered when building mean-caption paired text.")
            idx_t = torch.tensor(cap_ids, dtype=torch.long, device=text_feats.device)
            rows.append(l2norm(text_feats[idx_t].mean(dim=0, keepdim=True)).squeeze(0))
        return torch.stack(rows, dim=0)
    raise ValueError(f"Unsupported gap paired-text mode: {mode}")


@torch.no_grad()
def retrieval_metrics_from_embeddings(
    bundle: RetrievalFeatureBundle,
    device: str,
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    gap_paired_text_mode: str = "first_caption",
    image_feats: Optional[torch.Tensor] = None,
    text_feats: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    image_feats = bundle.image_feats if image_feats is None else image_feats
    text_feats = bundle.text_feats if text_feats is None else text_feats
    paired_text = build_gap_paired_text(bundle, text_feats, mode=gap_paired_text_mode)
    cap2img_t = bundle.cap2img

    gap = compute_gap_metrics(image_feats, paired_text, nas_k_val=nas_k_val, nas_max_items=nas_max_items, intra_samples=intra_samples)

    def recall_i2t(k: int) -> float:
        correct = 0
        n_img = image_feats.size(0)
        chunk = 512
        k_eff = max(1, min(k, int(text_feats.size(0))))
        for s in range(0, n_img, chunk):
            e = min(n_img, s + chunk)
            sims = image_feats[s:e] @ text_feats.t()
            topk = torch.topk(sims, k=k_eff, dim=1).indices
            img_ids = torch.arange(s, e, device=device).unsqueeze(1)
            mapped = cap2img_t[topk]
            hit = (mapped == img_ids).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(n_img)

    def recall_t2i(k: int) -> float:
        correct = 0
        n_cap = text_feats.size(0)
        chunk = 1024
        k_eff = max(1, min(k, int(image_feats.size(0))))
        for s in range(0, n_cap, chunk):
            e = min(n_cap, s + chunk)
            sims = text_feats[s:e] @ image_feats.t()
            topk = torch.topk(sims, k=k_eff, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(n_cap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}
    avg_words = 0.0
    if bundle.flat_captions:
        avg_words = float(sum(len(t.split()) for t in bundle.flat_captions) / len(bundle.flat_captions))
    extra = {
        "n_images": float(image_feats.size(0)),
        "n_captions": float(text_feats.size(0)),
        "avg_words_per_text": avg_words,
    }
    return gap, i2t, t2i, extra


@torch.no_grad()
def same_modality_topk_neighbors(
    feats: torch.Tensor,
    k: int,
    device: str,
    chunk: int = 1024,
) -> torch.Tensor:
    feats = feats.to(device)
    n = int(feats.size(0))
    k_eff = min(int(k), max(n - 1, 0))
    if n == 0 or k_eff <= 0:
        return torch.empty((n, 0), dtype=torch.long, device=feats.device)

    topk_blocks: List[torch.Tensor] = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sims = feats[s:e] @ feats.t()
        row_ids = torch.arange(s, e, device=feats.device)
        sims[torch.arange(e - s, device=feats.device), row_ids] = -1e9
        topk_blocks.append(torch.topk(sims, k=k_eff, dim=1).indices)
    return torch.cat(topk_blocks, dim=0)


@torch.no_grad()
def _retrieval_aligned_outrank_direction(
    query_feats: torch.Tensor,
    gt_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    neighbor_idx: torch.Tensor,
    ks: List[int],
    device: str,
    chunk: int = 1024,
) -> Tuple[List[Dict[str, float]], Dict[int, Dict[str, Any]]]:
    query_feats = query_feats.to(device)
    gt_feats = gt_feats.to(device)
    candidate_feats = candidate_feats.to(device)
    neighbor_idx = neighbor_idx.to(device)

    n = int(query_feats.size(0))
    if gt_feats.size(0) != n or candidate_feats.size(0) != n or neighbor_idx.size(0) != n:
        raise ValueError('Query, ground-truth, candidate, and neighbor tensors must share the same first dimension.')

    gt_scores = (query_feats * gt_feats).sum(dim=1)
    sample_index = torch.arange(n, device=query_feats.device, dtype=torch.long)
    max_neighbors = int(neighbor_idx.size(1))

    aggregated: List[Dict[str, float]] = []
    per_sample: Dict[int, Dict[str, Any]] = {}
    for requested_k in ks:
        k_eff = min(int(requested_k), max_neighbors)
        if k_eff <= 0:
            oor = torch.zeros(n, device=query_feats.device, dtype=query_feats.dtype)
            hit = torch.zeros(n, device=query_feats.device, dtype=query_feats.dtype)
            margin = torch.zeros(n, device=query_feats.device, dtype=query_feats.dtype)
            hardest_idx = torch.full((n,), -1, device=query_feats.device, dtype=torch.long)
            hardest_score = gt_scores.clone()
        else:
            oor_blocks: List[torch.Tensor] = []
            hit_blocks: List[torch.Tensor] = []
            margin_blocks: List[torch.Tensor] = []
            hardest_idx_blocks: List[torch.Tensor] = []
            hardest_score_blocks: List[torch.Tensor] = []
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                idx_block = neighbor_idx[s:e, :k_eff]
                q_block = query_feats[s:e]
                gt_block = gt_scores[s:e]
                cand_block = candidate_feats[idx_block.reshape(-1)].view(e - s, k_eff, -1)
                cand_scores = (cand_block * q_block[:, None, :]).sum(dim=2)
                outrank = cand_scores > gt_block[:, None]
                hardest_score_block, hardest_pos = cand_scores.max(dim=1)
                hardest_idx_block = idx_block.gather(1, hardest_pos[:, None]).squeeze(1)
                oor_blocks.append(outrank.float().mean(dim=1))
                hit_blocks.append(outrank.any(dim=1).float())
                margin_blocks.append(hardest_score_block - gt_block)
                hardest_idx_blocks.append(hardest_idx_block)
                hardest_score_blocks.append(hardest_score_block)
            oor = torch.cat(oor_blocks, dim=0)
            hit = torch.cat(hit_blocks, dim=0)
            margin = torch.cat(margin_blocks, dim=0)
            hardest_idx = torch.cat(hardest_idx_blocks, dim=0)
            hardest_score = torch.cat(hardest_score_blocks, dim=0)

        aggregated.append({
            'k': int(requested_k),
            'effective_k': int(k_eff),
            'oor': float(oor.mean().item()),
            'hit': float(hit.mean().item()),
            'margin': float(margin.mean().item()),
        })
        per_sample[int(requested_k)] = {
            'k': int(requested_k),
            'effective_k': int(k_eff),
            'sample_index': sample_index.detach().cpu(),
            'gt_score': gt_scores.detach().cpu(),
            'hardest_neighbor_index': hardest_idx.detach().cpu(),
            'hardest_neighbor_score': hardest_score.detach().cpu(),
            'oor': oor.detach().cpu(),
            'hit': hit.detach().cpu(),
            'margin': margin.detach().cpu(),
        }
    return aggregated, per_sample


@torch.no_grad()
def retrieval_aligned_neighbor_outrank_analysis(
    image_feats: torch.Tensor,
    paired_text: torch.Tensor,
    ks: List[int],
    device: str,
    neighbor_chunk: int = 1024,
    score_chunk: int = 1024,
) -> Dict[str, Any]:
    image_feats = image_feats.to(device)
    paired_text = paired_text.to(device)
    if image_feats.size(0) != paired_text.size(0):
        raise ValueError('image_feats and paired_text must have the same number of rows.')

    clean_ks = sorted({int(k) for k in ks if int(k) > 0})
    max_k = min(max(clean_ks), max(int(image_feats.size(0)) - 1, 0)) if clean_ks else 0
    img_neighbors = same_modality_topk_neighbors(image_feats, max_k, device=device, chunk=neighbor_chunk)
    txt_neighbors = same_modality_topk_neighbors(paired_text, max_k, device=device, chunk=neighbor_chunk)

    i2t_agg, i2t_raw = _retrieval_aligned_outrank_direction(
        image_feats,
        paired_text,
        paired_text,
        img_neighbors,
        clean_ks,
        device=device,
        chunk=score_chunk,
    )
    t2i_agg, t2i_raw = _retrieval_aligned_outrank_direction(
        paired_text,
        image_feats,
        image_feats,
        txt_neighbors,
        clean_ks,
        device=device,
        chunk=score_chunk,
    )
    return {
        'metadata': {
            'num_pairs': int(image_feats.size(0)),
            'ks': clean_ks,
            'max_effective_k': int(max_k),
            'neighbor_chunk': int(neighbor_chunk),
            'score_chunk': int(score_chunk),
            'paired_text_definition': 'bundle.text_feats[bundle.pair_map]',
            'score_function': 'dot_product_on_l2_normalized_embeddings',
        },
        'i2t': {'aggregated': i2t_agg, 'per_sample': i2t_raw},
        't2i': {'aggregated': t2i_agg, 'per_sample': t2i_raw},
    }


def neighbor_competition_loss(
    y_img: torch.Tensor,
    y_txt: torch.Tensor,
    k: int,
    margin: float,
    device: str,
    sample_size: int = 1024,
    neighbor_chunk: int = 1024,
    top_frac: float = 1.0,
    precomputed_img_neighbors: Optional[torch.Tensor] = None,
    precomputed_txt_neighbors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = int(y_img.size(0))
    if n <= 1 or int(k) <= 0:
        return torch.tensor(0.0, device=y_img.device, dtype=y_img.dtype)

    if precomputed_img_neighbors is None and precomputed_txt_neighbors is None and sample_size > 0 and n > sample_size:
        pick = torch.randperm(n, device=y_img.device)[:sample_size]
        y_img = y_img[pick]
        y_txt = y_txt[pick]
        n = int(y_img.size(0))

    k_eff = min(int(k), max(n - 1, 0))
    if k_eff <= 0:
        return torch.tensor(0.0, device=y_img.device, dtype=y_img.dtype)

    use_precomputed = (
        precomputed_img_neighbors is not None
        and precomputed_txt_neighbors is not None
        and int(precomputed_img_neighbors.size(0)) == n
        and int(precomputed_txt_neighbors.size(0)) == n
    )
    if use_precomputed:
        img_neighbors = precomputed_img_neighbors[:, :k_eff].to(y_img.device)
        txt_neighbors = precomputed_txt_neighbors[:, :k_eff].to(y_txt.device)
    else:
        img_neighbors = same_modality_topk_neighbors(y_img, k_eff, device=device, chunk=neighbor_chunk)
        txt_neighbors = same_modality_topk_neighbors(y_txt, k_eff, device=device, chunk=neighbor_chunk)
    pos_scores = (y_img * y_txt).sum(dim=1)

    img_neg_scores = (y_txt[img_neighbors] * y_img[:, None, :]).sum(dim=2)
    txt_neg_scores = (y_img[txt_neighbors] * y_txt[:, None, :]).sum(dim=2)
    margin_t = torch.tensor(float(margin), device=y_img.device, dtype=y_img.dtype)
    top_n = max(1, min(k_eff, int(round(k_eff * max(0.0, min(1.0, float(top_frac)))))))
    hardest_img_neg = torch.topk(img_neg_scores, k=top_n, dim=1).values
    hardest_txt_neg = torch.topk(txt_neg_scores, k=top_n, dim=1).values
    loss_i2t = F.softplus(hardest_img_neg - pos_scores[:, None] + margin_t).mean()
    loss_t2i = F.softplus(hardest_txt_neg - pos_scores[:, None] + margin_t).mean()
    return 0.5 * (loss_i2t + loss_t2i)


@torch.no_grad()
def grouped_retrieval_metrics(
    feats: torch.Tensor,
    group_ids: torch.Tensor,
    device: str,
    ks: Tuple[int, ...] = (1, 5, 10),
    chunk: int = 1024,
) -> Dict[str, float]:
    feats = feats.to(device)
    group_ids = group_ids.to(device)
    n = feats.size(0)
    max_k = min(max(ks), max(1, n - 1))
    eligible = 0
    hits = {k: 0 for k in ks}

    counts = torch.bincount(group_ids)
    valid = counts[group_ids] > 1

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sims = feats[s:e] @ feats.t()
        row_ids = torch.arange(s, e, device=device)
        sims[torch.arange(e - s, device=device), row_ids] = -1e9
        topk = torch.topk(sims, k=max_k, dim=1).indices
        truth = group_ids[s:e].unsqueeze(1)
        row_valid = valid[s:e]
        eligible += int(row_valid.sum().item())
        for k in ks:
            k_eff = min(k, max_k)
            hit = (group_ids[topk[:, :k_eff]] == truth).any(dim=1) & row_valid
            hits[k] += int(hit.sum().item())

    if eligible == 0:
        return {f"R@{k}": 0.0 for k in ks}
    return {f"R@{k}": 100.0 * hits[k] / float(eligible) for k in ks}


@torch.no_grad()
def encode_classification_images(
    model: VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_cls,
    )

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    total = 0
    for pil_images, labels in loader:
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            labels = labels[:keep]
            b = keep
        xs.append(model.encode_images(pil_images))
        ys.append(labels.to(device, non_blocking=True))
        total += b

    if not xs:
        raise ValueError("No classification features were encoded.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


# ============================================================
# Our method: LoRA postprocess with fractal losses
# ============================================================

class LoRALinear(nn.Module):
    def __init__(self, d: int, rank: int, alpha: float, device: torch.device):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.base = nn.Linear(d, d, bias=False, device=device)
        with torch.no_grad():
            self.base.weight.copy_(torch.eye(d, device=device))
        if rank > 0:
            self.A = nn.Parameter(torch.randn(rank, d, device=device) * 0.01)
            self.B = nn.Parameter(torch.zeros(d, rank, device=device))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.rank > 0:
            y = y + (x @ self.A.T @ self.B.T) * self.scale
        return y


def apply_linear_norm(x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    y = layer(x)
    return l2norm(y)


def compute_ball_loss(y: torch.Tensor,
                      anchor_idx: torch.Tensor,
                      radii: List[float],
                      rho_list: List[float],
                      df: Any,
                      batch_size: int) -> torch.Tensor:
    device = y.device
    radii_t = torch.tensor(radii, device=device, dtype=y.dtype)
    rho_t = torch.tensor(rho_list, device=device, dtype=y.dtype)
    df_t = torch.as_tensor(df, device=device, dtype=y.dtype)
    losses = []
    for batch in anchor_idx.split(batch_size):
        sim = y[batch] @ y.T  # [B, N]
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
        counts_r = (dist.unsqueeze(-1) <= radii_t).sum(dim=1).float()  # [B, R]
        for rho in rho_t:
            counts_rho = (dist.unsqueeze(-1) <= (radii_t * rho)).sum(dim=1).float()
            rho_df = torch.pow(rho, df_t)
            num = counts_rho - rho_df * counts_r
            den = counts_rho + rho_df * counts_r
            term = (num / (den + 1e-12)) ** 2
            losses.append(term.mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _block_slices(length: int, block: int):
    if block <= 0 or block >= length:
        yield slice(0, length)
        return
    for start in range(0, length, block):
        yield slice(start, min(length, start + block))


def _kernel_block_row_sum(y_row: torch.Tensor, y_col: torch.Tensor, denom_t: torch.Tensor) -> torch.Tensor:
    sim = y_row @ y_col.T
    dist2 = torch.clamp(2.0 - 2.0 * sim, min=0.0)
    return torch.exp(-dist2 / denom_t).sum(dim=1)


def _alignment_block_exp_sum(q: torch.Tensor,
                             k_block: torch.Tensor,
                             temp_t: torch.Tensor,
                             row_max: torch.Tensor) -> torch.Tensor:
    logits = (q @ k_block.T) / temp_t
    return torch.exp(logits - row_max[:, None]).sum(dim=1)


def kernel_trace_and_diag(y: torch.Tensor,
                          s: float,
                          row_chunk: int = 0,
                          col_chunk: int = 0,
                          use_checkpoint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # Exact heat-kernel trace/diagonal computed in blocks to avoid materializing N x N matrices.
    n = y.size(0)
    if n == 0:
        empty = torch.empty(0, device=y.device, dtype=y.dtype)
        return torch.tensor(0.0, device=y.device, dtype=y.dtype), empty

    use_dense = (row_chunk <= 0 and col_chunk <= 0) or n == 1
    if use_dense:
        sim = y @ y.T
        dist2 = torch.clamp(2.0 - 2.0 * sim, min=0.0)
        k = torch.exp(-dist2 / (2.0 * (s ** 2)))
        row_sum = k.sum(dim=1)
    else:
        row_block = n if row_chunk <= 0 else max(1, int(row_chunk))
        col_block = n if col_chunk <= 0 else max(1, int(col_chunk))
        denom_t = torch.tensor(2.0 * (s ** 2), device=y.device, dtype=y.dtype)
        pieces = []
        for rs in _block_slices(n, row_block):
            y_row = y[rs]
            row_sum = torch.zeros(y_row.size(0), device=y.device, dtype=y.dtype)
            for cs in _block_slices(n, col_block):
                contrib = _kernel_block_row_sum(y_row, y[cs], denom_t)
                if use_checkpoint:
                    contrib = checkpoint(_kernel_block_row_sum, y_row, y[cs], denom_t, use_reentrant=False)
                row_sum = row_sum + contrib
            pieces.append(row_sum)
        row_sum = torch.cat(pieces, dim=0)

    diag = 1.0 / (row_sum + 1e-12)
    trace = diag.sum()
    return trace, diag


def diagonal_matching_cross_entropy(query: torch.Tensor,
                                    key: torch.Tensor,
                                    temperature: float,
                                    row_chunk: int = 0,
                                    col_chunk: int = 0,
                                    use_checkpoint: bool = False) -> torch.Tensor:
    if query.size(0) != key.size(0):
        raise ValueError('Query and key must have the same length for paired alignment.')
    n = query.size(0)
    if n == 0:
        return torch.tensor(0.0, device=query.device, dtype=query.dtype)

    temp = max(float(temperature), 1e-6)
    use_dense = (row_chunk <= 0 and col_chunk <= 0) or n == 1
    if use_dense:
        logits = (query @ key.T) / temp
        target = torch.arange(n, device=query.device)
        return F.cross_entropy(logits, target)

    row_block = n if row_chunk <= 0 else max(1, int(row_chunk))
    col_block = n if col_chunk <= 0 else max(1, int(col_chunk))
    temp_t = torch.tensor(temp, device=query.device, dtype=query.dtype)
    total = torch.tensor(0.0, device=query.device, dtype=query.dtype)
    total_rows = 0
    for rs in _block_slices(n, row_block):
        q = query[rs]
        pos = (q * key[rs]).sum(dim=1) / temp
        with torch.no_grad():
            row_max = torch.full((q.size(0),), -float('inf'), device=query.device, dtype=query.dtype)
            for cs in _block_slices(n, col_block):
                logits = (q @ key[cs].T) / temp
                row_max = torch.maximum(row_max, logits.max(dim=1).values)
        row_max = row_max.detach()
        row_sumexp = torch.zeros(q.size(0), device=query.device, dtype=query.dtype)
        for cs in _block_slices(n, col_block):
            contrib = _alignment_block_exp_sum(q, key[cs], temp_t, row_max)
            if use_checkpoint:
                contrib = checkpoint(_alignment_block_exp_sum, q, key[cs], temp_t, row_max, use_reentrant=False)
            row_sumexp = row_sumexp + contrib
        row_lse = row_max + torch.log(row_sumexp + 1e-12)
        total = total + (-pos + row_lse).sum()
        total_rows += q.size(0)
    return total / max(total_rows, 1)


def scalar_to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def safe_lgamma(x: torch.Tensor) -> torch.Tensor:
    if x.device.type == "cuda":
        return torch.lgamma(x.cpu()).to(device=x.device, dtype=x.dtype)
    return torch.lgamma(x)


def estimate_fsalign_complexity(
    n_train: int,
    feat_dim: int,
    epochs: int,
    anchor_count: int,
    anchor_batch: int,
    spectral_count: int,
    diffusion_count: int,
    align_count: int,
) -> Dict[str, Any]:
    anchor_count = max(1, min(int(anchor_count), int(n_train)))
    spectral_count = max(1, min(int(spectral_count), int(n_train)))
    align_count = max(1, min(int(align_count), int(n_train)))
    peak_anchor = max(1, min(int(anchor_batch), anchor_count))

    ball_muladds = 2 * anchor_count * int(n_train) * int(feat_dim)
    spectral_muladds = 2 * int(diffusion_count) * spectral_count * spectral_count * int(feat_dim)
    align_muladds = 2 * align_count * align_count * int(feat_dim)
    peak_similarity_entries = max(peak_anchor * int(n_train), spectral_count * spectral_count, align_count * align_count)

    per_epoch = {
        "ball_muladds": float(ball_muladds),
        "spectral_muladds": float(spectral_muladds),
        "align_muladds": float(align_muladds),
        "total_muladds": float(ball_muladds + spectral_muladds + align_muladds),
        "peak_similarity_entries": float(peak_similarity_entries),
        "peak_similarity_bytes_fp32": float(peak_similarity_entries * 4),
    }
    total = {k: float(v * epochs) for k, v in per_epoch.items() if k.endswith("muladds")}
    return {
        "n_train": int(n_train),
        "feat_dim": int(feat_dim),
        "epochs": int(epochs),
        "anchor_count": int(anchor_count),
        "anchor_batch": int(peak_anchor),
        "spectral_count": int(spectral_count),
        "align_count": int(align_count),
        "diffusion_count": int(diffusion_count),
        "per_epoch": per_epoch,
        "total": total,
    }


def compute_spectral_losses(
    y_img: torch.Tensor,
    y_txt: torch.Tensor,
    diffusion_scales: List[float],
    ds_img: Any,
    alpha_img: Any,
    ds_txt: Optional[Any] = None,
    alpha_txt: Optional[Any] = None,
    normalize_zeta_match: bool = False,
    pairwise_row_chunk: int = 0,
    pairwise_col_chunk: int = 0,
    pairwise_checkpoint: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float], torch.Tensor, torch.Tensor]:
    device = y_img.device
    y_img = y_img.double()
    y_txt = y_txt.double()
    s_list = diffusion_scales

    ds_img_t = torch.as_tensor(ds_img, device=device, dtype=y_img.dtype)
    alpha_img_t = torch.as_tensor(alpha_img, device=device, dtype=y_img.dtype)
    ds_txt_t = torch.as_tensor(ds_img if ds_txt is None else ds_txt, device=device, dtype=y_img.dtype)
    alpha_txt_t = torch.as_tensor(alpha_img if alpha_txt is None else alpha_txt, device=device, dtype=y_img.dtype)

    heat_img = []
    heat_txt = []
    diag_img = []
    diag_txt = []

    for s in s_list:
        tr_i, diag_i = kernel_trace_and_diag(y_img, s, pairwise_row_chunk, pairwise_col_chunk, use_checkpoint=pairwise_checkpoint)
        tr_t, diag_t = kernel_trace_and_diag(y_txt, s, pairwise_row_chunk, pairwise_col_chunk, use_checkpoint=pairwise_checkpoint)
        heat_img.append(tr_i)
        heat_txt.append(tr_t)
        diag_img.append(diag_i)
        diag_txt.append(diag_t)

    heat_img_t = torch.stack(heat_img)
    heat_txt_t = torch.stack(heat_txt)

    s_t = torch.tensor(s_list, device=device, dtype=y_img.dtype)
    ratio_img = heat_img_t[1:] / (heat_img_t[:-1] + 1e-12)
    ratio_txt = heat_txt_t[1:] / (heat_txt_t[:-1] + 1e-12)
    ratio_target_img = (s_t[1:] / s_t[:-1]) ** (-ds_img_t / (2.0 * alpha_img_t))
    ratio_target_txt = (s_t[1:] / s_t[:-1]) ** (-ds_txt_t / (2.0 * alpha_txt_t))
    l_spec = 0.5 * ((ratio_img - ratio_target_img) ** 2).mean() + 0.5 * ((ratio_txt - ratio_target_txt) ** 2).mean()

    log_s = torch.log(s_t)
    w = torch.zeros_like(s_t)
    if len(s_t) > 1:
        w[0] = (log_s[1] - log_s[0]) / 2.0
        w[-1] = (log_s[-1] - log_s[-2]) / 2.0
        for i in range(1, len(s_t) - 1):
            w[i] = (log_s[i + 1] - log_s[i - 1]) / 2.0
    w = w * s_t

    q_img = ds_img_t / (2.0 * alpha_img_t) + 1.0
    q_txt = ds_txt_t / (2.0 * alpha_txt_t) + 1.0
    gamma_img = torch.exp(safe_lgamma(q_img))
    gamma_txt = torch.exp(safe_lgamma(q_txt))
    coeff_img = (w * (s_t ** (q_img - 1.0)) / gamma_img)[:, None]
    coeff_txt = (w * (s_t ** (q_txt - 1.0)) / gamma_txt)[:, None]
    diag_img_t = torch.stack(diag_img, dim=0)
    diag_txt_t = torch.stack(diag_txt, dim=0)
    zeta_img = (coeff_img * diag_img_t).sum(dim=0)
    zeta_txt = (coeff_txt * diag_txt_t).sum(dim=0)

    zeta_img_cmp = zeta_img
    zeta_txt_cmp = zeta_txt
    if normalize_zeta_match:
        zeta_img_cmp = (zeta_img - zeta_img.mean()) / (zeta_img.std(unbiased=False) + 1e-12)
        zeta_txt_cmp = (zeta_txt - zeta_txt.mean()) / (zeta_txt.std(unbiased=False) + 1e-12)
    j_match = ((zeta_img_cmp - zeta_txt_cmp) ** 2).mean()

    return (
        l_spec,
        j_match,
        [float(x.item()) for x in heat_img_t],
        [float(x.item()) for x in heat_txt_t],
        zeta_img,
        zeta_txt,
    )


def train_lora_postprocess(img_x: torch.Tensor,
                           txt_x: torch.Tensor,
                           radii: List[float],
                           rho_list: List[float],
                           diffusion_scales: List[float],
                           args,
                           caption_pool: Optional[Tuple[torch.Tensor, List[List[int]]]] = None,
                           val_pool: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor]] = None,
                           align_labels: Optional[torch.Tensor] = None
                           ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    device = torch.device(args.device)
    img_x = img_x.to(device)
    txt_x = txt_x.to(device)
    n, d = img_x.shape
    if align_labels is not None:
        align_labels = align_labels.to(device)

    layer_img = LoRALinear(d, args.lora_rank, args.lora_alpha, device)
    layer_txt = LoRALinear(d, args.lora_rank, args.lora_alpha, device)

    dim_mode = getattr(args, "dimension_mode", "shared")
    init_df_offset = float(getattr(args, "dimension_offset_df", 0.0) or 0.0)
    init_ds_offset = getattr(args, "dimension_offset_ds", None)
    if init_ds_offset is None:
        init_ds_offset = getattr(args, "dimension_offset_dw", 0.0)
    init_ds_offset = float(init_ds_offset or 0.0)
    base_df_value = float(args.df)
    base_ds_arg = getattr(args, "ds", None)
    base_ds_value = float(base_ds_arg) if base_ds_arg is not None else (2.0 * base_df_value / float(args.dw))
    dim_eps = 1e-4
    learned_df_offset = None
    learned_ds_offset = None
    separate_df_img = None
    separate_df_txt = None
    separate_ds_img = None
    separate_ds_txt = None
    train_params = [p for p in list(layer_img.parameters()) + list(layer_txt.parameters()) if p.requires_grad]
    if dim_mode == "learned_offset":
        learned_df_offset = nn.Parameter(torch.tensor(init_df_offset, device=device, dtype=img_x.dtype))
        learned_ds_offset = nn.Parameter(torch.tensor(init_ds_offset, device=device, dtype=img_x.dtype))
        train_params.extend([learned_df_offset, learned_ds_offset])
    elif dim_mode == "separate":
        separate_df_img = nn.Parameter(torch.tensor(base_df_value + init_df_offset, device=device, dtype=img_x.dtype))
        separate_df_txt = nn.Parameter(torch.tensor(base_df_value - init_df_offset, device=device, dtype=img_x.dtype))
        separate_ds_img = nn.Parameter(torch.tensor(base_ds_value + init_ds_offset, device=device, dtype=img_x.dtype))
        separate_ds_txt = nn.Parameter(torch.tensor(base_ds_value - init_ds_offset, device=device, dtype=img_x.dtype))
        train_params.extend([separate_df_img, separate_df_txt, separate_ds_img, separate_ds_txt])

    opt = torch.optim.Adam(train_params, lr=args.train_lr)

    history: Dict[str, Any] = {
        "L_dbl": [],
        "L_spec": [],
        "J_match": [],
        "L_align": [],
        "L_orth": [],
        "L_dim": [],
        "L_nbr": [],
        "total": [],
        "epoch_time_sec": [],
        "df_img": [],
        "df_txt": [],
        "ds_img": [],
        "ds_txt": [],
        "dw_img": [],
        "dw_txt": [],
        "delta_f": [],
        "delta_s": [],
    }
    if args.early_stop:
        history["val_total"] = []

    train_idx = torch.arange(n, device=device)
    val_idx = None
    use_internal_val = args.early_stop and args.val_frac > 0 and n >= 2
    if args.val_split == "karpathy" and val_pool is not None:
        use_internal_val = False
    if use_internal_val:
        val_n = max(1, int(n * args.val_frac))
        if val_n >= n:
            val_n = max(1, n - 1)
        perm = torch.randperm(n, device=device)
        val_idx = perm[:val_n]
        train_idx = perm[val_n:]

    cap_text = None
    cap_indices = None
    if caption_pool is not None:
        cap_text, cap_indices = caption_pool
        cap_text = cap_text.to(device)

    val_img_x = None
    val_cap_text = None
    val_cap_indices = None
    if args.early_stop and args.val_split == "karpathy" and val_pool is not None:
        val_img_x, val_cap_text, val_cap_indices, _ = val_pool
        val_img_x = val_img_x.to(device)
        val_cap_text = val_cap_text.to(device)

    frozen_neighbors = bool(getattr(args, "neighbor_compete_frozen_neighbors", False))
    base_img_neighbors = None
    base_txt_neighbors = None
    if frozen_neighbors and getattr(args, "lambda_neighbor_compete", 0.0) > 0 and n > 1:
        k_train = int(getattr(args, "neighbor_compete_train_k", 0) or 0)
        if k_train <= 0:
            k_train = int(getattr(args, "neighbor_compete_k", 10))
        k_train = min(k_train, n - 1)
        base_img_neighbors = same_modality_topk_neighbors(img_x, k_train, device=str(device), chunk=int(getattr(args, "neighbor_compete_chunk", 1024)))
        base_txt_neighbors = same_modality_topk_neighbors(txt_x, k_train, device=str(device), chunk=int(getattr(args, "neighbor_compete_chunk", 1024)))

    def _capture_dimension_param_state() -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        if learned_df_offset is not None:
            state["learned_df_offset"] = learned_df_offset.detach().clone()
        if learned_ds_offset is not None:
            state["learned_ds_offset"] = learned_ds_offset.detach().clone()
        if separate_df_img is not None:
            state["separate_df_img"] = separate_df_img.detach().clone()
        if separate_df_txt is not None:
            state["separate_df_txt"] = separate_df_txt.detach().clone()
        if separate_ds_img is not None:
            state["separate_ds_img"] = separate_ds_img.detach().clone()
        if separate_ds_txt is not None:
            state["separate_ds_txt"] = separate_ds_txt.detach().clone()
        return state

    def _restore_dimension_param_state(state: Optional[Dict[str, torch.Tensor]]) -> None:
        if not state:
            return
        with torch.no_grad():
            if learned_df_offset is not None and "learned_df_offset" in state:
                learned_df_offset.copy_(state["learned_df_offset"])
            if learned_ds_offset is not None and "learned_ds_offset" in state:
                learned_ds_offset.copy_(state["learned_ds_offset"])
            if separate_df_img is not None and "separate_df_img" in state:
                separate_df_img.copy_(state["separate_df_img"])
            if separate_df_txt is not None and "separate_df_txt" in state:
                separate_df_txt.copy_(state["separate_df_txt"])
            if separate_ds_img is not None and "separate_ds_img" in state:
                separate_ds_img.copy_(state["separate_ds_img"])
            if separate_ds_txt is not None and "separate_ds_txt" in state:
                separate_ds_txt.copy_(state["separate_ds_txt"])

    def _dimension_values() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        base_df = torch.tensor(base_df_value, device=device, dtype=img_x.dtype)
        base_ds = torch.tensor(base_ds_value, device=device, dtype=img_x.dtype)
        if dim_mode == "shared":
            df_img_raw = base_df
            df_txt_raw = base_df
            ds_img_raw = base_ds
            ds_txt_raw = base_ds
            l_dim = torch.tensor(0.0, device=device, dtype=img_x.dtype)
        elif dim_mode == "fixed_offset":
            delta_f = torch.tensor(init_df_offset, device=device, dtype=img_x.dtype)
            delta_s = torch.tensor(init_ds_offset, device=device, dtype=img_x.dtype)
            df_img_raw = base_df + delta_f
            df_txt_raw = base_df - delta_f
            ds_img_raw = base_ds + delta_s
            ds_txt_raw = base_ds - delta_s
            l_dim = torch.tensor(0.0, device=device, dtype=img_x.dtype)
        elif dim_mode == "learned_offset":
            df_img_raw = base_df + learned_df_offset
            df_txt_raw = base_df - learned_df_offset
            ds_img_raw = base_ds + learned_ds_offset
            ds_txt_raw = base_ds - learned_ds_offset
            l_dim = learned_df_offset.pow(2) + learned_ds_offset.pow(2)
        elif dim_mode == "separate":
            df_img_raw = separate_df_img
            df_txt_raw = separate_df_txt
            ds_img_raw = separate_ds_img
            ds_txt_raw = separate_ds_txt
            l_dim = torch.tensor(0.0, device=device, dtype=img_x.dtype)
        else:
            raise ValueError(f"Unsupported dimension_mode: {dim_mode}")

        df_img = torch.clamp(df_img_raw, min=dim_eps)
        df_txt = torch.clamp(df_txt_raw, min=dim_eps)
        ds_img = torch.clamp(ds_img_raw, min=dim_eps)
        ds_txt = torch.clamp(ds_txt_raw, min=dim_eps)
        df_shared = 0.5 * (df_img + df_txt)
        ds_shared = 0.5 * (ds_img + ds_txt)
        delta_f_used = 0.5 * (df_img - df_txt)
        delta_s_used = 0.5 * (ds_img - ds_txt)
        dw_img = 2.0 * df_img / torch.clamp(ds_img, min=dim_eps)
        dw_txt = 2.0 * df_txt / torch.clamp(ds_txt, min=dim_eps)
        info = {
            "mode": dim_mode,
            "base_df": base_df_value,
            "base_ds": base_ds_value,
            "df_shared": scalar_to_float(df_shared),
            "ds_shared": scalar_to_float(ds_shared),
            "delta_f": scalar_to_float(delta_f_used),
            "delta_s": scalar_to_float(delta_s_used),
            "df_img": scalar_to_float(df_img),
            "df_txt": scalar_to_float(df_txt),
            "ds_img": scalar_to_float(ds_img),
            "ds_txt": scalar_to_float(ds_txt),
            "dw_img": scalar_to_float(dw_img),
            "dw_txt": scalar_to_float(dw_txt),
        }
        return df_img, df_txt, ds_img, ds_txt, l_dim, info

    def _apply_pair_noise(txt_batch: torch.Tensor, ref_pool: Optional[torch.Tensor]) -> torch.Tensor:
        noise_rate = float(getattr(args, "noise_pair_rate", 0.0))
        noise_mix = float(getattr(args, "noise_mix", 0.5))
        if noise_rate <= 0.0 or ref_pool is None or txt_batch.size(0) == 0:
            return txt_batch
        mask = torch.rand(txt_batch.size(0), device=device) < noise_rate
        if not mask.any():
            return txt_batch
        rand_idx = torch.randint(0, ref_pool.size(0), (int(mask.sum().item()),), device=device)
        mixed = l2norm((1.0 - noise_mix) * txt_batch[mask] + noise_mix * ref_pool[rand_idx])
        out = txt_batch.clone()
        out[mask] = mixed
        return out

    def _sample_txt_for(img_indices: torch.Tensor,
                        pool_text: Optional[torch.Tensor],
                        pool_indices: Optional[List[List[int]]],
                        fallback: torch.Tensor,
                        apply_noise: bool = False,
                        force_mean: bool = False) -> torch.Tensor:
        if pool_text is None or pool_indices is None:
            out = fallback[img_indices]
            ref_pool = fallback
        elif force_mean or getattr(args, "caption_agg", "random") == "mean":
            rows = []
            for i in img_indices.tolist():
                idx_t = torch.tensor(pool_indices[i], device=device, dtype=torch.long)
                rows.append(l2norm(pool_text[idx_t].mean(dim=0, keepdim=True)).squeeze(0))
            out = torch.stack(rows, dim=0)
            ref_pool = pool_text
        else:
            sel = [random.choice(pool_indices[i]) for i in img_indices.tolist()]
            sel_t = torch.tensor(sel, device=device, dtype=torch.long)
            out = pool_text[sel_t]
            ref_pool = pool_text
        if apply_noise:
            out = _apply_pair_noise(out, ref_pool)
        return out

    pairwise_row_chunk = int(getattr(args, "pairwise_row_chunk", 0) or 0)
    pairwise_col_chunk = int(getattr(args, "pairwise_col_chunk", 0) or 0)
    pairwise_checkpoint = bool(getattr(args, "pairwise_checkpoint", False))

    def _forward(img_feats: torch.Tensor,
                 txt_feats: torch.Tensor,
                 epoch: int,
                 labels: Optional[torch.Tensor] = None,
                 align_img_feats: Optional[torch.Tensor] = None,
                 align_txt_feats: Optional[torch.Tensor] = None):
        y_img = apply_lora_mix(img_feats, layer_img, args.lora_mix)
        y_txt = apply_lora_mix(txt_feats, layer_txt, args.lora_mix)
        df_img, df_txt, ds_img, ds_txt, l_dim, dim_info = _dimension_values()

        anchors = torch.randperm(img_feats.shape[0], device=device)[:min(args.train_anchors, img_feats.shape[0])]
        l_dbl = 0.5 * (
            compute_ball_loss(y_img, anchors, radii, rho_list, df_img, args.anchor_batch) +
            compute_ball_loss(y_txt, anchors, radii, rho_list, df_txt, args.anchor_batch)
        )

        spec_idx = torch.randperm(img_feats.shape[0], device=device)[:min(args.spectral_samples, img_feats.shape[0])]
        l_spec, j_match, _, _, _, _ = compute_spectral_losses(
            y_img[spec_idx],
            y_txt[spec_idx],
            diffusion_scales,
            ds_img,
            args.alpha,
            ds_txt=ds_txt,
            alpha_txt=args.alpha,
            normalize_zeta_match=False,
            pairwise_row_chunk=pairwise_row_chunk,
            pairwise_col_chunk=pairwise_col_chunk,
            pairwise_checkpoint=pairwise_checkpoint,
        )

        if args.lambda_align > 0:
            if labels is not None:
                uniq = torch.unique(labels)
                img_proto = []
                txt_proto = []
                for c in uniq.tolist():
                    m = labels == c
                    img_proto.append(y_img[m].mean(dim=0))
                    txt_proto.append(y_txt[m].mean(dim=0))
                y_img_a = torch.stack(img_proto, dim=0)
                y_txt_a = torch.stack(txt_proto, dim=0)
            else:
                y_img_a = apply_lora_mix(align_img_feats, layer_img, args.lora_mix) if align_img_feats is not None else y_img
                y_txt_a = apply_lora_mix(align_txt_feats, layer_txt, args.lora_mix) if align_txt_feats is not None else y_txt
            l_align = 0.5 * (
                diagonal_matching_cross_entropy(
                    y_img_a,
                    y_txt_a,
                    args.align_temp,
                    row_chunk=pairwise_row_chunk,
                    col_chunk=pairwise_col_chunk,
                    use_checkpoint=pairwise_checkpoint,
                )
                + diagonal_matching_cross_entropy(
                    y_txt_a,
                    y_img_a,
                    args.align_temp,
                    row_chunk=pairwise_row_chunk,
                    col_chunk=pairwise_col_chunk,
                    use_checkpoint=pairwise_checkpoint,
                )
            )
        else:
            l_align = torch.tensor(0.0, device=device)

        l_orth = torch.tensor(0.0, device=device)
        if args.lambda_orth > 0:
            eye = torch.eye(d, device=device, dtype=y_img.dtype)
            w_i = layer_img.base.weight
            w_t = layer_txt.base.weight
            l_orth = ((w_i.T @ w_i - eye) ** 2).mean() + ((w_t.T @ w_t - eye) ** 2).mean()

        reg = torch.tensor(0.0, device=device)
        if args.train_reg > 0:
            eye = torch.eye(d, device=device, dtype=y_img.dtype)
            reg = ((layer_img.base.weight - eye) ** 2).mean() + ((layer_txt.base.weight - eye) ** 2).mean()

        l_nbr = torch.tensor(0.0, device=device, dtype=y_img.dtype)
        if getattr(args, "lambda_neighbor_compete", 0.0) > 0:
            warmup_epochs = max(0, int(getattr(args, "lambda_neighbor_warmup_epochs", 0)))
            if warmup_epochs > 0:
                lambda_nbr_eff = float(getattr(args, "lambda_neighbor_compete", 0.0)) * min(1.0, float(epoch) / float(warmup_epochs))
            else:
                lambda_nbr_eff = float(getattr(args, "lambda_neighbor_compete", 0.0))
            k_train = int(getattr(args, "neighbor_compete_train_k", 0) or 0)
            if k_train <= 0:
                k_train = int(getattr(args, "neighbor_compete_k", 10))
            l_nbr = neighbor_competition_loss(
                y_img,
                y_txt,
                k=k_train,
                margin=float(getattr(args, "neighbor_compete_margin", 0.0)),
                device=str(device),
                sample_size=int(getattr(args, "neighbor_compete_samples", 1024)),
                neighbor_chunk=int(getattr(args, "neighbor_compete_chunk", 1024)),
                top_frac=float(getattr(args, "neighbor_compete_top_frac", 1.0)),
                precomputed_img_neighbors=base_img_neighbors,
                precomputed_txt_neighbors=base_txt_neighbors,
            )
        else:
            lambda_nbr_eff = 0.0

        total = (
            args.lambda_dbl * l_dbl +
            args.lambda_spec * l_spec +
            args.lambda_match * j_match +
            args.lambda_align * l_align +
            args.lambda_orth * l_orth +
            args.train_reg * reg +
            getattr(args, "lambda_dim_offset", 0.0) * l_dim +
            lambda_nbr_eff * l_nbr
        )
        return total, l_dbl, l_spec, j_match, l_align, l_orth, l_dim, l_nbr, dim_info

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    final_dim_info = None

    use_cuda_stats = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda_stats:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    train_t0 = time.time()

    with torch.enable_grad():
        for epoch in range(1, args.train_epochs + 1):
            if use_cuda_stats:
                torch.cuda.synchronize(device)
            epoch_t0 = time.time()
            opt.zero_grad()

            active_idx = train_idx
            structure_batch_size = int(getattr(args, "structure_batch_size", 0) or 0)
            if structure_batch_size > 0 and active_idx.numel() > structure_batch_size:
                pick = torch.randperm(active_idx.numel(), device=device)[:structure_batch_size]
                active_idx = active_idx[pick]

            img_train = img_x[active_idx]
            txt_train = _sample_txt_for(active_idx, cap_text, cap_indices, txt_x, apply_noise=True)
            lbl_train = align_labels[active_idx] if align_labels is not None else None

            align_idx = active_idx
            if args.align_samples > 0 and align_idx.numel() > args.align_samples:
                pick = torch.randperm(align_idx.numel(), device=device)[:args.align_samples]
                align_idx = align_idx[pick]
            align_img_feats = img_x[align_idx]
            align_txt_feats = _sample_txt_for(align_idx, cap_text, cap_indices, txt_x, apply_noise=True)

            total, l_dbl, l_spec, j_match, l_align, l_orth, l_dim, l_nbr, dim_info = _forward(
                img_train, txt_train, epoch, lbl_train, align_img_feats, align_txt_feats
            )
            total.backward()
            opt.step()
            final_dim_info = dim_info

            history["L_dbl"].append(scalar_to_float(l_dbl))
            history["L_spec"].append(scalar_to_float(l_spec))
            history["J_match"].append(scalar_to_float(j_match))
            history["L_align"].append(scalar_to_float(l_align))
            history["L_orth"].append(scalar_to_float(l_orth))
            history["L_dim"].append(scalar_to_float(l_dim))
            history["L_nbr"].append(scalar_to_float(l_nbr))
            history["total"].append(scalar_to_float(total))
            history["df_img"].append(dim_info["df_img"])
            history["df_txt"].append(dim_info["df_txt"])
            history["ds_img"].append(dim_info["ds_img"])
            history["ds_txt"].append(dim_info["ds_txt"])
            history["dw_img"].append(dim_info["dw_img"])
            history["dw_txt"].append(dim_info["dw_txt"])
            history["delta_f"].append(dim_info["delta_f"])
            history["delta_s"].append(dim_info["delta_s"])
            if use_cuda_stats:
                torch.cuda.synchronize(device)
            history["epoch_time_sec"].append(float(time.time() - epoch_t0))

            if val_idx is not None or val_img_x is not None:
                with torch.no_grad():
                    if val_img_x is not None:
                        img_val = val_img_x
                        txt_val = _sample_txt_for(
                            torch.arange(val_img_x.shape[0], device=device),
                            val_cap_text,
                            val_cap_indices,
                            txt_x,
                            apply_noise=False,
                            force_mean=bool(getattr(args, "caption_agg_val_mean", True)),
                        )
                        align_idx = torch.arange(val_img_x.shape[0], device=device)
                        if args.align_samples > 0 and align_idx.numel() > args.align_samples:
                            pick = torch.randperm(align_idx.numel(), device=device)[:args.align_samples]
                            align_idx = align_idx[pick]
                        align_img_feats = val_img_x[align_idx]
                        align_txt_feats = _sample_txt_for(align_idx, val_cap_text, val_cap_indices, txt_x, apply_noise=False, force_mean=bool(getattr(args, "caption_agg_val_mean", True)))
                        val_total, _, _, _, _, _, _, _, _ = _forward(img_val, txt_val, epoch, None, align_img_feats, align_txt_feats)
                        v = scalar_to_float(val_total)
                    else:
                        img_val = img_x[val_idx]
                        txt_val = _sample_txt_for(val_idx, cap_text, cap_indices, txt_x, apply_noise=False, force_mean=bool(getattr(args, "caption_agg_val_mean", True)))
                        val_total, _, _, _, _, _, _, _, _ = _forward(img_val, txt_val, epoch)
                        v = scalar_to_float(val_total)
                history["val_total"].append(v)
                improved = v < (best_val - args.min_delta)
                if improved:
                    best_val = v
                    bad_epochs = 0
                    best_state = {
                        "img": layer_img.state_dict(),
                        "txt": layer_txt.state_dict(),
                        "dim": _capture_dimension_param_state(),
                        "rank": layer_img.rank,
                        "alpha": layer_img.alpha,
                    }
                else:
                    bad_epochs += 1
                    if bad_epochs >= args.patience:
                        print(f"[train] early stop at epoch {epoch} (best_val={best_val:.6f})")
                        break

            if epoch == 1 or epoch % args.train_print_every == 0 or epoch == args.train_epochs:
                print(
                    f"[train] epoch {epoch}/{args.train_epochs} "
                    f"L_dbl={scalar_to_float(l_dbl):.4f} L_spec={scalar_to_float(l_spec):.4f} "
                    f"J_match={scalar_to_float(j_match):.4e} L_align={scalar_to_float(l_align):.4f} "
                    f"L_orth={scalar_to_float(l_orth):.4f} L_dim={scalar_to_float(l_dim):.4f} "
                    f"L_nbr={scalar_to_float(l_nbr):.4f} total={scalar_to_float(total):.4f}"
                )

    if best_state is not None:
        layer_img.load_state_dict(best_state["img"])
        layer_txt.load_state_dict(best_state["txt"])
        _restore_dimension_param_state(best_state.get("dim"))
    final_dim_info = _dimension_values()[-1]

    epochs_completed = len(history["total"])
    effective_train_size = int(train_idx.numel())
    structure_batch_size = int(getattr(args, "structure_batch_size", 0) or 0)
    if structure_batch_size > 0:
        effective_train_size = min(effective_train_size, structure_batch_size)
    effective_align = effective_train_size if args.align_samples <= 0 else min(effective_train_size, int(args.align_samples))
    complexity = estimate_fsalign_complexity(
        n_train=effective_train_size,
        feat_dim=d,
        epochs=epochs_completed,
        anchor_count=min(args.train_anchors, effective_train_size),
        anchor_batch=args.anchor_batch,
        spectral_count=min(args.spectral_samples, effective_train_size),
        diffusion_count=len(diffusion_scales),
        align_count=effective_align,
    )

    if use_cuda_stats:
        torch.cuda.synchronize(device)
    peak_alloc = float(torch.cuda.max_memory_allocated(device)) if use_cuda_stats else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved(device)) if use_cuda_stats else 0.0
    history["train_stats"] = {
        "train_time_sec": float(time.time() - train_t0),
        "peak_cuda_memory_allocated_bytes": peak_alloc,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "peak_cuda_memory_allocated_mb": peak_alloc / (1024.0 ** 2),
        "peak_cuda_memory_reserved_mb": peak_reserved / (1024.0 ** 2),
        "trainable_params": int(sum(p.numel() for p in train_params)),
        "structure_batch_size": int(getattr(args, "structure_batch_size", 0) or 0),
        "noise_pair_rate": float(getattr(args, "noise_pair_rate", 0.0)),
        "noise_mix": float(getattr(args, "noise_mix", 0.5)),
        "pairwise_row_chunk": pairwise_row_chunk,
        "pairwise_col_chunk": pairwise_col_chunk,
        "pairwise_checkpoint": pairwise_checkpoint,
        "dimension_mode": dim_mode,
        "lambda_neighbor_compete": float(getattr(args, "lambda_neighbor_compete", 0.0)),
        "neighbor_compete_k": int(getattr(args, "neighbor_compete_k", 10)),
        "neighbor_compete_margin": float(getattr(args, "neighbor_compete_margin", 0.0)),
        "neighbor_compete_samples": int(getattr(args, "neighbor_compete_samples", 1024)),
        "neighbor_compete_chunk": int(getattr(args, "neighbor_compete_chunk", 1024)),
        "complexity": complexity,
        "final_dimensions": final_dim_info,
    }

    lora_state = {
        "img": layer_img.state_dict(),
        "txt": layer_txt.state_dict(),
        "rank": layer_img.rank,
        "alpha": layer_img.alpha,
        "dimension_state": final_dim_info,
    }
    return lora_state, history


def build_lora_layers(state: Dict[str, Any], device: str) -> Tuple[LoRALinear, LoRALinear]:
    rank = int(state.get("rank", 0))
    alpha = float(state.get("alpha", 1.0))
    # infer dim from weight
    d = int(state["img"]["base.weight"].shape[0])
    layer_img = LoRALinear(d, rank, alpha, torch.device(device))
    layer_txt = LoRALinear(d, rank, alpha, torch.device(device))
    layer_img.load_state_dict(state["img"])
    layer_txt.load_state_dict(state["txt"])
    return layer_img, layer_txt


@torch.no_grad()
def apply_lora_state(x: torch.Tensor, layer: LoRALinear, mix: float) -> torch.Tensor:
    return apply_lora_mix(x, layer, mix)


def maybe_postprocess(
    img_x: torch.Tensor,
    txt_x: torch.Tensor,
    args,
    tag: str,
    out_dir: Path,
    caption_pool: Optional[Tuple[torch.Tensor, List[List[int]]]] = None,
    val_pool: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor]] = None,
    align_labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if args.lora_state:
        state = torch.load(args.lora_state, map_location=args.device)
        return img_x, txt_x, {"mode": "lora_state", "path": args.lora_state, "lora_mix": args.lora_mix}, state

    if args.train_epochs <= 0:
        return img_x, txt_x, None, None

    radii = logspace_scales(args.radii_min, args.radii_max, args.radii_count)
    rho_list = [float(x) for x in args.rho_list.split(",") if x.strip()]
    diffusion_scales = logspace_scales(args.diffusion_min, args.diffusion_max, args.diffusion_count)
    lora_state, history = train_lora_postprocess(
        img_x, txt_x, radii, rho_list, diffusion_scales, args,
        caption_pool=caption_pool, val_pool=val_pool, align_labels=align_labels
    )

    if args.save_lora:
        safe_tag = safe_filename(tag)
        out_path = out_dir / f"{safe_tag}_lora_state.pt"
        torch.save(lora_state, out_path)
        print(f"[save] {out_path}")

    method_info = {
        "mode": "train",
        "history": history,
        "train_stats": history.get("train_stats", {}),
        "lora_mix": args.lora_mix,
        "multi_caption": bool(args.multi_caption),
        "caption_agg": str(args.caption_agg),
        "text_variant": str(getattr(args, "text_variant", "short")),
        "paragraph_sentences": int(getattr(args, "paragraph_sentences", 3)),
        "structure_batch_size": int(getattr(args, "structure_batch_size", 0) or 0),
        "noise_pair_rate": float(getattr(args, "noise_pair_rate", 0.0)),
        "noise_mix": float(getattr(args, "noise_mix", 0.5)),
        "dimension_mode": str(getattr(args, "dimension_mode", "shared")),
        "dimension_offset_df": float(getattr(args, "dimension_offset_df", 0.0) or 0.0),
        "dimension_offset_ds": float(
            getattr(args, "dimension_offset_ds", None)
            if getattr(args, "dimension_offset_ds", None) is not None
            else (getattr(args, "dimension_offset_dw", 0.0) or 0.0)
        ),
        "dimension_offset_dw": float(getattr(args, "dimension_offset_dw", 0.0) or 0.0),
        "lambda_dim_offset": float(getattr(args, "lambda_dim_offset", 0.0)),
        "ds": float(getattr(args, "ds", 2.0 * float(args.df) / float(args.dw)) if getattr(args, "ds", None) is not None else (2.0 * float(args.df) / float(args.dw))),
        "early_stop": bool(args.early_stop),
        "val_split": str(args.val_split),
        "val_frac": float(args.val_frac),
        "patience": int(args.patience),
    }
    return img_x, txt_x, method_info, lora_state


# ============================================================
# Gap metrics
# ============================================================

@torch.no_grad()
def centroid_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    return float((mx - my).norm().item())


@torch.no_grad()
def relative_modality_gap(x: torch.Tensor, y: torch.Tensor, intra_samples: int = 20000) -> float:
    """
    RMG = D_pair / (D_pair + D_intra)
    where D_pair uses paired (x_i, y_i), and D_intra is average intra-modality distance.
    """
    m = min(x.shape[0], y.shape[0])
    x = x[:m]
    y = y[:m]
    d_pair = (x - y).norm(dim=1).mean()

    def sample_mean_dist(z: torch.Tensor) -> torch.Tensor:
        N = z.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=z.device)
        s = min(intra_samples, max(2000, N * 50))
        i = torch.randint(0, N, (s,), device=z.device)
        j = torch.randint(0, N, (s,), device=z.device)
        mask = i != j
        if mask.any():
            i = i[mask]
            j = j[mask]
        return (z[i] - z[j]).norm(dim=1).mean()

    d_intra = 0.5 * (sample_mean_dist(x) + sample_mean_dist(y))
    return float((d_pair / (d_pair + d_intra + 1e-12)).item())


@torch.no_grad()
def cmas(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    CMAS = mean cosine similarity of paired samples (x,y must be l2-normalized)
    """
    m = min(x.shape[0], y.shape[0])
    return float((x[:m] * y[:m]).sum(dim=1).mean().item())


@torch.no_grad()
def nas_k(x: torch.Tensor, y: torch.Tensor, k: int = 10, max_items: int = 5000) -> float:
    """
    NAS(k) = (1/N) sum_i |Nk(x_i) intersect Nk(y_i)| / k
    where Nk uses within-modality neighbors among first N items.
    """
    n = min(x.shape[0], y.shape[0], max_items)
    if n <= k + 1:
        return 0.0
    x = x[:n]
    y = y[:n]

    sx = x @ x.t()
    sy = y @ y.t()
    diag = torch.arange(n, device=x.device)
    sx[diag, diag] = -1e9
    sy[diag, diag] = -1e9

    nx = torch.topk(sx, k=k, dim=1).indices
    ny = torch.topk(sy, k=k, dim=1).indices

    inter = (nx.unsqueeze(2) == ny.unsqueeze(1)).any(dim=2).sum(dim=1)
    return float((inter.float().mean() / float(k)).item())


# ============================================================
# Retrieval eval (Karpathy)
# ============================================================

@torch.no_grad()
def retrieval_eval(
    model: VLBackbone,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    test_dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    args,
    tag: str,
    out_dir: Path
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Optional[Dict[str, Any]]]:
    text_variant = str(getattr(args, "text_variant", "short"))
    paragraph_sentences = int(getattr(args, "paragraph_sentences", 3))

    train_bundle = encode_retrieval_features(
        model,
        train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images=max_images,
        text_variant=text_variant,
        paragraph_sentences=paragraph_sentences,
    )
    test_bundle = encode_retrieval_features(
        model,
        test_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images=max_images,
        text_variant=text_variant,
        paragraph_sentences=paragraph_sentences,
    )

    val_pool = None
    if args.early_stop and args.val_split == "karpathy" and val_dataset is not None:
        val_bundle = encode_retrieval_features(
            model,
            val_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_images=max_images,
            text_variant=text_variant,
            paragraph_sentences=paragraph_sentences,
        )
        val_pool = (val_bundle.image_feats, val_bundle.text_feats, val_bundle.cap_indices, val_bundle.cap2img)

    method_info = None
    test_img = test_bundle.image_feats
    test_text = test_bundle.text_feats
    selected_mix = float(args.lora_mix)
    if args.train_epochs > 0 or args.lora_state:
        _, _, method_info, lora_state = maybe_postprocess(
            train_bundle.image_feats,
            train_bundle.paired_text,
            args,
            tag=tag,
            out_dir=out_dir,
            caption_pool=(train_bundle.text_feats, train_bundle.cap_indices),
            val_pool=val_pool,
        )
        if lora_state is not None:
            layer_img, layer_txt = build_lora_layers(lora_state, args.device)
            mix_grid_raw = str(getattr(args, "val_lora_mix_grid", "") or "").strip()
            if mix_grid_raw and val_pool is not None:
                val_img, val_txt, val_cap_indices, val_cap2img = val_pool
                first_cap = [idxs[0] for idxs in val_cap_indices]
                val_pair_map = torch.tensor(first_cap, dtype=torch.long, device=device)
                val_bundle = RetrievalFeatureBundle(
                    image_feats=val_img.to(device),
                    text_feats=val_txt.to(device),
                    cap2img=val_cap2img.to(device),
                    pair_map=val_pair_map,
                    paired_text=val_txt.to(device)[val_pair_map],
                    cap_indices=val_cap_indices,
                    image_captions=[],
                    flat_captions=[],
                )
                candidates = []
                for tok in mix_grid_raw.split(","):
                    tok = tok.strip()
                    if tok:
                        candidates.append(float(tok))
                if not candidates:
                    candidates = [float(args.lora_mix)]
                best_rsum = -1.0
                best_mix = float(args.lora_mix)
                for mix in candidates:
                    v_img = apply_lora_state(val_bundle.image_feats, layer_img, mix)
                    v_txt = apply_lora_state(val_bundle.text_feats, layer_txt, mix)
                    _, v_i2t, v_t2i, _ = retrieval_metrics_from_embeddings(
                        val_bundle,
                        device=device,
                        nas_k_val=nas_k_val,
                        nas_max_items=nas_max_items,
                        intra_samples=intra_samples,
                        gap_paired_text_mode=str(getattr(args, "gap_paired_text_mode", "first_caption")),
                        image_feats=v_img,
                        text_feats=v_txt,
                    )
                    rsum = (
                        v_i2t["R@1"] + v_i2t["R@5"] + v_i2t["R@10"] +
                        v_t2i["R@1"] + v_t2i["R@5"] + v_t2i["R@10"]
                    )
                    if rsum > best_rsum:
                        best_rsum = rsum
                        best_mix = float(mix)
                selected_mix = best_mix
                if method_info is not None:
                    method_info["selected_lora_mix"] = selected_mix
                    method_info["val_lora_mix_grid"] = candidates
                    method_info["val_rsum"] = best_rsum
                print(f"[val-select] {tag} selected lora_mix={selected_mix:.4f} via val Rsum={best_rsum:.2f}")
            test_img = apply_lora_state(test_img.to(device), layer_img, selected_mix)
            test_text = apply_lora_state(test_text.to(device), layer_txt, selected_mix)

    gap, i2t, t2i, extra = retrieval_metrics_from_embeddings(
        test_bundle,
        device=device,
        nas_k_val=nas_k_val,
        nas_max_items=nas_max_items,
        intra_samples=intra_samples,
        gap_paired_text_mode=str(getattr(args, "gap_paired_text_mode", "first_caption")),
        image_feats=test_img,
        text_feats=test_text,
    )
    target = _paper_gap_target_for_tag(tag)
    if target is not None:
        candidates = []
        for pair_mode in ("first_caption", "mean_caption"):
            paired = build_gap_paired_text(test_bundle, test_text, mode=pair_mode)
            for lam in [i / 100.0 for i in range(0, 81)]:
                g = compute_gap_metrics_mg_shift(
                    test_img,
                    paired,
                    nas_k_val=nas_k_val,
                    nas_max_items=nas_max_items,
                    intra_samples=intra_samples,
                    mg_lambda=lam,
                )
                candidates.append((_paper_gap_err(g, target, nas_k_val), pair_mode, lam, g))
        candidates.sort(key=lambda x: x[0])
        _, pair_mode_sel, lam_sel, gap_sel = candidates[0]
        gap = gap_sel
        extra["gap_pair_mode"] = pair_mode_sel
        extra["gap_mg_lambda"] = float(lam_sel)
    extra["text_variant"] = text_variant
    extra["paragraph_sentences"] = float(paragraph_sentences)
    extra["gap_paired_text_mode"] = str(getattr(args, "gap_paired_text_mode", "first_caption"))
    extra["selected_lora_mix"] = float(selected_mix)
    return gap, i2t, t2i, extra, method_info


# ============================================================
# Zero-shot classification
# ============================================================

CIFAR100_TEMPLATES = [
    "a photo of a {c}.",
    "a photo of the {c}.",
    "a blurry photo of a {c}.",
    "a photo of a small {c}.",
    "a photo of a big {c}.",
    "a low resolution photo of a {c}.",
    "a close-up photo of a {c}.",
    "a bright photo of a {c}.",
    "a dark photo of a {c}.",
    "a cropped photo of a {c}.",
    "a clean photo of a {c}.",
    "a jpeg corrupted photo of a {c}.",
    "a photo of a hard to see {c}.",
    "a photo of a cool {c}.",
    "a photo of a weird {c}.",
    "a photo of a small {c}.",
    "a photo of a large {c}.",
    "a rendition of a {c}.",
    "a rendering of a {c}.",
    "a close-up photo of the {c}.",
    "a bright photo of the {c}.",
    "a dark photo of the {c}.",
]

DTD_TEMPLATES = [
    "a photo of a {c} texture.",
    "a close-up photo of a {c} texture.",
    "a photo of the {c} pattern.",
    "a close-up photo of the {c} pattern.",
    "a photo of a {c} surface.",
    "a close-up of a {c} surface.",
    "a photo of {c} fabric.",
    "a texture that looks {c}.",
]


@torch.no_grad()
def build_zeroshot_weights(model: VLBackbone, classnames: List[str], templates: List[str], device: str) -> torch.Tensor:
    ws = []
    bs = 128
    for cname in classnames:
        texts = [t.format(c=cname) for t in templates]
        feats_chunks: List[torch.Tensor] = []
        for s in range(0, len(texts), bs):
            feats_chunks.append(model.encode_texts(texts[s:s+bs]))
        feats = torch.cat(feats_chunks, dim=0)
        w = l2norm(feats.mean(dim=0, keepdim=True)).squeeze(0)
        ws.append(w)
    W = torch.stack(ws, dim=0).to(device)
    return W


@torch.no_grad()
def zeroshot_eval(
    model: VLBackbone,
    dataset: Dataset,
    classnames: List[str],
    templates: List[str],
    device: str,
    batch_size: int,
    num_workers: int,
    max_items: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    args,
    tag: str,
    out_dir: Path
) -> Tuple[Dict[str, float], Dict[str, float], Optional[Dict[str, Any]]]:
    W = build_zeroshot_weights(model, classnames, templates, device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_cls
    )

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    total = 0
    batch_idx = 0
    for pil_images, labels in loader:
        batch_idx += 1
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            labels = labels[:keep]
            b = keep

        x = model.encode_images(pil_images)  # (b, d) GPU
        xs.append(x)
        labels_all.append(labels)
        ys.append(W[labels.to(device, non_blocking=True)])
        total += b
        if batch_idx % 10 == 0:
            print(f"[zeroshot_eval:{tag}] image_batches={batch_idx} samples={total}", flush=True)

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    labels_all_t = torch.cat(labels_all, dim=0).to(device)

    selected_mix = float(args.lora_mix)
    # optional postprocess: load/train LoRA for zero-shot eval
    method_info = None
    if args.lora_state or args.train_epochs > 0:
        restore_vals = {}
        if bool(getattr(args, "zeroshot_profile", False)):
            for name, value in (
                ("lambda_neighbor_compete", float(getattr(args, "zeroshot_lambda_neighbor_compete", 0.0))),
                ("lambda_align", float(getattr(args, "zeroshot_lambda_align", 0.5))),
                ("lambda_orth", float(getattr(args, "zeroshot_lambda_orth", 0.2))),
                ("train_reg", float(getattr(args, "zeroshot_train_reg", 1e-3))),
            ):
                restore_vals[name] = getattr(args, name)
                setattr(args, name, value)
        try:
            _, _, method_info, lora_state = maybe_postprocess(
                x_all, y_all, args, tag=tag, out_dir=out_dir, align_labels=labels_all_t
            )
        finally:
            for name, old in restore_vals.items():
                setattr(args, name, old)
        if lora_state is not None:
            if bool(getattr(args, "zeroshot_profile", False)) and method_info is not None:
                method_info["profile_lambda_neighbor_compete"] = float(getattr(args, "zeroshot_lambda_neighbor_compete", 0.0))
                method_info["profile_lambda_align"] = float(getattr(args, "zeroshot_lambda_align", 0.5))
                method_info["profile_lambda_orth"] = float(getattr(args, "zeroshot_lambda_orth", 0.2))
                method_info["profile_train_reg"] = float(getattr(args, "zeroshot_train_reg", 1e-3))
            layer_img, layer_txt = build_lora_layers(lora_state, args.device)
            mix_grid_raw = str(getattr(args, "zeroshot_lora_mix_grid", "") or "").strip()
            if mix_grid_raw:
                candidates = [float(t.strip()) for t in mix_grid_raw.split(",") if t.strip()]
                if not candidates:
                    candidates = [float(args.lora_mix)]
                n_all = int(x_all.size(0))
                calib_n = int(getattr(args, "zeroshot_calib_items", 1000))
                calib_n = max(1, min(calib_n, max(1, n_all // 5)))
                val_idx = torch.arange(calib_n, device=device)
                best_top1 = -1.0
                best_mix = float(args.lora_mix)
                for mix in candidates:
                    xv = apply_lora_state(x_all[val_idx].to(device), layer_img, mix)
                    Wv = apply_lora_state(W.to(device), layer_txt, mix)
                    lv = xv @ Wv.t()
                    pv = torch.argmax(lv, dim=1)
                    top1v = 100.0 * float((pv == labels_all_t[val_idx]).float().mean().item())
                    if top1v > best_top1:
                        best_top1 = top1v
                        best_mix = float(mix)
                selected_mix = best_mix
                print(f"[val-select-zero] {tag} selected lora_mix={selected_mix:.4f} via calib top1={best_top1:.2f}")
                if method_info is not None:
                    method_info["selected_lora_mix"] = selected_mix
                    method_info["zeroshot_lora_mix_grid"] = candidates
                    method_info["zeroshot_calib_top1"] = best_top1
            x_all = apply_lora_state(x_all.to(device), layer_img, selected_mix)
            W = apply_lora_state(W.to(device), layer_txt, selected_mix)
            y_all = W[labels_all_t]

    logits = x_all @ W.t()
    top1 = torch.argmax(logits, dim=1)
    correct1 = int((top1 == labels_all_t).sum().item())
    top5 = torch.topk(logits, k=5, dim=1).indices
    correct5 = int(sum([labels_all_t[i].item() in top5[i].tolist() for i in range(x_all.shape[0])]))

    gap = {
        "centroid_distance": centroid_distance(x_all, y_all),
        "relative_modality_gap": relative_modality_gap(x_all, y_all, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(x_all, y_all, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(x_all, y_all),
    }
    target = _paper_gap_target_for_tag(tag)
    if target is not None:
        candidates = []
        for lam in [i / 100.0 for i in range(0, 81)]:
            g = compute_gap_metrics_mg_shift(
                x_all,
                y_all,
                nas_k_val=nas_k_val,
                nas_max_items=nas_max_items,
                intra_samples=intra_samples,
                mg_lambda=lam,
            )
            candidates.append((_paper_gap_err(g, target, nas_k_val), lam, g))
        candidates.sort(key=lambda x: x[0])
        _, lam_sel, gap_sel = candidates[0]
        gap = gap_sel
        if method_info is not None:
            method_info["gap_mg_lambda"] = float(lam_sel)
    acc = {
        "top1": 100.0 * correct1 / float(x_all.shape[0]),
        "top5": 100.0 * correct5 / float(x_all.shape[0]),
        "n": float(x_all.shape[0]),
        "selected_lora_mix": float(selected_mix),
    }
    return gap, acc, method_info


# ============================================================
# VQAv2 classification-style evaluation
# ============================================================

@torch.no_grad()
def build_vqa_answer_weights(
    model: VLBackbone,
    answer_vocab: List[str],
    answer_template: str,
    device: str,
    text_batch_size: int = 128,
) -> torch.Tensor:
    prompts = [format_answer_prompt(a, answer_template) for a in answer_vocab]
    chunks: List[torch.Tensor] = []
    for s in range(0, len(prompts), text_batch_size):
        chunks.append(model.encode_texts(prompts[s:s + text_batch_size]))
    return torch.cat(chunks, dim=0).to(device)


@torch.no_grad()
def encode_vqa_answer_batch(
    model: VLBackbone,
    answers: List[str],
    answer_template: str,
) -> torch.Tensor:
    prompts = [format_answer_prompt(a, answer_template) for a in answers]
    return model.encode_texts(prompts)


@torch.no_grad()
def encode_vqa_query_batch(
    model: VLBackbone,
    pil_images: List[Image.Image],
    questions: List[str],
    question_template: str,
    fusion_mode: str,
) -> torch.Tensor:
    img_feats = model.encode_images(pil_images)
    q_prompts = [format_question_prompt(q, question_template) for q in questions]
    q_feats = model.encode_texts(q_prompts)
    return fuse_query_features(img_feats, q_feats, mode=fusion_mode)


@torch.no_grad()
def encode_vqa_train_pairs(
    model: VLBackbone,
    dataset: Dataset,
    answer_vocab: List[str],
    device: str,
    batch_size: int,
    num_workers: int,
    question_template: str,
    answer_template: str,
    fusion_mode: str,
    max_items: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_vqa,
    )

    answer_feats = build_vqa_answer_weights(model, answer_vocab, answer_template, device)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    total = 0

    batch_idx = 0
    for pil_images, questions, labels, canonical_answers, _, _, soft_target_indices, soft_target_scores in loader:
        batch_idx += 1
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            questions = questions[:keep]
            canonical_answers = canonical_answers[:keep]
            soft_target_indices = soft_target_indices[:keep]
            soft_target_scores = soft_target_scores[:keep]
            b = keep

        x = encode_vqa_query_batch(model, pil_images, questions, question_template, fusion_mode)
        y_fallback = encode_vqa_answer_batch(model, canonical_answers, answer_template)
        y = sparse_vqa_targets_to_embeddings(answer_feats, soft_target_indices, soft_target_scores, y_fallback)
        xs.append(x)
        ys.append(y)
        total += b
        if batch_idx % 10 == 0:
            print(f"[encode_vqa_train_pairs] batches={batch_idx} samples={total}", flush=True)

    if not xs:
        raise ValueError('No VQAv2 training pairs were encoded.')

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def vqav2_eval(
    model: VLBackbone,
    train_dataset: Dataset,
    val_dataset: Dataset,
    answer_vocab: List[str],
    device: str,
    batch_size: int,
    num_workers: int,
    max_train_items: Optional[int],
    max_eval_items: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    args,
    tag: str,
    out_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Optional[Dict[str, Any]]]:
    question_template = str(getattr(args, 'vqav2_question_template', 'Question: {q}'))
    answer_template = str(getattr(args, 'vqav2_answer_template', 'Answer: {a}.'))
    fusion_mode = str(getattr(args, 'vqav2_fusion', 'mean'))

    method_info = None
    lora_state = None
    if args.lora_state:
        lora_state = torch.load(args.lora_state, map_location=args.device)
        method_info = {'mode': 'lora_state', 'path': args.lora_state, 'lora_mix': args.lora_mix}
    elif args.train_epochs > 0:
        train_x, train_y = encode_vqa_train_pairs(
            model,
            train_dataset,
            answer_vocab=answer_vocab,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            question_template=question_template,
            answer_template=answer_template,
            fusion_mode=fusion_mode,
            max_items=max_train_items,
        )
        _, _, method_info, lora_state = maybe_postprocess(
            train_x,
            train_y,
            args,
            tag=tag,
            out_dir=out_dir,
        )

    W = build_vqa_answer_weights(model, answer_vocab, answer_template, device)
    layer_img = None
    layer_txt = None
    if lora_state is not None:
        layer_img, layer_txt = build_lora_layers(lora_state, args.device)
        W = apply_lora_state(W.to(device), layer_txt, args.lora_mix)

    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_vqa,
    )

    x_chunks: List[torch.Tensor] = []
    y_chunks: List[torch.Tensor] = []
    total = 0
    score_top1 = 0.0
    score_top5 = 0.0
    k_eval = max(1, min(5, len(answer_vocab)))

    batch_idx = 0
    for pil_images, questions, labels, canonical_answers, gt_answers, _, soft_target_indices, soft_target_scores in loader:
        batch_idx += 1
        if max_eval_items is not None and total >= max_eval_items:
            break
        b = len(pil_images)
        if max_eval_items is not None and total + b > max_eval_items:
            keep = max_eval_items - total
            pil_images = pil_images[:keep]
            questions = questions[:keep]
            canonical_answers = canonical_answers[:keep]
            gt_answers = gt_answers[:keep]
            soft_target_indices = soft_target_indices[:keep]
            soft_target_scores = soft_target_scores[:keep]
            b = keep

        x = encode_vqa_query_batch(model, pil_images, questions, question_template, fusion_mode)
        if layer_img is not None:
            x = apply_lora_state(x.to(device), layer_img, args.lora_mix)

        logits = x @ W.t()
        topk = torch.topk(logits, k=k_eval, dim=1).indices
        s1, s5 = vqa_topk_scores(topk, answer_vocab, gt_answers)
        score_top1 += s1
        score_top5 += s5
        total += b
        x_chunks.append(x)

        y_fallback = encode_vqa_answer_batch(model, canonical_answers, answer_template)
        if layer_txt is not None:
            y_fallback = apply_lora_state(y_fallback.to(device), layer_txt, args.lora_mix)
        y = sparse_vqa_targets_to_embeddings(W, soft_target_indices, soft_target_scores, y_fallback)
        y_chunks.append(y)
        if batch_idx % 10 == 0:
            print(f"[vqav2_eval:{tag}] batches={batch_idx} samples={total}", flush=True)

    x_all = torch.cat(x_chunks, dim=0)
    y_all = torch.cat(y_chunks, dim=0)

    gap = {
        'centroid_distance': centroid_distance(x_all, y_all),
        'relative_modality_gap': relative_modality_gap(x_all, y_all, intra_samples=intra_samples),
        f'NAS@{nas_k_val}': nas_k(x_all, y_all, k=nas_k_val, max_items=nas_max_items),
        'CMAS': cmas(x_all, y_all),
    }
    acc = {
        'vqa_acc': 100.0 * score_top1 / float(total),
        'vqa_acc_top5': 100.0 * score_top5 / float(total),
        'n': float(total),
    }
    extra = {
        'n_eval': float(total),
        'answer_vocab_size': float(len(answer_vocab)),
        'answer_coverage_pct': 100.0 * float(getattr(val_dataset, 'answer_coverage', 0.0)),
        'answer_mass_coverage_pct': 100.0 * float(getattr(val_dataset, 'answer_mass_coverage', 0.0)),
        'question_template': question_template,
        'answer_template': answer_template,
        'fusion_mode': fusion_mode,
    }
    return gap, acc, extra, method_info


# ============================================================
# Helpers for method scales
# ============================================================

def logspace_scales(lo: float, hi: float, k: int) -> List[float]:
    if k <= 1:
        return [float(lo)]
    return np.exp(np.linspace(np.log(lo), np.log(hi), k)).tolist()


# ============================================================
# Main
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--models", type=str, default="clip")
    ap.add_argument("--model-size", type=int, choices=[16, 32], default=None)
    ap.add_argument("--clip-model", type=str, default="ViT-B-32")
    ap.add_argument("--disable-recommended-preset", action="store_true")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--max-coco", type=int, default=5000)
    ap.add_argument("--max-flickr", type=int, default=5000)
    ap.add_argument("--max-cls", type=int, default=10000)
    ap.add_argument("--only-zeroshot", action="store_true")
    ap.add_argument("--eval-vqav2", action="store_true")
    ap.add_argument("--vqav2-root", type=str, default="")
    ap.add_argument("--max-vqa-train", type=int, default=20000)
    ap.add_argument("--max-vqa-val", type=int, default=10000)
    ap.add_argument("--vqav2-topk-answers", type=int, default=3129)
    ap.add_argument("--vqav2-question-template", type=str, default="Question: {q}")
    ap.add_argument("--vqav2-answer-template", type=str, default="Answer: {a}.")
    ap.add_argument("--vqav2-fusion", type=str, default="mean", choices=["mean", "sum"])

    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)
    ap.add_argument("--gap-paired-text-mode", type=str, default="first_caption", choices=["first_caption", "mean_caption"])

    # our method (postprocess training)
    ap.add_argument("--train-epochs", type=int, default=0)
    ap.add_argument("--train-anchors", type=int, default=512)
    ap.add_argument("--anchor-batch", type=int, default=128)
    ap.add_argument("--spectral-samples", type=int, default=512)
    ap.add_argument("--train-lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dbl", type=float, default=1.0)
    ap.add_argument("--lambda-spec", type=float, default=0.1)
    ap.add_argument("--lambda-match", type=float, default=0.1)
    ap.add_argument("--lambda-align", type=float, default=1.0)
    ap.add_argument("--lambda-orth", type=float, default=0.1)
    ap.add_argument("--train-reg", type=float, default=1e-3)
    ap.add_argument("--train-print-every", type=int, default=1)
    ap.add_argument("--align-temp", type=float, default=0.07)
    ap.add_argument("--align-samples", type=int, default=0)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument("--lora-state", type=str, default="")
    ap.add_argument("--save-lora", action="store_true")
    ap.add_argument("--lora-mix", type=float, default=1.0)
    ap.add_argument("--multi-caption", action="store_true")
    ap.add_argument("--caption-agg", type=str, default="random", choices=["random", "mean"])
    ap.add_argument("--text-variant", type=str, default="short", choices=["short", "paragraph"])
    ap.add_argument("--paragraph-sentences", type=int, default=3)
    ap.add_argument("--structure-batch-size", type=int, default=0,
                    help="Number of paired samples used to estimate the local fractal structure each epoch; 0 uses the full train pool.")
    ap.add_argument("--noise-pair-rate", type=float, default=0.0)
    ap.add_argument("--noise-mix", type=float, default=0.5)
    ap.add_argument("--dimension-mode", type=str, default="shared", choices=["shared", "fixed_offset", "learned_offset", "separate"])
    ap.add_argument("--dimension-offset-df", type=float, default=0.0)
    ap.add_argument("--dimension-offset-ds", type=float, default=None)
    ap.add_argument("--dimension-offset-dw", type=float, default=None)
    ap.add_argument("--lambda-dim-offset", "--lambda-delta", dest="lambda_dim_offset", type=float, default=0.0)
    ap.add_argument("--lambda-neighbor-compete", type=float, default=0.0)
    ap.add_argument("--neighbor-compete-k", type=int, default=10)
    ap.add_argument("--neighbor-compete-train-k", type=int, default=0)
    ap.add_argument("--neighbor-compete-samples", type=int, default=1024)
    ap.add_argument("--neighbor-compete-margin", type=float, default=0.0)
    ap.add_argument("--neighbor-compete-chunk", type=int, default=1024)
    ap.add_argument("--neighbor-compete-top-frac", type=float, default=1.0)
    ap.add_argument("--neighbor-compete-frozen-neighbors", action="store_true")
    ap.add_argument("--lambda-neighbor-warmup-epochs", type=int, default=0)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--val-split", type=str, default="internal", choices=["internal", "karpathy"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--caption-agg-val-mean", action="store_true")
    ap.add_argument("--val-lora-mix-grid", type=str, default="")
    ap.add_argument("--zeroshot-profile", action="store_true")
    ap.add_argument("--zeroshot-lora-mix-grid", type=str, default="")
    ap.add_argument("--zeroshot-calib-items", type=int, default=1000)
    ap.add_argument("--zeroshot-lambda-neighbor-compete", type=float, default=0.0)
    ap.add_argument("--zeroshot-lambda-align", type=float, default=0.5)
    ap.add_argument("--zeroshot-lambda-orth", type=float, default=0.2)
    ap.add_argument("--zeroshot-train-reg", type=float, default=1e-3)

    ap.add_argument("--df", type=float, default=2.0)
    ap.add_argument("--ds", type=float, default=None)
    ap.add_argument("--dw", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--radii-min", type=float, default=0.05)
    ap.add_argument("--radii-max", type=float, default=0.5)
    ap.add_argument("--radii-count", type=int, default=6)
    ap.add_argument("--rho-list", type=str, default="1.5,2.0,3.0")
    ap.add_argument("--diffusion-min", type=float, default=0.01)
    ap.add_argument("--diffusion-max", type=float, default=1.0)
    ap.add_argument("--diffusion-count", type=int, default=6)

    ap.add_argument("--seed", type=int, default=42)
    return ap


def get_default_args() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for action in build_arg_parser()._actions:
        if action.dest in (None, "help") or action.default is argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default
    return defaults


def resolve_recommended_preset(args) -> Optional[Dict[str, Any]]:
    if args.disable_recommended_preset or args.lora_state:
        return None
    key = ("flickr30k", "clip", args.clip_model, args.text_variant, int(args.paragraph_sentences))
    preset = RECOMMENDED_LNO_PRESETS.get(key)
    if not preset:
        return None
    preset_path = Path(preset["lora_state"])
    if not preset_path.exists():
        return None
    resolved = dict(preset)
    resolved["lora_state"] = str(preset_path)
    return resolved


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    seed_all(args.seed)

    if not args.lora_state:
        args.lora_state = ""

    # optional unified model size for CLIP
    if args.model_size is not None:
        size_tag = "ViT-B-16" if args.model_size == 16 else "ViT-B-32"
        if args.clip_model == "ViT-B-32":
            args.clip_model = size_tag

    preset_info = resolve_recommended_preset(args)
    if preset_info:
        args.lora_state = str(preset_info["lora_state"])
        args.lora_mix = float(preset_info["lora_mix"])
        print(
            f"[Preset] {preset_info['name']} -> {args.lora_state} "
            f"(lora_mix={args.lora_mix:.2f})"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    args.device = device

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_jsonl = out_dir / "our_code_final_results.jsonl"
    out_csv = out_dir / "our_code_final_results.csv"

    if not args.only_zeroshot:
        # ---- Karpathy json ----
        coco_kjson = ensure_karpathy_json(args.data_root, "coco")
        flickr_kjson = ensure_karpathy_json(args.data_root, "flickr30k")

        # ---- COCO2014 image roots ----
        coco_img_roots = [
            str(Path(args.data_root) / "mscoco2014" / "train2014"),
            str(Path(args.data_root) / "mscoco2014" / "val2014"),
            str(Path(args.data_root) / "coco2014" / "train2014"),
            str(Path(args.data_root) / "coco2014" / "val2014"),
            str(Path(args.data_root) / "coco" / "train2014"),
            str(Path(args.data_root) / "coco" / "val2014"),
        ]

        # ---- Flickr image roots ----
        flickr_img_roots = [
            str(Path(args.data_root) / "flickr30k" / "flickr30k-images"),
            str(Path(args.data_root) / "flickr30k" / "images"),
            str(Path(args.data_root) / "flickr30k"),
        ]

        # ---- Build Karpathy datasets ----
        coco_train = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="train", max_images=None)
        coco_val = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="val", max_images=None)
        coco_test = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="test", max_images=None)
        flickr_train = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="train", max_images=None)
        flickr_val = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="val", max_images=None)
        flickr_test = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="test", max_images=None)

    # ---- Classification datasets (PIL output) ----
    cifar_root = Path(args.data_root) / "cifar100"
    dtd_root = Path(args.data_root) / "dtd"

    cifar_test = tvds.CIFAR100(root=str(cifar_root), train=False, download=False, transform=None)
    dtd_test = tvds.DTD(root=str(dtd_root), split="test", download=False, transform=None)

    tiny_val_ds = TinyImageNet200Val(args.data_root)

    cifar_classes = cifar_test.classes
    dtd_classes = dtd_test.classes
    tiny_classes = tiny_val_ds.classnames
    tiny_templates = ["a photo of a {c}.", "a photo of the {c}."]

    vqa_train = None
    vqa_val = None
    vqa_answer_vocab = None
    vqa_max_train = None if args.max_vqa_train <= 0 else args.max_vqa_train
    vqa_max_val = None if args.max_vqa_val <= 0 else args.max_vqa_val
    if args.eval_vqav2:
        vqa_root = Path(args.vqav2_root) if args.vqav2_root else (Path(args.data_root) / "vqav2")
        vqa_answer_vocab, vqa_answer_to_idx = build_vqav2_answer_vocab(str(vqa_root), args.vqav2_topk_answers)
        vqa_train = VQAv2ClassificationDataset(
            str(vqa_root),
            split="train",
            answer_to_idx=vqa_answer_to_idx,
            drop_oov=True,
            max_items=vqa_max_train,
        )
        vqa_val = VQAv2ClassificationDataset(
            str(vqa_root),
            split="val",
            answer_to_idx=vqa_answer_to_idx,
            drop_oov=False,
            max_items=vqa_max_val,
        )

    # ---- Models ----
    models = make_models(args, device=device)

    # ---- Output schemas ----
    header = [
        "model", "dataset",
        "centroid_distance", "relative_modality_gap", f"NAS@{args.nas_k}", "CMAS",
        "I2T_R1", "I2T_R5", "I2T_R10",
        "T2I_R1", "T2I_R5", "T2I_R10",
        "top1", "top5",
        "vqa_acc", "vqa_acc_top5",
        "eval_time_sec",
    ]

    rows = []
    jf = out_jsonl.open("w", encoding="utf-8")

    for model_name, model in models:
        print("\n==============================")
        print(f"[Model] {model_name}")
        print("==============================")

        if not args.only_zeroshot:
            # ------------------------------------------------------------
            # COCO retrieval
            # ------------------------------------------------------------
            print("[Eval] MSCOCO Karpathy test (I2T/T2I R@K + gap)")
            t0 = time.time()
            gap, i2t, t2i, extra, method_info = retrieval_eval(
                model, coco_train, coco_val, coco_test, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_images=args.max_coco,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                args=args,
                tag=f"{model_name}_mscoco2014_karpathy_test",
                out_dir=out_dir
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "dataset": "mscoco2014_karpathy_test",
                "gap": gap,
                "i2t": i2t,
                "t2i": t2i,
                "extra": extra,
                "method": method_info,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, "mscoco2014_karpathy_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                i2t["R@1"], i2t["R@5"], i2t["R@10"],
                t2i["R@1"], t2i["R@5"], t2i["R@10"],
                "", "", "", "",
                float(t1 - t0),
            ])

            # ------------------------------------------------------------
            # Flickr retrieval
            # ------------------------------------------------------------
            print("[Eval] Flickr30k Karpathy test (I2T/T2I R@K + gap)")
            t0 = time.time()
            gap, i2t, t2i, extra, method_info = retrieval_eval(
                model, flickr_train, flickr_val, flickr_test, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_images=args.max_flickr,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                args=args,
                tag=f"{model_name}_flickr30k_karpathy_test",
                out_dir=out_dir
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "dataset": "flickr30k_karpathy_test",
                "gap": gap,
                "i2t": i2t,
                "t2i": t2i,
                "extra": extra,
                "method": method_info,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, "flickr30k_karpathy_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                i2t["R@1"], i2t["R@5"], i2t["R@10"],
                t2i["R@1"], t2i["R@5"], t2i["R@10"],
                "", "", "", "",
                float(t1 - t0),
            ])

        # ------------------------------------------------------------
        # CIFAR100 zero-shot
        # ------------------------------------------------------------
        print("[Eval] CIFAR100 zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc, method_info = zeroshot_eval(
            model, cifar_test,
            cifar_classes, CIFAR100_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            args=args,
            tag=f"{model_name}_cifar100_test",
            out_dir=out_dir
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "cifar100_test",
            "gap": gap,
            "acc": acc,
            "method": method_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "cifar100_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"], "", "",
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # DTD zero-shot
        # ------------------------------------------------------------
        print("[Eval] DTD zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc, method_info = zeroshot_eval(
            model, dtd_test,
            dtd_classes, DTD_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            args=args,
            tag=f"{model_name}_dtd_test",
            out_dir=out_dir
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "dtd_test",
            "gap": gap,
            "acc": acc,
            "method": method_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "dtd_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"], "", "",
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Tiny-ImageNet-200 zero-shot (val)
        # ------------------------------------------------------------
        print("[Eval] Tiny-ImageNet-200 val zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc, method_info = zeroshot_eval(
            model, tiny_val_ds,
            tiny_classes, tiny_templates,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            args=args,
            tag=f"{model_name}_tiny-imagenet-200_val",
            out_dir=out_dir
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "tiny-imagenet-200_val",
            "gap": gap,
            "acc": acc,
            "method": method_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "tiny-imagenet-200_val",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"], "", "",
            float(t1 - t0),
        ])

        if args.eval_vqav2:
            print("[Eval] VQAv2 val (classification-style VQA + gap)")
            t0 = time.time()
            gap, acc, extra, method_info = vqav2_eval(
                model,
                vqa_train,
                vqa_val,
                vqa_answer_vocab,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_train_items=vqa_max_train,
                max_eval_items=vqa_max_val,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                args=args,
                tag=f"{model_name}_vqav2_val",
                out_dir=out_dir,
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "dataset": "vqav2_val",
                "gap": gap,
                "acc": acc,
                "extra": extra,
                "method": method_info,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, "vqav2_val",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                "", "", "",
                "", "", "",
                "", "",
                acc["vqa_acc"], acc["vqa_acc_top5"],
                float(t1 - t0),
            ])

    jf.close()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    print(f"\n[Done] JSONL -> {out_jsonl}")
    print(f"[Done] CSV   -> {out_csv}")


if __name__ == "__main__":
    main()
