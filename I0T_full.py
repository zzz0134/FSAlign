#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-2: I0T (Embedding Standardization) on top of CLIP / SigLIP / OpenCLIP
- SAME setting as baseline1_ground_truth_karpathy_mscoco2014.py:
  * Karpathy split for MSCOCO-2014 + Flickr30k retrieval
  * Zero-shot image classification on CIFAR100 / Tiny-ImageNet-200 / DTD
  * Gap metrics: centroid distance, Relative Modality Gap (RMG), NAS(k), CMAS
  * Record runtime per (model, dataset)
  * Save JSONL + CSV

I0T core:
  - Fit per-modality mean/std on TRAIN split (Karpathy train for retrieval)
  - Standardize embedding: z = (e - mean) / (std + eps)
  - Then l2-normalize: z = z / ||z||
  - Evaluate on TEST split with same metrics/protocol as baseline1.

Run:
  python baseline2_i0t_karpathy_mscoco2014.py \
    --data-root /work/was598/modilty_gap/tools/data \
    --out-dir /work/was598/modilty_gap/results/baseline2_i0t \
    --models clip,siglip,openclip \
    --batch-size 128 --num-workers 8 \
    --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
    --nas-k 10 \
    --fit-max-coco 20000 --fit-max-flickr 20000 --fit-max-cls 20000
"""

import os
import csv
import json
import time
import math
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import torchvision.datasets as tvds

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

# optional deps
try:
    import open_clip
except Exception:
    open_clip = None

try:
    from transformers import SiglipModel, SiglipProcessor
except Exception:
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

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

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
        for r in self.image_roots:
            if not r.exists():
                continue
            p = r / filename
            if p.exists():
                return p

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
    images = [b[0] for b in batch]
    caps_list = []
    for _, caps in batch:
        if isinstance(caps, (list, tuple)):
            caps_list.append([str(c) for c in caps])
        else:
            caps_list.append([str(caps)])
    return images, caps_list

def collate_cls(batch):
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
    def __init__(self, device: str):
        super().__init__(model_name="ViT-B-32", pretrained="openai", device=device)


class SigLIPWrapper(VLBackbone):
    def __init__(self, hf_name: str, device: str):
        super().__init__(device=device)
        assert SiglipModel is not None and SiglipProcessor is not None, "SigLIP requires transformers with SiglipModel."
        self.hf_name = hf_name
        self.model = SiglipModel.from_pretrained(hf_name).to(device).eval()
        self.proc = SiglipProcessor.from_pretrained(hf_name)

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
    keys = [k.strip().lower() for k in args.models.split(",") if k.strip()]
    out: List[Tuple[str, VLBackbone]] = []
    for k in keys:
        if k == "clip":
            out.append(("clip:ViT-B-32:openai", CLIPWrapper(device=device)))
        elif k == "openclip":
            out.append((f"open_clip:{args.openclip_model}:{args.openclip_pretrained}",
                        OpenCLIPWrapper(args.openclip_model, args.openclip_pretrained, device=device)))
        elif k == "siglip":
            out.append((f"siglip:{args.siglip_name}", SigLIPWrapper(args.siglip_name, device=device)))
        else:
            raise ValueError(f"Unknown model key: {k}")
    return out


# ============================================================
# I0T: per-modality standardization (fit mean/std) + l2norm
# ============================================================

class I0TStandardizer:
    """
    Fit per-modality mean/std on some reference set, then apply:
      z = (x - mean) / (std + eps)
      z = l2norm(z)
    All computations are on GPU (x expected on GPU).
    """
    def __init__(self, dim: int, device: str, eps: float = 1e-6):
        self.dim = dim
        self.device = device
        self.eps = eps
        self.img_mean = None
        self.img_std = None
        self.txt_mean = None
        self.txt_std = None

    @torch.no_grad()
    def _fit_stats(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feats: (N, D) on GPU
        mean = feats.mean(dim=0)
        var = feats.var(dim=0, unbiased=False)
        std = torch.sqrt(var + self.eps)
        return mean, std

    @torch.no_grad()
    def fit_retrieval_from_backbone(
        self,
        backbone: VLBackbone,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        max_images: Optional[int],
        text_bs: int = 256
    ):
        """
        Fit stats using a retrieval dataset (PIL, [captions]).
        Image stats: over image embeddings.
        Text stats: over ALL caption embeddings (flattened).
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_retrieval
        )

        img_chunks: List[torch.Tensor] = []
        all_caps: List[str] = []

        n_images = 0
        for pil_images, caps_list in loader:
            if max_images is not None and n_images >= max_images:
                break
            b = len(pil_images)
            if max_images is not None and n_images + b > max_images:
                keep = max_images - n_images
                pil_images = pil_images[:keep]
                caps_list = caps_list[:keep]
                b = keep

            feats = backbone.encode_images(pil_images)  # (b,d) GPU
            img_chunks.append(feats)

            for i in range(b):
                caps = caps_list[i]
                for c in caps:
                    all_caps.append(c)

            n_images += b

        img_feats = torch.cat(img_chunks, dim=0) if len(img_chunks) > 0 else torch.empty(0, self.dim, device=self.device)

        # encode texts in chunks
        txt_chunks: List[torch.Tensor] = []
        n_caps = len(all_caps)
        for s in range(0, n_caps, text_bs):
            tf = backbone.encode_texts(all_caps[s:s+text_bs])  # (bs,d) GPU
            txt_chunks.append(tf)
        txt_feats = torch.cat(txt_chunks, dim=0) if len(txt_chunks) > 0 else torch.empty(0, self.dim, device=self.device)

        self.img_mean, self.img_std = self._fit_stats(img_feats)
        self.txt_mean, self.txt_std = self._fit_stats(txt_feats)

    @torch.no_grad()
    def fit_classification_from_backbone(
        self,
        backbone: VLBackbone,
        dataset: Dataset,
        classnames: List[str],
        templates: List[str],
        batch_size: int,
        num_workers: int,
        max_items: Optional[int],
        text_bs: int = 128
    ):
        """
        Fit stats for classification scenario:
          - image stats: over dataset images embeddings (max_items)
          - text stats: over zeroshot class weights construction pool
                       (we aggregate all template prompts for all classes).
        """
        # image stats
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_cls
        )

        img_chunks: List[torch.Tensor] = []
        total = 0
        for pil_images, labels in loader:
            if max_items is not None and total >= max_items:
                break
            b = len(pil_images)
            if max_items is not None and total + b > max_items:
                keep = max_items - total
                pil_images = pil_images[:keep]
                b = keep

            feats = backbone.encode_images(pil_images)  # (b,d) GPU
            img_chunks.append(feats)
            total += b

        img_feats = torch.cat(img_chunks, dim=0) if len(img_chunks) > 0 else torch.empty(0, self.dim, device=self.device)
        self.img_mean, self.img_std = self._fit_stats(img_feats)

        # text stats: all prompts for all classes
        all_prompts: List[str] = []
        for cname in classnames:
            for t in templates:
                all_prompts.append(t.format(c=cname))

        txt_chunks: List[torch.Tensor] = []
        for s in range(0, len(all_prompts), text_bs):
            tf = backbone.encode_texts(all_prompts[s:s+text_bs])
            txt_chunks.append(tf)
        txt_feats = torch.cat(txt_chunks, dim=0) if len(txt_chunks) > 0 else torch.empty(0, self.dim, device=self.device)
        self.txt_mean, self.txt_std = self._fit_stats(txt_feats)

    @torch.no_grad()
    def transform_image(self, x: torch.Tensor) -> torch.Tensor:
        assert self.img_mean is not None and self.img_std is not None, "I0T image stats not fitted."
        z = (x - self.img_mean) / (self.img_std + self.eps)
        return l2norm(z)

    @torch.no_grad()
    def transform_text(self, y: torch.Tensor) -> torch.Tensor:
        assert self.txt_mean is not None and self.txt_std is not None, "I0T text stats not fitted."
        z = (y - self.txt_mean) / (self.txt_std + self.eps)
        return l2norm(z)


class I0TBackbone(VLBackbone):
    """
    Wrap a VLBackbone and apply I0T standardization after encoding.
    """
    def __init__(self, base: VLBackbone, stdizer: I0TStandardizer, device: str):
        super().__init__(device=device)
        self.base = base
        self.stdizer = stdizer

    @property
    def dim(self) -> int:
        return self.base.dim

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        x = self.base.encode_images(pil_images)  # already l2norm in base
        x = self.stdizer.transform_image(x)
        return x

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        y = self.base.encode_texts(texts)  # already l2norm in base
        y = self.stdizer.transform_text(y)
        return y


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
    NAS(k) = (1/N) sum_i |Nk(x_i) ∩ Nk(y_i)| / k
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
# Retrieval eval (Karpathy) - same protocol as baseline1
# ============================================================

@torch.no_grad()
def retrieval_eval(
    model: VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Dataset item: (PIL, [captions])
    We compute:
      - image feats for images
      - text feats for all captions
      - i2t recall: image -> any caption that belongs to that image
      - t2i recall: caption -> its image
      - gap metrics using paired (image, first-caption) for each image
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_retrieval
    )

    image_feats_chunks: List[torch.Tensor] = []
    all_caps: List[str] = []
    cap2img: List[int] = []

    n_images = 0
    for pil_images, caps_list in loader:
        if max_images is not None and n_images >= max_images:
            break

        b = len(pil_images)
        if max_images is not None and n_images + b > max_images:
            keep = max_images - n_images
            pil_images = pil_images[:keep]
            caps_list = caps_list[:keep]
            b = keep

        feats = model.encode_images(pil_images)  # (b, d) on GPU
        image_feats_chunks.append(feats)

        for i in range(b):
            caps = caps_list[i]
            for c in caps:
                all_caps.append(c)
                cap2img.append(n_images + i)

        n_images += b

    image_feats = torch.cat(image_feats_chunks, dim=0)  # (Nimg, d) GPU
    n_caps = len(all_caps)

    # encode captions
    text_feats_chunks: List[torch.Tensor] = []
    bs_t = 256
    for s in range(0, n_caps, bs_t):
        tf = model.encode_texts(all_caps[s:s+bs_t])  # (bs, d) GPU
        text_feats_chunks.append(tf)
    text_feats = torch.cat(text_feats_chunks, dim=0)  # (Ncap, d) GPU

    # paired text = first caption per image
    first_cap = [-1] * image_feats.size(0)
    for cap_idx, img_idx in enumerate(cap2img):
        if first_cap[img_idx] < 0:
            first_cap[img_idx] = cap_idx
    pair_map = torch.tensor(first_cap, dtype=torch.long, device=device)
    paired_text = text_feats[pair_map]

    gap = {
        "centroid_distance": centroid_distance(image_feats, paired_text),
        "relative_modality_gap": relative_modality_gap(image_feats, paired_text, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(image_feats, paired_text, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(image_feats, paired_text),
    }

    cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)

    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = image_feats.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = image_feats[s:e] @ text_feats.t()  # GPU
            topk = torch.topk(sims, k=K, dim=1).indices
            img_ids = torch.arange(s, e, device=device).unsqueeze(1)
            mapped = cap2img_t[topk]
            hit = (mapped == img_ids).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Nimg)

    def recall_t2i(K: int) -> float:
        correct = 0
        Ncap = text_feats.size(0)
        chunk = 1024
        for s in range(0, Ncap, chunk):
            e = min(Ncap, s + chunk)
            sims = text_feats[s:e] @ image_feats.t()  # GPU
            topk = torch.topk(sims, k=K, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Ncap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}
    extra = {"n_images": float(image_feats.size(0)), "n_captions": float(text_feats.size(0))}
    return gap, i2t, t2i, extra


# ============================================================
# Zero-shot classification - same protocol as baseline1
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
]

DTD_TEMPLATES = [
    "a photo of a {c} texture.",
    "a close-up photo of a {c} texture.",
    "a photo of the {c} pattern.",
    "a close-up photo of the {c} pattern.",
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
    intra_samples: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    W = build_zeroshot_weights(model, classnames, templates, device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_cls
    )

    correct1 = 0
    correct5 = 0
    total = 0

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    for pil_images, labels in loader:
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            labels = labels[:keep]
            b = keep

        x = model.encode_images(pil_images)  # (b, d) GPU
        logits = x @ W.t()                   # (b, C) GPU

        top1 = torch.argmax(logits, dim=1).cpu()
        correct1 += int((top1 == labels).sum().item())

        top5 = torch.topk(logits, k=5, dim=1).indices.cpu()
        correct5 += int(sum([labels[i].item() in top5[i].tolist() for i in range(b)]))

        total += b

        xs.append(x)
        ys.append(W[labels.to(device, non_blocking=True)])

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)

    gap = {
        "centroid_distance": centroid_distance(x_all, y_all),
        "relative_modality_gap": relative_modality_gap(x_all, y_all, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(x_all, y_all, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(x_all, y_all),
    }
    acc = {"top1": 100.0 * correct1 / float(total), "top5": 100.0 * correct5 / float(total), "n": float(total)}
    return gap, acc


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
def fit_vqa_i0t_stats(
    stdizer: I0TStandardizer,
    backbone: VLBackbone,
    train_dataset: Dataset,
    answer_vocab: List[str],
    question_template: str,
    answer_template: str,
    fusion_mode: str,
    batch_size: int,
    num_workers: int,
    max_items: Optional[int],
):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_vqa,
    )

    query_chunks: List[torch.Tensor] = []
    total = 0
    for pil_images, questions, labels, canonical_answers, gt_answers, qids, _, _ in loader:
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            questions = questions[:keep]
            b = keep
        query_chunks.append(encode_vqa_query_batch(backbone, pil_images, questions, question_template, fusion_mode))
        total += b

    if not query_chunks:
        raise ValueError('No VQAv2 query features were encoded for I0T fitting.')

    query_feats = torch.cat(query_chunks, dim=0)
    stdizer.img_mean, stdizer.img_std = stdizer._fit_stats(query_feats)

    answer_feats = build_vqa_answer_weights(backbone, answer_vocab, answer_template, stdizer.device)
    stdizer.txt_mean, stdizer.txt_std = stdizer._fit_stats(answer_feats)


@torch.no_grad()
def vqav2_eval_i0t(
    backbone: VLBackbone,
    stdizer: I0TStandardizer,
    val_dataset: Dataset,
    answer_vocab: List[str],
    device: str,
    batch_size: int,
    num_workers: int,
    max_items: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    question_template: str,
    answer_template: str,
    fusion_mode: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    W_raw = build_vqa_answer_weights(backbone, answer_vocab, answer_template, device)
    W = stdizer.transform_text(W_raw)

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

    for pil_images, questions, labels, canonical_answers, gt_answers, _, soft_target_indices, soft_target_scores in loader:
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            questions = questions[:keep]
            canonical_answers = canonical_answers[:keep]
            gt_answers = gt_answers[:keep]
            soft_target_indices = soft_target_indices[:keep]
            soft_target_scores = soft_target_scores[:keep]
            b = keep

        x_raw = encode_vqa_query_batch(backbone, pil_images, questions, question_template, fusion_mode)
        x = stdizer.transform_image(x_raw)
        logits = x @ W.t()
        topk = torch.topk(logits, k=k_eval, dim=1).indices
        s1, s5 = vqa_topk_scores(topk, answer_vocab, gt_answers)
        score_top1 += s1
        score_top5 += s5
        total += b
        x_chunks.append(x)

        y_fallback = stdizer.transform_text(encode_vqa_answer_batch(backbone, canonical_answers, answer_template))
        y_chunks.append(sparse_vqa_targets_to_embeddings(W, soft_target_indices, soft_target_scores, y_fallback))

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
    return gap, acc, extra


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--models", type=str, default="clip,siglip,openclip")
    ap.add_argument("--openclip-model", type=str, default="ViT-B-32")
    ap.add_argument("--openclip-pretrained", type=str, default="openai")
    ap.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--max-coco", type=int, default=5000)
    ap.add_argument("--max-flickr", type=int, default=5000)
    ap.add_argument("--max-cls", type=int, default=10000)

    # I0T fit limits
    ap.add_argument("--fit-max-coco", type=int, default=20000)
    ap.add_argument("--fit-max-flickr", type=int, default=20000)
    ap.add_argument("--fit-max-cls", type=int, default=20000)
    ap.add_argument("--eval-vqav2", action="store_true")
    ap.add_argument("--vqav2-root", type=str, default="")
    ap.add_argument("--fit-max-vqa", type=int, default=20000)
    ap.add_argument("--max-vqa-val", type=int, default=10000)
    ap.add_argument("--vqav2-topk-answers", type=int, default=3129)
    ap.add_argument("--vqav2-question-template", type=str, default="Question: {q}")
    ap.add_argument("--vqav2-answer-template", type=str, default="Answer: {a}.")
    ap.add_argument("--vqav2-fusion", type=str, default="mean", choices=["mean", "sum"])

    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_jsonl = out_dir / "baseline2_i0t_results.jsonl"
    out_csv = out_dir / "baseline2_i0t_results.csv"

    # ---- Karpathy json ----
    coco_kjson = ensure_karpathy_json(args.data_root, "coco")
    flickr_kjson = ensure_karpathy_json(args.data_root, "flickr30k")

    # ---- COCO2014 image roots (DIRECT MSCOCO-2014) ----
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

    # ---- Build Karpathy TRAIN + TEST datasets (for I0T fit + eval) ----
    coco_train = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="train", max_images=None)
    coco_test  = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="test",  max_images=None)

    flickr_train = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="train", max_images=None)
    flickr_test  = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="test",  max_images=None)

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
    vqa_fit_max = None if args.fit_max_vqa <= 0 else args.fit_max_vqa
    vqa_max_val = None if args.max_vqa_val <= 0 else args.max_vqa_val
    if args.eval_vqav2:
        vqa_root = Path(args.vqav2_root) if args.vqav2_root else (Path(args.data_root) / "vqav2")
        vqa_answer_vocab, vqa_answer_to_idx = build_vqav2_answer_vocab(str(vqa_root), args.vqav2_topk_answers)
        vqa_train = VQAv2ClassificationDataset(
            str(vqa_root),
            split="train",
            answer_to_idx=vqa_answer_to_idx,
            drop_oov=False,
            max_items=vqa_fit_max,
        )
        vqa_val = VQAv2ClassificationDataset(
            str(vqa_root),
            split="val",
            answer_to_idx=vqa_answer_to_idx,
            drop_oov=False,
            max_items=vqa_max_val,
        )

    # ---- Models ----
    base_models = make_models(args, device=device)

    # ---- Output schemas ----
    header = [
        "model", "dataset",
        "centroid_distance", "relative_modality_gap", f"NAS@{args.nas_k}", "CMAS",
        "I2T_R1", "I2T_R5", "I2T_R10",
        "T2I_R1", "T2I_R5", "T2I_R10",
        "top1", "top5",
        "vqa_acc", "vqa_acc_top5",
        "fit_time_sec", "eval_time_sec", "total_time_sec",
    ]

    rows = []
    jf = out_jsonl.open("w", encoding="utf-8")

    for model_name, base_model in base_models:
        print("\n==============================")
        print(f"[Model] {model_name}")
        print("==============================")

        # ------------------------------------------------------------
        # I0T fit: COCO train
        # ------------------------------------------------------------
        std_coco = I0TStandardizer(dim=base_model.dim, device=device)
        print("[I0T Fit] MSCOCO Karpathy train (fit mean/std)")
        t_fit0 = time.time()
        std_coco.fit_retrieval_from_backbone(
            backbone=base_model,
            dataset=coco_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.fit_max_coco
        )
        t_fit1 = time.time()
        coco_fit_time = float(t_fit1 - t_fit0)
        coco_model = I0TBackbone(base=base_model, stdizer=std_coco, device=device)

        # ------------------------------------------------------------
        # COCO retrieval eval (test)
        # ------------------------------------------------------------
        print("[Eval] MSCOCO Karpathy test (I2T/T2I R@K + gap) [I0T]")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval(
            coco_model, coco_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_coco,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        coco_eval_time = float(t1 - t0)
        coco_total_time = coco_fit_time + coco_eval_time

        rec = {
            "model": model_name,
            "dataset": "mscoco2014_karpathy_test",
            "method": "I0T_standardization",
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "fit_time_sec": coco_fit_time,
            "eval_time_sec": coco_eval_time,
            "total_time_sec": coco_total_time,
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "mscoco2014_karpathy_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "", "", "",
            coco_fit_time, coco_eval_time, coco_total_time,
        ])

        # ------------------------------------------------------------
        # I0T fit: Flickr train
        # ------------------------------------------------------------
        std_flickr = I0TStandardizer(dim=base_model.dim, device=device)
        print("[I0T Fit] Flickr30k Karpathy train (fit mean/std)")
        t_fit0 = time.time()
        std_flickr.fit_retrieval_from_backbone(
            backbone=base_model,
            dataset=flickr_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.fit_max_flickr
        )
        t_fit1 = time.time()
        flickr_fit_time = float(t_fit1 - t_fit0)
        flickr_model = I0TBackbone(base=base_model, stdizer=std_flickr, device=device)

        # ------------------------------------------------------------
        # Flickr retrieval eval (test)
        # ------------------------------------------------------------
        print("[Eval] Flickr30k Karpathy test (I2T/T2I R@K + gap) [I0T]")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval(
            flickr_model, flickr_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_flickr,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        flickr_eval_time = float(t1 - t0)
        flickr_total_time = flickr_fit_time + flickr_eval_time

        rec = {
            "model": model_name,
            "dataset": "flickr30k_karpathy_test",
            "method": "I0T_standardization",
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "fit_time_sec": flickr_fit_time,
            "eval_time_sec": flickr_eval_time,
            "total_time_sec": flickr_total_time,
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "flickr30k_karpathy_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "", "", "",
            flickr_fit_time, flickr_eval_time, flickr_total_time,
        ])

        # ------------------------------------------------------------
        # CIFAR100: fit I0T stats on classification scenario, then eval
        # ------------------------------------------------------------
        std_cifar = I0TStandardizer(dim=base_model.dim, device=device)
        print("[I0T Fit] CIFAR100 test (fit mean/std for classification protocol)")
        t_fit0 = time.time()
        std_cifar.fit_classification_from_backbone(
            backbone=base_model,
            dataset=cifar_test,
            classnames=cifar_classes,
            templates=CIFAR100_TEMPLATES,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.fit_max_cls
        )
        t_fit1 = time.time()
        cifar_fit_time = float(t_fit1 - t_fit0)
        cifar_model = I0TBackbone(base=base_model, stdizer=std_cifar, device=device)

        print("[Eval] CIFAR100 zero-shot (top1/top5 + gap) [I0T]")
        t0 = time.time()
        gap, acc = zeroshot_eval(
            cifar_model, cifar_test,
            cifar_classes, CIFAR100_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        cifar_eval_time = float(t1 - t0)
        cifar_total_time = cifar_fit_time + cifar_eval_time

        rec = {
            "model": model_name,
            "dataset": "cifar100_test",
            "method": "I0T_standardization",
            "gap": gap,
            "acc": acc,
            "fit_time_sec": cifar_fit_time,
            "eval_time_sec": cifar_eval_time,
            "total_time_sec": cifar_total_time,
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "cifar100_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"], "", "",
            cifar_fit_time, cifar_eval_time, cifar_total_time,
        ])

        # ------------------------------------------------------------
        # DTD: fit I0T stats, then eval
        # ------------------------------------------------------------
        std_dtd = I0TStandardizer(dim=base_model.dim, device=device)
        print("[I0T Fit] DTD test (fit mean/std for classification protocol)")
        t_fit0 = time.time()
        std_dtd.fit_classification_from_backbone(
            backbone=base_model,
            dataset=dtd_test,
            classnames=dtd_classes,
            templates=DTD_TEMPLATES,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.fit_max_cls
        )
        t_fit1 = time.time()
        dtd_fit_time = float(t_fit1 - t_fit0)
        dtd_model = I0TBackbone(base=base_model, stdizer=std_dtd, device=device)

        print("[Eval] DTD zero-shot (top1/top5 + gap) [I0T]")
        t0 = time.time()
        gap, acc = zeroshot_eval(
            dtd_model, dtd_test,
            dtd_classes, DTD_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        dtd_eval_time = float(t1 - t0)
        dtd_total_time = dtd_fit_time + dtd_eval_time

        rec = {
            "model": model_name,
            "dataset": "dtd_test",
            "method": "I0T_standardization",
            "gap": gap,
            "acc": acc,
            "fit_time_sec": dtd_fit_time,
            "eval_time_sec": dtd_eval_time,
            "total_time_sec": dtd_total_time,
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "dtd_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"], "", "",
            dtd_fit_time, dtd_eval_time, dtd_total_time,
        ])

        # ------------------------------------------------------------
        # Tiny-ImageNet-200: fit I0T stats, then eval
        # ------------------------------------------------------------
        std_tiny = I0TStandardizer(dim=base_model.dim, device=device)
        print("[I0T Fit] Tiny-ImageNet-200 val (fit mean/std for classification protocol)")
        t_fit0 = time.time()
        std_tiny.fit_classification_from_backbone(
            backbone=base_model,
            dataset=tiny_val_ds,
            classnames=tiny_classes,
            templates=tiny_templates,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.fit_max_cls
        )
        t_fit1 = time.time()
        tiny_fit_time = float(t_fit1 - t_fit0)
        tiny_model = I0TBackbone(base=base_model, stdizer=std_tiny, device=device)

        print("[Eval] Tiny-ImageNet-200 val zero-shot (top1/top5 + gap) [I0T]")
        t0 = time.time()
        gap, acc = zeroshot_eval(
            tiny_model, tiny_val_ds,
            tiny_classes, tiny_templates,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        tiny_eval_time = float(t1 - t0)
        tiny_total_time = tiny_fit_time + tiny_eval_time

        rec = {
            "model": model_name,
            "dataset": "tiny-imagenet-200_val",
            "method": "I0T_standardization",
            "gap": gap,
            "acc": acc,
            "fit_time_sec": tiny_fit_time,
            "eval_time_sec": tiny_eval_time,
            "total_time_sec": tiny_total_time,
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "tiny-imagenet-200_val",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"], "", "",
            tiny_fit_time, tiny_eval_time, tiny_total_time,
        ])

        if args.eval_vqav2:
            std_vqa = I0TStandardizer(dim=base_model.dim, device=device)
            print("[I0T Fit] VQAv2 train (fit mean/std for query-answer protocol)")
            t_fit0 = time.time()
            fit_vqa_i0t_stats(
                stdizer=std_vqa,
                backbone=base_model,
                train_dataset=vqa_train,
                answer_vocab=vqa_answer_vocab,
                question_template=args.vqav2_question_template,
                answer_template=args.vqav2_answer_template,
                fusion_mode=args.vqav2_fusion,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_items=vqa_fit_max,
            )
            t_fit1 = time.time()
            vqa_fit_time = float(t_fit1 - t_fit0)

            print("[Eval] VQAv2 val (classification-style VQA + gap) [I0T]")
            t0 = time.time()
            gap, acc, extra = vqav2_eval_i0t(
                backbone=base_model,
                stdizer=std_vqa,
                val_dataset=vqa_val,
                answer_vocab=vqa_answer_vocab,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_items=vqa_max_val,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                question_template=args.vqav2_question_template,
                answer_template=args.vqav2_answer_template,
                fusion_mode=args.vqav2_fusion,
            )
            t1 = time.time()
            vqa_eval_time = float(t1 - t0)
            vqa_total_time = vqa_fit_time + vqa_eval_time

            rec = {
                "model": model_name,
                "dataset": "vqav2_val",
                "method": "I0T_standardization",
                "gap": gap,
                "acc": acc,
                "extra": extra,
                "fit_time_sec": vqa_fit_time,
                "eval_time_sec": vqa_eval_time,
                "total_time_sec": vqa_total_time,
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
                vqa_fit_time, vqa_eval_time, vqa_total_time,
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
