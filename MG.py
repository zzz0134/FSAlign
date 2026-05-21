#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-5: Mind-the-Gap (MG) post-hoc embedding shifting (Liang et al. 2022)
- SAME evaluation protocol as Baseline-1:
  - Karpathy split for MSCOCO + Flickr30k retrieval
  - Zero-shot image classification on CIFAR100 / Tiny-ImageNet-200 / DTD
  - Gap metrics: centroid distance, Relative Modality Gap (RMG), NAS(k), CMAS
  - Record runtime per (model, dataset)
  - Save JSONL + CSV

Core difference vs Baseline-1:
  Apply MG shift:
    Δ_gap = mean(x) - mean(y)
    x_shift = Normalize(x - λ Δ_gap)
    y_shift = Normalize(y + λ Δ_gap)
  where x,y are L2-normalized embeddings.

Run:
  python baseline5_mind_the_gap_karpathy_mscoco2014.py \
    --data-root /work/was598/modilty_gap/tools/data \
    --out-dir /work/was598/modilty_gap/results/baseline5 \
    --models clip,siglip,openclip \
    --batch-size 128 --num-workers 8 \
    --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
    --nas-k 10 \
    --mg-lambda 0.375
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
        assert SiglipModel is not None and SiglipProcessor is not None, "SigLIP requires transformers>=4.40 with SiglipModel."
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
# Gap metrics (same as baseline1)
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
# MG shift (Baseline-5 core)
# ============================================================

@torch.no_grad()
def mg_shift_pairwise(
    x: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mind-the-Gap shift:
      Δ_gap = mean(x) - mean(y)
      x' = Normalize(x - lam * Δ_gap)
      y' = Normalize(y + lam * Δ_gap)
    Inputs x,y are assumed l2-normalized already (as in CLIP-style).
    Returns (x_shift, y_shift, delta_gap).
    """
    m = min(x.shape[0], y.shape[0])
    x0 = x[:m]
    y0 = y[:m]
    delta = x0.mean(dim=0) - y0.mean(dim=0)  # (d,)
    x_shift = l2norm(x0 - lam * delta, dim=1, eps=eps)
    y_shift = l2norm(y0 + lam * delta, dim=1, eps=eps)
    return x_shift, y_shift, delta


# ============================================================
# Retrieval eval (Karpathy) + MG shift
# ============================================================

@torch.no_grad()
def retrieval_eval_mg(
    model: VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    mg_lambda: float
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Dataset item: (PIL, [captions])
    We compute:
      - image feats for images
      - text feats for all captions
      - paired text = first caption per image
      - MG shift using (image, paired_text) pairs
      - retrieval using shifted embeddings
      - gap metrics using shifted (image, paired_text)
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

        feats = model.encode_images(pil_images)  # (b, d) on GPU (l2-normalized)
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
        tf = model.encode_texts(all_caps[s:s+bs_t])  # (bs, d) GPU (l2-normalized)
        text_feats_chunks.append(tf)
    text_feats = torch.cat(text_feats_chunks, dim=0)  # (Ncap, d) GPU

    # paired text = first caption per image
    first_cap = [-1] * image_feats.size(0)
    for cap_idx, img_idx in enumerate(cap2img):
        if first_cap[img_idx] < 0:
            first_cap[img_idx] = cap_idx
    pair_map = torch.tensor(first_cap, dtype=torch.long, device=device)
    paired_text = text_feats[pair_map]  # (Nimg, d)

    # MG shift computed on (image_feats, paired_text) pairs
    img_shift, paired_shift, delta_gap = mg_shift_pairwise(image_feats, paired_text, lam=mg_lambda)

    # Apply the SAME delta to ALL captions (important: keep alignment with MG definition)
    # text_shift = Normalize(text + lam*delta_gap)
    text_shift = l2norm(text_feats + mg_lambda * delta_gap, dim=1)

    # gap metrics on shifted (image, paired_text)
    gap = {
        "centroid_distance": centroid_distance(img_shift, paired_shift),
        "relative_modality_gap": relative_modality_gap(img_shift, paired_shift, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(img_shift, paired_shift, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(img_shift, paired_shift),
    }

    cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)

    # chunked recall for GPU memory control
    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = img_shift.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = img_shift[s:e] @ text_shift.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            img_ids = torch.arange(s, e, device=device).unsqueeze(1)
            mapped = cap2img_t[topk]
            hit = (mapped == img_ids).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Nimg)

    def recall_t2i(K: int) -> float:
        correct = 0
        Ncap = text_shift.size(0)
        chunk = 1024
        for s in range(0, Ncap, chunk):
            e = min(Ncap, s + chunk)
            sims = text_shift[s:e] @ img_shift.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Ncap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}
    extra = {
        "n_images": float(img_shift.size(0)),
        "n_captions": float(text_shift.size(0)),
        "mg_lambda": float(mg_lambda),
        "delta_gap_norm": float(delta_gap.norm().item()),
    }
    return gap, i2t, t2i, extra


# ============================================================
# Zero-shot classification + MG shift
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
def zeroshot_eval_mg(
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
    mg_lambda: float
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    - Build class text weights W (l2-normalized)
    - For each image, compute x (l2-normalized), logits = x @ W^T
    - For MG shift:
        pair y_i = W[label_i]
      compute delta_gap on (x_i, y_i), then shift x and ALL class weights W.
    - Evaluate accuracy using shifted x and shifted W.
    - Gap metrics on shifted (x, y_label).
    """
    W = build_zeroshot_weights(model, classnames, templates, device)  # (C, d), l2-normalized

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_cls
    )

    # First pass: collect embeddings up to max_items to compute delta_gap correctly on the evaluated subset
    xs: List[torch.Tensor] = []
    ylab: List[torch.Tensor] = []
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

        x = model.encode_images(pil_images)  # (b, d), l2-normalized
        y = W[labels.to(device, non_blocking=True)]  # (b, d)
        xs.append(x)
        ylab.append(y)
        total += b

    x_all = torch.cat(xs, dim=0)        # (N, d)
    y_all = torch.cat(ylab, dim=0)      # (N, d)

    # MG shift computed on (x_all, y_all)
    x_shift, y_shift, delta_gap = mg_shift_pairwise(x_all, y_all, lam=mg_lambda)

    # Shift ALL class weights (so logits are consistent)
    W_shift = l2norm(W + mg_lambda * delta_gap, dim=1)  # (C, d)

    # Accuracy on shifted representations
    logits = x_shift @ W_shift.t()
    pred1 = torch.argmax(logits, dim=1)
    correct1 = int((pred1 == torch.argmax((y_all @ W.t()), dim=1)).sum().item())  # sanity placeholder, not used

    # We must compare to true labels; but we didn't store labels tensor.
    # Reconstruct labels via nearest original W index from y_all? NOT safe.
    # Instead: store labels during first pass.
    # Fix: redo first pass while storing labels.
    # (We keep this code path robust by doing it properly below.)

    # Proper accuracy computation (second pass but without recomputing embeddings):
    # We stored only y vectors; we need labels. So store labels in first pass.
    # To keep the script self-contained and correct, we redo the first pass quickly but
    # using cached x_shift ordering. We'll store labels in first pass now.

    # --- redo properly: store labels while collecting ---
    # Note: To avoid recomputing x_all, we already have x_all and x_shift. Just redo the
    # dataloader once to fetch labels in the same order for the evaluated subset.
    labels_all: List[torch.Tensor] = []
    total2 = 0
    for _, labels in loader:
        if max_items is not None and total2 >= max_items:
            break
        b = labels.shape[0]
        if max_items is not None and total2 + b > max_items:
            keep = max_items - total2
            labels = labels[:keep]
            b = keep
        labels_all.append(labels.cpu())
        total2 += b
    labels_all_t = torch.cat(labels_all, dim=0)  # (N,)

    # now compute top1/top5 using shifted logits
    logits = x_shift @ W_shift.t()
    top1 = torch.argmax(logits, dim=1).cpu()
    correct1 = int((top1 == labels_all_t).sum().item())

    top5 = torch.topk(logits, k=5, dim=1).indices.cpu()
    correct5 = int(sum([labels_all_t[i].item() in top5[i].tolist() for i in range(labels_all_t.shape[0])]))

    gap = {
        "centroid_distance": centroid_distance(x_shift, y_shift),
        "relative_modality_gap": relative_modality_gap(x_shift, y_shift, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(x_shift, y_shift, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(x_shift, y_shift),
    }
    acc = {"top1": 100.0 * correct1 / float(labels_all_t.shape[0]),
           "top5": 100.0 * correct5 / float(labels_all_t.shape[0]),
           "n": float(labels_all_t.shape[0])}
    extra = {"mg_lambda": float(mg_lambda), "delta_gap_norm": float(delta_gap.norm().item())}
    return gap, acc, extra


# ============================================================
# Main (same structure as baseline1)
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

    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)

    ap.add_argument("--mg-lambda", type=float, default=0.375, help="MG shift scalar λ (can be negative).")

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[MG] lambda = {args.mg_lambda}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_jsonl = out_dir / "baseline5_results.jsonl"
    out_csv = out_dir / "baseline5_results.csv"

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

    # ---- Build Karpathy TEST datasets ----
    coco_test = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="test", max_images=None)
    flickr_test = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="test", max_images=None)

    # ---- Classification datasets ----
    cifar_root = Path(args.data_root) / "cifar100"
    dtd_root = Path(args.data_root) / "dtd"

    cifar_test = tvds.CIFAR100(root=str(cifar_root), train=False, download=False, transform=None)
    dtd_test = tvds.DTD(root=str(dtd_root), split="test", download=False, transform=None)
    tiny_val_ds = TinyImageNet200Val(args.data_root)

    cifar_classes = cifar_test.classes
    dtd_classes = dtd_test.classes
    tiny_classes = tiny_val_ds.classnames
    tiny_templates = ["a photo of a {c}.", "a photo of the {c}."]

    # ---- Models ----
    models = make_models(args, device=device)

    # ---- Output schemas ----
    header = [
        "model", "dataset",
        "centroid_distance", "relative_modality_gap", f"NAS@{args.nas_k}", "CMAS",
        "I2T_R1", "I2T_R5", "I2T_R10",
        "T2I_R1", "T2I_R5", "T2I_R10",
        "top1", "top5",
        "eval_time_sec",
    ]

    rows = []
    jf = out_jsonl.open("w", encoding="utf-8")

    for model_name, model in models:
        print("\n==============================")
        print(f"[Model] {model_name}")
        print("==============================")

        # ------------------------------------------------------------
        # COCO retrieval (MG)
        # ------------------------------------------------------------
        print("[Eval] MSCOCO Karpathy test (MG shift) (I2T/T2I R@K + gap)")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval_mg(
            model, coco_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_coco,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            mg_lambda=args.mg_lambda
        )
        t1 = time.time()
        ds_name = f"mscoco2014_karpathy_test_MG_lambda={args.mg_lambda}"
        rec = {
            "model": model_name,
            "dataset": ds_name,
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, ds_name,
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "",
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Flickr retrieval (MG)
        # ------------------------------------------------------------
        print("[Eval] Flickr30k Karpathy test (MG shift) (I2T/T2I R@K + gap)")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval_mg(
            model, flickr_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_flickr,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            mg_lambda=args.mg_lambda
        )
        t1 = time.time()
        ds_name = f"flickr30k_karpathy_test_MG_lambda={args.mg_lambda}"
        rec = {
            "model": model_name,
            "dataset": ds_name,
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, ds_name,
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "",
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # CIFAR100 zero-shot (MG)
        # ------------------------------------------------------------
        print("[Eval] CIFAR100 zero-shot (MG shift) (top1/top5 + gap)")
        t0 = time.time()
        gap, acc, extra = zeroshot_eval_mg(
            model, cifar_test,
            cifar_classes, CIFAR100_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            mg_lambda=args.mg_lambda
        )
        t1 = time.time()
        ds_name = f"cifar100_test_MG_lambda={args.mg_lambda}"
        rec = {
            "model": model_name,
            "dataset": ds_name,
            "gap": gap,
            "acc": acc,
            "extra": extra,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, ds_name,
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # DTD zero-shot (MG)
        # ------------------------------------------------------------
        print("[Eval] DTD zero-shot (MG shift) (top1/top5 + gap)")
        t0 = time.time()
        gap, acc, extra = zeroshot_eval_mg(
            model, dtd_test,
            dtd_classes, DTD_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            mg_lambda=args.mg_lambda
        )
        t1 = time.time()
        ds_name = f"dtd_test_MG_lambda={args.mg_lambda}"
        rec = {
            "model": model_name,
            "dataset": ds_name,
            "gap": gap,
            "acc": acc,
            "extra": extra,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, ds_name,
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Tiny-ImageNet-200 zero-shot (MG) (val)
        # ------------------------------------------------------------
        print("[Eval] Tiny-ImageNet-200 val zero-shot (MG shift) (top1/top5 + gap)")
        t0 = time.time()
        gap, acc, extra = zeroshot_eval_mg(
            model, tiny_val_ds,
            tiny_classes, tiny_templates,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples,
            mg_lambda=args.mg_lambda
        )
        t1 = time.time()
        ds_name = f"tiny-imagenet-200_val_MG_lambda={args.mg_lambda}"
        rec = {
            "model": model_name,
            "dataset": ds_name,
            "gap": gap,
            "acc": acc,
            "extra": extra,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, ds_name,
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
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
