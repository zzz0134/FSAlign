#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-6: GR-CLIP (Gap-Removed CLIP; mean-centering calibration)
- Same evaluation protocol & metrics as Baseline-1:
  - Karpathy split retrieval for MSCOCO-2014 + Flickr30k (I2T/T2I Recall@K)
  - Zero-shot image classification on CIFAR100 / Tiny-ImageNet-200 / DTD
  - Gap metrics: centroid distance, Relative Modality Gap (RMG), NAS(k), CMAS
  - Record runtime per (model, dataset)
  - Save JSONL + CSV

Core idea (GR-CLIP):
  For each modality, compute a global mean embedding on a calibration set, then subtract
  this mean from embeddings before similarity / downstream evaluation.
  We implement it as:
    e_img'  = normalize( e_img_raw  - mu_img )
    e_txt'  = normalize( e_txt_raw  - mu_txt )
  (and optionally distinguish query-text mean mu_q vs document-text mean mu_T; in our
   image-text retrieval we expose both, but by default we compute both from the same
   caption pool unless you provide separate calibration pools.)

Calibration sets used here (to avoid test leakage):
  - For MSCOCO retrieval:
      use Karpathy "train" split captions/images to compute means
  - For Flickr30k retrieval:
      use Karpathy "train" split captions/images to compute means
  - For CIFAR100:
      use CIFAR100 train split images as image calibration; text calibration from class prompts
  - For DTD:
      use DTD train split images; text calibration from class prompts
  - For Tiny-ImageNet-200:
      Tiny-ImageNet official train images exist under tiny-imagenet-200/train.
      We build a train dataset for calibration and use val for evaluation.

If some train split is missing in your local data, the script will fall back to using
a subset of the evaluation split for calibration (and will print a warning).

Run (mirrors baseline-1):
  python baseline6_gr_clip_karpathy_mscoco2014.py \
    --data-root /work/was598/modilty_gap/tools/data \
    --out-dir /work/was598/modilty_gap/results/baseline6 \
    --models clip,siglip,openclip \
    --batch-size 128 --num-workers 8 \
    --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
    --nas-k 10 \
    --calib-n 10000 \
    --calib-cache /work/was598/modilty_gap/results/baseline6/calib_cache
"""

import os
import csv
import json
import time
import math
import argparse
import random
import hashlib
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

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

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

class TinyImageNet200TrainForCalib(Dataset):
    """
    Tiny-ImageNet-200 train split for calibration:
      tiny-imagenet-200/train/<wnid>/images/*.JPEG
    """
    def __init__(self, data_root: str):
        super().__init__()
        self.root = Path(data_root) / "tiny-imagenet-200"
        train_dir = self.root / "train"
        wnids_path = self.root / "wnids.txt"
        words_path = self.root / "words.txt"
        assert train_dir.exists(), f"Tiny-ImageNet train dir not found: {train_dir}"
        assert wnids_path.exists(), f"wnids.txt not found under {self.root}"
        assert words_path.exists(), f"words.txt not found under {self.root}"

        self.wnids = [l.strip() for l in wnids_path.read_text().splitlines() if l.strip()]

        wnid_to_words = {}
        for line in words_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                wnid_to_words[parts[0]] = parts[1].split(",")[0].strip()

        self.classnames = [wnid_to_words.get(w, w) for w in self.wnids]

        samples: List[str] = []
        for wnid in self.wnids:
            img_dir = train_dir / wnid / "images"
            if not img_dir.exists():
                continue
            for p in img_dir.glob("*.JPEG"):
                samples.append(str(p))

        if len(samples) == 0:
            raise AssertionError(f"No train images found under {train_dir}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return img

# ============================================================
# Collate functions
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

def collate_images_only(batch):
    images = [b for b in batch]
    return images

# ============================================================
# Backbones: CLIP / OpenCLIP / SigLIP
#   Important for GR-CLIP: we need RAW embeddings (pre-normalization)
# ============================================================

class VLBackbone(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def encode_images_raw(self, pil_images: List[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_texts_raw(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        return l2norm(self.encode_images_raw(pil_images).float())

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return l2norm(self.encode_texts_raw(texts).float())

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
    def encode_images_raw(self, pil_images: List[Image.Image]) -> torch.Tensor:
        tens = torch.stack([self.preprocess(im) for im in pil_images], dim=0)
        tens = tens.to(self.device, non_blocking=True)
        feat = self.model.encode_image(tens).float()
        return feat

    @torch.no_grad()
    def encode_texts_raw(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts)
        if isinstance(toks, dict):
            toks = {k: v.to(self.device) for k, v in toks.items()}
            feat = self.model.encode_text(**toks)
        else:
            toks = toks.to(self.device)
            feat = self.model.encode_text(toks)
        feat = feat.float()
        return feat

class CLIPWrapper(OpenCLIPWrapper):
    def __init__(self, device: str):
        super().__init__(model_name="ViT-B-32", pretrained="openai", device=device)

class SigLIPWrapper(VLBackbone):
    def __init__(self, hf_name: str, device: str):
        super().__init__(device=device)
        assert SiglipModel is not None and SiglipProcessor is not None, \
            "SigLIP requires transformers>=4.40 with SiglipModel."
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
    def encode_images_raw(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inp = self.proc(images=pil_images, return_tensors="pt")
        pv = inp["pixel_values"].to(self.device, non_blocking=True)
        feat = self.model.get_image_features(pixel_values=pv).float()
        return feat

    @torch.no_grad()
    def encode_texts_raw(self, texts: List[str]) -> torch.Tensor:
        inp = self.proc(text=texts, padding=True, truncation=True, return_tensors="pt")
        inp = {k: v.to(self.device, non_blocking=True) for k, v in inp.items()}
        feat = self.model.get_text_features(**inp).float()
        return feat

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
# GR-CLIP Calibrator (mean-centering)
# ============================================================

class GRCalibrator:
    """
    Stores modality means and applies GR calibration:
      e_img' = normalize(e_img_raw - mu_img)
      e_txt' = normalize(e_txt_raw - mu_txt_query or mu_txt_doc)
    We keep:
      - mu_img
      - mu_txt_query (mu_q)
      - mu_txt_doc   (mu_T)
    """
    def __init__(self, dim: int, device: str):
        self.dim = dim
        self.device = device
        self.mu_img = torch.zeros(dim, device=device)
        self.mu_txt_q = torch.zeros(dim, device=device)
        self.mu_txt_d = torch.zeros(dim, device=device)
        self.ready = False

    @torch.no_grad()
    def apply_image(self, e_img_raw: torch.Tensor) -> torch.Tensor:
        return l2norm(e_img_raw - self.mu_img)

    @torch.no_grad()
    def apply_text_query(self, e_txt_raw: torch.Tensor) -> torch.Tensor:
        return l2norm(e_txt_raw - self.mu_txt_q)

    @torch.no_grad()
    def apply_text_doc(self, e_txt_raw: torch.Tensor) -> torch.Tensor:
        return l2norm(e_txt_raw - self.mu_txt_d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mu_img": self.mu_img.detach().cpu(),
            "mu_txt_q": self.mu_txt_q.detach().cpu(),
            "mu_txt_d": self.mu_txt_d.detach().cpu(),
        }

    def load_state_dict(self, sd: Dict[str, torch.Tensor]):
        self.mu_img = sd["mu_img"].to(self.device)
        self.mu_txt_q = sd["mu_txt_q"].to(self.device)
        self.mu_txt_d = sd["mu_txt_d"].to(self.device)
        self.ready = True

class GRWrappedBackbone(VLBackbone):
    """
    Wrap a base VLBackbone and apply GR calibration.
    """
    def __init__(self, base: VLBackbone, calibrator: GRCalibrator):
        super().__init__(device=base.device)
        self.base = base
        self.cal = calibrator

    @property
    def dim(self) -> int:
        return self.base.dim

    @torch.no_grad()
    def encode_images_raw(self, pil_images: List[Image.Image]) -> torch.Tensor:
        return self.base.encode_images_raw(pil_images)

    @torch.no_grad()
    def encode_texts_raw(self, texts: List[str]) -> torch.Tensor:
        return self.base.encode_texts_raw(texts)

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        raw = self.base.encode_images_raw(pil_images).float()
        return self.cal.apply_image(raw)

    @torch.no_grad()
    def encode_texts_query(self, texts: List[str]) -> torch.Tensor:
        raw = self.base.encode_texts_raw(texts).float()
        return self.cal.apply_text_query(raw)

    @torch.no_grad()
    def encode_texts_doc(self, texts: List[str]) -> torch.Tensor:
        raw = self.base.encode_texts_raw(texts).float()
        return self.cal.apply_text_doc(raw)

    # keep default encode_texts as document-text for backward compatibility in classification
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return self.encode_texts_doc(texts)

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
# Calibration helpers
# ============================================================

@torch.no_grad()
def sample_karpathy_calib_texts_and_images(
    ds: Dataset,
    n: int,
    seed: int
) -> Tuple[List[Image.Image], List[str]]:
    """
    ds item: (PIL, [captions])
    Returns:
      - images: length <= n
      - texts : length <= n (we take one caption per image for calibration)
    """
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:min(n, len(idxs))]

    images: List[Image.Image] = []
    texts: List[str] = []
    for i in idxs:
        img, caps = ds[i]
        images.append(img)
        if isinstance(caps, (list, tuple)) and len(caps) > 0:
            texts.append(str(caps[0]))
        else:
            texts.append(str(caps))
    return images, texts

@torch.no_grad()
def compute_mean_image(
    model: VLBackbone,
    pil_images: List[Image.Image],
    device: str,
    batch_size: int
) -> torch.Tensor:
    if len(pil_images) == 0:
        raise ValueError("No images for calibration.")
    chunks = []
    for s in range(0, len(pil_images), batch_size):
        raw = model.encode_images_raw(pil_images[s:s+batch_size]).float()
        chunks.append(raw)
    feats = torch.cat(chunks, dim=0).to(device)
    mu = feats.mean(dim=0)
    return mu

@torch.no_grad()
def compute_mean_text(
    model: VLBackbone,
    texts: List[str],
    device: str,
    batch_size: int
) -> torch.Tensor:
    if len(texts) == 0:
        raise ValueError("No texts for calibration.")
    chunks = []
    for s in range(0, len(texts), batch_size):
        raw = model.encode_texts_raw(texts[s:s+batch_size]).float()
        chunks.append(raw)
    feats = torch.cat(chunks, dim=0).to(device)
    mu = feats.mean(dim=0)
    return mu

def maybe_load_calibrator(cache_dir: Optional[str], cache_key: str, calibrator: GRCalibrator) -> bool:
    if cache_dir is None:
        return False
    p = Path(cache_dir) / f"{cache_key}.pt"
    if p.exists() and p.stat().st_size > 0:
        try:
            sd = torch.load(str(p), map_location="cpu")
            calibrator.load_state_dict(sd)
            print(f"[CalibCache] loaded: {p}")
            return True
        except Exception as e:
            print(f"[CalibCache] failed to load {p}: {e}")
    return False

def save_calibrator(cache_dir: Optional[str], cache_key: str, calibrator: GRCalibrator):
    if cache_dir is None:
        return
    ensure_dir(Path(cache_dir))
    p = Path(cache_dir) / f"{cache_key}.pt"
    try:
        torch.save(calibrator.state_dict(), str(p))
        print(f"[CalibCache] saved: {p}")
    except Exception as e:
        print(f"[CalibCache] failed to save {p}: {e}")

# ============================================================
# Retrieval eval (Karpathy) with GR-CLIP
# ============================================================

@torch.no_grad()
def retrieval_eval_gr(
    model_gr: GRWrappedBackbone,
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
    Compute:
      - calibrated image feats for images
      - calibrated text feats for all captions
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

        feats = model_gr.encode_images(pil_images)  # calibrated (b, d)
        image_feats_chunks.append(feats)

        for i in range(b):
            caps = caps_list[i]
            for c in caps:
                all_caps.append(c)
                cap2img.append(n_images + i)

        n_images += b

    image_feats = torch.cat(image_feats_chunks, dim=0)  # (Nimg, d)
    n_caps = len(all_caps)

    # Encode captions for retrieval corpus (document text mean)
    text_feats_chunks: List[torch.Tensor] = []
    bs_t = 256
    for s in range(0, n_caps, bs_t):
        tf = model_gr.encode_texts_doc(all_caps[s:s+bs_t])  # calibrated
        text_feats_chunks.append(tf)
    text_feats = torch.cat(text_feats_chunks, dim=0)  # (Ncap, d)

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

    # chunked recall
    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = image_feats.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = image_feats[s:e] @ text_feats.t()
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
            sims = text_feats[s:e] @ image_feats.t()
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
# Zero-shot classification (same as baseline-1, but GR-calibrated)
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
def build_zeroshot_weights_gr(
    model_gr: GRWrappedBackbone,
    classnames: List[str],
    templates: List[str],
    device: str
) -> torch.Tensor:
    ws = []
    bs = 128
    for cname in classnames:
        texts = [t.format(c=cname) for t in templates]
        feats_chunks: List[torch.Tensor] = []
        for s in range(0, len(texts), bs):
            feats_chunks.append(model_gr.encode_texts_doc(texts[s:s+bs]))
        feats = torch.cat(feats_chunks, dim=0)
        w = l2norm(feats.mean(dim=0, keepdim=True)).squeeze(0)
        ws.append(w)
    W = torch.stack(ws, dim=0).to(device)
    return W

@torch.no_grad()
def zeroshot_eval_gr(
    model_gr: GRWrappedBackbone,
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
    W = build_zeroshot_weights_gr(model_gr, classnames, templates, device)

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

        x = model_gr.encode_images(pil_images)  # (b, d) calibrated
        logits = x @ W.t()                       # (b, C)

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
def build_vqa_answer_weights_gr(
    base_model: VLBackbone,
    calibrator: GRCalibrator,
    answer_vocab: List[str],
    answer_template: str,
    device: str,
    text_batch_size: int = 128,
) -> torch.Tensor:
    prompts = [format_answer_prompt(a, answer_template) for a in answer_vocab]
    chunks: List[torch.Tensor] = []
    for s in range(0, len(prompts), text_batch_size):
        raw = base_model.encode_texts_raw(prompts[s:s + text_batch_size]).float()
        chunks.append(calibrator.apply_text_doc(raw))
    return torch.cat(chunks, dim=0).to(device)


@torch.no_grad()
def encode_vqa_answer_batch_gr(
    base_model: VLBackbone,
    calibrator: GRCalibrator,
    answers: List[str],
    answer_template: str,
) -> torch.Tensor:
    prompts = [format_answer_prompt(a, answer_template) for a in answers]
    raw = base_model.encode_texts_raw(prompts).float()
    return calibrator.apply_text_doc(raw)


@torch.no_grad()
def encode_vqa_query_batch_gr(
    base_model: VLBackbone,
    calibrator: GRCalibrator,
    pil_images: List[Image.Image],
    questions: List[str],
    question_template: str,
    fusion_mode: str,
) -> torch.Tensor:
    img_raw = base_model.encode_images_raw(pil_images).float()
    q_prompts = [format_question_prompt(q, question_template) for q in questions]
    q_raw = base_model.encode_texts_raw(q_prompts).float()
    img = calibrator.apply_image(img_raw)
    q = calibrator.apply_text_query(q_raw)
    return fuse_query_features(img, q, mode=fusion_mode)


@torch.no_grad()
def fit_vqa_gr_calibrator(
    base_model: VLBackbone,
    calibrator: GRCalibrator,
    train_dataset: Dataset,
    answer_vocab: List[str],
    question_template: str,
    answer_template: str,
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

    img_chunks: List[torch.Tensor] = []
    q_chunks: List[torch.Tensor] = []
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

        img_chunks.append(base_model.encode_images_raw(pil_images).float())
        q_prompts = [format_question_prompt(q, question_template) for q in questions]
        q_chunks.append(base_model.encode_texts_raw(q_prompts).float())
        total += b

    if not img_chunks or not q_chunks:
        raise ValueError('No VQAv2 calibration features were encoded for GR-CLIP.')

    calibrator.mu_img = torch.cat(img_chunks, dim=0).mean(dim=0).to(calibrator.device)
    calibrator.mu_txt_q = torch.cat(q_chunks, dim=0).mean(dim=0).to(calibrator.device)

    answer_prompts = [format_answer_prompt(a, answer_template) for a in answer_vocab]
    answer_chunks: List[torch.Tensor] = []
    for s in range(0, len(answer_prompts), batch_size):
        answer_chunks.append(base_model.encode_texts_raw(answer_prompts[s:s + batch_size]).float())
    calibrator.mu_txt_d = torch.cat(answer_chunks, dim=0).mean(dim=0).to(calibrator.device)
    calibrator.ready = True


@torch.no_grad()
def vqav2_eval_gr(
    base_model: VLBackbone,
    calibrator: GRCalibrator,
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
    W = build_vqa_answer_weights_gr(base_model, calibrator, answer_vocab, answer_template, device)

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

        x = encode_vqa_query_batch_gr(base_model, calibrator, pil_images, questions, question_template, fusion_mode)
        logits = x @ W.t()
        topk = torch.topk(logits, k=k_eval, dim=1).indices
        s1, s5 = vqa_topk_scores(topk, answer_vocab, gt_answers)
        score_top1 += s1
        score_top5 += s5
        total += b
        x_chunks.append(x)
        y_fallback = encode_vqa_answer_batch_gr(base_model, calibrator, canonical_answers, answer_template)
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

    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)

    # GR-CLIP calibration
    ap.add_argument("--calib-n", type=int, default=10000, help="number of samples per calibration pool")
    ap.add_argument("--calib-batch", type=int, default=128, help="batch size for calibration encoding")
    ap.add_argument("--calib-cache", type=str, default=None, help="dir to cache calibration means per (model, dataset)")
    ap.add_argument("--eval-vqav2", action="store_true")
    ap.add_argument("--vqav2-root", type=str, default="")
    ap.add_argument("--max-vqa-val", type=int, default=10000)
    ap.add_argument("--vqav2-topk-answers", type=int, default=3129)
    ap.add_argument("--vqav2-question-template", type=str, default="Question: {q}")
    ap.add_argument("--vqav2-answer-template", type=str, default="Answer: {a}.")
    ap.add_argument("--vqav2-fusion", type=str, default="mean", choices=["mean", "sum"])

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_jsonl = out_dir / "baseline6_results.jsonl"
    out_csv = out_dir / "baseline6_results.csv"

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
    coco_test = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="test", max_images=None)
    flickr_test = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="test", max_images=None)

    # Calibration pools (train split preferred)
    try:
        coco_train_calib = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="train", max_images=None)
    except Exception as e:
        print(f"[Warn] COCO train split not usable for calibration, fallback to test subset: {e}")
        coco_train_calib = coco_test

    try:
        flickr_train_calib = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="train", max_images=None)
    except Exception as e:
        print(f"[Warn] Flickr train split not usable for calibration, fallback to test subset: {e}")
        flickr_train_calib = flickr_test

    # ---- Classification datasets ----
    cifar_root = Path(args.data_root) / "cifar100"
    dtd_root = Path(args.data_root) / "dtd"

    cifar_test = tvds.CIFAR100(root=str(cifar_root), train=False, download=False, transform=None)
    dtd_test = tvds.DTD(root=str(dtd_root), split="test", download=False, transform=None)

    # calibration images from train splits if available
    try:
        cifar_train_calib = tvds.CIFAR100(root=str(cifar_root), train=True, download=False, transform=None)
    except Exception as e:
        print(f"[Warn] CIFAR100 train split not usable for calibration, fallback to test subset: {e}")
        cifar_train_calib = cifar_test

    try:
        dtd_train_calib = tvds.DTD(root=str(dtd_root), split="train", download=False, transform=None)
    except Exception as e:
        print(f"[Warn] DTD train split not usable for calibration, fallback to test subset: {e}")
        dtd_train_calib = dtd_test

    tiny_val_ds = TinyImageNet200Val(args.data_root)
    try:
        tiny_train_calib = TinyImageNet200TrainForCalib(args.data_root)
    except Exception as e:
        print(f"[Warn] Tiny-ImageNet train split not usable for calibration, fallback to val subset: {e}")
        tiny_train_calib = None

    cifar_classes = cifar_test.classes
    dtd_classes = dtd_test.classes
    tiny_classes = tiny_val_ds.classnames
    tiny_templates = ["a photo of a {c}.", "a photo of the {c}."]

    vqa_train = None
    vqa_val = None
    vqa_answer_vocab = None
    vqa_calib_max = None if args.calib_n <= 0 else args.calib_n
    vqa_max_val = None if args.max_vqa_val <= 0 else args.max_vqa_val
    if args.eval_vqav2:
        vqa_root = Path(args.vqav2_root) if args.vqav2_root else (Path(args.data_root) / "vqav2")
        vqa_answer_vocab, vqa_answer_to_idx = build_vqav2_answer_vocab(str(vqa_root), args.vqav2_topk_answers)
        vqa_train = VQAv2ClassificationDataset(
            str(vqa_root),
            split="train",
            answer_to_idx=vqa_answer_to_idx,
            drop_oov=False,
            max_items=vqa_calib_max,
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
        "eval_time_sec",
    ]

    rows = []
    jf = out_jsonl.open("w", encoding="utf-8")

    for model_name, base_model in base_models:
        print("\n==============================")
        print(f"[Model] {model_name} (GR-CLIP calibrated)")
        print("==============================")

        calibrator = GRCalibrator(dim=base_model.dim, device=device)

        # ------------------------------------------------------------
        # COCO calibration (means for image/text in COCO domain)
        # ------------------------------------------------------------
        coco_cache_key = sha1_text(f"{model_name}|coco|calib_n={args.calib_n}")
        loaded = maybe_load_calibrator(args.calib_cache, coco_cache_key, calibrator)

        if not loaded:
            print("[Calib] Computing COCO means (train split preferred)...")
            imgs, caps = sample_karpathy_calib_texts_and_images(coco_train_calib, n=args.calib_n, seed=args.seed)
            mu_img = compute_mean_image(base_model, imgs, device=device, batch_size=args.calib_batch)
            mu_txt = compute_mean_text(base_model, caps, device=device, batch_size=args.calib_batch)

            calibrator.mu_img = mu_img
            calibrator.mu_txt_q = mu_txt
            calibrator.mu_txt_d = mu_txt
            calibrator.ready = True
            save_calibrator(args.calib_cache, coco_cache_key, calibrator)

        model_gr = GRWrappedBackbone(base_model, calibrator)

        # ------------------------------------------------------------
        # COCO retrieval
        # ------------------------------------------------------------
        print("[Eval] MSCOCO Karpathy test (I2T/T2I R@K + gap) [GR-CLIP]")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval_gr(
            model_gr, coco_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_coco,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()

        rec = {
            "model": model_name,
            "dataset": "mscoco2014_karpathy_test",
            "method": "GR-CLIP",
            "calibration": {
                "source": "karpathy_train_preferred",
                "calib_n": int(args.calib_n),
                "cache_key": coco_cache_key,
            },
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
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
        # Flickr calibration (separate means in Flickr domain)
        # ------------------------------------------------------------
        flickr_cal = GRCalibrator(dim=base_model.dim, device=device)
        flickr_cache_key = sha1_text(f"{model_name}|flickr|calib_n={args.calib_n}")
        loaded = maybe_load_calibrator(args.calib_cache, flickr_cache_key, flickr_cal)

        if not loaded:
            print("[Calib] Computing Flickr means (train split preferred)...")
            imgs, caps = sample_karpathy_calib_texts_and_images(flickr_train_calib, n=args.calib_n, seed=args.seed)
            mu_img = compute_mean_image(base_model, imgs, device=device, batch_size=args.calib_batch)
            mu_txt = compute_mean_text(base_model, caps, device=device, batch_size=args.calib_batch)

            flickr_cal.mu_img = mu_img
            flickr_cal.mu_txt_q = mu_txt
            flickr_cal.mu_txt_d = mu_txt
            flickr_cal.ready = True
            save_calibrator(args.calib_cache, flickr_cache_key, flickr_cal)

        model_gr_flickr = GRWrappedBackbone(base_model, flickr_cal)

        # ------------------------------------------------------------
        # Flickr retrieval
        # ------------------------------------------------------------
        print("[Eval] Flickr30k Karpathy test (I2T/T2I R@K + gap) [GR-CLIP]")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval_gr(
            model_gr_flickr, flickr_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_flickr,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()

        rec = {
            "model": model_name,
            "dataset": "flickr30k_karpathy_test",
            "method": "GR-CLIP",
            "calibration": {
                "source": "karpathy_train_preferred",
                "calib_n": int(args.calib_n),
                "cache_key": flickr_cache_key,
            },
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
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
        # CIFAR100 calibration + zero-shot
        # ------------------------------------------------------------
        print("[Calib] CIFAR100 means (image from train preferred; text from class prompts)")
        cifar_cal = GRCalibrator(dim=base_model.dim, device=device)
        cifar_cache_key = sha1_text(f"{model_name}|cifar100|calib_n={args.calib_n}")
        loaded = maybe_load_calibrator(args.calib_cache, cifar_cache_key, cifar_cal)

        if not loaded:
            # sample images
            rng = random.Random(args.seed)
            idxs = list(range(len(cifar_train_calib)))
            rng.shuffle(idxs)
            idxs = idxs[:min(args.calib_n, len(idxs))]
            imgs = []
            for i in idxs:
                im, _ = cifar_train_calib[i]
                imgs.append(im)

            # texts: prompt pool for calibration (all classes x templates)
            txts = []
            for cname in cifar_classes:
                for t in CIFAR100_TEMPLATES:
                    txts.append(t.format(c=cname))

            # subsample texts to calib-n *some factor (cap)
            rng.shuffle(txts)
            txts = txts[:min(len(txts), max(args.calib_n, 2000))]

            mu_img = compute_mean_image(base_model, imgs, device=device, batch_size=args.calib_batch)
            mu_txt = compute_mean_text(base_model, txts, device=device, batch_size=args.calib_batch)

            cifar_cal.mu_img = mu_img
            cifar_cal.mu_txt_q = mu_txt
            cifar_cal.mu_txt_d = mu_txt
            cifar_cal.ready = True
            save_calibrator(args.calib_cache, cifar_cache_key, cifar_cal)

        model_gr_cifar = GRWrappedBackbone(base_model, cifar_cal)

        print("[Eval] CIFAR100 zero-shot (top1/top5 + gap) [GR-CLIP]")
        t0 = time.time()
        gap, acc = zeroshot_eval_gr(
            model_gr_cifar, cifar_test,
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

        rec = {
            "model": model_name,
            "dataset": "cifar100_test",
            "method": "GR-CLIP",
            "calibration": {
                "source": "cifar_train_images + class_prompts",
                "calib_n": int(args.calib_n),
                "cache_key": cifar_cache_key,
            },
            "gap": gap,
            "acc": acc,
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
        # DTD calibration + zero-shot
        # ------------------------------------------------------------
        print("[Calib] DTD means (image from train preferred; text from class prompts)")
        dtd_cal = GRCalibrator(dim=base_model.dim, device=device)
        dtd_cache_key = sha1_text(f"{model_name}|dtd|calib_n={args.calib_n}")
        loaded = maybe_load_calibrator(args.calib_cache, dtd_cache_key, dtd_cal)

        if not loaded:
            rng = random.Random(args.seed)
            idxs = list(range(len(dtd_train_calib)))
            rng.shuffle(idxs)
            idxs = idxs[:min(args.calib_n, len(idxs))]
            imgs = []
            for i in idxs:
                im, _ = dtd_train_calib[i]
                imgs.append(im)

            txts = []
            for cname in dtd_classes:
                for t in DTD_TEMPLATES:
                    txts.append(t.format(c=cname))

            rng.shuffle(txts)
            txts = txts[:min(len(txts), max(args.calib_n, 2000))]

            mu_img = compute_mean_image(base_model, imgs, device=device, batch_size=args.calib_batch)
            mu_txt = compute_mean_text(base_model, txts, device=device, batch_size=args.calib_batch)

            dtd_cal.mu_img = mu_img
            dtd_cal.mu_txt_q = mu_txt
            dtd_cal.mu_txt_d = mu_txt
            dtd_cal.ready = True
            save_calibrator(args.calib_cache, dtd_cache_key, dtd_cal)

        model_gr_dtd = GRWrappedBackbone(base_model, dtd_cal)

        print("[Eval] DTD zero-shot (top1/top5 + gap) [GR-CLIP]")
        t0 = time.time()
        gap, acc = zeroshot_eval_gr(
            model_gr_dtd, dtd_test,
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

        rec = {
            "model": model_name,
            "dataset": "dtd_test",
            "method": "GR-CLIP",
            "calibration": {
                "source": "dtd_train_images + class_prompts",
                "calib_n": int(args.calib_n),
                "cache_key": dtd_cache_key,
            },
            "gap": gap,
            "acc": acc,
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
        # Tiny-ImageNet calibration + zero-shot
        # ------------------------------------------------------------
        print("[Calib] Tiny-ImageNet-200 means (train preferred; text from class prompts)")
        tiny_cal = GRCalibrator(dim=base_model.dim, device=device)
        tiny_cache_key = sha1_text(f"{model_name}|tinyimagenet|calib_n={args.calib_n}")
        loaded = maybe_load_calibrator(args.calib_cache, tiny_cache_key, tiny_cal)

        if not loaded:
            imgs = []
            rng = random.Random(args.seed)

            if tiny_train_calib is not None:
                idxs = list(range(len(tiny_train_calib)))
                rng.shuffle(idxs)
                idxs = idxs[:min(args.calib_n, len(idxs))]
                for i in idxs:
                    im = tiny_train_calib[i]
                    imgs.append(im)
            else:
                # fallback: sample from val
                idxs = list(range(len(tiny_val_ds)))
                rng.shuffle(idxs)
                idxs = idxs[:min(args.calib_n, len(idxs))]
                for i in idxs:
                    im, _ = tiny_val_ds[i]
                    imgs.append(im)

            txts = []
            for cname in tiny_classes:
                for t in tiny_templates:
                    txts.append(t.format(c=cname))
            rng.shuffle(txts)
            txts = txts[:min(len(txts), max(args.calib_n, 2000))]

            mu_img = compute_mean_image(base_model, imgs, device=device, batch_size=args.calib_batch)
            mu_txt = compute_mean_text(base_model, txts, device=device, batch_size=args.calib_batch)

            tiny_cal.mu_img = mu_img
            tiny_cal.mu_txt_q = mu_txt
            tiny_cal.mu_txt_d = mu_txt
            tiny_cal.ready = True
            save_calibrator(args.calib_cache, tiny_cache_key, tiny_cal)

        model_gr_tiny = GRWrappedBackbone(base_model, tiny_cal)

        print("[Eval] Tiny-ImageNet-200 val zero-shot (top1/top5 + gap) [GR-CLIP]")
        t0 = time.time()
        gap, acc = zeroshot_eval_gr(
            model_gr_tiny, tiny_val_ds,
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

        rec = {
            "model": model_name,
            "dataset": "tiny-imagenet-200_val",
            "method": "GR-CLIP",
            "calibration": {
                "source": "tiny_train_images_preferred + class_prompts",
                "calib_n": int(args.calib_n),
                "cache_key": tiny_cache_key,
            },
            "gap": gap,
            "acc": acc,
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
            print("[Calib] VQAv2 train means (image/question from train; answer from vocab)")
            vqa_cal = GRCalibrator(dim=base_model.dim, device=device)
            vqa_cache_key = sha1_text(
                f"{model_name}|vqav2|answers={args.vqav2_topk_answers}|calib_n={args.calib_n}|fusion={args.vqav2_fusion}"
            )
            loaded = maybe_load_calibrator(args.calib_cache, vqa_cache_key, vqa_cal)
            if not loaded:
                fit_vqa_gr_calibrator(
                    base_model=base_model,
                    calibrator=vqa_cal,
                    train_dataset=vqa_train,
                    answer_vocab=vqa_answer_vocab,
                    question_template=args.vqav2_question_template,
                    answer_template=args.vqav2_answer_template,
                    batch_size=args.calib_batch,
                    num_workers=args.num_workers,
                    max_items=vqa_calib_max,
                )
                save_calibrator(args.calib_cache, vqa_cache_key, vqa_cal)

            print("[Eval] VQAv2 val (classification-style VQA + gap) [GR-CLIP]")
            t0 = time.time()
            gap, acc, extra = vqav2_eval_gr(
                base_model=base_model,
                calibrator=vqa_cal,
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

            rec = {
                "model": model_name,
                "dataset": "vqav2_val",
                "method": "GR-CLIP",
                "calibration": {
                    "source": "vqav2_train_images + train_questions + answer_vocab",
                    "calib_n": int(args.calib_n),
                    "cache_key": vqa_cache_key,
                },
                "gap": gap,
                "acc": acc,
                "extra": extra,
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
