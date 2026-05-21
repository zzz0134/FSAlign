#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-5: AlignCLIP Method (Shared Parameter Space + IMSep) on external backbones
- Backbones: CLIP (OpenCLIP ViT-B-32 openai), SigLIP (HF), OpenCLIP (user-chosen)
- Train: fine-tune with AlignCLIP objective:
    L = L_CRsep + alpha * L_IMSep
  where IMSep is semantically-regularized intra-modality separation (paper Eq.6-14)
  and "shared parameter space" is implemented as a shared trainable adapter head applied
  to BOTH modalities (universal across CLIP/SigLIP/OpenCLIP).
- Eval: identical to baseline1:
    * MSCOCO-2014 + Flickr30k Karpathy split: I2T/T2I Recall@{1,5,10}
    * CIFAR100, Tiny-ImageNet-200, DTD: zero-shot top1/top5
    * modality gap metrics per dataset: centroid distance, RMG, NAS(k), CMAS
- Save JSONL + CSV per (model, dataset)
- Optionally save checkpoints per model.

Expected data layout is identical to baseline1_ground_truth_karpathy_mscoco2014.py

Run (example):
  # Train + Eval
  python baseline5_alignclip_method_on_backbones.py \
    --data-root /work/was598/modilty_gap/tools/data \
    --out-dir   /work/was598/modilty_gap/results/baseline5_alignclip_method \
    --models clip,siglip,openclip \
    --do-train \
    --train-splits train,restval \
    --train-source coco+flickr \
    --epochs 1 --lr 1e-5 --batch-size 128 --num-workers 8 \
    --alpha 0.5 --nas-k 10 --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
    --finetune-scope adapter

  # Eval only (load checkpoints)
  python baseline5_alignclip_method_on_backbones.py \
    --data-root ... --out-dir ... --models clip,siglip,openclip \
    --eval-only \
    --ckpt-dir /work/was598/modilty_gap/results/baseline5_alignclip_method/checkpoints

Notes:
- This script is designed to be faithful to AlignCLIP's *objective* (Eq.6-14) and
  implement "shared learnable parameter space" in a way that works for all three backbones:
  a single shared adapter head applied to both modalities.
- If you want strict "share the transformer encoder + projection layer" (SharedCLIP Figure 2)
  you must use an architecture where both modalities are tokenized to the same transformer width,
  which is not directly true for most off-the-shelf CLIP/SigLIP configurations. This baseline
  keeps your requested external backbones intact and applies AlignCLIP's method on top.

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
from torch.utils.data import Dataset, DataLoader, ConcatDataset

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

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None


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


class KarpathyTrainPairDataset(Dataset):
    """
    Training pairs from Karpathy json. Each item yields:
      (PIL.Image, one_caption_str)
    We sample a random caption per image on each __getitem__ call.
    """
    def __init__(
        self,
        karpathy_json: str,
        image_roots: List[str],
        splits: List[str],
        max_images: Optional[int] = None
    ):
        super().__init__()
        self.karpathy_json = Path(karpathy_json)
        assert self.karpathy_json.exists(), f"Karpathy json not found: {self.karpathy_json}"
        self.image_roots = [Path(p) for p in image_roots]
        self.splits = set([s.strip() for s in splits if s.strip()])

        data = json.loads(self.karpathy_json.read_text(encoding="utf-8"))
        images = data["images"]

        items = []
        missing = 0
        for img in images:
            if img.get("split", "") not in self.splits:
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
                f"No train items for splits={sorted(list(self.splits))} from {karpathy_json}. "
                f"Check image roots. (missing_paths={missing})"
            )

        print(f"[KarpathyTrainPairDataset] splits={sorted(list(self.splits))} items={len(items)} (missing_paths={missing})")
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
        cap = caps[random.randrange(len(caps))]
        return img, str(cap)


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

def collate_train_pairs(batch):
    images = [b[0] for b in batch]
    texts = [str(b[1]) for b in batch]
    return images, texts


# ============================================================
# Backbones: CLIP / OpenCLIP / SigLIP
# ============================================================

class VLBackbone(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self._use_train_preprocess = False

    def set_use_train_preprocess(self, flag: bool):
        self._use_train_preprocess = bool(flag)

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """Differentiable image encoder -> l2-normalized embeddings on device."""
        raise NotImplementedError

    def forward_texts(self, texts: List[str]) -> torch.Tensor:
        """Differentiable text encoder -> l2-normalized embeddings on device."""
        raise NotImplementedError

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        self.eval()
        return self.forward_images(pil_images)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self.eval()
        return self.forward_texts(texts)

    def try_get_logit_scale(self) -> Optional[torch.nn.Parameter]:
        """Return underlying logit_scale if exists (OpenCLIP/CLIP), else None."""
        return None


class OpenCLIPWrapper(VLBackbone):
    def __init__(self, model_name: str, pretrained: str, device: str):
        super().__init__(device=device)
        assert open_clip is not None, "open_clip is not installed."

        self.model_name = model_name
        self.pretrained = pretrained

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model.to(device)
        self.model.eval()

        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self.tokenizer = tokenizer

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feat = self.model.encode_image(dummy)
            self._dim = int(feat.shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    def try_get_logit_scale(self) -> Optional[torch.nn.Parameter]:
        if hasattr(self.model, "logit_scale"):
            ls = getattr(self.model, "logit_scale")
            if isinstance(ls, torch.nn.Parameter):
                return ls
        return None

    def _preprocess(self, pil_images: List[Image.Image]) -> torch.Tensor:
        pp = self.preprocess_train if self._use_train_preprocess else self.preprocess_val
        tens = torch.stack([pp(im) for im in pil_images], dim=0)
        return tens.to(self.device, non_blocking=True)

    def forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        tens = self._preprocess(pil_images)
        feat = self.model.encode_image(tens).float()
        return l2norm(feat)

    def forward_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts)
        if isinstance(toks, dict):
            toks = {k: v.to(self.device, non_blocking=True) for k, v in toks.items()}
            feat = self.model.encode_text(**toks).float()
        else:
            toks = toks.to(self.device, non_blocking=True)
            feat = self.model.encode_text(toks).float()
        return l2norm(feat)


class CLIPWrapper(OpenCLIPWrapper):
    def __init__(self, device: str):
        super().__init__(model_name="ViT-B-32", pretrained="openai", device=device)


class SigLIPWrapper(VLBackbone):
    def __init__(self, hf_name: str, device: str):
        super().__init__(device=device)
        assert SiglipModel is not None and SiglipProcessor is not None, \
            "SigLIP requires transformers with SiglipModel/SiglipProcessor."

        self.hf_name = hf_name
        self.model = SiglipModel.from_pretrained(hf_name).to(device)
        self.model.eval()
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

    def forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inp = self.proc(images=pil_images, return_tensors="pt")
        pv = inp["pixel_values"].to(self.device, non_blocking=True)
        feat = self.model.get_image_features(pixel_values=pv).float()
        return l2norm(feat)

    def forward_texts(self, texts: List[str]) -> torch.Tensor:
        inp = self.proc(text=texts, padding=True, truncation=True, return_tensors="pt")
        inp = {k: v.to(self.device, non_blocking=True) for k, v in inp.items()}
        feat = self.model.get_text_features(**inp).float()
        return l2norm(feat)


def make_base_backbones(args, device: str) -> List[Tuple[str, VLBackbone]]:
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
# Semantic text encoder (SBERT all-mpnet-base-v2 via transformers)
# ============================================================

class SemanticTextEncoder(nn.Module):
    """
    Encodes texts to semantic embeddings Es used for D=1-S in IMSep.
    Default model: sentence-transformers/all-mpnet-base-v2 (paper uses SBERT all-mpnet-base-v2).
    Implemented with transformers for fewer dependencies.
    """
    def __init__(self, model_name: str, device: str):
        super().__init__()
        assert AutoTokenizer is not None and AutoModel is not None, \
            "transformers is required for SemanticTextEncoder."
        self.model_name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device)
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for s in range(0, len(texts), batch_size):
            t = texts[s:s+batch_size]
            inp = self.tok(t, padding=True, truncation=True, return_tensors="pt")
            inp = {k: v.to(self.device, non_blocking=True) for k, v in inp.items()}
            out = self.enc(**inp)
            last = out.last_hidden_state  # (B, L, H)
            mask = inp["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
            pooled = (last * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1e-6))
            pooled = l2norm(pooled)
            outs.append(pooled)
        return torch.cat(outs, dim=0)


# ============================================================
# AlignCLIP Method Module (shared parameter space + IMSep objective)
# ============================================================

class SharedAdapterHead(nn.Module):
    """
    Shared learnable parameter space across modalities:
    A single adapter applied to BOTH image and text embeddings (shared weights).
    """
    def __init__(
        self,
        dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        residual: bool = True,
        use_layernorm: bool = True,
        act: str = "gelu",
    ):
        super().__init__()
        self.dim = dim
        self.hidden = dim * hidden_mult
        self.residual = residual
        self.use_layernorm = use_layernorm

        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, dim)
        self.drop = nn.Dropout(dropout)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=False)
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        if self.residual:
            y = x + y
        return l2norm(y)


class AlignCLIPOnBackbone(nn.Module):
    """
    Wrap an external backbone and apply AlignCLIP objective:
      L = L_CRsep + alpha L_IMSep
    with shared adapter head as "shared learnable parameter space".
    """
    def __init__(
        self,
        backbone: VLBackbone,
        alpha: float = 0.5,
        adapter_hidden_mult: int = 4,
        adapter_dropout: float = 0.0,
        adapter_residual: bool = True,
        adapter_layernorm: bool = True,
        adapter_act: str = "gelu",
        init_logit_scale: float = 1/0.07,  # typical CLIP temperature
        train_logit_scale: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.alpha = float(alpha)

        self.adapter = SharedAdapterHead(
            dim=backbone.dim,
            hidden_mult=adapter_hidden_mult,
            dropout=adapter_dropout,
            residual=adapter_residual,
            use_layernorm=adapter_layernorm,
            act=adapter_act
        )

        base_ls = backbone.try_get_logit_scale()
        if base_ls is not None:
            # Use underlying model logit_scale parameter
            self.logit_scale = base_ls
        else:
            # Create our own
            self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale), dtype=torch.float32))

        if not train_logit_scale:
            self.logit_scale.requires_grad_(False)

    @property
    def dim(self) -> int:
        return self.backbone.dim

    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        x = self.backbone.encode_images(pil_images)
        return self.adapter(x)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        y = self.backbone.encode_texts(texts)
        return self.adapter(y)

    def forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        x = self.backbone.forward_images(pil_images)
        return self.adapter(x)

    def forward_texts(self, texts: List[str]) -> torch.Tensor:
        y = self.backbone.forward_texts(texts)
        return self.adapter(y)

    def compute_losses(
        self,
        pil_images: List[Image.Image],
        texts: List[str],
        semantic_encoder: Optional[SemanticTextEncoder],
        semantic_batch_size: int = 64,
        semantic_device_fallback_cpu: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Implements paper Eq.(8)-(14) exactly (with stable normalization for S):
          V = Ev Ev^T
          S = Es Es^T (cosine)  -> D = 1 - S
          V_D = V ⊙ D
          M = Ev Et^T ; diag(M)=I⊙M
          y_vsep = exp(tau) * [diag(M) + V_D]
          L_IMSep = CE(y_vsep, labels)
          L_CRsep = CE(y_v, labels) + CE(y_t, labels)
          L = L_CRsep + alpha*L_IMSep
        """
        b = len(pil_images)
        assert b == len(texts), "batch images/texts mismatch"
        if b < 2:
            raise RuntimeError("Batch size must be >= 2 for IMSep/contrastive loss.")

        device = self.backbone.device

        Ev = self.forward_images(pil_images)   # (b,d) normalized
        Et = self.forward_texts(texts)         # (b,d) normalized

        logit_scale = self.logit_scale.exp()

        labels = torch.arange(b, device=device, dtype=torch.long)

        # Cross-modal logits (paper Eq.13 uses yv and yt)
        yv = logit_scale * (Ev @ Et.t())       # (b,b)
        yt = yv.t()                            # (b,b)

        L_cr_v = F.cross_entropy(yv, labels)
        L_cr_t = F.cross_entropy(yt, labels)
        L_CRsep = L_cr_v + L_cr_t

        # IMSep: semantic D from paired texts (paper Eq.9), V from images (paper Eq.8)
        V = Ev @ Ev.t()                        # (b,b) cosine similarity (since normalized)

        if semantic_encoder is None:
            # fallback: no semantic supervision -> D=1-S -> set S=I -> D=0 on diag, 1 elsewhere
            S = torch.eye(b, device=device, dtype=V.dtype)
            D = (1.0 - S)
        else:
            # encode semantics (Es) without grad; then cosine sim
            # If semantic encoder is on CPU and main is GPU, we transfer.
            with torch.no_grad():
                Es = semantic_encoder.encode(texts, batch_size=semantic_batch_size)  # (b,hs), normalized
            if semantic_device_fallback_cpu:
                Es = Es.to(device, non_blocking=True)
            else:
                # semantic_encoder already on target device; still ensure
                Es = Es.to(device, non_blocking=True)

            # robust cosine sim: S = Es Es^T (Es already normalized)
            S = Es @ Es.t()
            # clamp to avoid numerical issues
            S = S.clamp(-1.0, 1.0)
            D = (1.0 - S)  # in [-0,2], effectively [0,2] if S in [ -1, 1 ]

        # V_D (paper Eq.10)
        V_D = V * D

        # diag(M) (paper uses M=Ev Et^T)
        M = Ev @ Et.t()                        # (b,b)
        diagM = torch.diag(torch.diagonal(M))  # (b,b)

        # y_vsep (paper Eq.11)
        y_vsep = logit_scale * (diagM + V_D)   # (b,b)

        L_IMSep = F.cross_entropy(y_vsep, labels)

        L = L_CRsep + (self.alpha * L_IMSep)

        logs = {
            "loss_total": float(L.detach().item()),
            "loss_CRsep": float(L_CRsep.detach().item()),
            "loss_IMSep": float(L_IMSep.detach().item()),
            "logit_scale": float(logit_scale.detach().item()),
        }
        return L, logs


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def configure_finetune_scope(model: AlignCLIPOnBackbone, scope: str):
    """
    scope:
      - "adapter": train only shared adapter (+ logit_scale if it requires grad)
      - "adapter+proj": adapter + (if available) backbone projection params
      - "all": full backbone + adapter (+ logit_scale)
    Since different backbones expose projection differently, "adapter+proj" tries best-effort.
    """
    scope = scope.strip().lower()

    # default: freeze everything, then enable selected parts
    set_requires_grad(model.backbone, False)
    set_requires_grad(model.adapter, False)

    # logit_scale may belong to backbone or to wrapper; keep its current requires_grad
    # (user can control via args.train_logit_scale)

    if scope == "adapter":
        set_requires_grad(model.adapter, True)

    elif scope == "adapter+proj":
        set_requires_grad(model.adapter, True)

        # Best-effort: unfreeze projection-like params
        # OpenCLIP/CLIP: model.backbone.model may have text_projection / visual.proj
        bb = model.backbone
        if isinstance(bb, OpenCLIPWrapper):
            oc = bb.model
            # common names in open_clip
            for name, p in oc.named_parameters():
                if any(k in name.lower() for k in ["proj", "projection", "text_projection"]):
                    p.requires_grad_(True)

        elif isinstance(bb, SigLIPWrapper):
            # SigLIP: projection heads live inside SiglipModel modules
            for name, p in bb.model.named_parameters():
                if any(k in name.lower() for k in ["projection", "proj"]):
                    p.requires_grad_(True)

    elif scope == "all":
        set_requires_grad(model.backbone, True)
        set_requires_grad(model.adapter, True)

    else:
        raise ValueError(f"Unknown finetune scope: {scope}")


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
    m = min(x.shape[0], y.shape[0])
    return float((x[:m] * y[:m]).sum(dim=1).mean().item())

@torch.no_grad()
def nas_k(x: torch.Tensor, y: torch.Tensor, k: int = 10, max_items: int = 5000) -> float:
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
# Retrieval eval (Karpathy) - identical behavior to baseline1
# ============================================================

@torch.no_grad()
def retrieval_eval_alignclip(
    model: AlignCLIPOnBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

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

        feats = model.encode_images(pil_images)  # (b,d) GPU
        image_feats_chunks.append(feats)

        for i in range(b):
            caps = caps_list[i]
            for c in caps:
                all_caps.append(c)
                cap2img.append(n_images + i)

        n_images += b

    image_feats = torch.cat(image_feats_chunks, dim=0)
    n_caps = len(all_caps)

    text_feats_chunks: List[torch.Tensor] = []
    bs_t = 256
    for s in range(0, n_caps, bs_t):
        tf = model.encode_texts(all_caps[s:s+bs_t])
        text_feats_chunks.append(tf)
    text_feats = torch.cat(text_feats_chunks, dim=0)

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
# Zero-shot classification (same as baseline1 behavior)
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
def build_zeroshot_weights_alignclip(
    model: AlignCLIPOnBackbone,
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
            feats_chunks.append(model.encode_texts(texts[s:s+bs]))
        feats = torch.cat(feats_chunks, dim=0)
        w = l2norm(feats.mean(dim=0, keepdim=True)).squeeze(0)
        ws.append(w)
    W = torch.stack(ws, dim=0).to(device)
    return W

@torch.no_grad()
def zeroshot_eval_alignclip(
    model: AlignCLIPOnBackbone,
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

    W = build_zeroshot_weights_alignclip(model, classnames, templates, device)

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

        x = model.encode_images(pil_images)
        logits = x @ W.t()

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
# Training loop for AlignCLIP objective
# ============================================================

def train_alignclip_method(
    model: AlignCLIPOnBackbone,
    train_ds: Dataset,
    device: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    grad_accum: int,
    max_steps: Optional[int],
    amp: bool,
    semantic_encoder: Optional[SemanticTextEncoder],
    semantic_bs: int,
    log_every: int = 50,
) -> Dict[str, float]:

    model.train()
    model.backbone.set_use_train_preprocess(True)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_train_pairs,
        drop_last=True
    )

    # collect trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters. Check --finetune-scope / requires_grad flags.")

    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.startswith("cuda")))

    step = 0
    t_start = time.time()

    running = {"loss_total": 0.0, "loss_CRsep": 0.0, "loss_IMSep": 0.0}

    for ep in range(epochs):
        for pil_images, texts in loader:
            step += 1

            with torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda"))):
                loss, logs = model.compute_losses(
                    pil_images, texts,
                    semantic_encoder=semantic_encoder,
                    semantic_batch_size=semantic_bs,
                    semantic_device_fallback_cpu=True
                )
                loss = loss / float(max(1, grad_accum))

            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            for k in running.keys():
                if k in logs:
                    running[k] += logs[k]

            if log_every > 0 and step % log_every == 0:
                denom = float(log_every)
                msg = (
                    f"[Train] ep={ep+1}/{epochs} step={step} "
                    f"loss={running['loss_total']/denom:.4f} "
                    f"CRsep={running['loss_CRsep']/denom:.4f} "
                    f"IMSep={running['loss_IMSep']/denom:.4f} "
                    f"logit_scale={logs.get('logit_scale', -1.0):.3f}"
                )
                print(msg)
                for k in running.keys():
                    running[k] = 0.0

            if max_steps is not None and step >= max_steps:
                break

        if max_steps is not None and step >= max_steps:
            break

    t_end = time.time()

    model.eval()
    model.backbone.set_use_train_preprocess(False)

    return {
        "train_time_sec": float(t_end - t_start),
        "train_steps": float(step),
        "epochs": float(epochs),
    }


def save_checkpoint(model: AlignCLIPOnBackbone, path: Path):
    ensure_dir(path.parent)
    ckpt = {
        "adapter": model.adapter.state_dict(),
        "logit_scale": float(model.logit_scale.detach().cpu().item()) if isinstance(model.logit_scale, torch.Tensor) else None,
        "alpha": model.alpha,
    }
    # optionally save backbone trainable params (if user fine-tuned backbone)
    # safest: save full backbone state_dict (can be large). controlled by user in args.
    torch.save(ckpt, str(path))

def load_checkpoint(model: AlignCLIPOnBackbone, path: Path, strict: bool = True):
    ckpt = torch.load(str(path), map_location="cpu")
    model.adapter.load_state_dict(ckpt["adapter"], strict=strict)
    if "logit_scale" in ckpt and ckpt["logit_scale"] is not None:
        with torch.no_grad():
            model.logit_scale.copy_(torch.tensor(ckpt["logit_scale"], dtype=model.logit_scale.dtype))
    if "alpha" in ckpt:
        model.alpha = float(ckpt["alpha"])


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

    # eval controls (same as baseline1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-coco", type=int, default=5000)
    ap.add_argument("--max-flickr", type=int, default=5000)
    ap.add_argument("--max-cls", type=int, default=10000)
    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)

    # AlignCLIP method training options
    ap.add_argument("--do-train", action="store_true", help="Run AlignCLIP objective training before eval.")
    ap.add_argument("--eval-only", action="store_true", help="Skip training; optionally load ckpts from --ckpt-dir.")
    ap.add_argument("--ckpt-dir", type=str, default="", help="Directory containing saved adapter checkpoints per model tag.")
    ap.add_argument("--save-ckpt", action="store_true", help="Save adapter checkpoints after training.")

    ap.add_argument("--train-source", type=str, default="coco+flickr",
                    help="Training data source: coco, flickr, coco+flickr")
    ap.add_argument("--train-splits", type=str, default="train,restval",
                    help="Karpathy splits for training pairs, e.g. 'train,restval'")
    ap.add_argument("--max-train-images", type=int, default=200000,
                    help="Cap number of training images per dataset (for quick runs). None-like: -1")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--amp", action="store_true")

    # AlignCLIP loss hyperparams
    ap.add_argument("--alpha", type=float, default=0.5, help="IMSep weight alpha (paper uses 0.5).")
    ap.add_argument("--train-logit-scale", action="store_true", help="Train logit_scale/temperature.")
    ap.add_argument("--finetune-scope", type=str, default="adapter",
                    help="adapter | adapter+proj | all")

    # adapter architecture
    ap.add_argument("--adapter-hidden-mult", type=int, default=4)
    ap.add_argument("--adapter-dropout", type=float, default=0.0)
    ap.add_argument("--adapter-no-residual", action="store_true")
    ap.add_argument("--adapter-no-layernorm", action="store_true")
    ap.add_argument("--adapter-act", type=str, default="gelu")

    # semantic encoder
    ap.add_argument("--semantic-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--semantic-device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--semantic-batch-size", type=int, default=64)
    ap.add_argument("--no-semantic", action="store_true", help="Disable semantic encoder (D becomes trivial).")

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_jsonl = out_dir / "baseline5_alignclip_method_results.jsonl"
    out_csv = out_dir / "baseline5_alignclip_method_results.csv"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (out_dir / "checkpoints")

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

    # ---- Train datasets (Karpathy splits) ----
    train_splits = [s.strip() for s in args.train_splits.split(",") if s.strip()]
    max_train = None if args.max_train_images < 0 else int(args.max_train_images)

    train_ds_list: List[Dataset] = []
    if args.train_source.lower() in ["coco", "coco+flickr", "flickr+coco"]:
        train_ds_list.append(
            KarpathyTrainPairDataset(str(coco_kjson), coco_img_roots, splits=train_splits, max_images=max_train)
        )
    if args.train_source.lower() in ["flickr", "coco+flickr", "flickr+coco"]:
        train_ds_list.append(
            KarpathyTrainPairDataset(str(flickr_kjson), flickr_img_roots, splits=train_splits, max_images=max_train)
        )
    train_ds = ConcatDataset(train_ds_list) if len(train_ds_list) > 1 else (train_ds_list[0] if len(train_ds_list) == 1 else None)

    if args.do_train and train_ds is None:
        raise RuntimeError("Training requested but no train dataset constructed. Check --train-source.")

    # ---- Semantic encoder ----
    semantic_encoder = None
    if not args.no_semantic:
        sem_dev = args.semantic_device.strip().lower()
        if sem_dev == "cuda" and not torch.cuda.is_available():
            sem_dev = "cpu"
        print(f"[SemanticEncoder] {args.semantic_model} on {sem_dev}")
        semantic_encoder = SemanticTextEncoder(args.semantic_model, device=sem_dev)

    # ---- Models ----
    base_backbones = make_base_backbones(args, device=device)

    # ---- Output schemas ----
    header = [
        "model", "dataset",
        "centroid_distance", "relative_modality_gap", f"NAS@{args.nas_k}", "CMAS",
        "I2T_R1", "I2T_R5", "I2T_R10",
        "T2I_R1", "T2I_R5", "T2I_R10",
        "top1", "top5",
        "train_time_sec",
        "eval_time_sec",
    ]

    rows = []
    jf = out_jsonl.open("w", encoding="utf-8")

    for base_name, backbone in base_backbones:
        print("\n==============================")
        print(f"[Base Backbone] {base_name}")
        print("==============================")

        model = AlignCLIPOnBackbone(
            backbone=backbone,
            alpha=args.alpha,
            adapter_hidden_mult=args.adapter_hidden_mult,
            adapter_dropout=args.adapter_dropout,
            adapter_residual=(not args.adapter_no_residual),
            adapter_layernorm=(not args.adapter_no_layernorm),
            adapter_act=args.adapter_act,
            train_logit_scale=args.train_logit_scale
        ).to(device)

        # finetune scope
        configure_finetune_scope(model, args.finetune_scope)

        # Load checkpoint if eval-only or ckpt exists
        model_tag_safe = base_name.replace("/", "_").replace(":", "_")
        ckpt_path = ckpt_dir / f"{model_tag_safe}.pt"

        train_info = {"train_time_sec": 0.0}
        if args.eval_only:
            if args.ckpt_dir and ckpt_path.exists():
                print(f"[Load] {ckpt_path}")
                load_checkpoint(model, ckpt_path, strict=True)
            else:
                print("[EvalOnly] No checkpoint loaded (running with random adapter).")
        else:
            if args.do_train:
                print(f"[Train] AlignCLIP method on {base_name}")
                if train_ds is None:
                    raise RuntimeError("train_ds is None but --do-train was set.")
                tinfo = train_alignclip_method(
                    model=model,
                    train_ds=train_ds,
                    device=device,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    grad_accum=max(1, int(args.grad_accum)),
                    max_steps=None if args.max_steps < 0 else int(args.max_steps),
                    amp=bool(args.amp),
                    semantic_encoder=semantic_encoder,
                    semantic_bs=int(args.semantic_batch_size),
                    log_every=50,
                )
                train_info.update(tinfo)

                if args.save_ckpt:
                    print(f"[Save] {ckpt_path}")
                    ensure_dir(ckpt_dir)
                    save_checkpoint(model, ckpt_path)

        # ------------------------------------------------------------
        # COCO retrieval
        # ------------------------------------------------------------
        print("[Eval] MSCOCO Karpathy test (I2T/T2I R@K + gap)")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval_alignclip(
            model, coco_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_coco,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        rec = {
            "model": base_name,
            "dataset": "mscoco2014_karpathy_test",
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "train": train_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            base_name, "mscoco2014_karpathy_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "",
            train_info.get("train_time_sec", 0.0),
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Flickr retrieval
        # ------------------------------------------------------------
        print("[Eval] Flickr30k Karpathy test (I2T/T2I R@K + gap)")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval_alignclip(
            model, flickr_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_flickr,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        rec = {
            "model": base_name,
            "dataset": "flickr30k_karpathy_test",
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "train": train_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            base_name, "flickr30k_karpathy_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "",
            train_info.get("train_time_sec", 0.0),
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # CIFAR100 zero-shot
        # ------------------------------------------------------------
        print("[Eval] CIFAR100 zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval_alignclip(
            model, cifar_test,
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
            "model": base_name,
            "dataset": "cifar100_test",
            "gap": gap,
            "acc": acc,
            "train": train_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            base_name, "cifar100_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            train_info.get("train_time_sec", 0.0),
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # DTD zero-shot
        # ------------------------------------------------------------
        print("[Eval] DTD zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval_alignclip(
            model, dtd_test,
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
            "model": base_name,
            "dataset": "dtd_test",
            "gap": gap,
            "acc": acc,
            "train": train_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            base_name, "dtd_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            train_info.get("train_time_sec", 0.0),
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Tiny-ImageNet-200 zero-shot (val)
        # ------------------------------------------------------------
        print("[Eval] Tiny-ImageNet-200 val zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval_alignclip(
            model, tiny_val_ds,
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
            "model": base_name,
            "dataset": "tiny-imagenet-200_val",
            "gap": gap,
            "acc": acc,
            "train": train_info,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            base_name, "tiny-imagenet-200_val",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            train_info.get("train_time_sec", 0.0),
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
