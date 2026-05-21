#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-2: INCL (ICLR 2025) Modality Inversion (OTI / OVI)
- Same datasets / splits / metrics / output format as baseline1_ground_truth_karpathy_mscoco2014.py

Retrieval on Karpathy test split for MSCOCO2014 + Flickr30k:
  * For CLIP/OpenCLIP:
      - I2T: use OTI to map each image -> text-encoder space, retrieve among native text features
      - T2I: use OVI to map each caption -> image-encoder space, retrieve among native image features
  * For SigLIP:
      - Fallback to baseline1-style native retrieval (no OTI/OVI) to keep same measurement standard.

Zero-shot image classification on CIFAR100 / Tiny-ImageNet-200 / DTD (same as baseline1)

Gap metrics: centroid distance, Relative Modality Gap (RMG), NAS(k), CMAS
  * For CLIP/OpenCLIP retrieval:
      - gap_i2t is measured in text space: OTI(image) vs paired native text
      - gap_t2i is measured in image space: native image vs OVI(paired text)
      - CSV reports gap_avg = average of scalar metrics (to match baseline1 columns)
      - JSONL includes gap_i2t, gap_t2i, gap_avg
  * For SigLIP fallback retrieval:
      - gap is computed in the usual paired space (native image vs paired native text) and stored as gap_avg.

Important fixes vs earlier buggy version:
  (1) open_clip transformer layout differs by version (batch_first True/False).
      We implement safe forward that tries batch_first (B,L,D) first, then fallback to (L,B,D).
      This fixes attn_mask shape errors like (7,7) vs (128,128).
  (2) OTI/OVI optimization must NOT backprop into model parameters.
      We freeze model parameters and detach constant embeddings (prefix/eot) to avoid
      "Trying to backward through the graph a second time" errors.

Run:
  python baseline2_incl_modality_inversion_karpathy_mscoco2014.py \
    --data-root /work/was598/modilty_gap/tools/data \
    --out-dir /work/was598/modilty_gap/results/baseline2_incl \
    --models clip,siglip,openclip \
    --batch-size 128 --num-workers 8 \
    --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
    --nas-k 10 \
    --oti-steps 150 --ovi-steps 1000 --oti-lr 0.02 --ovi-lr 0.02 \
    --ovi-p 4

Outputs:
  {out_dir}/baseline2_results.jsonl
  {out_dir}/baseline2_results.csv
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
# Backbones
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


def _freeze_module_params(m: nn.Module, freeze: bool = True):
    for p in m.parameters():
        p.requires_grad_(not freeze)

def _safe_transformer_forward(transformer, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    open_clip versions differ:
      - some expect x as (B,L,D) (batch_first=True)
      - some expect x as (L,B,D)
    We try batch_first first; if it errors with attn_mask mismatch / shape, fallback.
    """
    # Try batch_first (B,L,D)
    try:
        if attn_mask is not None:
            return transformer(x, attn_mask=attn_mask)
        return transformer(x)
    except Exception as e1:
        # fallback to (L,B,D)
        try:
            x2 = x.permute(1, 0, 2).contiguous()
            if attn_mask is not None:
                y2 = transformer(x2, attn_mask=attn_mask)
            else:
                y2 = transformer(x2)
            return y2.permute(1, 0, 2).contiguous()
        except Exception as e2:
            # re-raise with context
            raise RuntimeError(
                f"[TransformerForward] Both batch_first and seq_first failed.\n"
                f"batch_first error: {repr(e1)}\n"
                f"seq_first error: {repr(e2)}\n"
                f"x.shape={tuple(x.shape)} attn_mask.shape={(tuple(attn_mask.shape) if attn_mask is not None else None)}"
            )

class OpenCLIPWrapper(VLBackbone):
    """
    OpenCLIP / CLIP via open_clip.

    Adds helpers needed for OTI/OVI:
      - text_forward_from_embeddings()
      - vision_forward_from_patch_embeddings()
    """
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

        # We never want grads for normal eval; inversion will also freeze params explicitly.
        _freeze_module_params(self.model, freeze=True)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feat = self.model.encode_image(dummy)
            self._dim = int(feat.shape[-1])

        self.context_length = int(getattr(self.model, "context_length", 77))

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

        # open_clip encode_text expects token ids (B,L)
        if isinstance(toks, dict):
            if "input_ids" in toks:
                toks = toks["input_ids"]
            else:
                toks = list(toks.values())[0]

        if not torch.is_tensor(toks):
            toks = torch.tensor(toks)

        assert toks.ndim == 2, f"open_clip tokenizer must return 2D tokens (B,L), got {toks.shape}"
        assert toks.shape[0] == len(texts), f"token batch mismatch: toks.shape={toks.shape}, len(texts)={len(texts)}"

        toks = toks.to(self.device, non_blocking=True).long()
        feat = self.model.encode_text(toks).float()
        return l2norm(feat)

    def _get_text_modules(self):
        token_embedding = getattr(self.model, "token_embedding", None)
        positional_embedding = getattr(self.model, "positional_embedding", None)
        transformer = getattr(self.model, "transformer", None)
        ln_final = getattr(self.model, "ln_final", None)
        text_projection = getattr(self.model, "text_projection", None)
        attn_mask = getattr(self.model, "attn_mask", None)
        return token_embedding, positional_embedding, transformer, ln_final, text_projection, attn_mask

    def text_forward_from_embeddings(self, x_emb: torch.Tensor, eot_positions: torch.Tensor) -> torch.Tensor:
        """
        x_emb: (B, L, D_text) token embeddings (WITHOUT positional embedding).
        eot_positions: (B,) indices in [0,L-1] used to pool features.
        Returns l2-normalized projected text features: (B, D_out)
        """
        token_embedding, positional_embedding, transformer, ln_final, text_projection, attn_mask = self._get_text_modules()
        assert token_embedding is not None and positional_embedding is not None and transformer is not None, \
            "OpenCLIP text modules not found."

        B, L, D = x_emb.shape
        assert positional_embedding.shape[0] >= L, f"L={L} exceeds positional_embedding length={positional_embedding.shape[0]}"

        pos = positional_embedding[:L, :].unsqueeze(0).to(dtype=x_emb.dtype, device=x_emb.device)
        x = x_emb + pos  # (B,L,D)

        mask = None
        if attn_mask is not None:
            mask = attn_mask[:L, :L].to(device=x.device)

        # SAFE transformer forward for different open_clip versions
        x = _safe_transformer_forward(transformer, x, attn_mask=mask)  # (B,L,D)

        if ln_final is not None:
            x = ln_final(x)

        idx = eot_positions.view(B, 1, 1).expand(B, 1, x.shape[-1])
        pooled = x.gather(dim=1, index=idx).squeeze(1)  # (B,D)

        if text_projection is not None:
            pooled = pooled @ text_projection

        return l2norm(pooled.float())

    def _get_visual_modules(self):
        visual = getattr(self.model, "visual", None)
        assert visual is not None, "OpenCLIP visual module not found."
        return visual

    def vision_forward_from_patch_embeddings(self, patch_emb: torch.Tensor) -> torch.Tensor:
        """
        patch_emb: (B, U, Dv) patch embedding space (matches visual.positional_embedding dim)
        Returns l2-normalized image features (B, D_out)
        """
        visual = self._get_visual_modules()

        class_embedding = getattr(visual, "class_embedding", None)
        positional_embedding = getattr(visual, "positional_embedding", None)
        ln_pre = getattr(visual, "ln_pre", None)
        transformer = getattr(visual, "transformer", None)
        ln_post = getattr(visual, "ln_post", None)
        proj = getattr(visual, "proj", None)

        assert class_embedding is not None and positional_embedding is not None and transformer is not None, \
            "OpenCLIP visual internals not found (need ViT visual)."

        B, U, Dv = patch_emb.shape
        assert positional_embedding.shape[0] == U + 1, \
            f"Expected positional_embedding length {U+1}, got {positional_embedding.shape[0]}."

        cls = class_embedding.to(dtype=patch_emb.dtype, device=patch_emb.device)
        cls = cls.unsqueeze(0).unsqueeze(0).expand(B, 1, Dv)  # (B,1,Dv)

        x = torch.cat([cls, patch_emb], dim=1)  # (B, U+1, Dv)
        x = x + positional_embedding.to(dtype=x.dtype, device=x.device).unsqueeze(0)

        if ln_pre is not None:
            x = ln_pre(x)

        # visual transformer in open_clip is typically batch_first too; use safe forward
        x = _safe_transformer_forward(transformer, x, attn_mask=None)  # (B,U+1,Dv)

        x = x[:, 0, :]
        if ln_post is not None:
            x = ln_post(x)

        if proj is not None:
            x = x @ proj

        return l2norm(x.float())


class CLIPWrapper(OpenCLIPWrapper):
    def __init__(self, device: str):
        super().__init__(model_name="ViT-B-32", pretrained="openai", device=device)


class SigLIPWrapper(VLBackbone):
    """
    SigLIP: used for baseline1-style eval (retrieval + classification), NO OTI/OVI in this baseline2.
    """
    def __init__(self, hf_name: str, device: str):
        super().__init__(device=device)
        assert SiglipModel is not None and SiglipProcessor is not None, \
            "SigLIP requires transformers>=4.40 with SiglipModel."
        self.hf_name = hf_name
        self.model = SiglipModel.from_pretrained(hf_name).to(device).eval()
        self.proc = SiglipProcessor.from_pretrained(hf_name)

        _freeze_module_params(self.model, freeze=True)

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
# INCL: OTI / OVI
# ============================================================

def _build_nn_repeat_index(U: int, P: int, device: torch.device) -> torch.Tensor:
    """
    Nearest-neighbor interpolation indices mapping U target patches to P source pseudo-patches.
    Returns idx_u of shape (U,) with values in [0,P-1].
    """
    u = torch.arange(U, device=device).float()
    t = (u + 0.5) / float(U)
    p_float = t * float(P) - 0.5
    idx = torch.round(p_float).long().clamp(0, P - 1)
    return idx

@torch.no_grad()
def _first_caption_index_from_cap2img(cap2img: List[int], n_images: int, device: str) -> torch.Tensor:
    first_cap = [-1] * n_images
    for cap_idx, img_idx in enumerate(cap2img):
        if first_cap[img_idx] < 0:
            first_cap[img_idx] = cap_idx
    return torch.tensor(first_cap, dtype=torch.long, device=device)

def oti_invert_images_to_text_space(
    backbone: OpenCLIPWrapper,
    pil_images: List[Image.Image],
    prefix_tokens: torch.Tensor,
    oti_steps: int,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    device: str,
) -> torch.Tensor:
    """
    OTI (Algorithm 1): optimize pseudo-token v* (R=1) in token embedding space
    to minimize 1 - cos( psi_I , psi_T(v*) ).
    Returns: inverted text features psi_T (B, D_out), l2-normalized.
    """
    assert isinstance(backbone, OpenCLIPWrapper), "OTI implemented only for OpenCLIPWrapper/CLIPWrapper here."

    # Freeze model params; gradients only for v
    _freeze_module_params(backbone.model, freeze=True)
    backbone.model.eval()

    with torch.no_grad():
        psi_I = backbone.encode_images(pil_images)  # (B, D_out), normalized

    token_embedding, positional_embedding, transformer, ln_final, text_projection, attn_mask = backbone._get_text_modules()
    assert token_embedding is not None and positional_embedding is not None, "Text modules missing for OTI."

    B = psi_I.shape[0]
    D_text = token_embedding.weight.shape[1]

    prefix_tokens = prefix_tokens.to(backbone.device)
    # prefix_tokens is one sequence; count non-pad (pad is 0 in CLIP BPE)
    Lp = int((prefix_tokens != 0).sum().item())
    L = Lp + 2  # prefix + pseudo + eot
    assert L <= backbone.context_length, f"OTI sequence length L={L} exceeds context_length={backbone.context_length}"

    # Determine eot token id robustly
    # open_clip CLIP BPE typically: EOT=49407, SOT=49406
    # We try to locate EOT in model if possible; else fallback to max token in prefix
    eot_id = None
    if hasattr(backbone.model, "text"):
        # very rare; ignore
        pass
    # fallback:
    eot_id = int(prefix_tokens.max().item())

    prefix_ids = prefix_tokens[:Lp].unsqueeze(0).expand(B, Lp).to(backbone.device).long()

    # IMPORTANT: make prefix/eot embeddings constants WITHOUT grad graph
    with torch.no_grad():
        prefix_emb = token_embedding(prefix_ids).detach()  # (B,Lp,D_text)
        eot_ids = torch.full((B, 1), eot_id, device=backbone.device, dtype=torch.long)
        eot_emb = token_embedding(eot_ids).detach()        # (B,1,D_text)

    # Learnable pseudo-token v*
    v = torch.randn(B, 1, D_text, device=backbone.device, dtype=prefix_emb.dtype, requires_grad=True)

    opt = torch.optim.AdamW([v], lr=lr, betas=betas, weight_decay=weight_decay)

    psi_T = None
    for _ in range(int(oti_steps)):
        opt.zero_grad(set_to_none=True)

        x_emb = torch.cat([prefix_emb, v, eot_emb], dim=1)  # (B,L,D_text)
        eot_pos = torch.full((B,), L - 1, device=backbone.device, dtype=torch.long)

        psi_T = backbone.text_forward_from_embeddings(x_emb, eot_positions=eot_pos)  # (B,D_out)
        cos = (psi_I * psi_T).sum(dim=1).clamp(-1.0, 1.0)
        loss = (1.0 - cos).mean()

        loss.backward()
        opt.step()

    assert psi_T is not None
    return psi_T.detach()

def ovi_invert_texts_to_image_space(
    backbone: OpenCLIPWrapper,
    texts: List[str],
    ovi_p: int,
    ovi_steps: int,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    device: str,
) -> torch.Tensor:
    """
    OVI (Algorithm 2): optimize pseudo-patches w* in patch embedding space.
    Returns: inverted image features psi_I (B, D_out), l2-normalized.
    """
    assert isinstance(backbone, OpenCLIPWrapper), "OVI implemented only for OpenCLIPWrapper/CLIPWrapper here."

    _freeze_module_params(backbone.model, freeze=True)
    backbone.model.eval()

    with torch.no_grad():
        psi_T = backbone.encode_texts(texts)  # (B, D_out), normalized

    visual = backbone._get_visual_modules()
    positional_embedding = getattr(visual, "positional_embedding", None)
    assert positional_embedding is not None, "visual.positional_embedding not found (need ViT visual)."
    U = int(positional_embedding.shape[0] - 1)
    Dv = int(positional_embedding.shape[1])

    B = psi_T.shape[0]
    P = int(ovi_p)
    assert P >= 1
    assert P <= U, f"OVI P={P} cannot exceed U={U}."

    # Learnable pseudo-patches
    w = torch.randn(B, P, Dv, device=backbone.device, dtype=positional_embedding.dtype, requires_grad=True)

    idx_u = _build_nn_repeat_index(U=U, P=P, device=backbone.device)

    opt = torch.optim.AdamW([w], lr=lr, betas=betas, weight_decay=weight_decay)

    psi_I = None
    for _ in range(int(ovi_steps)):
        opt.zero_grad(set_to_none=True)

        patch_emb = w[:, idx_u, :]  # (B,U,Dv)
        psi_I = backbone.vision_forward_from_patch_embeddings(patch_emb)  # (B,D_out)
        cos = (psi_I * psi_T).sum(dim=1).clamp(-1.0, 1.0)
        loss = (1.0 - cos).mean()

        loss.backward()
        opt.step()

    assert psi_I is not None
    return psi_I.detach()


# ============================================================
# Retrieval eval (Karpathy) - baseline1 native version (fallback for SigLIP)
# ============================================================

@torch.no_grad()
def retrieval_eval_native(
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
    Same as baseline1: native retrieval without inversion.
    Dataset item: (PIL, [captions])
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

        feats = model.encode_images(pil_images)
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
# Retrieval eval (Karpathy) with INCL inversion (CLIP/OpenCLIP)
# ============================================================

@torch.no_grad()
def _encode_all_captions(backbone: VLBackbone, all_caps: List[str], device: str, bs_t: int = 256) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for s in range(0, len(all_caps), bs_t):
        chunks.append(backbone.encode_texts(all_caps[s:s+bs_t]))
    return torch.cat(chunks, dim=0)

def retrieval_eval_incl(
    backbone: VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    oti_steps: int,
    ovi_steps: int,
    oti_lr: float,
    ovi_lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    ovi_p: int,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    INCL retrieval for OpenCLIP/CLIP.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_retrieval
    )

    native_image_chunks: List[torch.Tensor] = []
    oti_image_chunks: List[torch.Tensor] = []
    all_caps: List[str] = []
    cap2img: List[int] = []

    n_images = 0

    prefix_tokens = None
    if isinstance(backbone, OpenCLIPWrapper):
        toks = backbone.tokenizer(["a photo of"])
        if isinstance(toks, dict):
            toks = toks["input_ids"] if "input_ids" in toks else list(toks.values())[0]
        if isinstance(toks, torch.Tensor):
            prefix_tokens = toks[0].detach().cpu()
        else:
            prefix_tokens = torch.tensor(toks[0], dtype=torch.long)

    for pil_images, caps_list in loader:
        if max_images is not None and n_images >= max_images:
            break

        b = len(pil_images)
        if max_images is not None and n_images + b > max_images:
            keep = max_images - n_images
            pil_images = pil_images[:keep]
            caps_list = caps_list[:keep]
            b = keep

        native_img = backbone.encode_images(pil_images)
        native_image_chunks.append(native_img)

        for i in range(b):
            caps = caps_list[i]
            for c in caps:
                all_caps.append(c)
                cap2img.append(n_images + i)

        if isinstance(backbone, OpenCLIPWrapper):
            oti_img = oti_invert_images_to_text_space(
                backbone=backbone,
                pil_images=pil_images,
                prefix_tokens=prefix_tokens,
                oti_steps=oti_steps,
                lr=oti_lr,
                betas=betas,
                weight_decay=weight_decay,
                device=device,
            )
        else:
            first_caps = [caps_list[i][0] for i in range(b)]
            oti_img = backbone.encode_texts(first_caps)

        oti_image_chunks.append(oti_img)

        n_images += b

    native_image_feats = torch.cat(native_image_chunks, dim=0)
    oti_image_feats = torch.cat(oti_image_chunks, dim=0)
    n_caps = len(all_caps)

    text_feats = _encode_all_captions(backbone, all_caps, device=device, bs_t=256)

    cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)
    n_images_total = int(native_image_feats.shape[0])

    pair_map = _first_caption_index_from_cap2img(cap2img, n_images_total, device=device)
    paired_text = text_feats[pair_map]

    gap_i2t = {
        "centroid_distance": centroid_distance(oti_image_feats, paired_text),
        "relative_modality_gap": relative_modality_gap(oti_image_feats, paired_text, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(oti_image_feats, paired_text, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(oti_image_feats, paired_text),
    }

    if isinstance(backbone, OpenCLIPWrapper):
        ovi_chunks: List[torch.Tensor] = []
        bs_ovi = max(4, min(32, batch_size))  # keep smaller to avoid massive memory for w
        for s in range(0, n_caps, bs_ovi):
            ovi_chunks.append(
                ovi_invert_texts_to_image_space(
                    backbone=backbone,
                    texts=all_caps[s:s+bs_ovi],
                    ovi_p=ovi_p,
                    ovi_steps=ovi_steps,
                    lr=ovi_lr,
                    betas=betas,
                    weight_decay=weight_decay,
                    device=device,
                )
            )
        ovi_text_feats = torch.cat(ovi_chunks, dim=0)
    else:
        ovi_text_feats = text_feats

    ovi_paired_text = ovi_text_feats[pair_map]
    gap_t2i = {
        "centroid_distance": centroid_distance(native_image_feats, ovi_paired_text),
        "relative_modality_gap": relative_modality_gap(native_image_feats, ovi_paired_text, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(native_image_feats, ovi_paired_text, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(native_image_feats, ovi_paired_text),
    }

    gap_avg = {}
    for k in gap_i2t.keys():
        gap_avg[k] = 0.5 * (float(gap_i2t[k]) + float(gap_t2i[k]))

    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = oti_image_feats.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = oti_image_feats[s:e] @ text_feats.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            img_ids = torch.arange(s, e, device=device).unsqueeze(1)
            mapped = cap2img_t[topk]
            hit = (mapped == img_ids).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Nimg)

    def recall_t2i(K: int) -> float:
        correct = 0
        Ncap = ovi_text_feats.size(0)
        chunk = 1024
        for s in range(0, Ncap, chunk):
            e = min(Ncap, s + chunk)
            sims = ovi_text_feats[s:e] @ native_image_feats.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Ncap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}

    gap_pack = {"gap_i2t": gap_i2t, "gap_t2i": gap_t2i, "gap_avg": gap_avg}
    extra = {"n_images": float(native_image_feats.size(0)), "n_captions": float(text_feats.size(0))}
    return gap_pack, i2t, t2i, extra


# ============================================================
# Zero-shot classification (same as baseline1)
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

    # INCL OTI/OVI hyperparams
    ap.add_argument("--oti-steps", type=int, default=150)
    ap.add_argument("--ovi-steps", type=int, default=1000)
    ap.add_argument("--oti-lr", type=float, default=0.02)
    ap.add_argument("--ovi-lr", type=float, default=0.02)
    ap.add_argument("--adamw-beta1", type=float, default=0.9)
    ap.add_argument("--adamw-beta2", type=float, default=0.999)
    ap.add_argument("--adamw-wd", type=float, default=0.01)

    ap.add_argument("--ovi-p", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_jsonl = out_dir / "baseline2_results.jsonl"
    out_csv = out_dir / "baseline2_results.csv"

    coco_kjson = ensure_karpathy_json(args.data_root, "coco")
    flickr_kjson = ensure_karpathy_json(args.data_root, "flickr30k")

    coco_img_roots = [
        str(Path(args.data_root) / "mscoco2014" / "train2014"),
        str(Path(args.data_root) / "mscoco2014" / "val2014"),
        str(Path(args.data_root) / "coco2014" / "train2014"),
        str(Path(args.data_root) / "coco2014" / "val2014"),
        str(Path(args.data_root) / "coco" / "train2014"),
        str(Path(args.data_root) / "coco" / "val2014"),
    ]

    flickr_img_roots = [
        str(Path(args.data_root) / "flickr30k" / "flickr30k-images"),
        str(Path(args.data_root) / "flickr30k" / "images"),
        str(Path(args.data_root) / "flickr30k"),
    ]

    coco_test = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="test", max_images=None)
    flickr_test = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="test", max_images=None)

    cifar_root = Path(args.data_root) / "cifar100"
    dtd_root = Path(args.data_root) / "dtd"

    cifar_test = tvds.CIFAR100(root=str(cifar_root), train=False, download=False, transform=None)
    dtd_test = tvds.DTD(root=str(dtd_root), split="test", download=False, transform=None)

    tiny_val_ds = TinyImageNet200Val(args.data_root)

    cifar_classes = cifar_test.classes
    dtd_classes = dtd_test.classes
    tiny_classes = tiny_val_ds.classnames
    tiny_templates = ["a photo of a {c}.", "a photo of the {c}."]

    models = make_models(args, device=device)

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

    betas = (float(args.adamw_beta1), float(args.adamw_beta2))
    weight_decay = float(args.adamw_wd)

    for model_name, model in models:
        print("\n==============================")
        print(f"[Model] {model_name}")
        print("==============================")

        # ------------------------------------------------------------
        # COCO retrieval
        # ------------------------------------------------------------
        if isinstance(model, OpenCLIPWrapper):
            print("[Eval] MSCOCO Karpathy test (INCL: I2T via OTI, T2I via OVI)")
            t0 = time.time()
            gap_pack, i2t, t2i, extra = retrieval_eval_incl(
                model, coco_test, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_images=args.max_coco,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                oti_steps=args.oti_steps,
                ovi_steps=args.ovi_steps,
                oti_lr=args.oti_lr,
                ovi_lr=args.ovi_lr,
                betas=betas,
                weight_decay=weight_decay,
                ovi_p=args.ovi_p,
            )
            t1 = time.time()

            rec = {
                "model": model_name,
                "dataset": "mscoco2014_karpathy_test",
                "gap": gap_pack,
                "i2t": i2t,
                "t2i": t2i,
                "extra": extra,
                "incl": {
                    "oti_steps": args.oti_steps,
                    "ovi_steps": args.ovi_steps,
                    "oti_lr": args.oti_lr,
                    "ovi_lr": args.ovi_lr,
                    "adamw_beta1": args.adamw_beta1,
                    "adamw_beta2": args.adamw_beta2,
                    "adamw_wd": args.adamw_wd,
                    "ovi_p": args.ovi_p,
                },
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            gap = gap_pack["gap_avg"]
            rows.append([
                model_name, "mscoco2014_karpathy_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                i2t["R@1"], i2t["R@5"], i2t["R@10"],
                t2i["R@1"], t2i["R@5"], t2i["R@10"],
                "", "",
                float(t1 - t0),
            ])
        else:
            print("[Eval] MSCOCO Karpathy test (fallback native retrieval: no OTI/OVI)")
            t0 = time.time()
            gap, i2t, t2i, extra = retrieval_eval_native(
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
                "model": model_name,
                "dataset": "mscoco2014_karpathy_test",
                "gap": {"gap_avg": gap},
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
                "", "",
                float(t1 - t0),
            ])

        # ------------------------------------------------------------
        # Flickr retrieval
        # ------------------------------------------------------------
        if isinstance(model, OpenCLIPWrapper):
            print("[Eval] Flickr30k Karpathy test (INCL: I2T via OTI, T2I via OVI)")
            t0 = time.time()
            gap_pack, i2t, t2i, extra = retrieval_eval_incl(
                model, flickr_test, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_images=args.max_flickr,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                oti_steps=args.oti_steps,
                ovi_steps=args.ovi_steps,
                oti_lr=args.oti_lr,
                ovi_lr=args.ovi_lr,
                betas=betas,
                weight_decay=weight_decay,
                ovi_p=args.ovi_p,
            )
            t1 = time.time()

            rec = {
                "model": model_name,
                "dataset": "flickr30k_karpathy_test",
                "gap": gap_pack,
                "i2t": i2t,
                "t2i": t2i,
                "extra": extra,
                "incl": {
                    "oti_steps": args.oti_steps,
                    "ovi_steps": args.ovi_steps,
                    "oti_lr": args.oti_lr,
                    "ovi_lr": args.ovi_lr,
                    "adamw_beta1": args.adamw_beta1,
                    "adamw_beta2": args.adamw_beta2,
                    "adamw_wd": args.adamw_wd,
                    "ovi_p": args.ovi_p,
                },
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            gap = gap_pack["gap_avg"]
            rows.append([
                model_name, "flickr30k_karpathy_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                i2t["R@1"], i2t["R@5"], i2t["R@10"],
                t2i["R@1"], t2i["R@5"], t2i["R@10"],
                "", "",
                float(t1 - t0),
            ])
        else:
            print("[Eval] Flickr30k Karpathy test (fallback native retrieval: no OTI/OVI)")
            t0 = time.time()
            gap, i2t, t2i, extra = retrieval_eval_native(
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
                "model": model_name,
                "dataset": "flickr30k_karpathy_test",
                "gap": {"gap_avg": gap},
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
                "", "",
                float(t1 - t0),
            ])

        # ------------------------------------------------------------
        # CIFAR100 zero-shot
        # ------------------------------------------------------------
        print("[Eval] CIFAR100 zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval(
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
            "model": model_name,
            "dataset": "cifar100_test",
            "gap": {"gap_avg": gap},
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
            acc["top1"], acc["top5"],
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # DTD zero-shot
        # ------------------------------------------------------------
        print("[Eval] DTD zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval(
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
            "model": model_name,
            "dataset": "dtd_test",
            "gap": {"gap_avg": gap},
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
            acc["top1"], acc["top5"],
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Tiny-ImageNet-200 zero-shot (val)
        # ------------------------------------------------------------
        print("[Eval] Tiny-ImageNet-200 val zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval(
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
            "model": model_name,
            "dataset": "tiny-imagenet-200_val",
            "gap": {"gap_avg": gap},
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
