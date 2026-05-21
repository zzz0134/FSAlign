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
    try:
        from transformers.models.siglip.modeling_siglip import SiglipModel
        from transformers.models.siglip.processing_siglip import SiglipProcessor
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
    keys = [k.strip().lower() for k in args.models.split(",") if k.strip()]
    out: List[Tuple[str, VLBackbone]] = []
    for k in keys:
        if k == "clip":
            out.append((f"clip:{args.clip_model}:openai", CLIPWrapper(args.clip_model, device=device)))
        elif k == "openclip":
            out.append((f"open_clip:{args.openclip_model}:{args.openclip_pretrained}",
                        OpenCLIPWrapper(args.openclip_model, args.openclip_pretrained, device=device)))
        elif k == "siglip":
            out.append((f"siglip:{args.siglip_name}", SigLIPWrapper(args.siglip_name, device=device)))
        else:
            raise ValueError(f"Unknown model key: {k}")
    return out


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
                      df: float,
                      batch_size: int) -> torch.Tensor:
    device = y.device
    radii_t = torch.tensor(radii, device=device, dtype=y.dtype)
    rho_t = torch.tensor(rho_list, device=device, dtype=y.dtype)
    losses = []
    for batch in anchor_idx.split(batch_size):
        sim = y[batch] @ y.T  # [B, N]
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
        counts_r = (dist.unsqueeze(-1) <= radii_t).sum(dim=1).float()  # [B, R]
        for rho in rho_t:
            counts_rho = (dist.unsqueeze(-1) <= (radii_t * rho)).sum(dim=1).float()
            num = counts_rho - (rho ** df) * counts_r
            den = counts_rho + (rho ** df) * counts_r
            term = (num / (den + 1e-12)) ** 2
            losses.append(term.mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def kernel_trace_and_diag(y: torch.Tensor, s: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # continuous operator approximation: heat kernel ~ Gaussian similarity
    sim = y @ y.T
    dist2 = torch.clamp(2.0 - 2.0 * sim, min=0.0)
    k = torch.exp(-dist2 / (2.0 * (s ** 2)))
    row_sum = k.sum(dim=1)
    diag = 1.0 / (row_sum + 1e-12)
    trace = diag.sum()
    return trace, diag


def compute_spectral_losses(y_img: torch.Tensor,
                            y_txt: torch.Tensor,
                            diffusion_scales: List[float],
                            df: float,
                            dw: float,
                            alpha: float) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float], torch.Tensor, torch.Tensor]:
    device = y_img.device
    y_img = y_img.double()
    y_txt = y_txt.double()
    s_list = diffusion_scales
    d_s = 2.0 * df / dw

    heat_img = []
    heat_txt = []
    diag_img = []
    diag_txt = []

    for s in s_list:
        tr_i, diag_i = kernel_trace_and_diag(y_img, s)
        tr_t, diag_t = kernel_trace_and_diag(y_txt, s)
        heat_img.append(tr_i)
        heat_txt.append(tr_t)
        diag_img.append(diag_i)
        diag_txt.append(diag_t)

    heat_img_t = torch.stack(heat_img)
    heat_txt_t = torch.stack(heat_txt)

    s_t = torch.tensor(s_list, device=device, dtype=y_img.dtype)
    ratio_target = (s_t[1:] / s_t[:-1]) ** (-d_s / (2.0 * alpha))
    ratio_img = heat_img_t[1:] / (heat_img_t[:-1] + 1e-12)
    ratio_txt = heat_txt_t[1:] / (heat_txt_t[:-1] + 1e-12)
    l_spec = 0.5 * ((ratio_img - ratio_target) ** 2).mean() + 0.5 * ((ratio_txt - ratio_target) ** 2).mean()

    # log-spaced quadrature weights
    log_s = torch.log(s_t)
    w = torch.zeros_like(s_t)
    if len(s_t) > 1:
        w[0] = (log_s[1] - log_s[0]) / 2.0
        w[-1] = (log_s[-1] - log_s[-2]) / 2.0
        for i in range(1, len(s_t) - 1):
            w[i] = (log_s[i + 1] - log_s[i - 1]) / 2.0
    w = w * s_t

    q = d_s / (2.0 * alpha) + 1.0
    coeff = (w * (s_t ** (q - 1.0)) / math.gamma(q))[:, None]
    diag_img_t = torch.stack(diag_img, dim=0)
    diag_txt_t = torch.stack(diag_txt, dim=0)
    zeta_img = (coeff * diag_img_t).sum(dim=0)
    zeta_txt = (coeff * diag_txt_t).sum(dim=0)
    j_match = ((zeta_img - zeta_txt) ** 2).mean()

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
                           ) -> Tuple[Dict[str, Dict], Dict[str, List[float]]]:
    device = torch.device(args.device)
    img_x = img_x.to(device)
    txt_x = txt_x.to(device)
    n, d = img_x.shape
    if align_labels is not None:
        align_labels = align_labels.to(device)

    layer_img = LoRALinear(d, args.lora_rank, args.lora_alpha, device)
    layer_txt = LoRALinear(d, args.lora_rank, args.lora_alpha, device)
    train_params = [p for p in list(layer_img.parameters()) + list(layer_txt.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(train_params, lr=args.train_lr)

    history = {"L_dbl": [], "L_spec": [], "J_match": [], "L_align": [], "L_orth": [], "total": []}
    if args.early_stop:
        history["val_total"] = []

    # split train/val indices for early stopping (internal split)
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

    # optional caption pool for multi-caption training
    cap_text = None
    cap_indices = None
    if caption_pool is not None:
        cap_text, cap_indices = caption_pool
        cap_text = cap_text.to(device)

    # optional external val pool for Karpathy split
    val_img_x = None
    val_cap_text = None
    val_cap_indices = None
    val_cap2img = None
    if args.early_stop and args.val_split == "karpathy" and val_pool is not None:
        val_img_x, val_cap_text, val_cap_indices, val_cap2img = val_pool
        val_img_x = val_img_x.to(device)
        val_cap_text = val_cap_text.to(device)

    def _sample_txt_for(img_indices: torch.Tensor,
                        pool_text: Optional[torch.Tensor],
                        pool_indices: Optional[List[List[int]]],
                        fallback: torch.Tensor) -> torch.Tensor:
        if pool_text is None or pool_indices is None:
            return fallback[img_indices]
        # sample one caption per image
        sel = [random.choice(pool_indices[i]) for i in img_indices.tolist()]
        sel_t = torch.tensor(sel, device=device, dtype=torch.long)
        return pool_text[sel_t]

    def _forward(img_feats: torch.Tensor,
                 txt_feats: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 align_img_feats: Optional[torch.Tensor] = None,
                 align_txt_feats: Optional[torch.Tensor] = None):
        y_img = apply_lora_mix(img_feats, layer_img, args.lora_mix)
        y_txt = apply_lora_mix(txt_feats, layer_txt, args.lora_mix)

        anchors = torch.randperm(img_feats.shape[0], device=device)[:min(args.train_anchors, img_feats.shape[0])]
        l_dbl = 0.5 * (
            compute_ball_loss(y_img, anchors, radii, rho_list, args.df, args.anchor_batch) +
            compute_ball_loss(y_txt, anchors, radii, rho_list, args.df, args.anchor_batch)
        )

        spec_idx = torch.randperm(img_feats.shape[0], device=device)[:min(args.spectral_samples, img_feats.shape[0])]
        l_spec, j_match, _, _, _, _ = compute_spectral_losses(
            y_img[spec_idx], y_txt[spec_idx], diffusion_scales, args.df, args.dw, args.alpha
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
                logits = (y_img_a @ y_txt_a.T) / max(args.align_temp, 1e-6)
                lab = torch.arange(y_img_a.shape[0], device=device)
                l_align = 0.5 * (
                    F.cross_entropy(logits, lab) +
                    F.cross_entropy(logits.T, lab)
                )
            else:
                if align_img_feats is not None:
                    y_img_a = apply_lora_mix(align_img_feats, layer_img, args.lora_mix)
                else:
                    y_img_a = y_img
                if align_txt_feats is not None:
                    y_txt_a = apply_lora_mix(align_txt_feats, layer_txt, args.lora_mix)
                else:
                    y_txt_a = y_txt
                logits = (y_img_a @ y_txt_a.T) / max(args.align_temp, 1e-6)
                lab = torch.arange(y_img_a.shape[0], device=device)
                l_align = 0.5 * (
                    F.cross_entropy(logits, lab) +
                    F.cross_entropy(logits.T, lab)
                )
        else:
            l_align = torch.tensor(0.0, device=device)

        l_orth = 0.0
        if args.lambda_orth > 0:
            eye = torch.eye(d, device=device, dtype=y_img.dtype)
            w_i = layer_img.base.weight
            w_t = layer_txt.base.weight
            l_orth = ((w_i.T @ w_i - eye) ** 2).mean() + ((w_t.T @ w_t - eye) ** 2).mean()

        reg = 0.0
        if args.train_reg > 0:
            eye = torch.eye(d, device=device, dtype=y_img.dtype)
            reg = ((layer_img.base.weight - eye) ** 2).mean() + ((layer_txt.base.weight - eye) ** 2).mean()

        total = (
            args.lambda_dbl * l_dbl +
            args.lambda_spec * l_spec +
            args.lambda_match * j_match +
            args.lambda_align * l_align +
            args.lambda_orth * l_orth +
            args.train_reg * reg
        )
        return total, l_dbl, l_spec, j_match, l_align, l_orth

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    with torch.enable_grad():
        for epoch in range(1, args.train_epochs + 1):
            opt.zero_grad()

            img_train = img_x[train_idx]
            txt_train = _sample_txt_for(train_idx, cap_text, cap_indices, txt_x)
            lbl_train = align_labels[train_idx] if align_labels is not None else None

            align_idx = train_idx
            if args.align_samples > 0 and align_idx.numel() > args.align_samples:
                pick = torch.randperm(align_idx.numel(), device=device)[:args.align_samples]
                align_idx = align_idx[pick]
            align_img_feats = img_x[align_idx]
            align_txt_feats = _sample_txt_for(align_idx, cap_text, cap_indices, txt_x)

            total, l_dbl, l_spec, j_match, l_align, l_orth = _forward(
                img_train, txt_train, lbl_train, align_img_feats, align_txt_feats
            )
            total.backward()
            opt.step()

            history["L_dbl"].append(float(l_dbl.item()))
            history["L_spec"].append(float(l_spec.item()))
            history["J_match"].append(float(j_match.item()))
            history["L_align"].append(float(l_align.item()))
            history["L_orth"].append(float(l_orth if isinstance(l_orth, float) else l_orth.item()))
            history["total"].append(float(total.item()))

            # validation for early stopping
            if val_idx is not None or val_img_x is not None:
                with torch.no_grad():
                    if val_img_x is not None:
                        img_val = val_img_x
                        txt_val = _sample_txt_for(
                            torch.arange(val_img_x.shape[0], device=device),
                            val_cap_text, val_cap_indices, txt_x
                        )
                        align_idx = torch.arange(val_img_x.shape[0], device=device)
                        if args.align_samples > 0 and align_idx.numel() > args.align_samples:
                            pick = torch.randperm(align_idx.numel(), device=device)[:args.align_samples]
                            align_idx = align_idx[pick]
                        align_img_feats = val_img_x[align_idx]
                        align_txt_feats = _sample_txt_for(align_idx, val_cap_text, val_cap_indices, txt_x)
                        val_total, _, _, _, _, _ = _forward(
                            img_val, txt_val, None, align_img_feats, align_txt_feats
                        )
                        v = float(val_total.item())
                    else:
                        img_val = img_x[val_idx]
                        txt_val = _sample_txt_for(val_idx, cap_text, cap_indices, txt_x)
                        val_total, _, _, _, _, _ = _forward(img_val, txt_val)
                        v = float(val_total.item())
                history["val_total"].append(v)
                improved = v < (best_val - args.min_delta)
                if improved:
                    best_val = v
                    bad_epochs = 0
                    best_state = {
                        "img": layer_img.state_dict(),
                        "txt": layer_txt.state_dict(),
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
                    f"L_dbl={l_dbl.item():.4f} L_spec={l_spec.item():.4f} "
                    f"J_match={j_match.item():.4e} L_align={l_align.item():.4f} "
                    f"L_orth={float(l_orth if isinstance(l_orth, float) else l_orth.item()):.4f} "
                    f"total={total.item():.4f}"
                )

    if best_state is not None:
        layer_img.load_state_dict(best_state["img"])
        layer_txt.load_state_dict(best_state["txt"])

    lora_state = {
        "img": layer_img.state_dict(),
        "txt": layer_txt.state_dict(),
        "rank": layer_img.rank,
        "alpha": layer_img.alpha,
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
        "lora_mix": args.lora_mix,
        "multi_caption": bool(args.multi_caption),
        "caption_agg": str(args.caption_agg),
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
    """
    Dataset item: (PIL, [captions])
    Train split is used for LoRA fitting, val split for early-stop (optional),
    and test split for final evaluation.
    We compute:
      - image feats for images
      - text feats for all captions
      - i2t recall: image -> any caption that belongs to that image
      - t2i recall: caption -> its image
      - gap metrics using paired (image, first-caption) for each image
    """
    def encode_retrieval_feats(ds: Dataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
        loader = DataLoader(
            ds,
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
        # caption pool per image for multi-caption training
        cap_indices = [[] for _ in range(image_feats.size(0))]
        for cap_idx, img_idx in enumerate(cap2img):
            cap_indices[img_idx].append(cap_idx)

        pair_map = torch.tensor(first_cap, dtype=torch.long, device=device)
        paired_text = text_feats[pair_map]
        cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)
        return image_feats, text_feats, cap2img_t, pair_map, paired_text, cap_indices

    train_img, train_text, _, train_pair_map, train_paired_text, train_cap_indices = encode_retrieval_feats(train_dataset)
    test_img, test_text, cap2img_t, pair_map, paired_text, _ = encode_retrieval_feats(test_dataset)

    val_pool = None
    if args.early_stop and args.val_split == "karpathy" and val_dataset is not None:
        val_img, val_text, val_cap2img_t, _, _, val_cap_indices = encode_retrieval_feats(val_dataset)
        val_pool = (val_img, val_text, val_cap_indices, val_cap2img_t)

    # optional postprocess
    method_info = None
    if args.train_epochs > 0 or args.lora_state:
        _, _, method_info, lora_state = maybe_postprocess(
            train_img, train_paired_text, args, tag=tag, out_dir=out_dir,
            caption_pool=(train_text, train_cap_indices),
            val_pool=val_pool
        )
        if lora_state is not None:
            layer_img, layer_txt = build_lora_layers(lora_state, args.device)
            test_img = apply_lora_state(test_img.to(device), layer_img, args.lora_mix)
            test_text = apply_lora_state(test_text.to(device), layer_txt, args.lora_mix)
            paired_text = test_text[pair_map]

    gap = {
        "centroid_distance": centroid_distance(test_img, paired_text),
        "relative_modality_gap": relative_modality_gap(test_img, paired_text, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(test_img, paired_text, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(test_img, paired_text),
    }

    # chunked recall for GPU memory control
    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = test_img.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = test_img[s:e] @ test_text.t()  # GPU
            topk = torch.topk(sims, k=K, dim=1).indices
            img_ids = torch.arange(s, e, device=device).unsqueeze(1)
            mapped = cap2img_t[topk]
            hit = (mapped == img_ids).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Nimg)

    def recall_t2i(K: int) -> float:
        correct = 0
        Ncap = test_text.size(0)
        chunk = 1024
        for s in range(0, Ncap, chunk):
            e = min(Ncap, s + chunk)
            sims = test_text[s:e] @ test_img.t()  # GPU
            topk = torch.topk(sims, k=K, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Ncap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}
    extra = {"n_images": float(test_img.size(0)), "n_captions": float(test_text.size(0))}
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
        xs.append(x)
        labels_all.append(labels)
        ys.append(W[labels.to(device, non_blocking=True)])
        total += b

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    labels_all_t = torch.cat(labels_all, dim=0).to(device)

    # optional postprocess: only load existing LoRA for zero-shot eval
    method_info = None
    if args.lora_state:
        _, _, method_info, lora_state = maybe_postprocess(
            x_all, y_all, args, tag=tag, out_dir=out_dir, align_labels=labels_all_t
        )
        if lora_state is not None:
            layer_img, layer_txt = build_lora_layers(lora_state, args.device)
            x_all = apply_lora_state(x_all.to(device), layer_img, args.lora_mix)
            W = apply_lora_state(W.to(device), layer_txt, args.lora_mix)
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
    acc = {"top1": 100.0 * correct1 / float(x_all.shape[0]), "top5": 100.0 * correct5 / float(x_all.shape[0]), "n": float(x_all.shape[0])}
    return gap, acc, method_info


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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--models", type=str, default="clip,siglip,openclip")
    ap.add_argument("--model-size", type=int, choices=[16, 32], default=None)
    ap.add_argument("--clip-model", type=str, default="ViT-B-32")
    ap.add_argument("--openclip-model", type=str, default="ViT-B-32")
    ap.add_argument("--openclip-pretrained", type=str, default="openai")
    ap.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--max-coco", type=int, default=5000)
    ap.add_argument("--max-flickr", type=int, default=5000)
    ap.add_argument("--max-cls", type=int, default=10000)
    ap.add_argument("--only-zeroshot", action="store_true")

    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)

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
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--val-split", type=str, default="internal", choices=["internal", "karpathy"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-delta", type=float, default=0.0)

    ap.add_argument("--df", type=float, default=2.0)
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

    args = ap.parse_args()
    seed_all(args.seed)

    if not args.lora_state:
        args.lora_state = ""

    # optional unified model size for CLIP/OpenCLIP
    if args.model_size is not None:
        size_tag = "ViT-B-16" if args.model_size == 16 else "ViT-B-32"
        if args.clip_model == "ViT-B-32":
            args.clip_model = size_tag
        if args.openclip_model == "ViT-B-32":
            args.openclip_model = size_tag

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
                "", "",
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
                "", "",
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
            acc["top1"], acc["top5"],
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
            acc["top1"], acc["top5"],
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
