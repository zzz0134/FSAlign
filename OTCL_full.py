#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-4 (Paper Reproduction): "Fill the Gap: Quantifying and Reducing the Modality Gap
in Image-Text Representation Learning" (Role et al., 2025)

What this baseline does (kept consistent with baseline1 pipeline):
- Same datasets:
  - Karpathy split retrieval: MSCOCO-2014 + Flickr30k (test split)
  - Zero-shot classification: CIFAR100 / DTD / Tiny-ImageNet-200 (val)
- Same baseline1 metrics exported to CSV:
  - centroid_distance, Relative Modality Gap (RMG), NAS@k, CMAS
  - Retrieval: I2T / T2I Recall@{1,5,10}
  - Classification: top1/top5
  - Runtime per (model, dataset, postproc)
- Same output formats:
  - JSONL (rich) + CSV (flat)

What is new (baseline4):
- Implements paper's post-processing methods to reduce modality gap:
  1) Spectral technique (SPECk): build bipartite graph between images and texts with W = X Y^T,
     adjacency A = [[0, W],[W^T, 0]], random-walk Laplacian L_rw = D^{-1}(D - A) = I - D^{-1}A,
     then take k smallest non-trivial eigenvectors and use rows of F as new embeddings.
     Practical note: A is huge if dense. We sparsify W with top-k edges per row/col and use
     scipy.sparse.linalg.eigsh for truncated eigendecomposition.
  2) Optimal Transport (OT): Laplacian-regularized OT via POT (Python Optimal Transport).
     We fit a transport plan gamma on a subset of paired embeddings, then map:
        X' = normalize_rows(gamma @ X), Y' = normalize_rows(gamma^T @ Y)
     (barycentric-style mapping), then evaluate with transformed embeddings.

- Implements paper's additional heterogeneity metrics on the MIXED database (images+texts):
  - ITR / TIR (ratio of top-1 neighbor modality bias)
  - TMR / IMR (mean rank of the best cross-modal item)
  - FID between image and text embedding distributions (Gaussian assumption)
  These are written into JSONL under "gap_paper" but CSV stays baseline1-compatible.

Run (example):
  python baseline4_fill_the_gap_spectral_ot.py \
    --data-root /work/was598/modilty_gap/tools/data \
    --out-dir /work/was598/modilty_gap/results/baseline4 \
    --models clip,siglip,openclip \
    --postprocs orig,spec60,ot \
    --spec-graph-topk 50 \
    --ot-fit-pairs 5000 \
    --batch-size 128 --num-workers 8 \
    --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
    --nas-k 10

Dependencies (optional):
  - open_clip_torch (for OpenCLIP/CLIP wrappers)
  - transformers (for SigLIP)
  - scipy (REQUIRED for spectral method at n~5000)
  - pot / ot (REQUIRED for OT method)

"""

import os
import csv
import json
import time
import math
import argparse
import random
import inspect
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

# spectral deps
try:
    import scipy
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    scipy = None
    sp = None
    spla = None

# OT deps (POT)
try:
    import ot  # POT library is usually imported as "ot"
except Exception:
    ot = None


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
# Gap metrics (baseline1)
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
# Paper metrics (Fill the Gap)
# ============================================================

@torch.no_grad()
def fid_gaussian(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> float:
    """
    FID^2 = ||mu_x - mu_y||^2 + Tr(Sx + Sy - 2 * (Sx Sy)^{1/2})
    Here compute on embeddings, Gaussian assumption.
    Implementation: do in float64 on CPU for stability.
    """
    x_np = x.detach().float().cpu().numpy().astype(np.float64)
    y_np = y.detach().float().cpu().numpy().astype(np.float64)

    mx = x_np.mean(axis=0)
    my = y_np.mean(axis=0)
    cx = np.cov(x_np, rowvar=False)
    cy = np.cov(y_np, rowvar=False)

    cx = cx + np.eye(cx.shape[0]) * eps
    cy = cy + np.eye(cy.shape[0]) * eps

    try:
        from scipy.linalg import sqrtm
        covmean = sqrtm(cx @ cy)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid2 = ((mx - my) @ (mx - my)) + np.trace(cx + cy - 2.0 * covmean)
        fid2 = float(max(fid2, 0.0))
        return float(math.sqrt(fid2))
    except Exception:
        return float("nan")

@torch.no_grad()
def heterogeneity_indices_itrtir_imrtmr(
    x: torch.Tensor,
    y: torch.Tensor,
    max_items: int = 5000
) -> Dict[str, float]:
    n = min(x.shape[0], y.shape[0], max_items)
    x = x[:n]
    y = y[:n]
    Z = torch.cat([x, y], dim=0)  # (2n,d)
    S = Z @ Z.t()
    diag = torch.arange(2*n, device=Z.device)
    S[diag, diag] = -1e9

    nn = torch.argmax(S, dim=1)  # (2n,)

    img_queries = torch.arange(0, n, device=Z.device)
    txt_queries = torch.arange(n, 2*n, device=Z.device)

    img_nn = nn[img_queries]
    txt_nn = nn[txt_queries]

    img_nn_is_img = (img_nn < n)
    img_nn_is_txt = (img_nn >= n)
    txt_nn_is_txt = (txt_nn >= n)
    txt_nn_is_img = (txt_nn < n)

    i_i = int(img_nn_is_img.sum().item())
    i_t = int(img_nn_is_txt.sum().item())
    t_t = int(txt_nn_is_txt.sum().item())
    t_i = int(txt_nn_is_img.sum().item())

    itr = float("inf") if i_t == 0 and i_i > 0 else (float(i_i) / float(max(i_t, 1)))
    tir = float("inf") if t_i == 0 and t_t > 0 else (float(t_t) / float(max(t_i, 1)))

    S_img = S[img_queries]  # (n,2n)
    S_txt = S[txt_queries]  # (n,2n)

    best_text_score = S_img[:, n:2*n].max(dim=1).values
    tmr = 1.0 + (S_img > best_text_score.unsqueeze(1)).sum(dim=1).float().mean().item()

    best_img_score = S_txt[:, 0:n].max(dim=1).values
    imr = 1.0 + (S_txt > best_img_score.unsqueeze(1)).sum(dim=1).float().mean().item()

    return {
        "ITR": float(itr),
        "TIR": float(tir),
        "TMR": float(tmr),
        "IMR": float(imr),
        "n_pairs": float(n),
    }


# ============================================================
# Post-processing methods (paper section 2.1)
# ============================================================

def _parse_postprocs(spec: str) -> List[Tuple[str, Dict[str, Any]]]:
    items = [s.strip().lower() for s in spec.split(",") if s.strip()]
    out = []
    for it in items:
        if it == "orig":
            out.append(("orig", {}))
        elif it.startswith("spec"):
            k_str = it.replace("spec", "")
            if not k_str.isdigit():
                raise ValueError(f"Bad spec format: {it} (use spec60 / spec120 etc.)")
            out.append((f"spec{k_str}", {"k": int(k_str)}))
        elif it == "ot":
            out.append(("ot", {}))
        else:
            raise ValueError(f"Unknown postproc: {it}")
    return out

@torch.no_grad()
def postproc_spectral(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    graph_topk: int = 50,
    max_items: int = 5000,
    seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if scipy is None or sp is None or spla is None:
        raise RuntimeError("Spectral postproc requires scipy (scipy.sparse + scipy.sparse.linalg).")

    n = min(x.shape[0], y.shape[0], max_items)
    x = x[:n].detach()
    y = y[:n].detach()

    W = (x @ y.t()).float()
    W_cpu = W.detach().cpu().numpy()

    rows = []
    cols = []
    vals = []

    k_row = min(graph_topk, n)
    for i in range(n):
        row = W_cpu[i]
        if k_row < n:
            idx = np.argpartition(row, -k_row)[-k_row:]
        else:
            idx = np.arange(n)
        v = row[idx]
        rows.append(np.full_like(idx, i))
        cols.append(idx)
        vals.append(v)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    rows2 = []
    cols2 = []
    vals2 = []
    for j in range(n):
        col = W_cpu[:, j]
        if k_row < n:
            idx = np.argpartition(col, -k_row)[-k_row:]
        else:
            idx = np.arange(n)
        v = col[idx]
        rows2.append(idx)
        cols2.append(np.full_like(idx, j))
        vals2.append(v)

    rows2 = np.concatenate(rows2)
    cols2 = np.concatenate(cols2)
    vals2 = np.concatenate(vals2)

    rr = np.concatenate([rows, rows2])
    cc = np.concatenate([cols, cols2])
    vv = np.concatenate([vals, vals2])

    W_sp = sp.coo_matrix((vv, (rr, cc)), shape=(n, n)).tocsr()

    zero = sp.csr_matrix((n, n))
    A = sp.bmat([[zero, W_sp], [W_sp.T, zero]], format="csr")

    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, 1e-12)
    Dinv = sp.diags(1.0 / deg)

    I = sp.identity(2*n, format="csr")
    Lrw = I - (Dinv @ A)

    k_eff = min(k + 1, 2*n - 1)
    try:
        evals, evecs = spla.eigsh(Lrw, k=k_eff, which="SM", tol=1e-3, maxiter=2000)
    except Exception as e:
        raise RuntimeError(f"eigsh failed (n={n}, k={k_eff}). Try smaller --max-coco/--max-flickr "
                           f"or smaller --spec-graph-topk. Error: {e}")

    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    if evecs.shape[1] > 1:
        Fmat = evecs[:, 1:1+k]
    else:
        Fmat = evecs[:, :1]

    F_t = torch.from_numpy(Fmat).to(x.device).float()
    x_new = F_t[:n]
    y_new = F_t[n:2*n]
    x_new = l2norm(x_new)
    y_new = l2norm(y_new)

    info = {
        "n_used": int(n),
        "k": int(k),
        "graph_topk": int(graph_topk),
        "eigvals_head": [float(v) for v in evals[:min(len(evals), 10)]],
        "sparsity_edges": int(A.nnz),
    }
    return x_new, y_new, info


# ============================================================
# POT API compatibility helpers (CRITICAL FIX)
# ============================================================

def _filter_kwargs_by_signature(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}

def _map_reg_eta_to_supported_name(init_func, reg_eta: float, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Different POT versions use different names for the Laplacian regularization strength.
    We try a prioritized mapping, and only set the first supported one.
    """
    sig = inspect.signature(init_func)
    params = set(sig.parameters.keys())

    # if eta is supported, keep it
    if "eta" in params:
        kwargs["eta"] = reg_eta
        return kwargs

    # common alternatives across POT versions / forks
    candidates = ["reg", "reg_e", "reg_eta", "epsilon", "eps", "alpha"]
    for name in candidates:
        if name in params:
            kwargs[name] = reg_eta
            return kwargs

    # nothing supported: do not pass any reg parameter (library default)
    return kwargs


@torch.no_grad()
def postproc_ot_laplacian(
    x: torch.Tensor,
    y: torch.Tensor,
    fit_pairs: int = 5000,
    max_items: int = 5000,
    reg_eta: float = 1.0,
    lambda_s: float = 1.0,
    lambda_t: float = 1.0,
    knn_graph: int = 20,
    seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    OT post-processing (paper section 2.1.2) via POT.

    Paper: uses POT "EMDLaplaceTransport" (Laplacian regularization) and fits on 5000 pairs.
    POT API differs across versions:
      - some versions: __init__(..., eta=...) exists; others use reg/reg_e/epsilon or none.
      - some versions: fit(..., a=..., b=...) exists; others do NOT accept a/b and assume uniform.
    This implementation reproduces the paper intent while being robust to POT version differences.
    """
    if ot is None:
        raise RuntimeError("OT postproc requires POT. Install: pip install pot")

    n = min(x.shape[0], y.shape[0], max_items)
    x = x[:n].detach()
    y = y[:n].detach()

    n_fit = min(fit_pairs, n)
    x_fit = x[:n_fit].detach().float().cpu().numpy()
    y_fit = y[:n_fit].detach().float().cpu().numpy()

    # uniform marginals (only used if fit() supports a/b)
    a = np.ones((n_fit,), dtype=np.float64) / float(n_fit)
    b = np.ones((n_fit,), dtype=np.float64) / float(n_fit)

    # build KNN similarity graphs in source/target for Laplacian regularization
    def knn_sim(A: np.ndarray, k: int) -> np.ndarray:
        S = A @ A.T
        np.fill_diagonal(S, -1e9)
        k_eff = min(k, A.shape[0] - 1)
        idx = np.argpartition(S, -k_eff, axis=1)[:, -k_eff:]
        W = np.zeros_like(S, dtype=np.float64)
        rows = np.arange(A.shape[0])[:, None]
        W[rows, idx] = 1.0
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0.0)
        return W

    S_s = knn_sim(x_fit, knn_graph)
    S_t = knn_sim(y_fit, knn_graph)

    # locate transport class
    TransportCls = None
    if hasattr(ot, "da") and hasattr(ot.da, "EMDLaplaceTransport"):
        TransportCls = ot.da.EMDLaplaceTransport
    if TransportCls is None:
        raise RuntimeError("Your POT installation does not provide ot.da.EMDLaplaceTransport. "
                           "Try upgrading POT: pip install -U pot")

    # --- robust __init__ kwargs (NO MORE eta crash) ---
    init_kwargs = {
        "reg_type": "pos",
        "lambda_s": lambda_s,
        "lambda_t": lambda_t,
        "metric": "sqeuclidean",
        # try to set reg parameter under whatever name this POT supports
        "eta": reg_eta,
    }
    init_kwargs = _map_reg_eta_to_supported_name(TransportCls.__init__, reg_eta, init_kwargs)
    init_kwargs = _filter_kwargs_by_signature(TransportCls.__init__, init_kwargs)

    transp = TransportCls(**init_kwargs)

    # --- robust fit kwargs (NO MORE a/b crash) ---
    fit_kwargs = {
        "Xs": x_fit,
        "Xt": y_fit,
        "a": a,
        "b": b,
        "Ss": S_s,
        "St": S_t,
    }
    fit_kwargs = _filter_kwargs_by_signature(transp.fit, fit_kwargs)

    transp.fit(**fit_kwargs)

    gamma = getattr(transp, "coupling_", None)
    if gamma is None:
        raise RuntimeError("OT transport fit did not produce coupling_. Check POT version / inputs.")

    gamma = gamma.astype(np.float64)

    def row_normalize(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        s = M.sum(axis=1, keepdims=True)
        s = np.maximum(s, eps)
        return M / s

    gamma_row = row_normalize(gamma)
    gamma_col = row_normalize(gamma.T)

    x_fit_mapped = gamma_row @ x_fit
    y_fit_mapped = gamma_col @ y_fit

    x_all = x.detach().float().cpu().numpy()
    y_all = y.detach().float().cpu().numpy()

    dx = x_fit_mapped - x_fit
    dy = y_fit_mapped - y_fit

    sim_x = x_all @ x_fit.T
    sim_y = y_all @ y_fit.T
    nn_x = sim_x.argmax(axis=1)
    nn_y = sim_y.argmax(axis=1)

    x_mapped = x_all + dx[nn_x]
    y_mapped = y_all + dy[nn_y]

    x_new = torch.from_numpy(x_mapped).to(x.device).float()
    y_new = torch.from_numpy(y_mapped).to(y.device).float()
    x_new = l2norm(x_new)
    y_new = l2norm(y_new)

    info = {
        "n_used": int(n),
        "n_fit": int(n_fit),
        "reg_eta_requested": float(reg_eta),
        "lambda_s": float(lambda_s),
        "lambda_t": float(lambda_t),
        "knn_graph": int(knn_graph),
        "coupling_sum": float(gamma.sum()),
        "pot_init_kwargs_used": init_kwargs,
        "pot_fit_kwargs_used": list(fit_kwargs.keys()),
    }
    return x_new, y_new, info

@torch.no_grad()
def apply_postproc(
    name: str,
    cfg: Dict[str, Any],
    x: torch.Tensor,
    y: torch.Tensor,
    args
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if name == "orig":
        return x, y, {"method": "orig"}
    if name.startswith("spec"):
        return postproc_spectral(
            x, y,
            k=int(cfg["k"]),
            graph_topk=args.spec_graph_topk,
            max_items=args.postproc_max_items,
            seed=args.seed
        )[0:3]
    if name == "ot":
        return postproc_ot_laplacian(
            x, y,
            fit_pairs=args.ot_fit_pairs,
            max_items=args.postproc_max_items,
            reg_eta=args.ot_eta,
            lambda_s=args.ot_lambda_s,
            lambda_t=args.ot_lambda_t,
            knn_graph=args.ot_knn_graph,
            seed=args.seed
        )[0:3]
    raise ValueError(name)


# ============================================================
# Retrieval eval (Karpathy) with post-processing
# ============================================================

@torch.no_grad()
def retrieval_eval_with_postproc(
    model: VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int,
    postproc_name: str,
    postproc_cfg: Dict[str, Any],
    args
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any]]:
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

    x_pp, y_pp, pp_info = apply_postproc(postproc_name, postproc_cfg, image_feats, paired_text, args)

    with torch.no_grad():
        anchors_orig = paired_text
        anchors_new = y_pp
        if postproc_name == "orig":
            text_pp = text_feats
        else:
            Ncap = text_feats.size(0)
            chunk = 2048
            mapped = []
            for s in range(0, Ncap, chunk):
                e = min(Ncap, s + chunk)
                sim = text_feats[s:e] @ anchors_orig.t()
                nn = torch.argmax(sim, dim=1)
                mapped.append(anchors_new[nn])
            text_pp = torch.cat(mapped, dim=0)
            text_pp = l2norm(text_pp)

    gap = {
        "centroid_distance": centroid_distance(x_pp, y_pp),
        "relative_modality_gap": relative_modality_gap(x_pp, y_pp, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(x_pp, y_pp, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(x_pp, y_pp),
    }

    try:
        paper_metrics = heterogeneity_indices_itrtir_imrtmr(x_pp, y_pp, max_items=min(args.paper_max_items, x_pp.size(0)))
        paper_metrics["FID"] = fid_gaussian(x_pp, y_pp)
    except Exception as e:
        paper_metrics = {"error": str(e)}

    cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)

    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = x_pp.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = x_pp[s:e] @ text_pp.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            img_ids = torch.arange(s, e, device=device).unsqueeze(1)
            mapped = cap2img_t[topk]
            hit = (mapped == img_ids).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Nimg)

    def recall_t2i(K: int) -> float:
        correct = 0
        Ncap = text_pp.size(0)
        chunk = 1024
        for s in range(0, Ncap, chunk):
            e = min(Ncap, s + chunk)
            sims = text_pp[s:e] @ x_pp.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Ncap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}
    extra = {"n_images": float(x_pp.size(0)), "n_captions": float(text_pp.size(0))}
    return gap, i2t, t2i, extra, {"postproc": postproc_name, "postproc_info": pp_info, "gap_paper": paper_metrics}


# ============================================================
# Zero-shot classification (kept baseline1 accuracy; optional gap postproc on (x, y_true))
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
def zeroshot_eval_with_postproc_gaponly(
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
    postproc_name: str,
    postproc_cfg: Dict[str, Any],
    args
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
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
    ys_true: List[torch.Tensor] = []

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
        ys_true.append(W[labels.to(device, non_blocking=True)])

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys_true, dim=0)

    try:
        x_pp, y_pp, pp_info = apply_postproc(postproc_name, postproc_cfg, x_all, y_all, args)
        gap = {
            "centroid_distance": centroid_distance(x_pp, y_pp),
            "relative_modality_gap": relative_modality_gap(x_pp, y_pp, intra_samples=intra_samples),
            f"NAS@{nas_k_val}": nas_k(x_pp, y_pp, k=nas_k_val, max_items=nas_max_items),
            "CMAS": cmas(x_pp, y_pp),
        }
        try:
            paper_metrics = heterogeneity_indices_itrtir_imrtmr(x_pp, y_pp, max_items=min(args.paper_max_items, x_pp.size(0)))
            paper_metrics["FID"] = fid_gaussian(x_pp, y_pp)
        except Exception as e:
            paper_metrics = {"error": str(e)}
        meta = {"postproc": postproc_name, "postproc_info": pp_info, "gap_paper": paper_metrics}
    except Exception as e:
        gap = {
            "centroid_distance": centroid_distance(x_all, y_all),
            "relative_modality_gap": relative_modality_gap(x_all, y_all, intra_samples=intra_samples),
            f"NAS@{nas_k_val}": nas_k(x_all, y_all, k=nas_k_val, max_items=nas_max_items),
            "CMAS": cmas(x_all, y_all),
        }
        meta = {"postproc": postproc_name, "error": str(e)}

    acc = {"top1": 100.0 * correct1 / float(total), "top5": 100.0 * correct5 / float(total), "n": float(total)}
    return gap, acc, meta


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

    ap.add_argument("--postprocs", type=str, default="orig,spec60,ot",
                    help="comma-separated: orig, spec60, spec120, ot ...")
    ap.add_argument("--postproc-max-items", type=int, default=5000,
                    help="cap n pairs used inside post-processing methods")

    ap.add_argument("--spec-graph-topk", type=int, default=50,
                    help="top-k sparsification for spectral bipartite graph (per row/col)")

    ap.add_argument("--ot-fit-pairs", type=int, default=5000,
                    help="number of paired samples used to fit OT coupling")
    ap.add_argument("--ot-eta", type=float, default=1.0, help="OT Laplacian reg eta (mapped to POT-supported name)")
    ap.add_argument("--ot-lambda-s", type=float, default=1.0, help="OT Laplacian reg lambda_s")
    ap.add_argument("--ot-lambda-t", type=float, default=1.0, help="OT Laplacian reg lambda_t")
    ap.add_argument("--ot-knn-graph", type=int, default=20, help="KNN size for Laplacian reg graphs")

    ap.add_argument("--paper-max-items", type=int, default=5000,
                    help="cap pairs used for paper heterogeneity metrics")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--max-coco", type=int, default=5000)
    ap.add_argument("--max-flickr", type=int, default=5000)
    ap.add_argument("--max-cls", type=int, default=10000)

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
    out_jsonl = out_dir / "baseline4_results.jsonl"
    out_csv = out_dir / "baseline4_results.csv"

    postprocs = _parse_postprocs(args.postprocs)
    print(f"[Postprocs] {postprocs}")

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
        "model", "postproc", "dataset",
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

        for pp_name, pp_cfg in postprocs:
            print("\n------------------------------")
            print(f"[Postproc] {pp_name}")
            print("------------------------------")

            print("[Eval] MSCOCO Karpathy test (I2T/T2I R@K + gap) + postproc")
            t0 = time.time()
            gap, i2t, t2i, extra, meta = retrieval_eval_with_postproc(
                model, coco_test, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_images=args.max_coco,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                postproc_name=pp_name,
                postproc_cfg=pp_cfg,
                args=args
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "postproc": pp_name,
                "dataset": "mscoco2014_karpathy_test",
                "gap": gap,
                "i2t": i2t,
                "t2i": t2i,
                "extra": extra,
                "meta": meta,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, pp_name, "mscoco2014_karpathy_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                i2t["R@1"], i2t["R@5"], i2t["R@10"],
                t2i["R@1"], t2i["R@5"], t2i["R@10"],
                "", "",
                float(t1 - t0),
            ])

            print("[Eval] Flickr30k Karpathy test (I2T/T2I R@K + gap) + postproc")
            t0 = time.time()
            gap, i2t, t2i, extra, meta = retrieval_eval_with_postproc(
                model, flickr_test, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_images=args.max_flickr,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                postproc_name=pp_name,
                postproc_cfg=pp_cfg,
                args=args
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "postproc": pp_name,
                "dataset": "flickr30k_karpathy_test",
                "gap": gap,
                "i2t": i2t,
                "t2i": t2i,
                "extra": extra,
                "meta": meta,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, pp_name, "flickr30k_karpathy_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                i2t["R@1"], i2t["R@5"], i2t["R@10"],
                t2i["R@1"], t2i["R@5"], t2i["R@10"],
                "", "",
                float(t1 - t0),
            ])

            print("[Eval] CIFAR100 zero-shot (top1/top5 + gap) + postproc(gap-only)")
            t0 = time.time()
            gap, acc, meta = zeroshot_eval_with_postproc_gaponly(
                model, cifar_test,
                cifar_classes, CIFAR100_TEMPLATES,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_items=args.max_cls,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                postproc_name=pp_name,
                postproc_cfg=pp_cfg,
                args=args
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "postproc": pp_name,
                "dataset": "cifar100_test",
                "gap": gap,
                "acc": acc,
                "meta": meta,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, pp_name, "cifar100_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                "", "", "",
                "", "", "",
                acc["top1"], acc["top5"],
                float(t1 - t0),
            ])

            print("[Eval] DTD zero-shot (top1/top5 + gap) + postproc(gap-only)")
            t0 = time.time()
            gap, acc, meta = zeroshot_eval_with_postproc_gaponly(
                model, dtd_test,
                dtd_classes, DTD_TEMPLATES,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_items=args.max_cls,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                postproc_name=pp_name,
                postproc_cfg=pp_cfg,
                args=args
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "postproc": pp_name,
                "dataset": "dtd_test",
                "gap": gap,
                "acc": acc,
                "meta": meta,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, pp_name, "dtd_test",
                gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
                "", "", "",
                "", "", "",
                acc["top1"], acc["top5"],
                float(t1 - t0),
            ])

            print("[Eval] Tiny-ImageNet-200 val zero-shot (top1/top5 + gap) + postproc(gap-only)")
            t0 = time.time()
            gap, acc, meta = zeroshot_eval_with_postproc_gaponly(
                model, tiny_val_ds,
                tiny_classes, tiny_templates,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_items=args.max_cls,
                nas_k_val=args.nas_k,
                nas_max_items=args.nas_max_items,
                intra_samples=args.intra_samples,
                postproc_name=pp_name,
                postproc_cfg=pp_cfg,
                args=args
            )
            t1 = time.time()
            rec = {
                "model": model_name,
                "postproc": pp_name,
                "dataset": "tiny-imagenet-200_val",
                "gap": gap,
                "acc": acc,
                "meta": meta,
                "eval_time_sec": float(t1 - t0),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            rows.append([
                model_name, pp_name, "tiny-imagenet-200_val",
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
