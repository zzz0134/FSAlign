#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperbolic (Angle-based) Baseline from:
  "Accept the Modality Gap: An Exploration in the Hyperbolic Space" (CVPR 2024)

- Map CLIP image/text features to Lorentz (hyperbolic) space via exponential map (Eq. 14).
- Use exterior angle (Eq. 9) as cross-modal similarity for zero-shot classification and retrieval.
- Metrics unified with baseline#1: Modality Gap (centroid, Frechet, RMG), zero-shot top1/top5, retrieval R@1/5/10.
- Add runtime (sec) for each dataset and overall.

Datasets:
- CIFAR-100, Tiny-ImageNet-200, DTD -> Zero-shot Top1/Top5
- MSCOCO, Flickr30k                 -> I2T/T2I R@1/R@5/R@10

Results: ./results/hyperclip_baseline_{timestamp}.json/.csv
"""

import os
import sys
import time
import json
import math
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tvds
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy.linalg import sqrtm

# ----------------------------
# CLIP
# ----------------------------
import open_clip


# ----------------------------
# Transforms (zero-shot friendly)
# ----------------------------
def build_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


# ----------------------------
# Tiny-ImageNet-200 (auto-download)
# ----------------------------
class TinyImageNet200(Dataset):
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root: str, split: str = "val", transform=None, auto_download: bool = True):
        self.root_base = Path(root).expanduser().resolve()
        self.root = self.root_base / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        assert split in ["train", "val"]

        self._ensure_exists(auto_download=auto_download)

        wnids_path = self.root / "wnids.txt"
        words_path = self.root / "words.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"{wnids_path} missing.")
        if not words_path.exists():
            raise FileNotFoundError(f"{words_path} missing.")

        self.wnids = [x.strip() for x in wnids_path.read_text().splitlines() if x.strip()]
        wnid_to_words = {}
        for line in words_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            wnid_to_words[parts[0]] = parts[1]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.idx_to_classname = {}
        for wnid in self.wnids:
            short = wnid_to_words.get(wnid, wnid).split(",")[0].split(";")[0]
            self.idx_to_classname[self.class_to_idx[wnid]] = short

        self.samples: List[Tuple[str, int]] = []
        if split == "train":
            train_dir = self.root / "train"
            for wnid in self.wnids:
                img_dir = train_dir / wnid / "images"
                if not img_dir.exists():
                    continue
                for img_name in img_dir.glob("*.JPEG"):
                    self.samples.append((str(img_name), self.class_to_idx[wnid]))
        else:
            val_dir = self.root / "val"
            annot = val_dir / "val_annotations.txt"
            if not annot.exists():
                raise FileNotFoundError(f"{annot} missing.")
            mapping = {}
            for line in annot.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                fname, wnid = parts[0], parts[1]
                mapping[fname] = wnid
            img_dir = val_dir / "images"
            for img_name in img_dir.glob("*.JPEG"):
                wnid = mapping[img_name.name]
                self.samples.append((str(img_name), self.class_to_idx[wnid]))
        if len(self.samples) == 0:
            raise RuntimeError("No images in Tiny-ImageNet-200.")

    def _ensure_exists(self, auto_download: bool):
        need = not self.root.exists() or not (self.root / "wnids.txt").exists()
        if not need:
            return
        if not auto_download:
            raise FileNotFoundError(f"{self.root} not found. Enable auto_download or place dataset.")
        self._download_and_extract()

    def _download_and_extract(self):
        self.root_base.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_base / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            print(f"[TinyImageNet] Downloading -> {zip_path}")
            import urllib.request, shutil as _shutil
            with urllib.request.urlopen(self.URL) as r, open(zip_path, "wb") as f:
                _shutil.copyfileobj(r, f)
        print(f"[TinyImageNet] Extracting -> {self.root_base}")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root_base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fp, label = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def classes(self) -> List[str]:
        return [self.idx_to_classname[i] for i in range(len(self.idx_to_classname))]


# ----------------------------
# COCO captions
# ----------------------------
class CocoCaptionsEval(Dataset):
    def __init__(self, root: str, split: str = "val", transform=None):
        assert split in ["train", "val"]
        data_root = Path(root).expanduser().resolve() / "coco2017"
        img_dir = data_root / "images" / (f"{'train2017' if split=='train' else 'val2017'}")
        ann = data_root / "annotations" / f"captions_{'train2017' if split=='train' else 'val2017'}.json"
        if not img_dir.exists() or not ann.exists():
            raise FileNotFoundError(f"COCO not found: {img_dir} / {ann}")
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, caps = self.ds[idx]
        return img, caps  # list[str]


# ----------------------------
# Flickr30k (oldest API: root=images folder, ann_file=no download)
# ----------------------------
class Flickr30kEval(Dataset):
    """
    Old torchvision signature:
      Flickr30k(root, ann_file, transform=None, target_transform=None)
    """
    def __init__(self, root: str, split: str = "test", transform=None):
        data_root = Path(root).expanduser().resolve() / "flickr30k"
        images_root = data_root / "flickr30k-images"
        ann_file = data_root / "results_20130124.token"
        if not images_root.exists():
            raise FileNotFoundError(f"Missing images folder: {images_root}")
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing token file: {ann_file}")
        self.ds = tvds.Flickr30k(root=str(images_root), ann_file=str(ann_file), transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, caps = self.ds[idx]
        if isinstance(caps, list):
            captions = caps
        else:
            captions = [str(caps)]
        return img, captions


# ----------------------------
# Collate for variable-length captions
# ----------------------------
def collate_varlen_captions(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    caps_list = []
    for _, caps in batch:
        if isinstance(caps, (list, tuple)):
            caps_list.append(list(caps))
        else:
            caps_list.append([str(caps)])
    return images, caps_list


# ----------------------------
# Hyperbolic (Lorentz) utilities
#   Eq. (2): Lorentz inner product
#   Eq. (3): time component
#   Eq. (14): exponential map (space component)
#   Eq. (9): exterior angle
# ----------------------------
def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x,y: (..., D+1) with [time, space...]
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

def to_lorentz(v: torch.Tensor, c: float) -> torch.Tensor:
    """
    v: (..., D) Euclidean/tangent vector
    returns x: (..., D+1) Lorentz coordinates (time, space...)
    Eq. (14) for space; Eq. (3) for time.
    """
    eps = 1e-9
    norm_v = v.norm(dim=-1, keepdim=True).clamp_min(eps)  # (...,1)
    sqrtc = math.sqrt(c)
    coef = torch.sinh(sqrtc * norm_v) / (sqrtc * norm_v)   # (...,1)
    x_space = coef * v                                     # (...,D)
    # time from Eq. (3): x0 = sqrt(1/c + ||x_space||^2)
    x0 = torch.sqrt(1.0 / c + (x_space * x_space).sum(dim=-1, keepdim=True))
    x = torch.cat([x0, x_space], dim=-1)                  # (..., D+1)
    return x

def exterior_angle(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """
    Eq. (9): ext(x,y) = arccos( ( ytime + xtime * c< x,y >_H ) / ( ||x_space|| * sqrt( (c< x,y >_H)^2 - 1 ) ) )
    x,y: (..., D+1)
    returns angle in radians
    """
    eps = 1e-9
    x0, y0 = x[..., 0], y[..., 0]
    x_space = x[..., 1:]
    lxy = lorentz_inner(x, y)  # (...,)

    num = y0 + x0 * (c * lxy)
    den = x_space.norm(dim=-1).clamp_min(eps) * torch.sqrt((c * lxy) ** 2 - 1.0 + eps)
    val = (num / den).clamp(min=-1.0, max=1.0)
    ang = torch.acos(val)
    return ang  # (...,)

def angle_similarity(x: torch.Tensor, ybank: torch.Tensor, c: float, batch: int = 4096) -> torch.Tensor:
    """
    Compute -angle between x and each row in ybank (so larger is better).
    x: (B, D+1), ybank: (N, D+1)
    return: (B, N)
    """
    out = []
    for i in range(0, ybank.size(0), batch):
        yb = ybank[i:i+batch]  # (b, D+1)
        # broadcast to compute angle for all pairs in current chunk
        # trick: compute lxy for all B x b
        # lxy = -x0*y0 + x_space@y_space^T
        x0 = x[:, :1]    # (B,1)
        y0 = yb[:, :1].T # (1,b)
        xs = x[:, 1:]    # (B,D)
        ys = yb[:, 1:].T # (D,b)
        lxy = -x0 @ y0 + xs @ ys  # (B,b)

        # num/den per Eq.(9)
        num = y0 + x0 * (c * lxy)             # (B,b)
        xs_norm = xs.norm(dim=1, keepdim=True)  # (B,1)
        den = xs_norm * torch.sqrt((c * lxy) ** 2 - 1.0 + 1e-9)
        val = (num / den).clamp(min=-1.0, max=1.0)
        ang = torch.acos(val)  # (B,b)
        out.append(-ang)       # use negative angle as similarity
    return torch.cat(out, dim=1)


# ----------------------------
# CLIP encode -> hyperbolic map
# ----------------------------
@torch.no_grad()
def load_clip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device: str, batch_size: int = 256) -> torch.Tensor:
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        f = model.encode_text(tokens)
        feats.append(f)
    return torch.cat(feats, dim=0)  # (N,D)

@torch.no_grad()
def encode_images(model, images: torch.Tensor) -> torch.Tensor:
    return model.encode_image(images)  # (B,D)


# ----------------------------
# Modality Gap metrics (欧氏统计，统一口径)
# 使用 Lorentz 向量的“空间分量” x_space 作为欧氏特征
# ----------------------------
def _np_trace(mat: np.ndarray) -> float:
    return float(np.trace(mat))

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    m_diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd_sq = m_diff.dot(m_diff) + _np_trace(sigma1 + sigma2 - 2.0 * covmean)
    fd_sq = max(0.0, float(fd_sq))
    return float(math.sqrt(fd_sq))

def modality_gap_metrics_from_lorentz(img_X: torch.Tensor, txt_X: torch.Tensor) -> Dict[str, float]:
    """
    img_X, txt_X: tensors of shape (N, D+1) and (M, D+1) in Lorentz coords.
    We take Euclidean stats on space component only.
    """
    img = img_X[:, 1:].cpu().numpy()
    txt = txt_X[:, 1:].cpu().numpy()
    mu_i = img.mean(axis=0); mu_t = txt.mean(axis=0)
    ci = np.cov(img.T);       ct = np.cov(txt.T)
    centroid = float(np.linalg.norm(mu_i - mu_t))
    fd = frechet_distance(mu_i, ci, mu_t, ct)
    denom = math.sqrt(0.5 * (_np_trace(ci) + _np_trace(ct)) + 1e-12)
    rmg = centroid / denom if denom > 0 else float("nan")
    return {
        "centroid_distance": centroid,
        "frechet_distance": fd,
        "relative_modality_gap": rmg
    }


# ----------------------------
# Zero-shot classification (angle metric)
# ----------------------------
ZS_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
]

def build_prompts(classnames: List[str]) -> List[List[str]]:
    return [[t.format(c) for t in ZS_TEMPLATES] for c in classnames]

@torch.no_grad()
def zeroshot_eval_hyper(
    model, tokenizer, dataloader: DataLoader, classnames: List[str], device: str, c_curv: float
) -> Tuple[float, float, Dict[str, float], torch.Tensor, torch.Tensor]:
    # Build class text bank in hyperbolic space
    class_prompts = build_prompts(classnames)
    class_lorentz = []
    for prompts in tqdm(class_prompts, desc="Encode class prompts (hyper)", leave=False):
        txt = encode_texts(model, tokenizer, prompts, device=device)  # (P,D)
        # optional L2 norm to stabilize exponential map
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-9)
        x = to_lorentz(txt, c=c_curv)  # (P,D+1)
        # 按论文 zero-shot：对每类多个 prompt 的“角度”平均
        # 我们先做 prototype：沿空间分量做均值再回到 Lorentz？严格做法是直接在推理时对角度求均值。
        # 这里保留所有 prompt，预测时对角度做均值。
        class_lorentz.append(x)  # list of (P,D+1)

    correct1 = 0
    correct5 = 0
    n = 0
    all_img_lorentz = []

    for images, labels in tqdm(dataloader, desc="Zero-shot (angle)", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        img_f = encode_images(model, images)
        img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-9)
        x_img = to_lorentz(img_f, c=c_curv)  # (B,D+1)
        all_img_lorentz.append(x_img.cpu())

        # 计算每张图片对每个类的平均角度
        # 为效率：分类分批
        B = x_img.size(0)
        C = len(classnames)
        scores = torch.zeros(B, C, device="cpu")  # use CPU to save VRAM
        for ci in range(C):
            x_txt = class_lorentz[ci].to(device)   # (P,D+1)
            # 对每个 prompt 计算角度，然后均值
            # 用块乘法 angle_similarity 返回 -angle，取均值后再取负号回成 angle
            sim = angle_similarity(x_img, x_txt, c=c_curv, batch=4096)  # (B,P), on GPU
            ang = (-sim).mean(dim=1).cpu()  # (B,)
            scores[:, ci] = -ang  # 越大越好（即角度越小）
        top5 = scores.topk(5, dim=-1).indices  # (B,5)
        correct1 += (top5[:, 0].to(labels.device) == labels).sum().item()
        correct5 += sum([(top5[:, i].to(labels.device) == labels).sum().item() for i in range(5)])
        n += labels.size(0)

    top1 = correct1 / n
    top5 = correct5 / n

    img_all = torch.cat(all_img_lorentz, dim=0)         # (N,D+1)
    # 构造“类文本原型”：把每类所有 prompt 的洛伦兹向量堆起来做统计（欧氏统计用空间分量）
    txt_all = torch.cat([t.cpu() for t in class_lorentz], dim=0)  # (sumP,D+1)
    mg = modality_gap_metrics_from_lorentz(img_all, txt_all)
    return top1, top5, mg, img_all, txt_all


# ----------------------------
# Retrieval (angle metric)
# ----------------------------
@torch.no_grad()
def retrieval_eval_hyper(
    model, tokenizer, ds: Dataset, device: str, c_curv: float,
    batch_size: int = 64, max_items: Optional[int] = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=collate_varlen_captions, persistent_workers=False
    )
    img_lorentz_list = []
    all_captions: List[str] = []
    img_to_caps: Dict[int, List[int]] = {}
    cap_to_imgs: Dict[int, List[int]] = {}

    cap_idx = 0
    img_idx = 0

    for images, caps_list in tqdm(loader, desc="Embed images & gather captions", leave=False):
        images = images.to(device, non_blocking=True)
        f_img = encode_images(model, images)
        f_img = f_img / (f_img.norm(dim=-1, keepdim=True) + 1e-9)
        x_img = to_lorentz(f_img, c=c_curv)  # (B,D+1)
        img_lorentz_list.append(x_img.cpu())

        bsz = images.size(0)
        for j in range(bsz):
            caps = caps_list[j] if isinstance(caps_list[j], (list, tuple)) else [str(caps_list[j])]
            idxs = []
            for c_text in caps:
                all_captions.append(c_text)
                idxs.append(cap_idx)
                cap_to_imgs[cap_idx] = [img_idx]
                cap_idx += 1
            img_to_caps[img_idx] = idxs
            img_idx += 1

        if max_items is not None and img_idx >= max_items:
            break

    img_X = torch.cat(img_lorentz_list, dim=0)  # (N_img,D+1)

    # encode captions -> hyperbolic
    txt_f = encode_texts(model, tokenizer, all_captions, device=device)
    txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-9)
    txt_X = to_lorentz(txt_f, c=c_curv).cpu()  # (N_cap,D+1)

    # Modality Gap（统一口径）
    mg = modality_gap_metrics_from_lorentz(img_X, txt_X)

    # Similarity by negative angle
    with torch.no_grad():
        sim_i2t = angle_similarity(img_X.to(device), txt_X.to(device), c=c_curv, batch=4096).cpu().numpy()
    # recalls
    def recalls(sim_mat: np.ndarray, gt_map: Dict[int, List[int]], ks=(1,5,10)) -> Dict[str, float]:
        order = np.argsort(-sim_mat, axis=1)
        out = {}
        for k in ks:
            correct = 0
            for i in range(order.shape[0]):
                topk = set(order[i, :k].tolist())
                if any(gt in topk for gt in gt_map[i]):
                    correct += 1
            out[f"R@{k}"] = correct / order.shape[0]
        return out

    i2t = recalls(sim_i2t, img_to_caps, ks=(1,5,10))
    t2i = recalls(sim_i2t.T, cap_to_imgs, ks=(1,5,10))
    return mg, i2t, t2i


# ----------------------------
# Runner
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--curvature", type=float, default=1.0, help="Hyperbolic curvature c (>0)")
    ap.add_argument("--max-coco", type=int, default=None)
    ap.add_argument("--max-flickr", type=int, default=None)
    ap.add_argument("--save-dir", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"[HyperCLIP] model={args.model} pretrained={args.pretrained} device={device} c={args.curvature}")
    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)
    model.eval()
    image_tf = build_transforms(args.image_size)

    results = {
        "config": {
            "data_root": args.data_root,
            "device": device,
            "model": args.model,
            "pretrained": args.pretrained,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "curvature": args.curvature,
        },
        "zero_shot": {},
        "retrieval": {},
        "runtime_sec": {}
    }

    t0_all = time.time()

    # ---------- CIFAR-100 ----------
    t0 = time.time()
    print("\n[Eval] CIFAR-100 (zero-shot, angle)")
    cifar_val = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=False, transform=image_tf, download=True)
    cifar_train = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=True, transform=image_tf, download=True)
    cifar_classes = cifar_train.classes
    cifar_loader = DataLoader(cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    cifar_top1, cifar_top5, cifar_mg, cifar_imgX, cifar_txtX = zeroshot_eval_hyper(
        model, tokenizer, cifar_loader, cifar_classes, device, args.curvature
    )
    rt = time.time() - t0
    results["zero_shot"]["cifar100"] = {"top1": cifar_top1, "top5": cifar_top5, "modality_gap": cifar_mg}
    results["runtime_sec"]["cifar100"] = rt
    print(f"CIFAR100: top1={cifar_top1:.4f} top5={cifar_top5:.4f} time={rt:.1f}s")

    # ---------- Tiny-ImageNet-200 ----------
    t0 = time.time()
    print("\n[Eval] Tiny-ImageNet-200 (zero-shot, angle)")
    tiny_ds = TinyImageNet200(root=args.data_root, split="val", transform=image_tf, auto_download=True)
    tiny_loader = DataLoader(tiny_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    tiny_top1, tiny_top5, tiny_mg, tiny_imgX, tiny_txtX = zeroshot_eval_hyper(
        model, tokenizer, tiny_loader, tiny_ds.classes, device, args.curvature
    )
    rt = time.time() - t0
    results["zero_shot"]["tiny_imagenet_200"] = {"top1": tiny_top1, "top5": tiny_top5, "modality_gap": tiny_mg}
    results["runtime_sec"]["tiny_imagenet_200"] = rt
    print(f"TinyImageNet-200: top1={tiny_top1:.4f} top5={tiny_top5:.4f} time={rt:.1f}s")

    # ---------- DTD ----------
    t0 = time.time()
    print("\n[Eval] DTD (zero-shot, angle)")
    dtd_val = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="test", transform=image_tf, download=True)
    dtd_train = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="train", transform=image_tf, download=True)
    dtd_classes = dtd_train.classes
    dtd_loader = DataLoader(dtd_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dtd_top1, dtd_top5, dtd_mg, dtd_imgX, dtd_txtX = zeroshot_eval_hyper(
        model, tokenizer, dtd_loader, dtd_classes, device, args.curvature
    )
    rt = time.time() - t0
    results["zero_shot"]["dtd"] = {"top1": dtd_top1, "top5": dtd_top5, "modality_gap": dtd_mg}
    results["runtime_sec"]["dtd"] = rt
    print(f"DTD: top1={dtd_top1:.4f} top5={dtd_top5:.4f} time={rt:.1f}s")

    # ---------- MSCOCO Retrieval ----------
    t0 = time.time()
    print("\n[Eval] MSCOCO (I2T/T2I, angle)")
    coco_ds = CocoCaptionsEval(root=args.data_root, split="val", transform=image_tf)
    if args.max_coco is not None:
        class _Sub(Dataset):
            def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.base[i]
        coco_ds = _Sub(coco_ds, args.max_coco)
    coco_mg, coco_i2t, coco_t2i = retrieval_eval_hyper(
        model, tokenizer, coco_ds, device, args.curvature, batch_size=args.batch_size, max_items=args.max_coco
    )
    rt = time.time() - t0
    results["retrieval"]["mscoco"] = {"I2T": coco_i2t, "T2I": coco_t2i, "modality_gap": coco_mg}
    results["runtime_sec"]["mscoco"] = rt
    print(f"MSCOCO: I2T={coco_i2t} T2I={coco_t2i} time={rt:.1f}s")

    # ---------- Flickr30k Retrieval ----------
    t0 = time.time()
    print("\n[Eval] Flickr30k (I2T/T2I, angle)")
    flickr_ds = Flickr30kEval(root=args.data_root, split="test", transform=image_tf)
    if args.max_flickr is not None:
        class _SubF(Dataset):
            def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.base[i]
        flickr_ds = _SubF(flickr_ds, args.max_flickr)
    flickr_mg, flickr_i2t, flickr_t2i = retrieval_eval_hyper(
        model, tokenizer, flickr_ds, device, args.curvature, batch_size=args.batch_size, max_items=args.max_flickr
    )
    rt = time.time() - t0
    results["retrieval"]["flickr30k"] = {"I2T": flickr_i2t, "T2I": flickr_t2i, "modality_gap": flickr_mg}
    results["runtime_sec"]["flickr30k"] = rt
    print(f"Flickr30k: I2T={flickr_i2t} T2I={flickr_t2i} time={rt:.1f}s")

    # ---------- Save ----------
    total_time = time.time() - t0_all
    results["runtime_sec"]["total"] = total_time
    print(f"\n[Done] total time: {total_time:.1f}s")

    json_path = os.path.join(args.save_dir, f"hyperclip_baseline_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON -> {json_path}")

    # CSV (flatten)
    rows = []
    def add_row(name: str, metrics: Dict[str, Any]):
        row = {"dataset": name}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for kkk, vvv in vv.items():
                            row[f"{k}.{kk}.{kkk}"] = vvv
                    else:
                        row[f"{k}.{kk}"] = vv
            else:
                row[k] = v
        rows.append(row)

    add_row("cifar100", {"top1": results["zero_shot"]["cifar100"]["top1"], "top5": results["zero_shot"]["cifar100"]["top5"],
                         **{f"mg.{k}": v for k, v in results["zero_shot"]["cifar100"]["modality_gap"].items()},
                         "runtime_sec": results["runtime_sec"]["cifar100"]})
    add_row("tiny_imagenet_200", {"top1": results["zero_shot"]["tiny_imagenet_200"]["top1"], "top5": results["zero_shot"]["tiny_imagenet_200"]["top5"],
                                   **{f"mg.{k}": v for k, v in results["zero_shot"]["tiny_imagenet_200"]["modality_gap"].items()},
                                   "runtime_sec": results["runtime_sec"]["tiny_imagenet_200"]})
    add_row("dtd", {"top1": results["zero_shot"]["dtd"]["top1"], "top5": results["zero_shot"]["dtd"]["top5"],
                    **{f"mg.{k}": v for k, v in results["zero_shot"]["dtd"]["modality_gap"].items()},
                    "runtime_sec": results["runtime_sec"]["dtd"]})
    add_row("mscoco.I2T", {**results["retrieval"]["mscoco"]["I2T"],
                           **{f"mg.{k}": v for k, v in results["retrieval"]["mscoco"]["modality_gap"].items()},
                           "runtime_sec": results["runtime_sec"]["mscoco"]})
    add_row("mscoco.T2I", {**results["retrieval"]["mscoco"]["T2I"],
                           **{f"mg.{k}": v for k, v in results["retrieval"]["mscoco"]["modality_gap"].items()},
                           "runtime_sec": results["runtime_sec"]["mscoco"]})
    add_row("flickr30k.I2T", {**results["retrieval"]["flickr30k"]["I2T"],
                              **{f"mg.{k}": v for k, v in results["retrieval"]["flickr30k"]["modality_gap"].items()},
                              "runtime_sec": results["runtime_sec"]["flickr30k"]})
    add_row("flickr30k.T2I", {**results["retrieval"]["flickr30k"]["T2I"],
                              **{f"mg.{k}": v for k, v in results["retrieval"]["flickr30k"]["modality_gap"].items()},
                              "runtime_sec": results["runtime_sec"]["flickr30k"]})
    rows.append({"dataset": "TOTAL", "runtime_sec": results["runtime_sec"]["total"]})

    csv_path = os.path.join(args.save_dir, f"hyperclip_baseline_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
