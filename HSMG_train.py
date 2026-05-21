#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR 2024 'Accept the Modality Gap' Strict Baseline
- Train (angle-based contrastive in Lorentz space) with *strict* hyperparameters from the paper appendix:
  AdamW (betas 0.9/0.98), weight decay 0.2 (no wd for bias/learnable scalars),
  120000 iterations, global batch 2048, peak LR 5e-4 with 4k warmup then cosine to 0,
  data aug: RandomResizedCrop scale [0.5,1.0] + resize to 224.
- Evaluate with unified metrics:
  * Modality Gap: centroid distance, Frechet distance, relative modality gap (on Lorentz space components).
  * Zero-shot Top1/Top5 on CIFAR100, Tiny-ImageNet-200, DTD (average exterior angle over class prompts).
  * Retrieval I2T/T2I R@1/5/10 on MSCOCO val2017 & Flickr30k test using exterior angle ranking.
- Measure runtime (sec) for each dataset + total.
- Save JSON/CSV results under --save-dir.

Author: modality_gap project
"""

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tvds
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy.linalg import sqrtm

import open_clip


# ------------------------
# Utils
# ------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device, non_blocking=True)
    return x


# ------------------------
# Transforms (strict: RRC scale [0.5,1.0] then resize 224)
# ------------------------
def build_transforms(image_size: int = 224, scale=(0.5, 1.0), is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    else:
        # eval：与 CLIP 风格一致（简洁稳健）
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])


# ------------------------
# Tiny-ImageNet-200 (auto download/extract)
# ------------------------
class TinyImageNet200(Dataset):
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root: str, split: str = "val", transform=None, auto_download: bool = True):
        self.root_base = Path(root).expanduser().resolve()
        self.root = self.root_base / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        assert split in ["train", "val"]

        self._ensure_exists(auto_download)

        wnids_path = self.root / "wnids.txt"
        words_path = self.root / "words.txt"
        if not wnids_path.exists() or not words_path.exists():
            raise FileNotFoundError("wnids.txt / words.txt missing in tiny-imagenet-200.")
        self.wnids = [x.strip() for x in wnids_path.read_text().splitlines() if x.strip()]
        wnid_to_words = {}
        for line in words_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            wnid_to_words[parts[0]] = parts[1]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.idx_to_classname = {
            self.class_to_idx[w]: wnid_to_words.get(w, w).split(",")[0].split(";")[0] for w in self.wnids
        }
        self.samples: List[Tuple[str, int]] = []
        if split == "train":
            tdir = self.root / "train"
            for w in self.wnids:
                idir = tdir / w / "images"
                if not idir.exists(): continue
                for p in idir.glob("*.JPEG"):
                    self.samples.append((str(p), self.class_to_idx[w]))
        else:
            vdir = self.root / "val"
            annot = vdir / "val_annotations.txt"
            if not annot.exists(): raise FileNotFoundError("val_annotations.txt missing.")
            mapping = {}
            for line in annot.read_text().splitlines():
                if not line.strip(): continue
                parts = line.split("\t")
                mapping[parts[0]] = parts[1]
            idir = vdir / "images"
            for p in idir.glob("*.JPEG"):
                w = mapping[p.name]
                self.samples.append((str(p), self.class_to_idx[w]))
        if len(self.samples) == 0:
            raise RuntimeError("Tiny-ImageNet-200 has no images.")

    def _ensure_exists(self, auto_download: bool):
        need = not self.root.exists() or not (self.root / "wnids.txt").exists()
        if not need: return
        if not auto_download:
            raise FileNotFoundError(f"{self.root} not found. Enable auto_download or place dataset.")
        self.root_base.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_base / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            print("[TinyImageNet] Downloading...")
            import urllib.request, shutil as _shutil
            with urllib.request.urlopen(self.URL) as r, open(zip_path, "wb") as f:
                _shutil.copyfileobj(r, f)
        print("[TinyImageNet] Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root_base)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        fp, label = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform is not None: img = self.transform(img)
        return img, label

    @property
    def classes(self) -> List[str]:
        return [self.idx_to_classname[i] for i in range(len(self.idx_to_classname))]


# ------------------------
# COCO & Flickr30k datasets
# ------------------------
class CocoCaptionsEval(Dataset):
    def __init__(self, root: str, split: str = "val", transform=None):
        assert split in ["train", "val"]
        droot = Path(root).expanduser().resolve() / "coco2017"
        img_dir = droot / "images" / ("train2017" if split == "train" else "val2017")
        ann = droot / "annotations" / f"captions_{'train2017' if split=='train' else 'val2017'}.json"
        if not img_dir.exists() or not ann.exists():
            raise FileNotFoundError(f"COCO not found: {img_dir} / {ann}")
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx: int): img, caps = self.ds[idx]; return img, caps  # list[str]

# 训练用：把每张图与其若干 caption 展平成 (image_path, caption) 多对（兼容各版本 torchvision）
class CocoTrainPairs(Dataset):
    """
    Flatten COCO train2017 into (image_path, caption) pairs using the public COCO API.
    This avoids private methods like _load_captions/_load_image so it works across torchvision versions.
    """
    def __init__(self, root: str, transform=None, max_pairs_per_image: int = 5):
        data_root = Path(root).expanduser().resolve() / "coco2017"
        img_dir = data_root / "images" / "train2017"
        ann = data_root / "annotations" / "captions_train2017.json"
        if not img_dir.exists() or not ann.exists():
            raise FileNotFoundError(f"COCO train not found: {img_dir} / {ann}")

        # 用官方 Dataset 只为拿到 COCO API（pycocotools）
        base = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=None)
        # 这些属性在稳定版本里都存在
        self.coco = base.coco            # COCO API 对象
        self.root = Path(base.root)      # 图片根目录
        self.ids = list(base.ids)        # all image ids

        self.items: List[Tuple[str, str]] = []
        for img_id in self.ids:
            # 1) 文件名 -> 拼出图片路径
            img_meta = self.coco.loadImgs([img_id])[0]
            file_name = img_meta["file_name"]
            img_path = str(self.root / file_name)

            # 2) 通过 COCO API 拿该图的所有 caption
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            caps = [a.get("caption", "") for a in anns if "caption" in a]
            if max_pairs_per_image is not None and max_pairs_per_image > 0:
                caps = caps[:max_pairs_per_image]

            # 3) 展平为 (image_path, caption)
            for c in caps:
                if c is None:  # 防御：极少数注释异常
                    continue
                self.items.append((img_path, str(c)))

        if len(self.items) == 0:
            raise RuntimeError("No (image, caption) pairs were found in COCO train2017 annotations.")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, txt = self.items[idx]
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, txt


class Flickr30kEval(Dataset):
    """Oldest API: root should point to images folder; provide ann_file (no download arg)."""
    def __init__(self, root: str, split: str = "test", transform=None):
        droot = Path(root).expanduser().resolve() / "flickr30k"
        images_root = droot / "flickr30k-images"
        ann_file = droot / "results_20130124.token"
        if not images_root.exists(): raise FileNotFoundError(f"Missing images folder: {images_root}")
        if not ann_file.exists(): raise FileNotFoundError(f"Missing token file: {ann_file}")
        self.ds = tvds.Flickr30k(root=str(images_root), ann_file=str(ann_file), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx: int):
        img, caps = self.ds[idx]
        caps = caps if isinstance(caps, list) else [str(caps)]
        return img, caps


# ------------------------
# Collate for variable-length captions
# ------------------------
def collate_varlen_captions(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    caps_list = []
    for _, caps in batch:
        if isinstance(caps, (list, tuple)): caps_list.append(list(caps))
        else: caps_list.append([str(caps)])
    return images, caps_list


# ------------------------
# Hyperbolic (Lorentz) math
# ------------------------
class Curvature(nn.Module):
    """learnable positive curvature c via softplus: c = softplus(raw) + eps"""
    def __init__(self, init_c: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.raw = nn.Parameter(torch.as_tensor(math.log(math.exp(init_c) - 1.0), dtype=torch.float32))
        self.eps = eps
    def forward(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw) + self.eps

def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

def to_lorentz(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map at origin from Euclidean/tangent v (...,D) to Lorentz (...,D+1)."""
    eps = 1e-9
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    sqrtc = torch.sqrt(c)
    coef = torch.sinh(sqrtc * nv) / (sqrtc * nv)
    x_space = coef * v
    x0 = torch.sqrt(1.0 / c + (x_space * x_space).sum(dim=-1, keepdim=True))
    return torch.cat([x0, x_space], dim=-1)

def angle_similarity(x: torch.Tensor, ybank: torch.Tensor, c: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Return negative exterior angle as similarity. x:(B,D+1), ybank:(N,D+1) -> (B,N)"""
    out = []
    x0 = x[:, :1]; xs = x[:, 1:]; xs_norm = xs.norm(dim=1, keepdim=True).clamp_min(1e-9)
    for i in range(0, ybank.size(0), chunk):
        yb = ybank[i:i+chunk]
        y0 = yb[:, :1].T
        ys = yb[:, 1:].T
        lxy = -x0 @ y0 + xs @ ys
        num = y0 + x0 * (c * lxy)
        den = xs_norm * torch.sqrt((c * lxy) ** 2 - 1.0 + 1e-9)
        val = (num / den).clamp(min=-1.0, max=1.0)
        ang = torch.acos(val)
        out.append(-ang)
    return torch.cat(out, dim=1)


# ------------------------
# HyperCLIP wrapper (learnable curvature/temperature)
# ------------------------
class HyperCLIP(nn.Module):
    def __init__(self, clip_model, learn_curvature: bool = True, init_c: float = 1.0,
                 learn_temp: bool = True, init_temp: float = 0.07):
        super().__init__()
        self.clip = clip_model
        self.curv = Curvature(init_c) if learn_curvature else None
        self._c_fixed = nn.Parameter(torch.tensor([init_c]), requires_grad=False) if not learn_curvature else None
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temp))) if learn_temp else None
        self._t_fixed = nn.Parameter(torch.tensor([init_temp]), requires_grad=False) if not learn_temp else None

    @property
    def curvature(self) -> torch.Tensor:
        return self.curv() if self.curv is not None else self._c_fixed

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.logit_scale) if self.logit_scale is not None else self._t_fixed

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_img = self.clip.encode_image(images)
        f_txt = self.clip.encode_text(tokens)
        f_img = f_img / (f_img.norm(dim=-1, keepdim=True) + 1e-9)
        f_txt = f_txt / (f_txt.norm(dim=-1, keepdim=True) + 1e-9)
        c = self.curvature
        x_img = to_lorentz(f_img, c)
        x_txt = to_lorentz(f_txt, c)
        tau = self.temperature
        return x_img, x_txt, c, tau


# ------------------------
# Losses
# ------------------------
def angle_contrastive_loss(x_img: torch.Tensor, x_txt: torch.Tensor, c: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """InfoNCE over negative angle similarities for i2t/t2i."""
    sim_i2t = angle_similarity(x_img, x_txt, c, chunk=4096)
    sim_t2i = angle_similarity(x_txt, x_img, c, chunk=4096)
    logits_i2t = sim_i2t / tau
    logits_t2i = sim_t2i / tau
    labels = torch.arange(x_img.size(0), device=x_img.device)
    loss_i2t = nn.CrossEntropyLoss()(logits_i2t, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits_t2i, labels)
    return 0.5 * (loss_i2t + loss_t2i)

def centroid_regularizer_euclid(x_img: torch.Tensor, x_txt: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
    """Regularize centroids on *space components* (Euclidean)."""
    if lam <= 0: return x_img.new_zeros(())
    xi = x_img[:, 1:]; xt = x_txt[:, 1:]
    mu_i = xi.mean(dim=0, keepdim=True); mu_t = xt.mean(dim=0, keepdim=True)
    return lam * (mu_i - mu_t).pow(2).sum().sqrt()


# ------------------------
# Optim param groups: wd=0 for bias/1D scalars; wd=0.2 otherwise
# ------------------------
def build_optimizer(model: nn.Module, lr: float, betas=(0.9, 0.98), wd: float = 0.2):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=1e-6)


# ------------------------
# LR schedule: warmup (linear) + cosine to 0
# ------------------------
def cosine_lr(step: int, total_steps: int, lr_max: float, warmup: int) -> float:
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


# ------------------------
# Tokenizer
# ------------------------
def get_tokenizer(model_name: str):
    return open_clip.get_tokenizer(model_name)


# ------------------------
# Modality gap (use Lorentz space components only)
# ------------------------
def _np_trace(m: np.ndarray) -> float: return float(np.trace(m))
def frechet_distance(mu1, s1, mu2, s2) -> float:
    md = mu1 - mu2
    covmean = sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    fd_sq = md.dot(md) + _np_trace(s1 + s2 - 2.0 * covmean)
    fd_sq = max(0.0, float(fd_sq))
    return float(math.sqrt(fd_sq))

def modality_gap_from_lorentz(imgX: torch.Tensor, txtX: torch.Tensor) -> Dict[str, float]:
    img = imgX[:, 1:].cpu().numpy(); txt = txtX[:, 1:].cpu().numpy()
    mu_i = img.mean(axis=0); mu_t = txt.mean(axis=0)
    ci = np.cov(img.T); ct = np.cov(txt.T)
    centroid = float(np.linalg.norm(mu_i - mu_t))
    fd = frechet_distance(mu_i, ci, mu_t, ct)
    denom = math.sqrt(0.5 * (_np_trace(ci) + _np_trace(ct)) + 1e-12)
    rmg = centroid / denom if denom > 0 else float("nan")
    return {"centroid_distance": centroid, "frechet_distance": fd, "relative_modality_gap": rmg}


# ------------------------
# Zero-shot (angle) helpers
# ------------------------
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
def encode_texts(clip_model, tokenizer, texts: List[str], device: str, batch: int = 256) -> torch.Tensor:
    feats = []
    for i in range(0, len(texts), batch):
        toks = tokenizer(texts[i:i+batch]).to(device)
        f = clip_model.encode_text(toks)
        feats.append(f)
    return torch.cat(feats, dim=0)

@torch.no_grad()
def zeroshot_eval(hc: HyperCLIP, clip_model, tokenizer, loader: DataLoader, classnames: List[str], device: str):
    c = (hc.curvature if isinstance(hc, HyperCLIP) else hc.module.curvature)
    class_prompts = build_prompts(classnames)
    class_lx = []
    for prompts in tqdm(class_prompts, desc="ZS encode class", leave=False):
        f = encode_texts(clip_model, tokenizer, prompts, device)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
        x = to_lorentz(f, c)
        class_lx.append(x)
    correct1 = 0; correct5 = 0; n = 0
    all_imgX = []
    for images, labels in tqdm(loader, desc="Zero-shot (angle)", leave=False):
        images = to_device(images, device); labels = to_device(labels, device)
        f_img = clip_model.encode_image(images)
        f_img = f_img / (f_img.norm(dim=-1, keepdim=True) + 1e-9)
        x_img = to_lorentz(f_img, c)
        all_imgX.append(x_img.cpu())
        B = x_img.size(0); C = len(classnames)
        scores = torch.zeros(B, C, device="cpu")
        for ci in range(C):
            sim = angle_similarity(x_img, class_lx[ci].to(device), c, chunk=4096)
            ang = (-sim).mean(dim=1).cpu()
            scores[:, ci] = -ang
        top5 = scores.topk(5, dim=-1).indices
        correct1 += (top5[:, 0].to(labels.device) == labels).sum().item()
        correct5 += sum([(top5[:, i].to(labels.device) == labels).sum().item() for i in range(5)])
        n += labels.size(0)
    top1 = correct1 / n; top5 = correct5 / n
    img_all = torch.cat(all_imgX, dim=0)
    txt_all = torch.cat([t.cpu() for t in class_lx], dim=0)
    mg = modality_gap_from_lorentz(img_all, txt_all)
    return top1, top5, mg, img_all, txt_all


# ------------------------
# Retrieval (angle)
# ------------------------
def recalls_from_sim(sim_mat: np.ndarray, gt_map: Dict[int, List[int]], ks=(1, 5, 10)) -> Dict[str, float]:
    order = np.argsort(-sim_mat, axis=1)
    out = {}
    for k in ks:
        correct = 0
        for i in range(order.shape[0]):
            topk = set(order[i, :k].tolist())
            if any(gt in topk for gt in gt_map[i]): correct += 1
        out[f"R@{k}"] = correct / order.shape[0]
    return out

@torch.no_grad()
def retrieval_eval(clip_model, tokenizer, ds: Dataset, hc: HyperCLIP, device: str, batch_size: int = 64, max_items: Optional[int] = None):
    c = (hc.curvature if isinstance(hc, HyperCLIP) else hc.module.curvature)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=collate_varlen_captions, persistent_workers=False)
    imgX_list = []; all_caps = []; img_to_caps = {}; cap_to_imgs = {}
    cap_idx = 0; img_idx = 0
    for images, caps_list in tqdm(loader, desc="Embed images & gather captions", leave=False):
        images = to_device(images, device)
        f_img = clip_model.encode_image(images)
        f_img = f_img / (f_img.norm(dim=-1, keepdim=True) + 1e-9)
        x_img = to_lorentz(f_img, c)
        imgX_list.append(x_img.cpu())
        bsz = images.size(0)
        for j in range(bsz):
            caps = caps_list[j] if isinstance(caps_list[j], (list, tuple)) else [str(caps_list[j])]
            idxs = []
            for ct in caps:
                all_caps.append(ct)
                idxs.append(cap_idx)
                cap_to_imgs[cap_idx] = [img_idx]
                cap_idx += 1
            img_to_caps[img_idx] = idxs
            img_idx += 1
        if max_items is not None and img_idx >= max_items: break
    imgX = torch.cat(imgX_list, dim=0)
    # text encode
    tok = tokenizer
    f_txt = []
    for i in range(0, len(all_caps), 256):
        t = tok(all_caps[i:i+256]).to(device)
        f = clip_model.encode_text(t)
        f_txt.append(f)
    f_txt = torch.cat(f_txt, dim=0)
    f_txt = f_txt / (f_txt.norm(dim=-1, keepdim=True) + 1e-9)
    txtX = to_lorentz(f_txt, c).cpu()
    mg = modality_gap_from_lorentz(imgX, txtX)
    sim_i2t = angle_similarity(imgX.to(device), txtX.to(device), c, chunk=4096).cpu().numpy()
    i2t = recalls_from_sim(sim_i2t, img_to_caps, ks=(1, 5, 10))
    t2i = recalls_from_sim(sim_i2t.T, cap_to_imgs, ks=(1, 5, 10))
    return mg, i2t, t2i


# ------------------------
# Training loop (strict hyperparams; step-based up to total_steps=120000)
# ------------------------
def train_strict(hc: HyperCLIP, clip_model, tokenizer, train_set: Dataset, args, device: str):
    # data
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    # optimizer
    optim = build_optimizer(hc, lr=args.lr, betas=(args.adam_betas[0], args.adam_betas[1]), wd=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and not args.bf16)

    # training
    global_step = 0
    epoch = 0
    hc.train()
    text_tok = tokenizer

    pbar = tqdm(total=args.total_steps, desc="Train (strict)", dynamic_ncols=True)
    while global_step < args.total_steps:
        epoch += 1
        for images, texts in loader:
            images = to_device(images, device)
            tokens = text_tok(texts).to(device)
            with torch.autocast(device_type="cuda", dtype=(torch.bfloat16 if args.bf16 else torch.float16), enabled=args.amp):
                x_img, x_txt, c, tau = hc(images, tokens)
                loss = angle_contrastive_loss(x_img, x_txt, c, tau)
                loss = loss + centroid_regularizer_euclid(x_img, x_txt, lam=args.centroid_lambda)
            # grad accumulation to form global batch = batch_size * grad_accum = 2048
            loss = loss / args.grad_accum
            if args.amp and not args.bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step % args.grad_accum) == (args.grad_accum - 1):
                if args.amp and not args.bf16:
                    scaler.unscale_(optim)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(hc.parameters(), args.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(hc.parameters(), args.grad_clip)
                    optim.step()
                optim.zero_grad(set_to_none=True)

                # step-wise LR schedule
                step_idx = (global_step + 1) // args.grad_accum
                cur_lr = cosine_lr(step_idx, args.total_steps, args.lr, args.warmup_iters)
                for g in optim.param_groups: g["lr"] = cur_lr

                pbar.set_postfix(step=step_idx, lr=f"{cur_lr:.2e}", c=f"{float(c.item()):.3f}", tau=f"{float(tau.item()):.4f}")
                pbar.update(1)
                if step_idx >= args.total_steps:
                    break
            global_step += 1
        if (global_step // args.grad_accum) >= args.total_steps:
            break
    pbar.close()
    # save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        "model": hc.state_dict(),
        "optimizer": optim.state_dict(),
        "args": vars(args)
    }, os.path.join(args.save_dir, "strict_ckpt.pt"))


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    # data/model
    ap.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--image-crop-scale", nargs=2, type=float, default=[0.5, 1.0])  # strict
    # training strict HP
    ap.add_argument("--do-train", action="store_true")
    ap.add_argument("--total-steps", type=int, default=120000)   # strict
    ap.add_argument("--batch-size", type=int, default=512)       # per-device
    ap.add_argument("--grad-accum", type=int, default=4)         # 512*4=2048 global (strict)
    ap.add_argument("--lr", type=float, default=5e-4)            # peak LR (strict)
    ap.add_argument("--warmup-iters", type=int, default=4000)    # strict
    ap.add_argument("--weight-decay", type=float, default=0.2)   # strict
    ap.add_argument("--adam-betas", nargs=2, type=float, default=[0.9, 0.98])  # strict
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--centroid-lambda", type=float, default=0.0)
    # eval
    ap.add_argument("--do-eval", action="store_true")
    ap.add_argument("--max-coco", type=int, default=None)
    ap.add_argument("--max-flickr", type=int, default=None)
    # hyperbolic
    ap.add_argument("--init-curvature", type=float, default=1.0)
    ap.add_argument("--fix-curvature", action="store_true")
    ap.add_argument("--init-temperature", type=float, default=0.07)
    ap.add_argument("--fix-temperature", action="store_true")
    # save/seed
    ap.add_argument("--save-dir", type=str, default="runs/hyperclip_cvpr2024")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Load CLIP and HyperCLIP wrapper
    clip_model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = get_tokenizer(args.model)
    hc = HyperCLIP(
        clip_model,
        learn_curvature=not args.fix_curvature, init_c=args.init_curvature,
        learn_temp=not args.fix_temperature, init_temp=args.init_temperature
    ).to(device)

    # TRAIN (strict)
    if args.do_train:
        print(f"[Strict Train] steps={args.total_steps}, global_batch={args.batch_size*args.grad_accum}, "
              f"peak_lr={args.lr}, warmup={args.warmup_iters}, wd={args.weight_decay}, betas={tuple(args.adam_betas)}")
        tf_train = build_transforms(args.image_size, tuple(args.image_crop_scale), is_train=True)
        # 用 COCO 训练对齐流程（若你已准备 RedCaps，可替换成你的 RedCapsPairs）
        train_set = CocoTrainPairs(args.data_root, transform=tf_train, max_pairs_per_image=5)
        t0 = time.time()
        train_strict(hc, clip_model, tokenizer, train_set, args, device)
        print(f"[Train] done in {(time.time()-t0):.1f}s, checkpoint -> {os.path.join(args.save_dir,'strict_ckpt.pt')}")

    # EVAL
    if args.do_eval:
        results = {"config": {
            "data_root": args.data_root, "device": device,
            "model": args.model, "pretrained": args.pretrained,
            "image_size": args.image_size, "batch_size": args.batch_size,
            "curvature": float(hc.curvature.item()),
            "temperature": float(hc.temperature.item()),
            "strict_hparams": {
                "total_steps": args.total_steps, "global_batch": args.batch_size*args.grad_accum,
                "peak_lr": args.lr, "warmup_iters": args.warmup_iters,
                "weight_decay": args.weight_decay, "adam_betas": args.adam_betas,
                "crop_scale": args.image_crop_scale
            }
        }, "zero_shot": {}, "retrieval": {}, "runtime_sec": {}}
        t_all = time.time()
        tf_eval = build_transforms(args.image_size, is_train=False)

        # CIFAR-100
        t0 = time.time()
        print("\n[Eval] CIFAR-100 (ZS, angle)")
        cifar_val = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=False, transform=tf_eval, download=True)
        cifar_train = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=True, transform=tf_eval, download=True)
        cifar_classes = cifar_train.classes
        cifar_loader = DataLoader(cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        top1, top5, mg, *_ = zeroshot_eval(hc, clip_model, tokenizer, cifar_loader, cifar_classes, device)
        results["zero_shot"]["cifar100"] = {"top1": top1, "top5": top5, "modality_gap": mg}
        results["runtime_sec"]["cifar100"] = time.time() - t0
        print(f"CIFAR100: top1={top1:.4f}, top5={top5:.4f}, time={results['runtime_sec']['cifar100']:.1f}s")

        # Tiny-ImageNet-200
        t0 = time.time()
        print("\n[Eval] Tiny-ImageNet-200 (ZS, angle)")
        tiny_ds = TinyImageNet200(args.data_root, split="val", transform=tf_eval, auto_download=True)
        tiny_loader = DataLoader(tiny_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        top1, top5, mg, *_ = zeroshot_eval(hc, clip_model, tokenizer, tiny_loader, tiny_ds.classes, device)
        results["zero_shot"]["tiny_imagenet_200"] = {"top1": top1, "top5": top5, "modality_gap": mg}
        results["runtime_sec"]["tiny_imagenet_200"] = time.time() - t0
        print(f"Tiny-ImageNet-200: top1={top1:.4f}, top5={top5:.4f}, time={results['runtime_sec']['tiny_imagenet_200']:.1f}s")

        # DTD
        t0 = time.time()
        print("\n[Eval] DTD (ZS, angle)")
        dtd_val = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="test", transform=tf_eval, download=True)
        dtd_train = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="train", transform=tf_eval, download=True)
        dtd_classes = dtd_train.classes
        dtd_loader = DataLoader(dtd_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        top1, top5, mg, *_ = zeroshot_eval(hc, clip_model, tokenizer, dtd_loader, dtd_classes, device)
        results["zero_shot"]["dtd"] = {"top1": top1, "top5": top5, "modality_gap": mg}
        results["runtime_sec"]["dtd"] = time.time() - t0
        print(f"DTD: top1={top1:.4f}, top5={top5:.4f}, time={results['runtime_sec']['dtd']:.1f}s")

        # MSCOCO retrieval
        t0 = time.time()
        print("\n[Eval] MSCOCO (I2T/T2I, angle)")
        coco_ds = CocoCaptionsEval(args.data_root, split="val", transform=tf_eval)
        if args.max_coco is not None:
            class _Sub(Dataset):
                def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
                def __len__(self): return self.n
                def __getitem__(self, i): return self.base[i]
            coco_ds = _Sub(coco_ds, args.max_coco)
        mg, i2t, t2i = retrieval_eval(clip_model, tokenizer, coco_ds, hc, device,
                                       batch_size=args.batch_size, max_items=args.max_coco)
        results["retrieval"]["mscoco"] = {"I2T": i2t, "T2I": t2i, "modality_gap": mg}
        results["runtime_sec"]["mscoco"] = time.time() - t0
        print(f"MSCOCO: I2T={i2t}, T2I={t2i}, time={results['runtime_sec']['mscoco']:.1f}s")

        # Flickr30k retrieval
        t0 = time.time()
        print("\n[Eval] Flickr30k (I2T/T2I, angle)")
        flickr_ds = Flickr30kEval(args.data_root, split="test", transform=tf_eval)
        if args.max_flickr is not None:
            class _SubF(Dataset):
                def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
                def __len__(self): return self.n
                def __getitem__(self, i): return self.base[i]
            flickr_ds = _SubF(flickr_ds, args.max_flickr)
        mg, i2t, t2i = retrieval_eval(clip_model, tokenizer, flickr_ds, hc, device,
                                       batch_size=args.batch_size, max_items=args.max_flickr)
        results["retrieval"]["flickr30k"] = {"I2T": i2t, "T2I": t2i, "modality_gap": mg}
        results["runtime_sec"]["flickr30k"] = time.time() - t0
        print(f"Flickr30k: I2T={i2t}, T2I={t2i}, time={results['runtime_sec']['flickr30k']:.1f}s")

        # Save
        total_time = time.time() - t_all
        results["runtime_sec"]["total"] = total_time
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.save_dir, exist_ok=True)
        json_path = os.path.join(args.save_dir, f"cvpr2024_hyperclip_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # CSV flatten
        rows = []
        def add_row(name: str, metrics: Dict[str, Any], runtime: Optional[float] = None):
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
            if runtime is not None: row["runtime_sec"] = runtime
            rows.append(row)
        add_row("cifar100", results["zero_shot"]["cifar100"], results["runtime_sec"]["cifar100"])
        add_row("tiny_imagenet_200", results["zero_shot"]["tiny_imagenet_200"], results["runtime_sec"]["tiny_imagenet_200"])
        add_row("dtd", results["zero_shot"]["dtd"], results["runtime_sec"]["dtd"])
        add_row("mscoco.I2T", {"I2T": results["retrieval"]["mscoco"]["I2T"], "mg": results["retrieval"]["mscoco"]["modality_gap"]},
                results["runtime_sec"]["mscoco"])
        add_row("mscoco.T2I", {"T2I": results["retrieval"]["mscoco"]["T2I"], "mg": results["retrieval"]["mscoco"]["modality_gap"]},
                results["runtime_sec"]["mscoco"])
        add_row("flickr30k.I2T", {"I2T": results["retrieval"]["flickr30k"]["I2T"], "mg": results["retrieval"]["flickr30k"]["modality_gap"]},
                results["runtime_sec"]["flickr30k"])
        add_row("flickr30k.T2I", {"T2I": results["retrieval"]["flickr30k"]["T2I"], "mg": results["retrieval"]["flickr30k"]["modality_gap"]},
                results["runtime_sec"]["flickr30k"])
        rows.append({"dataset": "TOTAL", "runtime_sec": results["runtime_sec"]["total"]})
        csv_path = os.path.join(args.save_dir, f"cvpr2024_hyperclip_{ts}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved JSON -> {json_path}")
        print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
