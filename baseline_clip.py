#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP Baseline:
- Zero-shot image classification on CIFAR-100, Tiny-ImageNet-200, DTD  -> Top1/Top5
- Cross-modal retrieval on MSCOCO, Flickr30k                         -> R@1/R@5/R@10 (I2T & T2I)
- Modality gap metrics (per dataset): Centroid Distance, Frechet Distance, Relative Modality Gap
Results are saved as JSON and CSV under ./results/
"""

import os
import sys
import json
import math
import time
import csv
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

# SciPy for stable matrix sqrt in Frechet distance
from scipy.linalg import sqrtm

# ----------------------------
# 1) CLIP (open_clip)
# ----------------------------
import open_clip


# ----------------------------
# 2) Utility: transforms
# ----------------------------
def build_transforms(image_size: int = 224, is_train: bool = False):
    # Zero-shot eval: 采用较保守的增广（中心裁剪）
    if is_train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

# ----------------------------
# 3) Tiny-ImageNet-200 Dataset (auto-download if missing)
# ----------------------------
class TinyImageNet200(Dataset):
    """
    Expect directory layout (auto-downloaded if missing):
    <root>/tiny-imagenet-200/
        train/
            <wnid>/images/*.JPEG
        val/
            images/*.JPEG
            val_annotations.txt
        wnids.txt
        words.txt
    """
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root: str, split: str = "val", transform=None, auto_download: bool = True):
        self.root_base = Path(root).expanduser().resolve()
        self.root = self.root_base / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        assert split in ["train", "val"], "Tiny-ImageNet-200 split must be 'train' or 'val'"

        # ensure exists
        self._ensure_exists(auto_download=auto_download)

        # Load wnids (class ids)
        wnids_path = self.root / "wnids.txt"
        self._assert_file(wnids_path, "wnids.txt not found under {self.root}")
        self.wnids = [x.strip() for x in wnids_path.read_text().splitlines() if x.strip()]

        # Load words mapping
        words_path = self.root / "words.txt"
        self._assert_file(words_path, "words.txt not found under {self.root}")
        wnid_to_words = {}
        for line in words_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            wnid_to_words[parts[0]] = parts[1]

        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.idx_to_classname = {}
        for wnid in self.wnids:
            name = wnid_to_words.get(wnid, wnid)
            short = name.split(",")[0].split(";")[0]
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
            self._assert_file(annot, "val_annotations.txt not found under {val_dir}")
            mapping = {}
            for line in annot.read_text().splitlines():
                if not line.strip():
                    continue
                # official file uses tabs
                parts = line.split("\t")
                fname, wnid = parts[0], parts[1]
                mapping[fname] = wnid
            img_dir = val_dir / "images"
            for img_name in img_dir.glob("*.JPEG"):
                wnid = mapping[img_name.name]
                self.samples.append((str(img_name), self.class_to_idx[wnid]))

        assert len(self.samples) > 0, f"No images found in Tiny-ImageNet-200 {split}"

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

    # -------- helpers --------
    def _assert_file(self, path: Path, msg: str):
        if not path.exists():
            raise AssertionError(msg.format(self=self, val_dir=path.parent))

    def _ensure_exists(self, auto_download: bool = True):
        """
        If dataset folder doesn't exist or missing key files, try to download & extract.
        """
        need = not self.root.exists() or not (self.root / "wnids.txt").exists()
        if not need:
            return
        if not auto_download:
            raise AssertionError(f"Tiny-ImageNet-200 not found at {self.root}. "
                                 f"Please place it or enable auto_download.")
        self._download_and_extract()

    def _download_and_extract(self):
        self.root_base.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_base / "tiny-imagenet-200.zip"
        # download
        if not zip_path.exists():
            print(f"[TinyImageNet] Downloading from {self.URL} -> {zip_path}")
            import urllib.request, shutil as _shutil
            with urllib.request.urlopen(self.URL) as r, open(zip_path, "wb") as f:
                _shutil.copyfileobj(r, f)
        # extract
        print(f"[TinyImageNet] Extracting {zip_path} -> {self.root_base}")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root_base)
        # final check
        if not (self.root / "wnids.txt").exists():
            raise RuntimeError(f"Extraction finished but wnids.txt not found under {self.root}. "
                               f"Please check the archive integrity.")

# ----------------------------
# 4) COCO Captions wrapper
# ----------------------------
class CocoCaptionsEval(Dataset):
    """
    Wrap torchvision.datasets.CocoCaptions to expose:
    - image tensor
    - list of captions (list[str])
    """
    def __init__(self, root: str, split: str = "val", transform=None):
        assert split in ["train", "val"]
        data_root = Path(root).expanduser().resolve() / "coco2017"
        img_dir = data_root / "images" / (f"{'train2017' if split=='train' else 'val2017'}")
        ann = data_root / "annotations" / f"captions_{'train2017' if split=='train' else 'val2017'}.json"
        assert img_dir.exists(), f"COCO images not found: {img_dir}"
        assert ann.exists(), f"COCO annotations not found: {ann}"
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, caps = self.ds[idx]  # caps: list[str]
        return img, caps

import inspect

# ----------------------------
# 5) Flickr30k wrapper (oldest torchvision API: root must be images folder)
# ----------------------------
class Flickr30kEval(Dataset):
    """
    Old torchvision signature:
        Flickr30k(root, ann_file, transform=None, target_transform=None)

    Expected structure:
      <data_root>/flickr30k/
          flickr30k-images/
              1000092795.jpg
              ...
          results_20130124.token
    """
    def __init__(self, root: str, split: str = "test", transform=None):
        # split 参数仅占位，不参与底层调用
        data_root = Path(root).expanduser().resolve() / "flickr30k"
        images_root = data_root / "flickr30k-images"   # 关键：root 指向图片目录
        ann_file = data_root / "results_20130124.token"

        if not images_root.exists():
            raise FileNotFoundError(
                f"Missing images folder: {images_root}\n"
                f"Expected: {images_root}/<image>.jpg"
            )
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Missing annotation file: {ann_file}\n"
                f"Expected: {ann_file}"
            )

        # 旧接口：root=images_folder, ann_file=token file
        self.ds = tvds.Flickr30k(
            root=str(images_root),
            ann_file=str(ann_file),
            transform=transform
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, caps = self.ds[idx]
        # 统一成 list[str]
        if isinstance(caps, list):
            captions = caps
        else:
            captions = [str(caps)]
        return img, captions



# ----------------------------
# 6) CLIP helper
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
        text_feat = model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        feats.append(text_feat)
    return torch.cat(feats, dim=0)

@torch.no_grad()
def encode_images(model, images: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(images)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


# ----------------------------
# 7) Modality gap metrics
# ----------------------------
def _np_trace(mat: np.ndarray) -> float:
    return float(np.trace(mat))

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Frechet distance (FID) between two Gaussians."""
    m_diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd_sq = m_diff.dot(m_diff) + _np_trace(sigma1 + sigma2 - 2.0 * covmean)
    fd_sq = max(0.0, float(fd_sq))
    return float(math.sqrt(fd_sq))

def modality_gap_metrics(img_feats: np.ndarray, txt_feats: np.ndarray) -> Dict[str, float]:
    mu_i = img_feats.mean(axis=0)
    mu_t = txt_feats.mean(axis=0)
    ci = np.cov(img_feats.T)
    ct = np.cov(txt_feats.T)

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
# 8) Zero-shot classification
# ----------------------------
def build_zeroshot_prompts(classnames: List[str]) -> List[str]:
    # 经典 CLIP prompt 套语（精简但有效）
    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a photo of a small {}.",
        "a photo of a large {}."
    ]
    texts = []
    for cname in classnames:
        for t in templates:
            texts.append(t.format(cname))
    return texts

@torch.no_grad()
def zeroshot_classification_eval(
    model, tokenizer, dataloader: DataLoader, classnames: List[str], device: str
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Returns:
      top1, top5, mg_metrics, image_feats_all (N x D), text_feats_all (C x D, class prototypes)
    """
    # Build zero-shot classifier: average over multiple prompts per class
    class_texts = []
    for cname in classnames:
        prompts = build_zeroshot_prompts([cname])
        text_emb = encode_texts(model, tokenizer, prompts, device=device)  # (#prompts, D)
        class_proto = text_emb.mean(dim=0, keepdim=True)
        class_proto = class_proto / class_proto.norm(dim=-1, keepdim=True)
        class_texts.append(class_proto)
    class_texts = torch.cat(class_texts, dim=0)  # (C, D)

    correct1 = 0
    correct5 = 0
    n = 0

    all_img_feats = []
    for images, labels in tqdm(dataloader, desc="Zero-shot eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        img_feats = encode_images(model, images)  # (B, D)
        logits = (img_feats @ class_texts.t())  # (B, C)

        # top-k
        top5 = logits.topk(5, dim=-1).indices  # (B, 5)
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += sum([(top5[:, i] == labels).sum().item() for i in range(5)])
        n += labels.size(0)

        all_img_feats.append(img_feats.cpu())

    top1 = correct1 / n
    top5 = correct5 / n

    # Modality gap: use all image feats + all class text prototypes (weighted equally)
    img_feats_all = torch.cat(all_img_feats, dim=0).numpy()
    txt_feats_all = class_texts.cpu().numpy()
    mg = modality_gap_metrics(img_feats_all, txt_feats_all)

    return top1, top5, mg, img_feats_all, txt_feats_all


# ----------------------------
# 9) Cross-modal retrieval (COCO / Flickr30k)
# ----------------------------
@torch.no_grad()
def embed_captions_list(model, tokenizer, all_captions: List[str], device: str, batch_size: int = 256) -> torch.Tensor:
    return encode_texts(model, tokenizer, all_captions, device=device, batch_size=batch_size)

@torch.no_grad()
def compute_retrieval_metrics(
    sim_matrix: np.ndarray, gt_map: Dict[int, List[int]], ranks: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    sim_matrix: (N_query, N_target), higher=better
    gt_map: query_index -> list of target indices considered correct
    Returns recall@k for given ranks
    """
    recalls = {}
    order = np.argsort(-sim_matrix, axis=1)  # desc
    for k in ranks:
        correct = 0
        for i in range(order.shape[0]):
            topk = set(order[i, :k].tolist())
            if any((gt in topk) for gt in gt_map[i]):
                correct += 1
        recalls[f"R@{k}"] = correct / order.shape[0]
    return recalls

def collate_varlen_captions(batch):
    """
    batch: list of tuples (image_tensor, captions)
      - image_tensor: CxHxW (already transformed to tensor)
      - captions: list[str] (variable length per sample)
    Returns:
      images: Tensor (B, C, H, W)
      caps_list: list[list[str]], len=B
    """
    # images 都是 tensor，直接 stack
    images = torch.stack([b[0] for b in batch], dim=0)
    # captions 保持 python list 结构（每个元素仍然是 list[str]）
    caps_list = []
    for _, caps in batch:
        if isinstance(caps, (list, tuple)):
            caps_list.append(list(caps))
        else:
            caps_list.append([str(caps)])
    return images, caps_list

@torch.no_grad()
def retrieval_eval(
    model, tokenizer, ds: Dataset, device: str, batch_size: int = 64, max_items: Optional[int] = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Returns:
      - mg_metrics (modality gap metrics using image feats + all caption feats)
      - I2T recalls {R@1,R@5,R@10}
      - T2I recalls {R@1,R@5,R@10}
    """
    # 用自定义 collate_fn 保证 images 是 Tensor、captions 是 list[list[str]]（变长）
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_varlen_captions,
        persistent_workers=False  # 可选：避免一些环境下 worker 复用问题
    )

    image_feats_list: List[torch.Tensor] = []
    all_captions: List[str] = []
    img_to_caps: Dict[int, List[int]] = {}  # image idx -> caption idx list
    cap_to_imgs: Dict[int, List[int]] = {}  # caption idx -> [image idx]

    cap_idx = 0
    img_idx = 0

    for images, caps_list in tqdm(loader, desc="Embed images & gather captions", leave=False):
        # images: (B, C, H, W) tensor，caps_list: list of list[str], len=B
        bsz = images.size(0)
        images = images.to(device, non_blocking=True)

        img_feats = encode_images(model, images)  # (B, D)
        image_feats_list.append(img_feats.cpu())

        # 将每张图片的若干 caption 线性展开到 all_captions 里，并维护映射
        for j in range(bsz):
            cur_caps = caps_list[j]
            if not isinstance(cur_caps, (list, tuple)):
                cur_caps = [str(cur_caps)]
            idxs = []
            for c in cur_caps:
                all_captions.append(c)
                idxs.append(cap_idx)
                cap_to_imgs[cap_idx] = [img_idx]  # 每条 caption 对应唯一图
                cap_idx += 1
            img_to_caps[img_idx] = idxs
            img_idx += 1

        if max_items is not None and img_idx >= max_items:
            break

    image_feats = torch.cat(image_feats_list, dim=0)  # (N_img, D)
    # 文本一次性编码
    text_feats = embed_captions_list(model, tokenizer, all_captions, device=device)  # (N_cap, D)
    text_feats = text_feats.cpu()

    # 模态差异指标
    mg = modality_gap_metrics(image_feats.numpy(), text_feats.cpu().numpy())

    # I2T
    sim_i2t = (image_feats @ text_feats.t()).cpu().numpy()  # (N_img, N_cap)
    i2t_recalls = compute_retrieval_metrics(sim_i2t, img_to_caps, ranks=[1, 5, 10])

    # T2I
    sim_t2i = sim_i2t.T  # (N_cap, N_img)
    t2i_recalls = compute_retrieval_metrics(sim_t2i, cap_to_imgs, ranks=[1, 5, 10])

    return mg, i2t_recalls, t2i_recalls



# ----------------------------
# 10) Main runner
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data", help="Datasets root")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-coco", type=int, default=None, help="limit number of images for COCO eval (debug)")
    parser.add_argument("--max-flickr", type=int, default=None, help="limit number of images for Flickr30k eval (debug)")
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda" if (args.device.lower().startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load CLIP
    print(f"Loading CLIP: {args.model} ({args.pretrained}) on {device}")
    model, preprocess, tokenizer = load_clip(model_name=args.model, pretrained=args.pretrained, device=device)
    model.eval()

    # --------- Zero-shot classification datasets ---------
    image_tf = build_transforms(args.image_size, is_train=False)

    # CIFAR-100
    print("\n[Eval] CIFAR-100 (zero-shot)")
    cifar_train = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=True, transform=image_tf, download=True)
    cifar_val = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=False, transform=image_tf, download=True)
    cifar_classes = cifar_train.classes  # 100 class names
    cifar_loader = DataLoader(cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    cifar_top1, cifar_top5, cifar_mg, cifar_img_feats, cifar_txt_feats = zeroshot_classification_eval(
        model, tokenizer, cifar_loader, cifar_classes, device
    )

    # Tiny-ImageNet-200
    print("\n[Eval] Tiny-ImageNet-200 (zero-shot)")
    tiny_ds = TinyImageNet200(root=args.data_root, split="val", transform=image_tf)
    tiny_loader = DataLoader(tiny_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    tiny_top1, tiny_top5, tiny_mg, tiny_img_feats, tiny_txt_feats = zeroshot_classification_eval(
        model, tokenizer, tiny_loader, tiny_ds.classes, device
    )

    # DTD
    print("\n[Eval] DTD (zero-shot)")
    dtd_val = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="test", transform=image_tf, download=True)
    dtd_classes = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="train", transform=image_tf, download=True).classes
    dtd_loader = DataLoader(dtd_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dtd_top1, dtd_top5, dtd_mg, dtd_img_feats, dtd_txt_feats = zeroshot_classification_eval(
        model, tokenizer, dtd_loader, dtd_classes, device
    )

    # --------- Cross-modal retrieval datasets ---------
    # MSCOCO val
    print("\n[Eval] MSCOCO (I2T/T2I R@K)")
    coco_ds = CocoCaptionsEval(root=args.data_root, split="val", transform=image_tf)
    if args.max_coco is not None:
        # Wrap a subset
        class _Sub(Dataset):
            def __init__(self, base, n):
                self.base = base
                self.n = min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.base[i]
        coco_ds = _Sub(coco_ds, args.max_coco)
    coco_mg, coco_i2t, coco_t2i = retrieval_eval(model, tokenizer, coco_ds, device, batch_size=args.batch_size, max_items=args.max_coco)

    # Flickr30k test
    print("\n[Eval] Flickr30k (I2T/T2I R@K)")
    flickr_ds = Flickr30kEval(root=args.data_root, split="test", transform=image_tf)
    if args.max_flickr is not None:
        class _SubF(Dataset):
            def __init__(self, base, n):
                self.base = base
                self.n = min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.base[i]
        flickr_ds = _SubF(flickr_ds, args.max_flickr)
    flickr_mg, flickr_i2t, flickr_t2i = retrieval_eval(model, tokenizer, flickr_ds, device, batch_size=args.batch_size, max_items=args.max_flickr)

    # --------- Aggregate & Save ---------
    results = {
        "config": {
            "data_root": args.data_root,
            "device": device,
            "model": args.model,
            "pretrained": args.pretrained,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
        },
        "zero_shot": {
            "cifar100": {
                "top1": cifar_top1, "top5": cifar_top5,
                "modality_gap": cifar_mg
            },
            "tiny_imagenet_200": {
                "top1": tiny_top1, "top5": tiny_top5,
                "modality_gap": tiny_mg
            },
            "dtd": {
                "top1": dtd_top1, "top5": dtd_top5,
                "modality_gap": dtd_mg
            }
        },
        "retrieval": {
            "mscoco": {
                "I2T": coco_i2t,
                "T2I": coco_t2i,
                "modality_gap": coco_mg
            },
            "flickr30k": {
                "I2T": flickr_i2t,
                "T2I": flickr_t2i,
                "modality_gap": flickr_mg
            }
        }
    }

    json_path = os.path.join(args.save_dir, f"clip_baseline_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON results to: {json_path}")

    # Also CSV (one row per dataset/setting)
    rows = []
    def add_row(ds_name: str, metrics: Dict[str, Any]):
        row = {"dataset": ds_name}
        # flatten dicts
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

    add_row("cifar100", {"top1": cifar_top1, "top5": cifar_top5, **{"mg."+k: v for k, v in cifar_mg.items()}})
    add_row("tiny_imagenet_200", {"top1": tiny_top1, "top5": tiny_top5, **{"mg."+k: v for k, v in tiny_mg.items()}})
    add_row("dtd", {"top1": dtd_top1, "top5": dtd_top5, **{"mg."+k: v for k, v in dtd_mg.items()}})
    add_row("mscoco.I2T", coco_i2t | {"mg."+k: v for k, v in coco_mg.items()})
    add_row("mscoco.T2I", coco_t2i | {"mg."+k: v for k, v in coco_mg.items()})
    add_row("flickr30k.I2T", flickr_i2t | {"mg."+k: v for k, v in flickr_mg.items()})
    add_row("flickr30k.T2I", flickr_t2i | {"mg."+k: v for k, v in flickr_mg.items()})

    csv_path = os.path.join(args.save_dir, f"clip_baseline_{timestamp}.csv")
    # write CSV with pandas for consistent ordering
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")


if __name__ == "__main__":
    main()
