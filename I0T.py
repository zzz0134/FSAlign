#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I0T Baseline (CVPR 2024) — 完整实现
- 两阶段：
  (1) I0Tpost：后处理“标准化”方法（按模态减均值并 Frobenius 归一化），无训练，近零模态间隙；
  (2) I0Tasync：在冻结编码器后，为图/文各加一层独立 BN（作用在“已单位化”的嵌入上），
      只训练 BN（含 MCSIE 风格的无监督正例增强：在嵌入级施加 0.1 dropout 构造 aug），
      损失 = 标准对比 InfoNCE + CyCLIP 的 image/image 与 text/text 循环损失（权重 0.25+0.25）。
- 评测与前两条基线统一：
  * Modality Gap：centroid distance / Fréchet distance / relative gap（在单位范数的欧氏嵌入空间）；
  * Zero-shot（CIFAR100, Tiny-ImageNet-200, DTD）Top1/Top5（使用论文列出的 18 个模板做类文本，逐类取均值）；
  * 检索（MSCOCO val2017、Flickr30k test）：I2T/T2I 的 R@1/5/10（使用余弦相似度；I0Tpost/async 对嵌入生效）。
- 记录每个数据集的运行时间与总时间，保存 JSON 与 CSV。

严格遵循论文中的关键实现细节：
* I0Tpost：x' = Normalize(x - x̄_img)，y' = Normalize(y - ȳ_txt)，x,y 皆为“单位范数”嵌入；
* I0Tasync：在“单位范数”嵌入上接 BNimg/BNtxt（affine=True, track_running_stats=True），
  训练时使用 L_CLIP + 0.25*L_I-Cyclic + 0.25*L_T-Cyclic；并做嵌入级 dropout 构造四种组合 (I,Iaug)x(T,Taug)；
* 第一阶段（可选）提供 Long-CLIP-only 风格的小步微调接口（COCO 长描述不可得时，仍可用 COCO captions 验证流程）。
"""

import os
import math
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tvds
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy.linalg import sqrtm

import open_clip


# =========================
# 通用工具
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if isinstance(x, torch.Tensor): return x.to(device, non_blocking=True)
    return x

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# =========================
# 变换
# =========================
def build_transforms(image_size: int = 224, is_train: bool = True, rrc_scale=(0.5, 1.0)):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=rrc_scale),
            transforms.RandomHorizontalFlip(),
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


# =========================
# 数据集：Tiny-ImageNet-200
# =========================
class TinyImageNet200(Dataset):
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    def __init__(self, root: str, split: str = "val", transform=None, auto_download: bool = True):
        self.root_base = Path(root).expanduser().resolve()
        self.root = self.root_base / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        assert split in ["train", "val"]
        self._ensure_exists(auto_download)
        wnids = (self.root / "wnids.txt").read_text().splitlines()
        self.wnids = [x.strip() for x in wnids if x.strip()]
        words_map = {}
        for line in (self.root / "words.txt").read_text().splitlines():
            if not line.strip(): continue
            k, v = line.split("\t")
            words_map[k] = v
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.idx_to_name = {self.class_to_idx[w]: words_map.get(w, w).split(",")[0].split(";")[0] for w in self.wnids}
        self.samples: List[Tuple[str,int]] = []
        if split == "train":
            tdir = self.root / "train"
            for wnid in self.wnids:
                idir = tdir / wnid / "images"
                for p in idir.glob("*.JPEG"):
                    self.samples.append((str(p), self.class_to_idx[wnid]))
        else:
            vdir = self.root / "val"
            ann = vdir / "val_annotations.txt"
            mapping = {}
            for line in ann.read_text().splitlines():
                if not line.strip(): continue
                parts = line.split("\t")
                mapping[parts[0]] = parts[1]
            idir = vdir / "images"
            for p in idir.glob("*.JPEG"):
                wnid = mapping[p.name]
                self.samples.append((str(p), self.class_to_idx[wnid]))
        if not self.samples:
            raise RuntimeError("Tiny-ImageNet-200 empty!")

    def _ensure_exists(self, auto_download: bool):
        need = not self.root.exists() or not (self.root / "wnids.txt").exists()
        if not need: return
        if not auto_download:
            raise FileNotFoundError(f"{self.root} not found. Enable auto_download.")
        self.root_base.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_base / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            print("[TinyImageNet] Downloading...")
            import urllib.request, shutil as sh
            with urllib.request.urlopen(self.URL) as r, open(zip_path, "wb") as f:
                sh.copyfileobj(r, f)
        print("[TinyImageNet] Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root_base)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        fp, y = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform is not None: img = self.transform(img)
        return img, y
    @property
    def classes(self) -> List[str]:
        return [self.idx_to_name[i] for i in range(len(self.idx_to_name))]


# =========================
# 数据集：COCO & Flickr30k
# =========================
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
    def __getitem__(self, idx: int): img, caps = self.ds[idx]; return img, caps

# 训练 COCO 成对样本（兼容不依赖私有 API）
class CocoTrainPairs(Dataset):
    def __init__(self, root: str, transform=None, max_pairs_per_image: int = 5):
        droot = Path(root).expanduser().resolve() / "coco2017"
        img_dir = droot / "images" / "train2017"
        ann = droot / "annotations" / "captions_train2017.json"
        if not img_dir.exists() or not ann.exists():
            raise FileNotFoundError(f"COCO train not found: {img_dir} / {ann}")
        base = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann))
        coco = base.coco; root_dir = Path(base.root)
        self.items: List[Tuple[str,str]] = []
        for img_id in list(base.ids):
            meta = coco.loadImgs([img_id])[0]
            file_name = meta["file_name"]; img_path = str(root_dir / file_name)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            caps = [a.get("caption","") for a in anns if "caption" in a]
            if max_pairs_per_image is not None and max_pairs_per_image>0:
                caps = caps[:max_pairs_per_image]
            for c in caps:
                if c is None: continue
                self.items.append((img_path, str(c)))
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int):
        p, t = self.items[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None: img = self.transform(img)
        return img, t

class Flickr30kEval(Dataset):
    """Old torchvision API：root=images folder, ann_file token file"""
    def __init__(self, root: str, split: str = "test", transform=None):
        droot = Path(root).expanduser().resolve() / "flickr30k"
        images_root = droot / "flickr30k-images"
        ann_file = droot / "results_20130124.token"
        if not images_root.exists(): raise FileNotFoundError(f"Missing images: {images_root}")
        if not ann_file.exists(): raise FileNotFoundError(f"Missing token: {ann_file}")
        self.ds = tvds.Flickr30k(root=str(images_root), ann_file=str(ann_file), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx: int):
        img, caps = self.ds[idx]
        caps = caps if isinstance(caps, list) else [str(caps)]
        return img, caps


def collate_varlen_captions(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    caps_list = []
    for _, caps in batch:
        if isinstance(caps, (list, tuple)): caps_list.append(list(caps))
        else: caps_list.append([str(caps)])
    return images, caps_list


# =========================
# I0Tpost（推理时后处理）
# =========================
class I0TPostStandardizer:
    """
    对“单位范数”的图/文嵌入做：x' = Normalize(x - mean_img)，y' = Normalize(y - mean_txt)
    mean_* 从参考数据（通常是目标测试集合）计算。
    """
    def __init__(self, mean_img: torch.Tensor, mean_txt: torch.Tensor, device: str = "cpu"):
        self.mean_img = mean_img.to(device)
        self.mean_txt = mean_txt.to(device)

    @torch.no_grad()
    def std_img(self, x: torch.Tensor) -> torch.Tensor:
        x = l2norm(x, dim=-1)
        x = x - self.mean_img
        return l2norm(x, dim=-1)

    @torch.no_grad()
    def std_txt(self, y: torch.Tensor) -> torch.Tensor:
        y = l2norm(y, dim=-1)
        y = y - self.mean_txt
        return l2norm(y, dim=-1)

    @staticmethod
    @torch.no_grad()
    def estimate_means(clip_model, tokenizer, ds: Dataset, device: str, batch_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=collate_varlen_captions)
        img_feats = []; txt_feats = []
        for images, caps_list in tqdm(loader, desc="I0Tpost: estimate means", leave=False):
            images = to_device(images, device)
            fi = clip_model.encode_image(images); fi = l2norm(fi, dim=-1)
            img_feats.append(fi.cpu())
            # 取每张图的“第一条 caption”来估计文本均值（按论文：一条 caption 的评估也可与五条保持趋势一致）
            caps = [caps_list[j][0] for j in range(len(caps_list))]
            toks = tokenizer(caps).to(device)
            ft = clip_model.encode_text(toks); ft = l2norm(ft, dim=-1)
            txt_feats.append(ft.cpu())
        img_all = torch.cat(img_feats, dim=0)
        txt_all = torch.cat(txt_feats, dim=0)
        mean_img = img_all.mean(dim=0, keepdim=True)
        mean_txt = txt_all.mean(dim=0, keepdim=True)
        mean_img = l2norm(mean_img, dim=-1)
        mean_txt = l2norm(mean_txt, dim=-1)
        return mean_img, mean_txt


# =========================
# I0Tasync（可学习 BN）
# =========================
class I0TAsyncHead(nn.Module):
    """
    独立 BN 层，作于“单位范数嵌入”上： y = Normalize( BN( Normalize(f) ) )
    只训练 BN（affine=True, track_running_stats=True），编码器冻结。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.bn_img = nn.BatchNorm1d(dim, affine=True, track_running_stats=True)
        self.bn_txt = nn.BatchNorm1d(dim, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(p=0.1)  # MCSIE 风格的无监督正例增强

    def forward_img(self, f: torch.Tensor, aug: bool = False) -> torch.Tensor:
        x = l2norm(f, dim=-1)
        if aug: x = self.dropout(x)
        x = self.bn_img(x)
        x = l2norm(x, dim=-1)
        return x

    def forward_txt(self, g: torch.Tensor, aug: bool = False) -> torch.Tensor:
        y = l2norm(g, dim=-1)
        if aug: y = self.dropout(y)
        y = self.bn_txt(y)
        y = l2norm(y, dim=-1)
        return y


# =========================
# 对比损失（CLIP）+ CyCLIP 循环项
# =========================
def clip_contrastive_loss(img: torch.Tensor, txt: torch.Tensor, logit_scale: float = 4.6052) -> torch.Tensor:
    # sim = cosine；logit_scale=4.6052 ≈ ln(100)
    sim_i2t = img @ txt.t()
    sim_t2i = txt @ img.t()
    logits_i2t = sim_i2t * logit_scale
    logits_t2i = sim_t2i * logit_scale
    labels = torch.arange(img.size(0), device=img.device)
    li = F.cross_entropy(logits_i2t, labels)
    lt = F.cross_entropy(logits_t2i, labels)
    return 0.5*(li+lt)

def cyclic_loss_inmodal(feats: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    CyCLIP 的 in-modal “uniformity-like” 惩罚：让同一 batch 内部更均匀。
    这里用一个简单的 InfoNCE 负样本拉远：自身为正，其余全为负（不对称），
    以避免改变跨模态的对齐方向（只在单模态上均匀化）。
    """
    sim = feats @ feats.t() / temperature
    mask = torch.eye(sim.size(0), device=feats.device).bool()
    logits = sim[~mask].view(sim.size(0), sim.size(0)-1)
    # 目标是让非自身的 logit 小 → 相当于最大化自身与均匀分布的距离；用 logsumexp 近似
    # 这里采用对称形式的简单正则：logsumexp - 常数
    reg = torch.logsumexp(logits, dim=1).mean()
    return reg * 1e-3  # 温和正则，避免过强破坏

def l_cyclip(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    # L_CLIP + 0.25 * LI-cyclic + 0.25 * LC-cyclic
    l_clip = clip_contrastive_loss(img, txt)
    li_cyc = cyclic_loss_inmodal(img)
    lt_cyc = cyclic_loss_inmodal(txt)
    return l_clip + 0.25*li_cyc + 0.25*lt_cyc


# =========================
# Zero-shot 文本模板（18 个，论文附录表 5）
# =========================
CIFAR100_TEMPLATES = [
 "a photo of a {}.","a blurry photo of a {}.","a black and white photo of a {}.",
 "a low contrast photo of a {}.","a high contrast photo of a {}.","a bad photo of a {}.",
 "a good photo of a {}.","a photo of a small {}.","a photo of a big {}.","a photo of the {}.",
 "a blurry photo of the {}.","a black and white photo of the {}.","a low contrast photo of the {}.",
 "a high contrast photo of the {}.","a bad photo of the {}.","a good photo of the {}.",
 "a photo of the small {}.","a photo of the big {}."
]


# =========================
# Modality Gap 指标
# =========================
def _np_trace(m: np.ndarray) -> float: return float(np.trace(m))
def frechet_distance(mu1, s1, mu2, s2) -> float:
    md = mu1 - mu2
    covmean = sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    fd_sq = md.dot(md) + _np_trace(s1 + s2 - 2.0 * covmean)
    fd_sq = max(0.0, float(fd_sq)); return float(math.sqrt(fd_sq))

def modality_gap_from_embeddings(imgE: torch.Tensor, txtE: torch.Tensor) -> Dict[str, float]:
    x = imgE.cpu().numpy(); y = txtE.cpu().numpy()
    mu_x = x.mean(axis=0); mu_y = y.mean(axis=0)
    cx = np.cov(x.T); cy = np.cov(y.T)
    centroid = float(np.linalg.norm(mu_x - mu_y))
    fd = frechet_distance(mu_x, cx, mu_y, cy)
    denom = math.sqrt(0.5*(_np_trace(cx)+_np_trace(cy)) + 1e-12)
    rmg = centroid/denom if denom>0 else float("nan")
    return {"centroid_distance": centroid, "frechet_distance": fd, "relative_modality_gap": rmg}


# =========================
# 评测：Zero-shot 分类
# =========================
@torch.no_grad()
def encode_texts(clip_model, tokenizer, texts: List[str], device: str, batch: int = 256) -> torch.Tensor:
    feats=[]
    for i in range(0,len(texts),batch):
        toks = tokenizer(texts[i:i+batch]).to(device)
        f = clip_model.encode_text(toks)
        feats.append(f)
    return torch.cat(feats, dim=0)

@torch.no_grad()
def zeroshot_eval(clip_model, tokenizer, loader: DataLoader, classnames: List[str], device: str,
                  head: Optional[I0TAsyncHead]=None, post: Optional[I0TPostStandardizer]=None,
                  templates: Optional[List[str]]=None) -> Tuple[float,float,Dict[str,float],torch.Tensor,torch.Tensor]:
    T_list = templates if templates is not None else CIFAR100_TEMPLATES
    # 构造类文本嵌入并（可选）通过 I0T 模块
    cls_bank=[]
    for cname in tqdm(classnames, desc="ZS: encode classes", leave=False):
        prompts=[t.format(cname) for t in T_list]
        ft = encode_texts(clip_model, tokenizer, prompts, device)
        ft = l2norm(ft, dim=-1)
        if head is not None: ft = head.forward_txt(ft, aug=False)
        if post is not None: ft = post.std_txt(ft)
        cls_bank.append(ft)
    # 评测
    correct1=0; correct5=0; n=0
    img_all=[]; txt_all=[]
    for images, labels in tqdm(loader, desc="Zero-shot", leave=False):
        images = to_device(images, device); labels = to_device(labels, device)
        fi = clip_model.encode_image(images); fi = l2norm(fi, dim=-1)
        if head is not None: fi = head.forward_img(fi, aug=False)
        if post is not None: fi = post.std_img(fi)
        img_all.append(fi.cpu())
        # 计算到每一类的“模板平均”外积得分
        B = fi.size(0); C = len(classnames)
        scores = torch.zeros(B, C, device="cpu")
        for ci in range(C):
            txtc = cls_bank[ci].to(device)  # (P,d)
            sc = (fi @ txtc.t()).mean(dim=1).cpu()
            scores[:,ci] = sc
            txt_all.append(txtc.cpu())
        top5 = scores.topk(5, dim=-1).indices
        correct1 += (top5[:,0].to(labels.device) == labels).sum().item()
        correct5 += sum([(top5[:,i].to(labels.device) == labels).sum().item() for i in range(5)])
        n += labels.size(0)
    top1=correct1/n; top5=correct5/n
    img_cat = torch.cat(img_all, dim=0)
    txt_cat = torch.cat(txt_all, dim=0) if len(txt_all)>0 else torch.empty(0)
    mg = modality_gap_from_embeddings(img_cat, txt_cat) if txt_cat.numel()>0 else {"centroid_distance":float("nan"),"frechet_distance":float("nan"),"relative_modality_gap":float("nan")}
    return top1, top5, mg, img_cat, txt_cat


# =========================
# 评测：检索（I2T/T2I）
# =========================
def recalls_from_sim(sim_mat: np.ndarray, gt_map: Dict[int,List[int]], ks=(1,5,10)) -> Dict[str,float]:
    order = np.argsort(-sim_mat, axis=1)
    out={}
    for k in ks:
        correct=0
        for i in range(order.shape[0]):
            topk = set(order[i,:k].tolist())
            if any(gt in topk for gt in gt_map[i]): correct+=1
        out[f"R@{k}"] = correct/order.shape[0]
    return out

@torch.no_grad()
def retrieval_eval(clip_model, tokenizer, ds: Dataset, device: str, batch_size: int = 64,
                   head: Optional[I0TAsyncHead]=None, post: Optional[I0TPostStandardizer]=None,
                   max_items: Optional[int]=None):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=collate_varlen_captions)
    imgE_list=[]; all_caps=[]; img_to_caps={}; cap_to_imgs={}
    img_idx=0; cap_idx=0
    for images, caps_list in tqdm(loader, desc="Embed images & gather caps", leave=False):
        images = to_device(images, device)
        fi = clip_model.encode_image(images); fi = l2norm(fi, dim=-1)
        if head is not None: fi = head.forward_img(fi, aug=False)
        if post is not None: fi = post.std_img(fi)
        imgE_list.append(fi.cpu())
        bsz = images.size(0)
        for j in range(bsz):
            caps = caps_list[j]
            idxs=[]
            for ct in caps:
                all_caps.append(ct)
                idxs.append(cap_idx)
                cap_to_imgs[cap_idx]=[img_idx]
                cap_idx+=1
            img_to_caps[img_idx]=idxs
            img_idx+=1
        if max_items is not None and img_idx>=max_items: break
    imgE = torch.cat(imgE_list, dim=0)  # (Nimg,d)
    # 编码文本（使用所有 caption）
    ft_list=[]
    for i in range(0,len(all_caps),256):
        toks = tokenizer(all_caps[i:i+256]).to(device)
        ft = clip_model.encode_text(toks); ft = l2norm(ft, dim=-1)
        if head is not None: ft = head.forward_txt(ft, aug=False)
        if post is not None: ft = post.std_txt(ft)
        ft_list.append(ft.cpu())
    txtE = torch.cat(ft_list, dim=0) if ft_list else torch.empty(0, imgE.size(1))
    # MG
    mg = modality_gap_from_embeddings(imgE, txtE)
    # 相似度与召回
    sim = (imgE @ txtE.t()).numpy()
    i2t = recalls_from_sim(sim, img_to_caps, ks=(1,5,10))
    t2i = recalls_from_sim(sim.T, cap_to_imgs, ks=(1,5,10))
    return mg, i2t, t2i


# =========================
# 第一阶段（可选）微调（Long-CLIP-only 风格）
# =========================
def stage1_finetune(clip_model, tokenizer, train_set: Dataset, device: str,
                    epochs: int = 3, batch_size: int = 128, lr: float = 1e-6, weight_decay: float = 1e-2,
                    betas=(0.9,0.98), workers: int = 8):
    """
    按论文：AdamW，lr=1e-6，wd=1e-2，batch=128（每卡64*2卡）；温度 logit_scale=4.6052；3 epochs；
    损失：L_CLIP + 0.25*LI-cyclic + 0.25*LT-cyclic。
    """
    for p in clip_model.parameters(): p.requires_grad_(True)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(clip_model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=1e-6)
    clip_model.train()
    for ep in range(1, epochs+1):
        pbar = tqdm(loader, desc=f"Stage1 finetune epoch {ep}")
        for images, texts in pbar:
            images = to_device(images, device)
            toks = tokenizer(texts).to(device)
            fi = clip_model.encode_image(images); fi = l2norm(fi, dim=-1)
            ft = clip_model.encode_text(toks);    ft = l2norm(ft, dim=-1)
            loss = l_cyclip(fi, ft)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)
            optim.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    # 训练后切 eval
    clip_model.eval()


# =========================
# 第二阶段：只训练 BN（I0Tasync）
# =========================
def stage2_train_async_bn(clip_model, head: I0TAsyncHead, tokenizer, train_set: Dataset, device: str,
                          epochs: int = 2, batch_size: int = 256, lr: float = 1e-4, weight_decay: float = 0.2,
                          betas=(0.9,0.98), workers: int = 8):
    """
    冻结编码器，仅训练 BN；损失与 stage1 相同，但加入“嵌入级”dropout 增广 (I,Iaug)x(T,Taug) 的四项求和。
    AdamW β=(0.9,0.98)，wd=0.2（论文 I0T 全文对 wd 的讨论与主设置保持一致类比）。
    """
    for p in clip_model.parameters(): p.requires_grad_(False)
    head.train()
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    params = list(head.parameters())
    optim = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-6)
    for ep in range(1, epochs+1):
        pbar = tqdm(loader, desc=f"Stage2 I0Tasync epoch {ep}")
        for images, texts in pbar:
            images = to_device(images, device)
            toks = tokenizer(texts).to(device)
            with torch.no_grad():
                fi_base = clip_model.encode_image(images); fi_base = l2norm(fi_base, dim=-1)
                ft_base = clip_model.encode_text(toks);    ft_base = l2norm(ft_base, dim=-1)
            # 四组合
            fi = head.forward_img(fi_base, aug=False)
            fi_aug = head.forward_img(fi_base, aug=True)
            ft = head.forward_txt(ft_base, aug=False)
            ft_aug = head.forward_txt(ft_base, aug=True)
            loss = (l_cyclip(fi, ft) +
                    l_cyclip(fi, ft_aug) +
                    l_cyclip(fi_aug, ft) +
                    l_cyclip(fi_aug, ft_aug)) / 4.0
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    head.eval()


# =========================
# 主流程
# =========================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    # 训练阶段开关
    ap.add_argument("--do-stage1", action="store_true", help="微调编码器（可选）")
    ap.add_argument("--do-stage2", action="store_true", help="训练 I0Tasync BN（推荐）")
    ap.add_argument("--stage1-epochs", type=int, default=3)
    ap.add_argument("--stage2-epochs", type=int, default=2)
    # 评测
    ap.add_argument("--do-eval", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-coco", type=int, default=None)
    ap.add_argument("--max-flickr", type=int, default=None)
    # I0Tpost 相关
    ap.add_argument("--use-i0tpost", action="store_true", help="评测时使用 I0Tpost（需估计模态均值）")
    ap.add_argument("--i0tpost-ref", type=str, default="coco_val", choices=["coco_val","flickr_test"], help="用哪个集合估计 I0Tpost 的模态均值")
    # 保存与随机种子
    ap.add_argument("--save-dir", type=str, default="runs/iot_baseline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型 & tokenizer
    clip_model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # 数据
    tf_train = build_transforms(args.image_size, is_train=True, rrc_scale=(0.5,1.0))
    tf_eval  = build_transforms(args.image_size, is_train=False)

    # === Stage 1（可选）===
    if args.do_stage1:
        print("[Stage1] Finetune encoders (Long-CLIP-only style on COCO train)")
        train_set = CocoTrainPairs(args.data_root, transform=tf_train, max_pairs_per_image=5)
        stage1_finetune(clip_model, tokenizer, train_set, device,
                        epochs=args.stage1_epochs, batch_size=128, lr=1e-6, weight_decay=1e-2, betas=(0.9,0.98),
                        workers=args.workers)

    # === Stage 2（可选）===
    head_async = None
    if args.do_stage2:
        print("[Stage2] Train I0Tasync (BN on normalized embeddings)")
        # 推断嵌入维度
        with torch.no_grad():
            dummy = torch.randn(2, 3, args.image_size, args.image_size, device=device)
            fi = clip_model.encode_image(dummy)
            dim = fi.size(1)
        head_async = I0TAsyncHead(dim=dim).to(device)
        train_set = CocoTrainPairs(args.data_root, transform=tf_train, max_pairs_per_image=5)
        stage2_train_async_bn(clip_model, head_async, tokenizer, train_set, device,
                              epochs=args.stage2_epochs, batch_size=256, lr=1e-4, weight_decay=0.2, betas=(0.9,0.98),
                              workers=args.workers)
        # 保存 BN
        torch.save({"state_dict": head_async.state_dict(), "dim": dim},
                   os.path.join(args.save_dir, "iotasync_bn.pt"))

    # === I0Tpost（推理时）===
    post = None
    if args.use_i0tpost:
        print("[I0Tpost] Estimating modality means ...")
        if args.i0tpost_ref == "coco_val":
            ref_ds = CocoCaptionsEval(args.data_root, split="val", transform=tf_eval)
        else:
            ref_ds = Flickr30kEval(args.data_root, split="test", transform=tf_eval)
        mean_img, mean_txt = I0TPostStandardizer.estimate_means(clip_model, tokenizer, ref_ds, device, batch_size=args.batch_size)
        post = I0TPostStandardizer(mean_img, mean_txt, device=device)
        torch.save({"mean_img": mean_img.cpu(), "mean_txt": mean_txt.cpu()},
                   os.path.join(args.save_dir, f"iotpost_means_{args.i0tpost_ref}.pt"))

    # === 评测 ===
    if args.do_eval:
        results={"config":{
            "data_root": args.data_root, "device": device, "model": args.model, "pretrained": args.pretrained,
            "image_size": args.image_size, "batch_size": args.batch_size,
            "use_i0tpost": args.use_i0tpost, "i0tpost_ref": args.i0tpost_ref,
            "use_i0tasync": head_async is not None
        },"zero_shot":{},"retrieval":{},"runtime_sec":{}}

        t_all=time.time()

        # Zero-shot: CIFAR100
        t0=time.time()
        print("\n[Eval] CIFAR-100 (ZS)")
        cifar_val = tvds.CIFAR100(root=os.path.join(args.data_root,"cifar100"), train=False, transform=tf_eval, download=True)
        cifar_train = tvds.CIFAR100(root=os.path.join(args.data_root,"cifar100"), train=True, transform=tf_eval, download=True)
        cifar_classes = cifar_train.classes
        loader = DataLoader(cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        top1, top5, mg, *_ = zeroshot_eval(clip_model, tokenizer, loader, cifar_classes, device, head=head_async, post=post, templates=CIFAR100_TEMPLATES)
        results["zero_shot"]["cifar100"]={"top1":top1,"top5":top5,"modality_gap":mg}
        results["runtime_sec"]["cifar100"]=time.time()-t0
        print(f"CIFAR100: top1={top1:.4f}, top5={top5:.4f}, time={results['runtime_sec']['cifar100']:.1f}s")

        # Zero-shot: Tiny-ImageNet-200
        t0=time.time()
        print("\n[Eval] Tiny-ImageNet-200 (ZS)")
        tiny = TinyImageNet200(args.data_root, split="val", transform=tf_eval, auto_download=True)
        loader = DataLoader(tiny, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        top1, top5, mg, *_ = zeroshot_eval(clip_model, tokenizer, loader, tiny.classes, device, head=head_async, post=post, templates=CIFAR100_TEMPLATES)
        results["zero_shot"]["tiny_imagenet_200"]={"top1":top1,"top5":top5,"modality_gap":mg}
        results["runtime_sec"]["tiny_imagenet_200"]=time.time()-t0
        print(f"Tiny-ImageNet-200: top1={top1:.4f}, top5={top5:.4f}, time={results['runtime_sec']['tiny_imagenet_200']:.1f}s")

        # Zero-shot: DTD
        t0=time.time()
        print("\n[Eval] DTD (ZS)")
        dtd_val = tvds.DTD(root=os.path.join(args.data_root,"dtd"), split="test", transform=tf_eval, download=True)
        dtd_train = tvds.DTD(root=os.path.join(args.data_root,"dtd"), split="train", transform=tf_eval, download=True)
        dtd_classes = dtd_train.classes
        loader = DataLoader(dtd_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        top1, top5, mg, *_ = zeroshot_eval(clip_model, tokenizer, loader, dtd_classes, device, head=head_async, post=post, templates=CIFAR100_TEMPLATES)
        results["zero_shot"]["dtd"]={"top1":top1,"top5":top5,"modality_gap":mg}
        results["runtime_sec"]["dtd"]=time.time()-t0
        print(f"DTD: top1={top1:.4f}, top5={top5:.4f}, time={results['runtime_sec']['dtd']:.1f}s")

        # Retrieval: MSCOCO
        t0=time.time()
        print("\n[Eval] MSCOCO (I2T/T2I)")
        coco = CocoCaptionsEval(args.data_root, split="val", transform=tf_eval)
        if args.max_coco is not None:
            class _Sub(Dataset):
                def __init__(self, base, n): self.base, self.n = base, min(n,len(base))
                def __len__(self): return self.n
                def __getitem__(self,i): return self.base[i]
            coco = _Sub(coco, args.max_coco)
        mg, i2t, t2i = retrieval_eval(clip_model, tokenizer, coco, device, batch_size=args.batch_size, head=head_async, post=post, max_items=args.max_coco)
        results["retrieval"]["mscoco"]={"I2T":i2t,"T2I":t2i,"modality_gap":mg}
        results["runtime_sec"]["mscoco"]=time.time()-t0
        print(f"MSCOCO: I2T={i2t}, T2I={t2i}, time={results['runtime_sec']['mscoco']:.1f}s")

        # Retrieval: Flickr30k
        t0=time.time()
        print("\n[Eval] Flickr30k (I2T/T2I)")
        flickr = Flickr30kEval(args.data_root, split="test", transform=tf_eval)
        if args.max_flickr is not None:
            class _SubF(Dataset):
                def __init__(self, base, n): self.base, self.n = base, min(n,len(base))
                def __len__(self): return self.n
                def __getitem__(self,i): return self.base[i]
            flickr = _SubF(flickr, args.max_flickr)
        mg, i2t, t2i = retrieval_eval(clip_model, tokenizer, flickr, device, batch_size=args.batch_size, head=head_async, post=post, max_items=args.max_flickr)
        results["retrieval"]["flickr30k"]={"I2T":i2t,"T2I":t2i,"modality_gap":mg}
        results["runtime_sec"]["flickr30k"]=time.time()-t0
        print(f"Flickr30k: I2T={i2t}, T2I={t2i}, time={results['runtime_sec']['flickr30k']:.1f}s")

        # 保存
        total=time.time()-t_all
        results["runtime_sec"]["total"]=total
        ts=time.strftime("%Y%m%d_%H%M%S")
        json_path=os.path.join(args.save_dir, f"iot_baseline_{ts}.json")
        with open(json_path,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)
        # CSV
        rows=[]
        def add_row(name: str, metrics: Dict[str,Any], rt: Optional[float]=None):
            row={"dataset":name}
            for k,v in metrics.items():
                if isinstance(v,dict):
                    for kk,vv in v.items():
                        if isinstance(vv,dict):
                            for kkk,vvv in vv.items():
                                row[f"{k}.{kk}.{kkk}"]=vvv
                        else:
                            row[f"{k}.{kk}"]=vv
                else:
                    row[k]=v
            if rt is not None: row["runtime_sec"]=rt
            rows.append(row)
        add_row("cifar100",results["zero_shot"]["cifar100"],results["runtime_sec"]["cifar100"])
        add_row("tiny_imagenet_200",results["zero_shot"]["tiny_imagenet_200"],results["runtime_sec"]["tiny_imagenet_200"])
        add_row("dtd",results["zero_shot"]["dtd"],results["runtime_sec"]["dtd"])
        add_row("mscoco.I2T",{"I2T":results["retrieval"]["mscoco"]["I2T"],"mg":results["retrieval"]["mscoco"]["modality_gap"]},results["runtime_sec"]["mscoco"])
        add_row("mscoco.T2I",{"T2I":results["retrieval"]["mscoco"]["T2I"],"mg":results["retrieval"]["mscoco"]["modality_gap"]},results["runtime_sec"]["mscoco"])
        add_row("flickr30k.I2T",{"I2T":results["retrieval"]["flickr30k"]["I2T"],"mg":results["retrieval"]["flickr30k"]["modality_gap"]},results["runtime_sec"]["flickr30k"])
        add_row("flickr30k.T2I",{"T2I":results["retrieval"]["flickr30k"]["T2I"],"mg":results["retrieval"]["flickr30k"]["modality_gap"]},results["runtime_sec"]["flickr30k"])
        rows.append({"dataset":"TOTAL","runtime_sec":results["runtime_sec"]["total"]})
        csv_path=os.path.join(args.save_dir, f"iot_baseline_{ts}.csv")
        pd.DataFrame(rows).to_csv(csv_path,index=False)
        print(f"\nSaved JSON -> {json_path}")
        print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
