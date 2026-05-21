#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAS Baseline — 完整复现版（Baseline #6）
- 论文：Bridging the Modality Gap: Dimension Information Alignment and Sparse Spatial Constraint
  for Image-Text Matching (ACM MM'24, arXiv:2410.16853)
- 关键组件：
  1) 局部交互与全局汇聚：区域-词级相似→更新局部→池化为全局
  2) Triplet 对齐（距离加权负采样）
  3) 维度信息对齐（DIA）正则
  4) 空间约束：跨模态 L_inter 与内模态 L_intra（稀疏相关）
  5) 总目标：L = L_triplet + w_dim*L_dim + w_inter*L_inter + w_intra*L_intra

- 统一评测口径：
  * Modality Gap: centroid distance / Fréchet distance / relative gap（单位范数）
  * Zero-Shot（CIFAR100, Tiny-ImageNet-200, DTD）Top1/Top5（18模板）
  * 检索（MSCOCO/Flickr30k）I2T & T2I R@1/5/10
  * 记录所有阶段与总运行时间；JSON/CSV 落盘
"""

import os
import math
import json
import time
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tvds, transforms
import torchvision
from PIL import Image
from tqdm import tqdm
import pandas as pd

import open_clip
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import to_pil_image



# ===================== 通用 & 工具 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2norm(x: torch.Tensor, dim=-1, eps=1e-9):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def to_device(x, device):
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x


def ts():
    return time.strftime("%Y%m%d_%H%M%S")


# ===================== 图像变换（与前基线一致） =====================
def build_tf(image_size=224, train=True, rrc_scale=(0.5, 1.0)):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=rrc_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])


# ===================== Tiny-ImageNet-200 =====================
class TinyImageNet200(Dataset):
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root: str, split="val", transform=None, auto_download=True):
        self.root_base = Path(root).expanduser().resolve()
        self.root = self.root_base / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        assert split in ["train", "val"]
        self._ensure(auto_download)
        wnids = (self.root / "wnids.txt").read_text().splitlines()
        self.wnids = [w.strip() for w in wnids if w.strip()]
        words_map = {}
        for line in (self.root / "words.txt").read_text().splitlines():
            if not line.strip():
                continue
            k, v = line.split("\t")
            words_map[k] = v
        self.class_to_idx = {w: i for i, w in enumerate(self.wnids)}
        self.idx_to_name = {self.class_to_idx[w]: words_map.get(w, w).split(",")[0].split(";")[0] for w in self.wnids}
        self.samples = []
        if split == "train":
            tdir = self.root / "train"
            for wnid in self.wnids:
                idir = tdir / wnid / "images"
                self.samples += [(str(p), self.class_to_idx[wnid]) for p in idir.glob("*.JPEG")]
        else:
            vdir = self.root / "val"
            ann = vdir / "val_annotations.txt"
            mapping = {}
            for line in ann.read_text().splitlines():
                if not line.strip():
                    continue
                fn, wnid = line.split("\t")[:2]
                mapping[fn] = wnid
            idir = vdir / "images"
            for p in idir.glob("*.JPEG"):
                self.samples.append((str(p), self.class_to_idx[mapping[p.name]]))
        if not self.samples:
            raise RuntimeError("Tiny-ImageNet-200 empty")

    def _ensure(self, auto_download):
        if self.root.exists() and (self.root / "wnids.txt").exists():
            return
        if not auto_download:
            raise FileNotFoundError(self.root)
        self.root_base.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_base / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            print("[Tiny-ImageNet] Downloading ...")
            import urllib.request, shutil
            with urllib.request.urlopen(self.URL) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)
        print("[Tiny-ImageNet] Extracting ...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root_base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fp, y = self.samples[i]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

    @property
    def classes(self):
        return [self.idx_to_name[i] for i in range(len(self.idx_to_name))]


# ===================== COCO / Flickr30k（老接口） =====================
class CocoCaptionsEval(Dataset):
    def __init__(self, root: str, split="val", transform=None):
        droot = Path(root).expanduser().resolve() / "coco2017"
        img_dir = droot / "images" / ("val2017" if split == "val" else "train2017")
        ann = droot / "annotations" / f"captions_{'val2017' if split == 'val' else 'train2017'}.json"
        if not img_dir.exists() or not ann.exists():
            raise FileNotFoundError(f"COCO not found: {img_dir} / {ann}")
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, caps = self.ds[i]
        return img, caps


class Flickr30kEval(Dataset):
    def __init__(self, root: str, split="test", transform=None):
        droot = Path(root).expanduser().resolve() / "flickr30k"
        images_root = droot / "flickr30k-images"
        ann_file = droot / "results_20130124.token"
        if not images_root.exists():
            raise FileNotFoundError(images_root)
        if not ann_file.exists():
            raise FileNotFoundError(ann_file)
        self.ds = tvds.Flickr30k(root=str(images_root), ann_file=str(ann_file), transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, caps = self.ds[i]
        caps = caps if isinstance(caps, list) else [str(caps)]
        return img, caps


def collate_caps(batch):
    images = torch.stack([b[0] for b in batch], 0)
    caps_list = []
    for _, caps in batch:
        caps_list.append(list(caps) if isinstance(caps, (list, tuple)) else [str(caps)])
    return images, caps_list


# ===================== Fréchet (GPU) & Modality Gap =====================
def _cov_torch(Z: torch.Tensor) -> torch.Tensor:
    Zc = Z - Z.mean(dim=0, keepdim=True)
    N = Z.shape(0) if callable(getattr(Z, "shape", None)) else Z.shape[0]
    N = Z.shape[0]
    return (Zc.T @ Zc) / (N - 1 + 1e-9)


def _sqrtm_psd_torch(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    Ahalf = (evecs * torch.sqrt(evals).unsqueeze(0)) @ evecs.mH
    return Ahalf


def _fid_torch(mu1: torch.Tensor, C1: torch.Tensor, mu2: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    diff = mu1 - mu2
    C1_half = _sqrtm_psd_torch(C1)
    inner = C1_half @ C2 @ C1_half
    inner_half = _sqrtm_psd_torch(inner)
    tr_term = torch.trace(C1 + C2 - 2.0 * inner_half)
    fid_sq = torch.clamp(diff @ diff + tr_term, min=0.0)
    return torch.sqrt(fid_sq + 1e-12)


def modality_gap(imgE: torch.Tensor, txtE: torch.Tensor) -> Dict[str, float]:
    x = imgE
    y = txtE
    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    Cx = _cov_torch(x)
    Cy = _cov_torch(y)
    centroid = torch.norm(mu_x - mu_y).item()
    fd = _fid_torch(mu_x, Cx, mu_y, Cy).item()
    denom = float(torch.sqrt(0.5 * (torch.trace(Cx) + torch.trace(Cy)) + 1e-12))
    rmg = (centroid / denom) if denom > 0 else float("nan")
    return {"centroid_distance": centroid, "frechet_distance": fd, "relative_modality_gap": rmg}


# ===================== BUTD 区域特征抽取（缓存） =====================
class RegionExtractor:
    """
    使用 torchvision Faster R-CNN (ResNet50-FPN) 区域提取。
    - 每图取 topK proposals，经 ROI head 得 region embedding (d=1024)。
    - 结果缓存 .npz：{ "feat": (K, D) }
    """
    def __init__(self, device="cuda", cache_dir="runs/dias_cache", topk=36):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.topk = topk
        self.det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.det.eval().to(device)
        self.box_roi_pool = self.det.roi_heads.box_roi_pool
        self.backbone = self.det.backbone
        self.box_head = self.det.roi_heads.box_head
        self.transform = self.det.transform

    @torch.no_grad()
    def extract_regions(self, img_path: str) -> np.ndarray:
        cache = self.cache_dir / (Path(img_path).stem + ".npz")
        if cache.exists():
            arr = np.load(cache)["feat"]
            return arr
        img = Image.open(img_path).convert("RGB")
        img_t = transforms.ToTensor()(img).to(self.device)
        images, _ = self.transform([img_t])
        features = self.backbone(images.tensors)
        proposals, _ = self.det.rpn(images, features)
        boxes = proposals[0]
        roi = self.box_roi_pool(features, [boxes], images.image_sizes)
        feat = self.box_head(roi)  # (N, 1024)
        K = min(self.topk, feat.size(0))
        feat = feat[:K].detach().cpu().numpy()
        np.savez_compressed(cache, feat=feat)
        return feat


# ===================== 文本编码器（BERT 或 BiGRU） =====================
class TextEncoder(nn.Module):
    def __init__(self, enc_type="bert", device="cuda", bert_name="bert-base-uncased",
                 word_dim=300, rnn_dim=1024, max_len=40):
        super().__init__()
        self.enc_type = enc_type
        self.device = device
        self.max_len = max_len
        if enc_type == "bert":
            self.tok = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(bert_name)
            self.proj = nn.Linear(self.bert.config.hidden_size, 1024)
        else:
            self.emb = nn.Embedding(50000, word_dim)
            self.rnn = nn.GRU(word_dim, rnn_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.proj = nn.Identity()
        self.to(device)

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = next(self.parameters()).device
        if self.enc_type == "bert":
            toks = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.bert(**toks).last_hidden_state  # (B,T,H)
            feats = self.proj(out)                     # (B,T,1024)
            attn_mask = toks["attention_mask"]         # (B,T)
            return feats, attn_mask
        else:
            B = len(texts)
            T = self.max_len
            idx = torch.zeros(B, T, dtype=torch.long, device=self.device)
            lens = []
            for i, s in enumerate(texts):
                words = s.strip().lower().split()[:T]
                lens.append(len(words))
                for j, w in enumerate(words):
                    h = (hash(w) % (self.emb.num_embeddings - 1)) + 1
                    idx[i, j] = h
            em = self.emb(idx)  # (B,T,word_dim)
            out, _ = self.rnn(em)  # (B,T,rnn_dim)
            attn_mask = torch.zeros(B, T, dtype=torch.long, device=self.device)
            for i, l in enumerate(lens):
                attn_mask[i, :l] = 1
            feats = self.proj(out)
            return feats, attn_mask


# ===================== DIAS 模块（局部交互 + 正则 + 稀疏） =====================
class DIAS(nn.Module):
    def __init__(self, d=1024, pool="mean"):
        super().__init__()
        self.d = d
        self.pool = pool

    @staticmethod
    def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return a @ b.transpose(-1, -2)

    def local_interaction(self, V: torch.Tensor, T: torch.Tensor, T_mask: torch.Tensor):
        S = self.cosine_sim(V, T)  # (B,Nv,Nt)
        S = S * T_mask.unsqueeze(1) + (1 - T_mask.unsqueeze(1)) * (-1e4)
        w_v = F.softmax(S, dim=-1)  # 区域由词加权
        v_hat = w_v @ T

        w_t = F.softmax(S.transpose(1, 2), dim=-1)  # (B,Nt,Nv)
        t_hat = w_t @ V

        return v_hat, t_hat, S

    def pool_global(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            return X.mean(dim=1) if self.pool == "mean" else X.max(dim=1).values
        else:
            if self.pool == "mean":
                m = mask.unsqueeze(-1).float()
                return (X * m).sum(dim=1) / (m.sum(dim=1) + 1e-9)
            else:
                X = X.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
                return X.max(dim=1).values

    def dim_info_align_loss(self, V_local: torch.Tensor, T_local: torch.Tensor, T_mask: torch.Tensor):
        """
        批内近似跨模态“维度相关矩阵”：
        - 将区域与词的局部特征在批维与位置维拼到一起，得到 mV∈R^{d×(B·Nv)}、mT∈R^{d×(B·Nt)}
        - 为避免 Nv≠Nt 导致形状不匹配，随机/等距采样到相同列数 K=min(B·Nv, B·Nt)
        - 用列向量的余弦相似构造 C∈R^{d×d}，再按对角占比（改进式 Eq.(6)）计算 L_dim
        """
        B, Nv, d = V_local.shape
        _, Nt, _ = T_local.shape

        # mask 无效词
        T_loc = T_local * T_mask.unsqueeze(-1)

        # 展平局部；转成“维度×样本数”的形式
        V_all = V_local.reshape(-1, d)           # (B*Nv, d)
        T_all = T_loc.reshape(-1, d)             # (B*Nt, d)
        mV = V_all.transpose(0, 1)               # (d, B*Nv)
        mT = T_all.transpose(0, 1)               # (d, B*Nt)

        # 列向量归一化（余弦）
        mV = F.normalize(mV, dim=0)
        mT = F.normalize(mT, dim=0)

        # 统一列数：采样到 K = min(#cols)
        KV = mV.size(1); KT = mT.size(1)
        K = int(min(KV, KT))
        if K == 0:
            # 极端情况下（mask 全 0），返回 0 损失
            return torch.zeros([], device=V_local.device, dtype=V_local.dtype)

        # 等距子采样（更稳定，避免频繁随机数导致不可复现）
        if KV != K:
            idxV = torch.linspace(0, KV - 1, steps=K, device=mV.device).round().long()
            mV = mV[:, idxV]                     # (d, K)
        if KT != K:
            idxT = torch.linspace(0, KT - 1, steps=K, device=mT.device).round().long()
            mT = mT[:, idxT]                     # (d, K)

        # C = mV @ mT^T -> (d, d)
        C = mV @ mT.transpose(0, 1)

        # 改进式 Eq.(6)：对角占比 + 行/列归一
        row_sum = C.sum(dim=1) + 1e-9
        col_sum = C.sum(dim=0) + 1e-9
        diag = torch.diag(C)
        term = -(diag / row_sum + diag / col_sum)
        Ldim = term.sum()
        return Ldim


    @staticmethod
    def pairwise_dist(a: torch.Tensor, b: torch.Tensor):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return 1.0 - (a @ b.t())

    def sparse_spatial_losses(self, Vg: torch.Tensor, Tg: torch.Tensor):
        N = Vg.size(0)
        X = self.pairwise_dist(Vg, Tg)
        Y = self.pairwise_dist(Vg, Vg)
        Z = self.pairwise_dist(Tg, Tg)

        Lx = (X - X.t()).abs()
        Pimg = torch.sigmoid(-Lx)
        mu_i = Pimg.mean(dim=1, keepdim=True)
        std_i = Pimg.std(dim=1, keepdim=True) + 1e-9
        kappa_i = mu_i + 1.0 * std_i
        mu_j = Pimg.mean(dim=0, keepdim=True)
        std_j = Pimg.std(dim=0, keepdim=True) + 1e-9
        kappa_j = mu_j + 1.0 * std_j
        K = torch.maximum(kappa_i, kappa_j)
        Bx = (Lx > K).float()
        L_inter = ((Bx * Lx) ** 2).sum()

        Lyz = (Y - Z).abs()
        Py = torch.sigmoid(-Lyz)
        mu_i2 = Py.mean(dim=1, keepdim=True)
        std_i2 = Py.std(dim=1, keepdim=True) + 1e-9
        kappa_i2 = mu_i2 + 1.0 * std_i2
        mu_j2 = Py.mean(dim=0, keepdim=True)
        std_j2 = Py.std(dim=0, keepdim=True) + 1e-9
        kappa_j2 = mu_j2 + 1.0 * std_j2
        K2 = torch.maximum(kappa_i2, kappa_j2)
        Byz = (Lyz > K2).float()
        L_intra = ((Byz * Lyz) ** 2).sum()

        return L_inter, L_intra

    def forward(self, V_local: torch.Tensor, T_local: torch.Tensor, T_mask: torch.Tensor):
        v_hat, t_hat, S = self.local_interaction(V_local, T_local, T_mask)
        Vg = self.pool_global(v_hat)                      # (B,d)
        Tg = self.pool_global(t_hat, mask=T_mask)         # (B,d)
        L_dim = self.dim_info_align_loss(V_local, T_local, T_mask)
        L_inter, L_intra = self.sparse_spatial_losses(Vg, Tg)
        return Vg, Tg, L_dim, L_inter, L_intra, S


# ===================== Triplet（距离加权负采样近似） =====================
def distance_weighted_sampling(emb: torch.Tensor, labels: torch.Tensor, cutoff=0.5, nonzero=True):
    """
    Wu et al. ICCV'17 距离加权负采样的数值稳定实现（同类为正，不同类为负）。
    关键改动：在 log 域计算权重，避免 dist**(d-2) 在高维上溢出/NaN。
    emb: (N,d) 已做单位范数（或近似）；labels: (N,)
    返回：三元组索引 (a,p,n)
    """
    with torch.no_grad():
        # 相似度矩阵（数值安全：先 normalize 再 matmul）
        e = F.normalize(emb, dim=-1, eps=1e-12)
        sim = (e @ e.t()).clamp(-1.0, 1.0)  # (N,N)

        N, d = emb.size(0), emb.size(1)
        a_idx, p_idx, n_idx = [], [], []

        for i in range(N):
            pos = torch.where(labels == labels[i])[0]
            pos = pos[pos != i]
            neg = torch.where(labels != labels[i])[0]
            if len(pos) == 0 or len(neg) == 0:
                continue

            s = sim[i, neg]                          # (M,)
            # 角距离：dist = sqrt(2 - 2s) ∈ [0,2]
            dist = torch.sqrt((2.0 - 2.0 * s).clamp_(0.0, 2.0))

            # ---- 数值稳定的权重：logw = (d-2)*log(dist + eps) ----
            logw = (d - 2.0) * torch.log(dist + 1e-12)

            # 可选：cutoff 将过近的负样本降权（避免过“难”的负样本）
            if cutoff is not None and cutoff > 0:
                # 将 dist < cutoff 的项降低权重：等价于在 log 域减去一个常数
                mask_near = (dist < cutoff)
                logw = torch.where(mask_near, logw - 50.0, logw)  # 50 相当于把权重乘以 ~1e-22

            # 处理非有限值（inf/NaN）→ 极小数，避免影响 softmax
            logw = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -1e9))

            # 归一化为概率（数值稳定）
            # 若全为 -1e9（即退化），softmax 会得到全 1/M 的分布（因为 e^{-1e9} 全接近 0，再由内部归一化）
            probs = torch.softmax(logw, dim=0)

            # 兜底：若仍出现 sum==0（极端情况），退化为均匀分布
            psum = probs.sum()
            if not torch.isfinite(psum) or psum.item() == 0.0:
                probs = torch.full_like(probs, 1.0 / probs.numel())

            # 采一个负样本
            n = neg[torch.multinomial(probs, 1)]
            # 正样本随机挑一个
            p = pos[random.randrange(len(pos))]

            a_idx.append(i)
            p_idx.append(p.item())
            n_idx.append(n.item())

        if len(a_idx) == 0:
            return (torch.empty(0, dtype=torch.long, device=emb.device),
                    torch.empty(0, dtype=torch.long, device=emb.device),
                    torch.empty(0, dtype=torch.long, device=emb.device))

        return (torch.tensor(a_idx, device=emb.device),
                torch.tensor(p_idx, device=emb.device),
                torch.tensor(n_idx, device=emb.device))


def triplet_loss_from_indices(emb: torch.Tensor, a, p, n, margin=0.2):
    if len(a) == 0:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    sim = emb @ emb.t()
    s_ap = sim[a, p]
    s_an = sim[a, n]
    loss = F.relu(margin - s_ap + s_an).mean()
    return loss


# ===================== 训练数据封装：COCO/Flickr30k =====================
class PairDataset(Dataset):
    """
    输出：局部区域特征 (Nv, d) + 正样本文本（多个 caption 中随机取1）
    在线抽取区域 & 文本向量并缓存
    """
    def __init__(self, base_ds, split_name: str, data_root: str, device="cuda",
                 cache_dir="runs/dias_cache", region_topk=36, text_type="bert"):
        self.base = base_ds
        self.split = split_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if "coco" in str(type(base_ds)).lower():
            sub = "coco_regions"
        elif "flickr" in str(type(base_ds)).lower():
            sub = "flickr_regions"
        else:
            sub = "regions"
        self.reg_cache = self.cache_dir / sub
        self.reg_cache.mkdir(exist_ok=True, parents=True)
        self.region_ext = RegionExtractor(device=device, cache_dir=str(self.reg_cache), topk=region_topk)
        self.text_type = text_type
        self.text_enc = TextEncoder(enc_type=text_type, device=device)

    def __len__(self):
        return len(self.base)

    # @torch.no_grad()
    def __getitem__(self, i):
        img, caps = self.base[i]
        if isinstance(img, torch.Tensor):
            x = img.detach().float().cpu().clamp(0, 1)
            img_pil = to_pil_image(x)
        else:
            img_pil = img

        tmp = f"tmp_{hash((self.split, i)) & 0xfffffff}.jpg"
        tmp_path = self.reg_cache / tmp
        if not tmp_path.exists():
            img_pil.save(tmp_path)

        reg = self.region_ext.extract_regions(str(tmp_path))  # (K,1024)
        reg = torch.tensor(reg, dtype=torch.float32, device=self.device)

        cap = random.choice(caps)
        # Tloc, Tmask = self.text_enc([cap])  # (1,Nt,d), (1,Nt)
        return reg, cap


# ===================== 评测：检索 R@K =====================
def recalls_from_sim(sim: torch.Tensor, gt: Dict[int, List[int]]):
    res = {}
    Kidx = torch.topk(sim, k=10, dim=1, largest=True, sorted=True).indices.cpu().numpy()
    for k in (1, 5, 10):
        ok = 0
        for i in range(Kidx.shape[0]):
            pred = set(Kidx[i, :k].tolist())
            if any(t in pred for t in gt[i]):
                ok += 1
        res[f"R@{k}"] = ok / Kidx.shape[0]
    return res


@torch.no_grad()
def retrieval_eval_global(encoder: "DIASSystem", loader, device):
    Vg_all, Tg_all = [], []
    img2caps, cap2imgs = {}, {}
    img_idx = cap_idx = 0
    for regs, caps in tqdm(loader, desc="Embed (global)", leave=False):
        # 批量编码文本（不需要梯度）
        Tloc, Tmask = encoder.text_enc(caps)          # (B,T,d), (B,T)
        B = regs.size(0)
        for b in range(B):
            Vg, Tg = encoder.encode_global(regs[b], Tloc[b:b + 1], Tmask[b:b + 1])
            Vg_all.append(Vg); Tg_all.append(Tg[0])
            img2caps[img_idx] = [cap_idx]; cap2imgs[cap_idx] = [img_idx]
            img_idx += 1; cap_idx += 1
    Vg_all = F.normalize(torch.stack(Vg_all).to(device), dim=-1)
    Tg_all = F.normalize(torch.stack(Tg_all).to(device), dim=-1)
    sim = Vg_all @ Tg_all.t()
    i2t = recalls_from_sim(sim, img2caps)
    t2i = recalls_from_sim(sim.t(), cap2imgs)
    mg = modality_gap(Vg_all, Tg_all)
    return mg, i2t, t2i


# ===================== 评测：Zero-Shot（与前基线一致，用 BERT 文本原型） =====================
ZS_TEMPLATES = [
    "a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
    "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
    "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.", "a photo of the {}.",
    "a blurry photo of the {}.", "a black and white photo of the {}.", "a low contrast photo of the {}.",
    "a high contrast photo of the {}.", "a bad photo of the {}.", "a good photo of the {}.",
    "a photo of the small {}.", "a photo of the big {}."
]


@torch.no_grad()
def zeroshot_eval_dias(encoder, text_enc: TextEncoder, loader, classnames, device, templates=None):
    T = templates or ZS_TEMPLATES
    cls_bank = []
    for cname in tqdm(classnames, desc="ZS: encode classes", leave=False):
        prompts = [t.format(cname) for t in T]
        Tloc, Tmask = text_enc(prompts)              # (P,L,d)
        Tg = encoder.encode_text_global(Tloc, Tmask) # (P,d)
        cls_bank.append(F.normalize(Tg, dim=-1))
    C = len(classnames)

    correct1 = correct5 = n = 0
    img_all = []
    for images, labels in tqdm(loader, desc="Zero-shot", leave=False):
        images = images.to(device, non_blocking=True)
        fi = F.normalize(encoder.encode_image_global(images), dim=-1)  # (B,d)
        img_all.append(fi)
        scores = torch.zeros(fi.size(0), C, device=device)
        for ci in range(C):
            proto = cls_bank[ci].to(device)
            scores[:, ci] = (fi @ proto.t()).mean(dim=1)
        top5 = scores.topk(5, dim=-1).indices
        labels = labels.to(device)
        correct1 += (top5[:, 0] == labels).sum().item()
        for k in range(5):
            correct5 += (top5[:, k] == labels).sum().item()
        n += labels.size(0)

    img_cat = torch.cat(img_all, 0)
    txt_cat = torch.cat([c for c in cls_bank], 0).to(device)
    mg = modality_gap(img_cat, txt_cat)
    return correct1 / n, correct5 / n, mg, img_cat, txt_cat


# ===================== 编码器封装（训练态/评测态） =====================
class DIASSystem(nn.Module):
    def __init__(self, text_type="bert", device="cuda", region_topk=36):
        super().__init__()
        self.device = device
        self.text_enc = TextEncoder(enc_type=text_type, device=device)
        self.dias = DIAS(d=1024)
        self.region_ext = RegionExtractor(device=device, cache_dir="runs/dias_cache", topk=region_topk)
        self.zs_img_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )

    @torch.no_grad()
    def encode_image_global(self, images: torch.Tensor):
        return l2norm(self.zs_img_model.encode_image(images), -1)

    @torch.no_grad()
    def encode_text_global(self, Tloc: torch.Tensor, Tmask: torch.Tensor):
        Tg = (Tloc * Tmask.unsqueeze(-1).float()).sum(dim=1) / (Tmask.sum(dim=1, keepdim=True).float() + 1e-9)
        return F.normalize(Tg, dim=-1)

    def forward_batch(self, batch):
        regs, caps = batch                          # caps: List[str]
        Tloc, Tmask = self.text_enc(caps)           # (B,T,d), (B,T) —— 带 grad 的计算
        V_local = regs                              # (B,Nv,d) —— 区域特征可视为常量输入
        Vg, Tg, Ldim, Linter, Lintra, _ = self.dias(V_local, Tloc, Tmask)
        return Vg, Tg, Ldim, Linter, Lintra

    @torch.no_grad()
    def encode_global(self, regs: torch.Tensor, Tloc: torch.Tensor, Tmask: torch.Tensor):
        Vg, Tg, *_ = self.dias(regs.unsqueeze(0), Tloc, Tmask)
        return Vg[0], Tg[0]


# ===================== 训练循环 =====================
def train_dias(
    sys: DIASSystem, train_loader: DataLoader, val_loader: Optional[DataLoader],
    epochs=30, lr=5e-4, weight_decay=0.0, margin=0.2,
    w_dim=10.0, w_inter=0.05, w_intra=0.1, device="cuda", save_dir="runs/dias_ckpt"
):
    os.makedirs(save_dir, exist_ok=True)
    optim = torch.optim.Adam(sys.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

    for epoch in range(1, epochs + 1):
        sys.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}")
        loss_meter = 0.0
        for regs, caps in pbar:
            Vg, Tg, Ldim, Linter, Lintra = sys.forward_batch((regs,caps))
            Vg = F.normalize(Vg, dim=-1)
            Tg = F.normalize(Tg, dim=-1)
            emb = torch.cat([Vg, Tg], dim=0)
            labels = torch.arange(Vg.size(0), device=device)
            labels = torch.cat([labels, labels], dim=0)
            a, p, n = distance_weighted_sampling(emb.detach(), labels)
            L_tri = triplet_loss_from_indices(emb, a, p, n, margin=margin)
            loss = L_tri + w_dim * Ldim + w_inter * Linter + w_intra * Lintra
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_meter += loss.item()
            pbar.set_postfix(loss=f"{loss_meter / ((pbar.n) + 1):.4f}")
        scheduler.step()

        if val_loader is not None and epoch % 5 == 0:
            sys.eval()
            with torch.no_grad():
                V_list, T_list = [], []
                for regs, caps in tqdm(val_loader, desc="Val embed"):
                    Vg, Tg, *_ = sys.forward_batch((regs, caps))
                    V_list.append(F.normalize(Vg, dim=-1))
                    T_list.append(F.normalize(Tg, dim=-1))
                V = torch.cat(V_list, 0)
                T = torch.cat(T_list, 0)
                mg = modality_gap(V, T)
                print(f"[Val] MG: {mg}")

        torch.save({"epoch": epoch, "state_dict": sys.state_dict()}, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))


# ===================== collate（训练/评测统一） =====================
def collate_pair_batch_caps(batch, device):
    """
    batch: List[(reg:(Nv,d)[cuda], cap:str)]
    return:
      R: (B, Nv_max, d)[cuda]
      caps: List[str] 长度为 B
    """
    regs, caps = [], []
    for (r, c) in batch:
        regs.append(r); caps.append(c)
    Nv = max([x.size(0) for x in regs])
    D = regs[0].size(-1)
    R = torch.zeros(len(batch), Nv, D, device=device)
    for i in range(len(batch)):
        R[i, :regs[i].size(0)] = regs[i]
    return R, caps


# ===================== 主程序 =====================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-dir", type=str, default="runs/dias_baseline")
    ap.add_argument("--seed", type=int, default=42)

    # 训练设定（论文）
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size-flickr", type=int, default=128)
    ap.add_argument("--batch-size-coco", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    # 正则权重（论文敏感性分析默认值）
    ap.add_argument("--w-dim", type=float, default=10.0)
    ap.add_argument("--w-inter", type=float, default=0.05)
    ap.add_argument("--w-intra", type=float, default=0.1)

    # 文本编码与区域
    ap.add_argument("--text-enc", type=str, default="bert", choices=["bert", "bigru"])
    ap.add_argument("--region-topk", type=int, default=36)

    # 限制样本数（调试可用）
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)

    # 统一评测（ZS & Retrieval）
    ap.add_argument("--do-train", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # ========= 数据集（PairDataset 底层必须 transform=None） =========
    coco_train_raw = CocoCaptionsEval(args.data_root, split="train", transform=None)
    coco_val_raw   = CocoCaptionsEval(args.data_root, split="val",   transform=None)
    coco_eval_raw  = CocoCaptionsEval(args.data_root, split="val",   transform=None)
    flickr_eval_raw= Flickr30kEval(args.data_root, split="test",     transform=None)

    # ========= 构建系统 =========
    sys = DIASSystem(text_type=args.text_enc, device=device, region_topk=args.region_topk).to(device)

    t_all = time.time()
    results = {"config": {
        "epochs": args.epochs, "lr": args.lr, "batch_size_flickr": args.batch_size_flickr,
        "batch_size_coco": args.batch_size_coco, "w_dim": args.w_dim, "w_inter": args.w_inter,
        "w_intra": args.w_intra, "text_enc": args.text_enc, "region_topk": args.region_topk, "device": device
    }, "runtime_sec": {}, "retrieval": {}, "zero_shot": {}}

    # ========= 训练（MSCOCO） =========
    if args.do_train:
        if args.max_train is not None:
            class Sub(Dataset):
                def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
                def __len__(self): return self.n
                def __getitem__(self, i): return self.base[i]
            coco_train_base = Sub(coco_train_raw, args.max_train)
        else:
            coco_train_base = coco_train_raw

        train_ds = PairDataset(coco_train_base, "train", args.data_root, device=device,
                               region_topk=args.region_topk, text_type=args.text_enc)
        val_ds   = PairDataset(coco_val_raw,   "val",   args.data_root, device=device,
                               region_topk=args.region_topk, text_type=args.text_enc)

        
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size_coco, shuffle=True, num_workers=0,
            pin_memory=False, collate_fn=lambda b: collate_pair_batch_caps(b, device)
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size_coco, shuffle=False, num_workers=0,
            pin_memory=False, collate_fn=lambda b: collate_pair_batch_caps(b, device)
        )

        t0 = time.time()
        train_dias(sys, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                   weight_decay=args.weight_decay, margin=args.margin,
                   w_dim=args.w_dim, w_inter=args.w_inter, w_intra=args.w_intra,
                   device=device, save_dir=os.path.join(args.save_dir, "ckpt"))
        results["runtime_sec"]["train"] = time.time() - t0

    # ========= 评测加载器（评测也走 PairDataset） =========
    def make_eval_loader(base_ds, split_name: str, bs: int):
        if args.max_eval is not None:
            class Sub(Dataset):
                def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
                def __len__(self): return self.n
                def __getitem__(self, i): return self.base[i]
            base_ds = Sub(base_ds, args.max_eval)
        eval_ds = PairDataset(base_ds, split_name, data_root=args.data_root, device=device,
                              region_topk=args.region_topk, text_type=args.text_enc)
        
        return DataLoader(eval_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False,
                      collate_fn=lambda b: collate_pair_batch_caps(b, device))

    # ========= 检索评测：MSCOCO / Flickr30k =========
    t0 = time.time()
    coco_eval_loader = make_eval_loader(coco_eval_raw, split_name="val", bs=64)
    mg_coco, i2t_coco, t2i_coco = retrieval_eval_global(sys, coco_eval_loader, device)
    results["retrieval"]["mscoco"] = {"I2T": i2t_coco, "T2I": t2i_coco, "modality_gap": mg_coco}
    results["runtime_sec"]["mscoco"] = time.time() - t0
    print("[MSCOCO] I2T:", i2t_coco, " T2I:", t2i_coco)

    t0 = time.time()
    flickr_eval_loader = make_eval_loader(flickr_eval_raw, split_name="test", bs=64)
    mg_flickr, i2t_flickr, t2i_flickr = retrieval_eval_global(sys, flickr_eval_loader, device)
    results["retrieval"]["flickr30k"] = {"I2T": i2t_flickr, "T2I": t2i_flickr, "modality_gap": mg_flickr}
    results["runtime_sec"]["flickr30k"] = time.time() - t0
    print("[Flickr30k] I2T:", i2t_flickr, " T2I:", t2i_flickr)

    # ========= Zero-Shot：CIFAR100 / Tiny-ImageNet-200 / DTD =========
    tf_infer = build_tf(224, False)

    t0 = time.time()
    cifar_val = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=False, transform=tf_infer, download=True)
    cifar_train = tvds.CIFAR100(root=os.path.join(args.data_root, "cifar100"), train=True, transform=tf_infer, download=True)
    cifar_classes = cifar_train.classes
    loader = DataLoader(cifar_val, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval_dias(sys, sys.text_enc, loader, cifar_classes, device)
    results["zero_shot"]["cifar100"] = {"top1": top1, "top5": top5, "modality_gap": mg}
    results["runtime_sec"]["cifar100"] = time.time() - t0

    t0 = time.time()
    tiny = TinyImageNet200(args.data_root, split="val", transform=tf_infer, auto_download=True)
    loader = DataLoader(tiny, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval_dias(sys, sys.text_enc, loader, tiny.classes, device)
    results["zero_shot"]["tiny_imagenet_200"] = {"top1": top1, "top5": top5, "modality_gap": mg}
    results["runtime_sec"]["tiny_imagenet_200"] = time.time() - t0

    t0 = time.time()
    dtd_val = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="test", transform=tf_infer, download=True)
    dtd_train = tvds.DTD(root=os.path.join(args.data_root, "dtd"), split="train", transform=tf_infer, download=True)
    dtd_classes = dtd_train.classes
    loader = DataLoader(dtd_val, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval_dias(sys, sys.text_enc, loader, dtd_classes, device)
    results["zero_shot"]["dtd"] = {"top1": top1, "top5": top5, "modality_gap": mg}
    results["runtime_sec"]["dtd"] = time.time() - t0

    # ========= 保存 JSON & CSV =========
    results["runtime_sec"]["total"] = time.time() - t_all
    js_path = os.path.join(args.save_dir, f"dias_baseline_{ts()}.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = []
    def add_row(name, metrics, rt=None):
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
        if rt is not None:
            row["runtime_sec"] = rt
        rows.append(row)

    add_row("mscoco", results["retrieval"]["mscoco"], results["runtime_sec"]["mscoco"])
    add_row("flickr30k", results["retrieval"]["flickr30k"], results["runtime_sec"]["flickr30k"])
    add_row("cifar100", results["zero_shot"]["cifar100"], results["runtime_sec"]["cifar100"])
    add_row("tiny_imagenet_200", results["zero_shot"]["tiny_imagenet_200"], results["runtime_sec"]["tiny_imagenet_200"])
    add_row("dtd", results["zero_shot"]["dtd"], results["runtime_sec"]["dtd"])
    rows.append({"dataset": "TOTAL", "runtime_sec": results["runtime_sec"]["total"]})
    csv_path = os.path.join(args.save_dir, f"dias_baseline_{ts()}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nSaved JSON -> {js_path}")
    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
