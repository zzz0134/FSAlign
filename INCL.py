#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-the-Gap Baseline (OTI & OVI) — 完整实现
- 复现论文两种单样本优化的“模态倒置”方法：
  * OTI: 从图像 -> 文本空间，优化 R 个“伪 token”嵌入（默认 R=1，150步，AdamW lr=0.02, betas=(0.9,0.999), wd=0.01）
  * OVI: 从文本 -> 视觉空间，优化 P 个“伪 patch”嵌入（默认 P 依模型自适应，1000步，同优化器）
  * OVI 直接在 ViT 的 patch-embedding 空间运行（无需像素优化），可重复伪 patch 到 U 个真实 patch。
- 评测与前三条基线对齐：
  * Modality Gap: centroid distance / Fréchet distance / relative modality gap（基于单位范数嵌入）
  * Zero-shot（CIFAR100, Tiny-ImageNet-200, DTD）Top1/Top5（18 模板）
  * I2T/T2I 检索（MSCOCO val2017、Flickr30k test）：R@1/5/10
  * 逐数据集与总运行时间；保存 JSON/CSV
- 额外（用于严格复现论文现象；默认关闭）：
  * “intra-modal”检索：Image↔Image（mAP），Text↔Text（mAP）
  * 开关：--do-i2i / --do-t2t 以及 --use-oti / --use-ovi 控制是否把同模态任务转为跨模态来查询

注意：
- 不改动编码器参数；OTI/OVI 仅优化输入级伪向量，完全“无数据训练”
- 支持 OpenCLIP / OpenAI-CLIP / SigLIP（通过 open_clip_torch）
"""

import os, math, json, time, random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tvds, transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy.linalg import sqrtm

import open_clip


# =========================================================
# 通用工具
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-9):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def to_device(x, device):
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x


# =========================================================
# 变换（与前面基线一致）
# =========================================================
def build_tf(image_size=224, train=True, rrc_scale=(0.5, 1.0)):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=rrc_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466,0.4578275,0.40821073],
                                 [0.26862954,0.26130258,0.27577711]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size*1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466,0.4578275,0.40821073],
                                 [0.26862954,0.26130258,0.27577711]),
        ])


# =========================================================
# 数据集封装（与之前保持兼容）
# =========================================================
class TinyImageNet200(Dataset):
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    def __init__(self, root: str, split="val", transform=None, auto_download=True):
        self.root_base = Path(root).expanduser().resolve()
        self.root = self.root_base / "tiny-imagenet-200"
        self.split = split; self.transform = transform
        assert split in ["train","val"]
        self._ensure(auto_download)
        wnids = (self.root/"wnids.txt").read_text().splitlines()
        self.wnids = [w.strip() for w in wnids if w.strip()]
        words_map={}
        for line in (self.root/"words.txt").read_text().splitlines():
            if not line.strip(): continue
            k,v = line.split("\t"); words_map[k]=v
        self.class_to_idx={w:i for i,w in enumerate(self.wnids)}
        self.idx_to_name={self.class_to_idx[w]: words_map.get(w,w).split(",")[0].split(";")[0] for w in self.wnids}
        self.samples=[]
        if split=="train":
            tdir=self.root/"train"
            for wnid in self.wnids:
                idir=tdir/wnid/"images"
                self.samples += [(str(p), self.class_to_idx[wnid]) for p in idir.glob("*.JPEG")]
        else:
            vdir=self.root/"val"
            ann=vdir/"val_annotations.txt"; mapping={}
            for line in ann.read_text().splitlines():
                if not line.strip(): continue
                fn,wnid = line.split("\t")[:2]; mapping[fn]=wnid
            idir=vdir/"images"
            for p in idir.glob("*.JPEG"):
                self.samples.append((str(p), self.class_to_idx[mapping[p.name]]))
        if not self.samples: raise RuntimeError("Tiny-ImageNet-200 empty")

    def _ensure(self, auto_download):
        if self.root.exists() and (self.root/"wnids.txt").exists(): return
        if not auto_download: raise FileNotFoundError(self.root)
        self.root_base.mkdir(parents=True, exist_ok=True)
        zip_path=self.root_base/"tiny-imagenet-200.zip"
        if not zip_path.exists():
            print("[Tiny-ImageNet] Downloading ...")
            import urllib.request, shutil
            with urllib.request.urlopen(self.URL) as r, open(zip_path,"wb") as f:
                shutil.copyfileobj(r,f)
        print("[Tiny-ImageNet] Extracting ...")
        import zipfile
        with zipfile.ZipFile(zip_path,"r") as z: z.extractall(self.root_base)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fp,y = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform: img = self.transform(img)
        return img,y

    @property
    def classes(self): return [self.idx_to_name[i] for i in range(len(self.idx_to_name))]


class CocoCaptionsEval(Dataset):
    def __init__(self, root: str, split="val", transform=None):
        droot = Path(root).expanduser().resolve()/ "coco2017"
        img_dir = droot/"images"/("val2017" if split=="val" else "train2017")
        ann = droot/"annotations"/f"captions_{'val2017' if split=='val' else 'train2017'}.json"
        if not img_dir.exists() or not ann.exists():
            raise FileNotFoundError(f"COCO not found: {img_dir} / {ann}")
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img,caps = self.ds[i]
        return img, caps


class Flickr30kEval(Dataset):
    """老接口：root=images目录, ann_file=token文件"""
    def __init__(self, root: str, split="test", transform=None):
        droot = Path(root).expanduser().resolve()/ "flickr30k"
        images_root = droot/"flickr30k-images"
        ann_file = droot/"results_20130124.token"
        if not images_root.exists(): raise FileNotFoundError(images_root)
        if not ann_file.exists(): raise FileNotFoundError(ann_file)
        self.ds = tvds.Flickr30k(root=str(images_root), ann_file=str(ann_file), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img,caps = self.ds[i]
        caps = caps if isinstance(caps,list) else [str(caps)]
        return img,caps


def collate_caps(batch):
    images = torch.stack([b[0] for b in batch],0)
    caps_list=[]
    for _,caps in batch:
        caps_list.append(list(caps) if isinstance(caps,(list,tuple)) else [str(caps)])
    return images, caps_list


# =========================================================
# OTI：图像 -> 文本（优化伪 token 嵌入）
# =========================================================
class OTIInverter:
    """
    OTI: 仅优化 token embedding 空间的 R 个伪 token；
    文本模板固定为: "a photo of" + v*（与论文一致），在 EOT 前插入。
    """
    def __init__(self, clip_model, tokenizer, lr=0.02, betas=(0.9,0.999), weight_decay=0.01,
                 steps=150, R=1, device="cuda", amp=True):
        self.model = clip_model
        self.tokenizer = tokenizer
        self.lr=lr; self.betas=betas; self.wd=weight_decay
        self.steps=steps; self.R=R; self.device=device; self.amp=amp

        # 取 text 侧结构
        self.text = clip_model.transformer  # open_clip: text transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.registered = True

        self.ctx_len = self.positional_embedding.shape[0]
        # 准备模板 token ids
        tok = tokenizer(["a photo of"])  # [1, L]
        self.templ_ids = tok.to(device)

        with torch.no_grad():
            self.templ_emb = self.token_embedding(self.templ_ids)  # (1,L,dim)
        self.dim = self.templ_emb.shape[-1]

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        f = self.model.encode_image(images)
        return l2norm(f, -1)

    def _forward_text_from_emb(self, emb_seq: torch.Tensor, eot_idx: torch.Tensor) -> torch.Tensor:
        """
        emb_seq: (B, L, dim) — 已经是 token embedding 序列（其中一段被伪 token 替换）
        eot_idx: (B,) 每条文本的 EOT 位置（用于取 CLS/last token）
        """
        x = emb_seq + self.positional_embedding.to(emb_seq.dtype)
        x = x.permute(1,0,2)  # L,B,C
        x = self.text(x)
        x = x.permute(1,0,2)  # B,L,C
        x = self.ln_final(x)
        # 取 EOT 位置的 token
        b_idx = torch.arange(x.size(0), device=x.device)
        x = x[b_idx, eot_idx] @ self.text_projection
        return l2norm(x, -1)

    def invert(self, image_feats: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """
        对一批 image features 做 OTI，返回文本空间的倒置特征（单位范数）
        image_feats: (N, D_img) —— 一般直接传 model.encode_image(images) 的结果
        """
        N = image_feats.size(0)
        out_feats = torch.empty(N, self.model.text_projection.shape[1], device="cpu")
        # 统一使用相同 token 长度布局： [templ_ids] + [R 个伪token] + [EOT] + PAD …（总长=ctx_len）
        base_ids = self.tokenizer(["a photo of"]).to(self.device)  # (1, L_templ)
        base_len = base_ids.shape[1]
        # 构造一条“空白”句，后面我们不关心 token id，只用 EOT 位置（最后一个非 PAD）
        null_ids = torch.zeros(1, self.ctx_len, dtype=torch.int32, device=self.device)
        null_ids[0, :base_len] = base_ids[0]
        # 在模板后面放 R 个“占位符”（用 "." 占位以便产生 EOT）
        dot_ids = self.tokenizer(["."]).to(self.device)[0,1]  # "." 的 token id
        for i in range(self.R):
            null_ids[0, base_len+i] = dot_ids
        # EOT 放在 base_len+R 处
        eot_idx = torch.full((1,), base_len+self.R-1, dtype=torch.long, device=self.device)

        for s in range(0, N, batch_size):
            idx = slice(s, min(N, s+batch_size))
            b = idx.stop-idx.start
            # 初始化伪 token 参数： (B, R, dim)
            v = nn.Parameter(torch.randn(b, self.R, self.dim, device=self.device)*0.02)
            opt = torch.optim.AdamW([v], lr=self.lr, betas=self.betas, weight_decay=self.wd, eps=1e-8)
            img = image_feats[idx].to(self.device)
            img = l2norm(img, -1)

            # 常量 embedding & eot
            templ_emb = self.token_embedding(base_ids.repeat(b,1))  # (B, L_templ, dim)
            eot = eot_idx.repeat(b)  # (B,)

            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    # 组装 embedding 序列
                    # [templ_emb] + [v] + [其余 PAD 用 0 embedding]，长度对齐 ctx_len
                    pad_len = self.ctx_len - templ_emb.size(1) - self.R
                    pad = torch.zeros(b, pad_len, self.dim, device=self.device, dtype=templ_emb.dtype)
                    emb_seq = torch.cat([templ_emb, v, pad], dim=1)  # (B,L,dim)
                    txt = self._forward_text_from_emb(emb_seq, eot)
                    loss = 1.0 - (txt*img).sum(dim=-1).mean()
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()

            # 最终一前向
            with torch.no_grad():
                pad_len = self.ctx_len - templ_emb.size(1) - self.R
                pad = torch.zeros(b, pad_len, self.dim, device=self.device, dtype=templ_emb.dtype)
                emb_seq = torch.cat([templ_emb, v, pad], dim=1)
                txt = self._forward_text_from_emb(emb_seq, eot)
            out_feats[idx] = txt.detach().cpu()
        return out_feats


# =========================================================
# OVI：文本 -> 图像（优化伪 patch 嵌入；ViT 专用）
# =========================================================
class OVIInverter:
    """
    OVI: 在 ViT 的 patch-embedding 空间优化 P 个伪 patch；重复/插值到 U 个真实 patch，再送入视觉 Transformer。
    """
    def __init__(self, clip_model, tokenizer, lr=0.02, betas=(0.9,0.999), weight_decay=0.01,
                 steps=1000, P: Optional[int]=None, device="cuda", amp=True):
        self.model = clip_model
        self.tokenizer = tokenizer
        self.lr=lr; self.betas=betas; self.wd=weight_decay
        self.steps=steps; self.device=device; self.amp=amp

        self.visual = clip_model.visual  # ViT
        # 取 ViT 维度 & patch 数
        self.embed_dim = getattr(self.visual, "width", None) or getattr(self.visual, "embed_dim")
        # num_patches: 不同实现命名不同
        if hasattr(self.visual, "patch_embed"):
            self.U = self.visual.patch_embed.num_patches
            self.grid = int(math.sqrt(self.U))
        elif hasattr(self.visual, "conv1"):
            # OpenAI-CLIP 风格：从 conv1 计算
            # 以 224 输入估出 grid
            gs = int((224 // self.visual.conv1.stride[0]))
            self.grid = gs; self.U = gs*gs
        else:
            raise RuntimeError("Unsupported ViT visual encoder")

        # P 未指定时，按论文建议对不同 backbone 使用 1~4 的小值（这里根据 U 自适应）
        if P is None:
            self.P = 1 if self.U <= 196 else (2 if self.U <= 256 else 4)
        else:
            self.P = P

        # 取 CLS / POS / block / norm / proj
        self.cls_token = getattr(self.visual, "cls_token", getattr(self.visual, "class_token", None))
        self.pos_embed = getattr(self.visual, "pos_embed", getattr(self.visual, "positional_embedding", None))
        self.blocks = getattr(self.visual, "blocks", getattr(self.visual, "transformer", None))
        self.ln_pre = getattr(self.visual, "ln_pre", None)
        self.ln_post = getattr(self.visual, "ln_post", getattr(self.visual, "norm", None))
        self.proj = getattr(self.visual, "proj", None)

    @torch.no_grad()
    def encode_text(self, texts: List[str], batch=256) -> torch.Tensor:
        outs=[]
        for i in range(0, len(texts), batch):
            tok = self.tokenizer(texts[i:i+batch]).to(self.device)
            ft = self.model.encode_text(tok)
            outs.append(l2norm(ft, -1).cpu())
        return torch.cat(outs,0)

    def _repeat_to_U(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (B, P, D)  -> repeat/interpolate 到 (B, U, D)
        这里做最简单的“最近邻重复”：把 P 个块均匀repeat到 U 个
        """
        B,P,D = w.shape
        if P==self.U: return w
        # 分配每个伪patch的重复次数
        rep = [self.U//P]*(P)
        for i in range(self.U - sum(rep)): rep[i]+=1
        chunks=[]
        for i in range(P):
            wi = w[:,i:i+1,:].expand(B, rep[i], D)
            chunks.append(wi)
        out = torch.cat(chunks, dim=1)  # (B,U,D)
        return out

    def _forward_from_patches(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: (B, U, D) —— 视觉 Transformer 的 patch token（不含 CLS）
        返回：单位范数后的图像嵌入（与 encode_image 对齐）
        """
        B,U,D = patch_tokens.shape
        # 组 CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patch_tokens], dim=1)  # (B, 1+U, D)
        # 位置编码（若需要插值）
        if self.pos_embed is not None:
            pe = self.pos_embed
            if pe.shape[1] != x.shape[1]:
                # 插值到 grid×grid
                pe_cls, pe_tok = pe[:, :1], pe[:, 1:]
                S = int(math.sqrt(pe_tok.shape[1]))
                pe_tok = F.interpolate(pe_tok.permute(0,2,1).reshape(1,D,S,S),
                                       size=(self.grid,self.grid), mode="bicubic", align_corners=False)
                pe_tok = pe_tok.reshape(1,D,self.grid*self.grid).permute(0,2,1)
                pe = torch.cat([pe_cls, pe_tok], dim=1)
            x = x + pe

        if self.ln_pre is not None: x = self.ln_pre(x)
        # blocks 既可能是 nn.Sequential(blocks)，也可能是 transformer(x)
        if isinstance(self.blocks, nn.Sequential):
            for blk in self.blocks: x = blk(x)
        else:
            x = self.blocks(x)
        if self.ln_post is not None: x = self.ln_post(x)
        x = x[:,0,:]  # 取 CLS
        # proj 既可能是 nn.Linear，也可能是 Parameter
        if self.proj is not None:
            if isinstance(self.proj, nn.Linear): x = self.proj(x)
            else: x = x @ self.proj
        return l2norm(x, -1)

    def invert(self, text_feats: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """
        对一批 text features 做 OVI，返回视觉空间的倒置特征（单位范数）
        text_feats: (N, D_txt) —— 一般传 model.encode_text(tokenizer(texts))
        """
        N = text_feats.size(0)
        out = torch.empty(N, text_feats.size(1), device="cpu")
        for s in range(0, N, batch_size):
            idx = slice(s, min(N, s+batch_size))
            b = idx.stop - idx.start
            tgt = l2norm(text_feats[idx].to(self.device), -1)

            # 初始化 P 个伪 patch：(B,P,D)
            w = nn.Parameter(torch.randn(b, self.P, self.embed_dim, device=self.device)*0.02)
            opt = torch.optim.AdamW([w], lr=self.lr, betas=(0.9,0.999), weight_decay=self.wd, eps=1e-8)
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    tokensU = self._repeat_to_U(w)     # (B,U,D)
                    img = self._forward_from_patches(tokensU)
                    loss = 1.0 - (img*tgt).sum(-1).mean()
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()

            with torch.no_grad():
                tokensU = self._repeat_to_U(w)
                img = self._forward_from_patches(tokensU)
            out[idx] = img.detach().cpu()
        return out


# =========================================================
# 评测度量（与前三基线一致）
# =========================================================
CIFAR100_TEMPLATES = [
 "a photo of a {}.","a blurry photo of a {}.","a black and white photo of a {}.",
 "a low contrast photo of a {}.","a high contrast photo of a {}.","a bad photo of a {}.",
 "a good photo of a {}.","a photo of a small {}.","a photo of a big {}.","a photo of the {}.",
 "a blurry photo of the {}.","a black and white photo of the {}.","a low contrast photo of the {}.",
 "a high contrast photo of the {}.","a bad photo of the {}.","a good photo of the {}.",
 "a photo of the small {}.","a photo of the big {}."
]

def frechet_distance(mu1, s1, mu2, s2):
    md = mu1-mu2
    covm = sqrtm(s1.dot(s2))
    if np.iscomplexobj(covm): covm = covm.real
    fd_sq = float(md.dot(md) + np.trace(s1+s2-2*covm))
    return math.sqrt(max(fd_sq, 0.0))

def modality_gap(imgE: torch.Tensor, txtE: torch.Tensor) -> Dict[str,float]:
    x = imgE.cpu().numpy(); y = txtE.cpu().numpy()
    mu_x = x.mean(0); mu_y = y.mean(0)
    cx = np.cov(x.T); cy = np.cov(y.T)
    centroid = float(np.linalg.norm(mu_x-mu_y))
    fd = frechet_distance(mu_x, cx, mu_y, cy)
    denom = math.sqrt(0.5*(np.trace(cx)+np.trace(cy))+1e-12)
    rmg = centroid/denom if denom>0 else float('nan')
    return {"centroid_distance":centroid,"frechet_distance":fd,"relative_modality_gap":rmg}

@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device, batch=256):
    outs=[]
    for i in range(0,len(texts),batch):
        toks = tokenizer(texts[i:i+batch]).to(device)
        ft = l2norm(model.encode_text(toks), -1)
        outs.append(ft)
    return torch.cat(outs,0)

@torch.no_grad()
def zeroshot_eval(model, tokenizer, loader, classnames, device, templates=None):
    T = templates or CIFAR100_TEMPLATES
    # 文本类中心
    cls_bank=[]
    for cname in tqdm(classnames, desc="ZS: encode classes", leave=False):
        prompts=[t.format(cname) for t in T]
        ft = encode_texts(model, tokenizer, prompts, device)
        cls_bank.append(ft)  # (P, D)
    # 评测
    correct1=correct5=n=0
    img_all=[]; txt_all=[]
    for images,labels in tqdm(loader, desc="Zero-shot", leave=False):
        images=to_device(images, device); labels=to_device(labels, device)
        fi = l2norm(model.encode_image(images), -1); img_all.append(fi.cpu())
        B=fi.size(0); C=len(classnames)
        scores = torch.zeros(B,C, device="cpu")
        for ci in range(C):
            txtc = cls_bank[ci].to(device)  # (P,D)
            sc = (fi @ txtc.t()).mean(1).cpu()
            scores[:,ci]=sc
            txt_all.append(txtc.cpu())
        top5 = scores.topk(5, dim=-1).indices
        correct1 += (top5[:,0].to(labels.device)==labels).sum().item()
        correct5 += sum([(top5[:,i].to(labels.device)==labels).sum().item() for i in range(5)])
        n += labels.size(0)
    img_cat=torch.cat(img_all,0)
    txt_cat=torch.cat(txt_all,0) if len(txt_all)>0 else torch.empty(0)
    mg = modality_gap(img_cat, txt_cat) if txt_cat.numel()>0 else {"centroid_distance":float("nan"),"frechet_distance":float("nan"),"relative_modality_gap":float("nan")}
    return correct1/n, correct5/n, mg, img_cat, txt_cat

def recalls_from_sim(sim: np.ndarray, gt: Dict[int,List[int]], ks=(1,5,10)):
    order = np.argsort(-sim, axis=1); res={}
    for k in ks:
        ok=0
        for i in range(order.shape[0]):
            if any(t in set(order[i,:k]) for t in gt[i]): ok+=1
        res[f"R@{k}"]=ok/order.shape[0]
    return res

@torch.no_grad()
def retrieval_eval(model, tokenizer, ds, device, batch_size=64, max_items=None):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=collate_caps)
    imgE=[]; all_caps=[]; img2caps={}; cap2imgs={}
    img_idx=0; cap_idx=0
    for images, caps_list in tqdm(loader, desc="Embed images & gather captions", leave=False):
        images=to_device(images, device)
        fi = l2norm(model.encode_image(images), -1).cpu()
        imgE.append(fi)
        b = images.size(0)
        for j in range(b):
            caps=caps_list[j]; idxs=[]
            for c in caps:
                all_caps.append(c); idxs.append(cap_idx); cap2imgs[cap_idx]=[img_idx]; cap_idx+=1
            img2caps[img_idx]=idxs; img_idx+=1
        if max_items is not None and img_idx>=max_items: break
    imgE = torch.cat(imgE,0)  # (Nimg, D)
    # 文本
    ft=[]
    for i in range(0,len(all_caps),256):
        toks = tokenizer(all_caps[i:i+256]).to(device)
        ft.append(l2norm(model.encode_text(toks), -1).cpu())
    txtE = torch.cat(ft,0) if ft else torch.empty(0, imgE.size(1))
    mg = modality_gap(imgE, txtE)
    sim = (imgE @ txtE.t()).numpy()
    i2t = recalls_from_sim(sim, img2caps, ks=(1,5,10))
    t2i = recalls_from_sim(sim.T, cap2imgs, ks=(1,5,10))
    return mg, i2t, t2i


# -----------------（论文核心）同模态检索 mAP，用于复现现象；默认不执行 -----------------
def mean_average_precision(sim: np.ndarray, gt_rel: List[List[int]]):
    """
    sim: (Nq, Ng) — 每个 query 对所有 gallery 的相似度（大为好）
    gt_rel: 长度 Nq 的列表，每个元素是该 query 的“相关 gallery 下标”列表
    """
    order = np.argsort(-sim, axis=1)
    mAP=0.0
    for i in range(order.shape[0]):
        rel = set(gt_rel[i]); hits=0; ap=0.0
        for rank, g in enumerate(order[i], start=1):
            if g in rel:
                hits+=1; ap += hits/rank
        if len(rel)>0: ap/=len(rel)
        mAP += ap
    return mAP/order.shape[0]


@torch.no_grad()
def image_to_image_map(model, ds, device, use_oti=False, oti: Optional[OTIInverter]=None, batch=128, max_items=None):
    """
    复现论文：图-图检索（同模态 vs 经 OTI 转为跨模态查询）
    - baseline: 直接 image-image 余弦
    - use_oti=True: 对 query 做 OTI，改为“文本-图像”跨模态相似度
    """
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    feats=[]; labels=[]
    for im,y in tqdm(loader, desc="i2i: encode images", leave=False):
        im=to_device(im, device)
        f=l2norm(model.encode_image(im), -1).cpu()
        feats.append(f); labels += y.tolist()
        if max_items and len(labels)>=max_items: break
    imgE=torch.cat(feats,0); labels=np.array(labels)
    # gallery 与 query 划分：按标签生成 gt
    N=imgE.shape[0]
    gallery=imgE; query=imgE
    if not use_oti:
        sim=(query@gallery.t()).numpy()
    else:
        assert oti is not None, "use_oti=True 需要 OTIInverter"
        q_txt = oti.invert(query, batch_size=batch)  # (N,D)
        # 与图像跨模态相似度
        sim = (q_txt @ gallery.t()).numpy()
    # gt：同类为相关
    gt_rel=[np.where(labels==labels[i])[0].tolist() for i in range(N)]
    # 去掉自身
    for i in range(N):
        if i in gt_rel[i]: gt_rel[i].remove(i)
    return mean_average_precision(sim, gt_rel)


@torch.no_grad()
def text_to_text_map(model, tokenizer, ds, device, use_ovi=False, ovi: Optional[OVIInverter]=None, batch=128, max_items=None):
    """
    复现论文：文-文检索（同模态 vs 经 OVI 转为跨模态查询）
    - baseline: 直接 text-text 余弦
    - use_ovi=True: 对 query 做 OVI，改为“图像-文本”跨模态相似度
    """
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_caps)
    all_caps=[]
    for _,caps_list in tqdm(loader, desc="t2t: gather caps", leave=False):
        for caps in caps_list:
            all_caps += caps
        if max_items and len(all_caps)>=max_items*5: break  # 每图5条
    # 编码全部 caption
    txtE = encode_texts(model, tokenizer, all_caps, device)
    if not use_ovi:
        sim = (txtE @ txtE.t()).numpy()
    else:
        assert ovi is not None, "use_ovi=True 需要 OVIInverter"
        inv_img = ovi.invert(txtE.to(device), batch_size=batch)  # (N,D)
        sim = (inv_img @ inv_img.t()).numpy()
    # 构造 gt：每 5 条为同图的相关（Flickr30k/COCO 典型设置）
    N = txtE.size(0); group=5
    gt_rel=[]
    for i in range(N):
        gid = i//group
        cand = list(range(gid*group, (gid+1)*group))
        cand.remove(i)
        gt_rel.append(cand)
    return mean_average_precision(sim, gt_rel)


# =========================================================
# 主函数
# =========================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--save-dir", type=str, default="runs/ctg_baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-coco", type=int, default=None)
    ap.add_argument("--max-flickr", type=int, default=None)

    # 论文现象复现（同模态检索）
    ap.add_argument("--do-i2i", action="store_true", help="评测 image->image 检索 mAP")
    ap.add_argument("--do-t2t", action="store_true", help="评测 text->text 检索 mAP")
    ap.add_argument("--use-oti", action="store_true", help="i2i 使用 OTI 把同模态任务转为跨模态")
    ap.add_argument("--use-ovi", action="store_true", help="t2t 使用 OVI 把同模态任务转为跨模态")

    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型 & tokenizer
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)

    tf_train = build_tf(args.image_size, True)
    tf_eval  = build_tf(args.image_size, False)

    results={"config":{
        "model":args.model,"pretrained":args.pretrained,"image_size":args.image_size,
        "batch_size":args.batch_size,"device":device
    },"zero_shot":{},"retrieval":{},"runtime_sec":{},"intra_modal":{}}

    t_all=time.time()

    # ---------------- Zero-shot: CIFAR100 ----------------
    t0=time.time()
    print("\n[Eval] CIFAR-100 (ZS)")
    cifar_val = tvds.CIFAR100(root=os.path.join(args.data_root,"cifar100"), train=False, transform=tf_eval, download=True)
    cifar_train = tvds.CIFAR100(root=os.path.join(args.data_root,"cifar100"), train=True, transform=tf_eval, download=True)
    cifar_classes = cifar_train.classes
    loader = DataLoader(cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval(model, tokenizer, loader, cifar_classes, device)
    results["zero_shot"]["cifar100"]={"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["cifar100"]=time.time()-t0
    print(f"CIFAR100: top1={top1:.4f} top5={top5:.4f} time={results['runtime_sec']['cifar100']:.1f}s")

    # ---------------- Zero-shot: Tiny-ImageNet-200 ----------------
    t0=time.time()
    print("\n[Eval] Tiny-ImageNet-200 (ZS)")
    tiny = TinyImageNet200(args.data_root, split="val", transform=tf_eval, auto_download=True)
    loader = DataLoader(tiny, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval(model, tokenizer, loader, tiny.classes, device)
    results["zero_shot"]["tiny_imagenet_200"]={"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["tiny_imagenet_200"]=time.time()-t0
    print(f"Tiny-ImageNet-200: top1={top1:.4f} top5={top5:.4f} time={results['runtime_sec']['tiny_imagenet_200']:.1f}s")

    # ---------------- Zero-shot: DTD ----------------
    t0=time.time()
    print("\n[Eval] DTD (ZS)")
    dtd_val = tvds.DTD(root=os.path.join(args.data_root,"dtd"), split="test", transform=tf_eval, download=True)
    dtd_train = tvds.DTD(root=os.path.join(args.data_root,"dtd"), split="train", transform=tf_eval, download=True)
    dtd_classes = dtd_train.classes
    loader = DataLoader(dtd_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval(model, tokenizer, loader, dtd_classes, device)
    results["zero_shot"]["dtd"]={"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["dtd"]=time.time()-t0
    print(f"DTD: top1={top1:.4f} top5={top5:.4f} time={results['runtime_sec']['dtd']:.1f}s")

    # ---------------- Retrieval: MSCOCO ----------------
    t0=time.time()
    print("\n[Eval] MSCOCO (I2T/T2I R@K)")
    coco = CocoCaptionsEval(args.data_root, split="val", transform=tf_eval)
    if args.max_coco is not None:
        class Sub(Dataset):
            def __init__(self, base,n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self,i): return self.base[i]
        coco = Sub(coco, args.max_coco)
    mg, i2t, t2i = retrieval_eval(model, tokenizer, coco, device, batch_size=args.batch_size, max_items=args.max_coco)
    results["retrieval"]["mscoco"]={"I2T":i2t,"T2I":t2i,"modality_gap":mg}
    results["runtime_sec"]["mscoco"]=time.time()-t0
    print(f"MSCOCO: I2T={i2t} T2I={t2i} time={results['runtime_sec']['mscoco']:.1f}s")

    # ---------------- Retrieval: Flickr30k ----------------
    t0=time.time()
    print("\n[Eval] Flickr30k (I2T/T2I R@K)")
    flickr = Flickr30kEval(args.data_root, split="test", transform=tf_eval)
    if args.max_flickr is not None:
        class SubF(Dataset):
            def __init__(self, base,n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self,i): return self.base[i]
        flickr = SubF(flickr, args.max_flickr)
    mg, i2t, t2i = retrieval_eval(model, tokenizer, flickr, device, batch_size=args.batch_size, max_items=args.max_flickr)
    results["retrieval"]["flickr30k"]={"I2T":i2t,"T2I":t2i,"modality_gap":mg}
    results["runtime_sec"]["flickr30k"]=time.time()-t0
    print(f"Flickr30k: I2T={i2t} T2I={t2i} time={results['runtime_sec']['flickr30k']:.1f}s")

    # ---------------- 论文核心：同模态检索（可选） ----------------
    if args.do_i2i or args.do_t2t:
        oti = OTIInverter(model, tokenizer, device=device) if args.use_oti else None
        ovi = OVIInverter(model, tokenizer, device=device) if args.use_ovi else None

    # i2i（用 Tiny-ImageNet-200，可切换 OTI）
    if args.do_i2i:
        t0=time.time()
        print("\n[Intra-Modal] Image-to-Image retrieval (mAP)")
        tiny_i2i = TinyImageNet200(args.data_root, split="val", transform=tf_eval, auto_download=True)
        mAP_i2i = image_to_image_map(model, tiny_i2i, device, use_oti=args.use_oti, oti=oti, batch=min(128, args.batch_size))
        results["intra_modal"]["i2i_mAP"] = {"dataset":"Tiny-ImageNet-200", "use_OTI":bool(args.use_oti), "mAP": float(mAP_i2i)}
        results["runtime_sec"]["i2i_map"]=time.time()-t0
        print(f"i2i mAP (Tiny-ImageNet-200) = {mAP_i2i:.4f}  use_OTI={bool(args.use_oti)}")

    # t2t（用 Flickr30k，可切换 OVI）
    if args.do_t2t:
        t0=time.time()
        print("\n[Intra-Modal] Text-to-Text retrieval (mAP)")
        flickr_t2t = Flickr30kEval(args.data_root, split="test", transform=tf_eval)
        mAP_t2t = text_to_text_map(model, tokenizer, flickr_t2t, device, use_ovi=args.use_ovi, ovi=ovi, batch=min(128,args.batch_size), max_items=args.max_flickr)
        results["intra_modal"]["t2t_mAP"] = {"dataset":"Flickr30k", "use_OVI":bool(args.use_ovi), "mAP": float(mAP_t2t)}
        results["runtime_sec"]["t2t_map"]=time.time()-t0
        print(f"t2t mAP (Flickr30k) = {mAP_t2t:.4f}  use_OVI={bool(args.use_ovi)}")

    # ---------------- 保存 ----------------
    total=time.time()-t_all
    results["runtime_sec"]["total"]=total
    ts=time.strftime("%Y%m%d_%H%M%S")
    json_path=os.path.join(args.save_dir, f"ctg_baseline_{ts}.json")
    with open(json_path,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)

    # CSV 摘要
    rows=[]
    def add_row(name, metrics, rt=None):
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

    add_row("cifar100", results["zero_shot"]["cifar100"], results["runtime_sec"]["cifar100"])
    add_row("tiny_imagenet_200", results["zero_shot"]["tiny_imagenet_200"], results["runtime_sec"]["tiny_imagenet_200"])
    add_row("dtd", results["zero_shot"]["dtd"], results["runtime_sec"]["dtd"])
    add_row("mscoco.I2T", {"I2T":results["retrieval"]["mscoco"]["I2T"],"mg":results["retrieval"]["mscoco"]["modality_gap"]}, results["runtime_sec"]["mscoco"])
    add_row("mscoco.T2I", {"T2I":results["retrieval"]["mscoco"]["T2I"],"mg":results["retrieval"]["mscoco"]["modality_gap"]}, results["runtime_sec"]["mscoco"])
    add_row("flickr30k.I2T", {"I2T":results["retrieval"]["flickr30k"]["I2T"],"mg":results["retrieval"]["flickr30k"]["modality_gap"]}, results["runtime_sec"]["flickr30k"])
    add_row("flickr30k.T2I", {"T2I":results["retrieval"]["flickr30k"]["T2I"],"mg":results["retrieval"]["flickr30k"]["modality_gap"]}, results["runtime_sec"]["flickr30k"])
    if "i2i_mAP" in results["intra_modal"]:
        rows.append({"dataset":"i2i_map.Tiny-ImageNet-200","use_OTI":results["intra_modal"]["i2i_mAP"]["use_OTI"],
                     "mAP":results["intra_modal"]["i2i_mAP"]["mAP"],"runtime_sec":results["runtime_sec"].get("i2i_map",None)})
    if "t2t_mAP" in results["intra_modal"]:
        rows.append({"dataset":"t2t_map.Flickr30k","use_OVI":results["intra_modal"]["t2t_mAP"]["use_OVI"],
                     "mAP":results["intra_modal"]["t2t_mAP"]["mAP"],"runtime_sec":results["runtime_sec"].get("t2t_map",None)})
    rows.append({"dataset":"TOTAL","runtime_sec":results["runtime_sec"]["total"]})
    csv_path=os.path.join(args.save_dir, f"ctg_baseline_{ts}.csv")
    pd.DataFrame(rows).to_csv(csv_path,index=False)

    print(f"\nSaved JSON -> {json_path}")
    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
