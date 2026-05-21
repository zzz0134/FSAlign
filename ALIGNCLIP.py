#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AlignCLIP Baseline (Baseline #7) — Full, No Omission

from __future__ import annotations
import os, math, json, time, random, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate

from torchvision import datasets as tvds, transforms
from torchvision.datasets.folder import default_loader as pil_loader

from tqdm import tqdm

# Text encoders (SBERT for semantic; tokenizer/model for text branch if needed)
from transformers import AutoTokenizer, AutoModel

# ========== Paper References ==========
# AlignCLIP: shared transformer + projection, IMSep with SBERT, CC12M pretrain
# Training hyperparams: H100, ViT-B/16, 30 epochs, bs 512, AdamW 1e-3, cosine, warmup 10k, wd 0.1, tau 0.07
# Retrieval finetune: COCO 8 ep, Flickr30k 20 ep, bs 128, lr 5e-6, wd 0.2
# Source: Eslami & de Melo, "Mitigate the Gap..." (arXiv:2406.17639v3)


# ===================== Utils =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ts_now():
    return time.strftime("%Y%m%d_%H%M%S")

def l2norm(x: torch.Tensor, dim=-1, eps=1e-9):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ===================== GPU Fréchet / Modality Gap =====================
def _cov_torch(Z: torch.Tensor) -> torch.Tensor:
    Zc = Z - Z.mean(dim=0, keepdim=True)
    n = Z.shape[0]
    return (Zc.T @ Zc) / (n - 1 + 1e-9)

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

def modality_gap_metrics(imgE: torch.Tensor, txtE: torch.Tensor) -> Dict[str, float]:
    x = imgE; y = txtE
    mu_x = x.mean(dim=0); mu_y = y.mean(dim=0)
    Cx = _cov_torch(x); Cy = _cov_torch(y)
    centroid = torch.norm(mu_x - mu_y).item()
    fd = _fid_torch(mu_x, Cx, mu_y, Cy).item()
    denom = float(torch.sqrt(0.5 * (torch.trace(Cx) + torch.trace(Cy)) + 1e-12))
    rmg = (centroid / denom) if denom > 0 else float("nan")
    return {"centroid_distance": centroid, "frechet_distance": fd, "relative_modality_gap": rmg}

# ===================== Data: CC12M (WebDataset or CSV/JSONL) =====================
class CC12MWebDataset(Dataset):
    """
    Very lightweight reader:
      - If you have WebDataset shards (.tar): put (image_path, caption) pairs extracted beforehand into a JSONL/CSV index.
      - Or directly provide a CSV/JSONL with columns: image_path, caption.
    This class just reads local files. (Real WebDataset streaming is omitted to keep offline reproducibility.)
    """
    def __init__(self, index_file: str, transform=None, max_items: Optional[int]=None):
        self.index_file = Path(index_file)
        assert self.index_file.exists(), f"Index file not found: {index_file}"
        if self.index_file.suffix.lower() in [".jsonl", ".json"]:
            import json
            rows = []
            with open(self.index_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    rows.append((obj["image_path"], obj["caption"]))
        else:
            df = pd.read_csv(self.index_file)
            rows = list(zip(df["image_path"].tolist(), df["caption"].tolist()))
        if max_items is not None:
            rows = rows[:max_items]
        self.items = rows
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        ip, cap = self.items[i]
        img = Image.open(ip).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, cap

# Use COCO train as a fallback mock "pretrain" (for quick smoke test)
class CocoTrainPairs(Dataset):
    def __init__(self, data_root: str, transform=None, split="train2017"):
        root = Path(data_root) / "coco2017"
        img_dir = root / "images" / split
        ann = root / "annotations" / f"captions_{split}.json"
        assert img_dir.exists() and ann.exists(), "COCO not found"
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[i]  # (PIL->TF), [caps]

# ===================== Data: Retrieval Eval =====================
class CocoCaptionsEval(Dataset):
    def __init__(self, root: str, split="val", transform=None):
        droot = Path(root) / "coco2017"
        img_dir = droot / "images" / ("val2017" if split == "val" else "train2017")
        ann = droot / "annotations" / f"captions_{'val2017' if split=='val' else 'train2017'}.json"
        assert img_dir.exists() and ann.exists(), "COCO not found"
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, caps = self.ds[i]
        return img, caps

class Flickr30kEval(Dataset):
    # old-style API: torchvision.datasets.Flickr30k(root=images_root, ann_file=token_file)
    def __init__(self, root: str, transform=None):
        droot = Path(root) / "flickr30k"
        images_root = droot / "flickr30k-images"
        ann_file = droot / "results_20130124.token"
        assert images_root.exists(), images_root
        assert ann_file.exists(), ann_file
        self.ds = tvds.Flickr30k(root=str(images_root), ann_file=str(ann_file), transform=transform)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, caps = self.ds[i]
        caps = caps if isinstance(caps, list) else [str(caps)]
        return img, caps

def collate_images_captions(b):
    imgs = [x[0] for x in b]
    caps = [random.choice(x[1]) for x in b]
    imgs = default_collate(imgs)
    return imgs, caps

# ===================== Transforms =====================
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

# ===================== Shared Transformer (ViT-B/16 like) =====================
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_dropout=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x) + h
        return x

class SharedTransformer(nn.Module):
    """
    A single Transformer encoder shared by image & text
    """
    def __init__(self, dim=768, depth=12, heads=12, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_ratio, attn_drop, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        return x

# ===================== AlignCLIP Model =====================
class AlignCLIP(nn.Module):
    """
    - Shared transformer & shared projection for both modalities (as in paper)
    - Vision: Conv patch embed (16x16), learnable CLS & 2D pos embedding, CLS pooled
    - Text: token embedding + 1D pos embedding, max pooling
    """
    def __init__(self, img_size=224, patch=16, vocab_size=49408, txt_maxlen=77,
                 width=768, layers=12, heads=12, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch == 0
        self.width = width
        self.txt_maxlen = txt_maxlen

        # Vision patch embed
        self.conv = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        num_patches = (img_size // patch) * (img_size // patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_img = nn.Parameter(torch.zeros(1, 1 + num_patches, width))

        # Text embedding
        self.token_emb = nn.Embedding(vocab_size, width)
        self.pos_txt = nn.Parameter(torch.zeros(1, txt_maxlen, width))

        # Shared encoder & projection
        self.encoder = SharedTransformer(dim=width, depth=layers, heads=heads, mlp_ratio=mlp_ratio)
        self.proj = nn.Linear(width, width, bias=False)  # shared projection
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))  # init tau=0.07

        # Init
        nn.init.trunc_normal_(self.pos_img, std=0.02)
        nn.init.trunc_normal_(self.pos_txt, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    # ----- Vision & Text encode -----
    def encode_image(self, images: torch.Tensor):
        x = self.conv(images)                      # (B, C=width, H', W')
        x = x.flatten(2).transpose(1, 2)           # (B, L, width)
        B, L, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_img[:, :L+1, :]
        x = self.encoder(x)
        x = x[:, 0, :]                             # CLS
        x = self.proj(x)
        x = l2norm(x)
        return x

    def encode_text(self, token_ids: torch.Tensor, attn_mask: torch.Tensor):
        # token_ids: (B, T), attn_mask: (B,T) (1 for valid)
        x = self.token_emb(token_ids) + self.pos_txt[:, :token_ids.size(1), :]
        x = self.encoder(x)
        # max pooling over tokens where mask=1
        mask = attn_mask.unsqueeze(-1).float()     # (B,T,1)
        x = x * mask + (1 - mask) * (-1e4)        # masked to very low
        x = x.max(dim=1).values
        x = self.proj(x)
        x = l2norm(x)
        return x

    # ----- Loss pieces -----
    @staticmethod
    def symmetric_ce(sim_logits: torch.Tensor, labels: torch.Tensor):
        # sim_logits: y_hat_v = exp(tau) Ev Et^T
        # produce both sides and CE
        yv = sim_logits
        yt = sim_logits.t()
        ce = (F.cross_entropy(yv, labels) + F.cross_entropy(yt, labels)) * 0.5
        return ce

    def forward_clip_loss(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (img_emb @ txt_emb.t())
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        loss = self.symmetric_ce(logits, labels)
        return loss, logits

    def forward_imsep_loss(self, img_emb: torch.Tensor, txt_emb: torch.Tensor,
                           sbert_txt_emb: torch.Tensor):
        """
        Eq.(6)-(13) with re-scaling:
        V = Ev Ev^T, S = Es Es^T / ||Es||^2, D=1-S, VD=V ⊙ D
        M = Ev Et^T, diag(M) keep, y_hat_vsep = exp(tau)*(diag(M)+VD), LIMSep = CE(y_hat_vsep, Y)
        """
        # Normalize
        Ev = l2norm(img_emb); Et = l2norm(txt_emb); Es = l2norm(sbert_txt_emb)
        # Pairwise
        V = Ev @ Ev.t()                 # (B,B)
        S = Es @ Es.t()                 # (B,B) cosine already
        D = 1.0 - S
        VD = V * D
        M = Ev @ Et.t()                 # (B,B)
        diagM = torch.diag(M)           # (B,)
        yvsep = VD.clone()
        yvsep[range(yvsep.size(0)), range(yvsep.size(0))] = yvsep.diag() + diagM
        logits = self.logit_scale.exp() * yvsep
        labels = torch.arange(Ev.size(0), device=Ev.device)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

# ===================== Tokenizers & SBERT =====================
class TextPipeline:
    def __init__(self, model_name="roberta-base", max_len=77, device="cuda"):
        # NOTE: CLIP uses BPE-Byte pair with max 77; here we use roberta tokenizer, close enough for baseline reproduction
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len
        self.device = device
    def encode(self, texts: List[str]):
        tok = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        return tok["input_ids"], tok["attention_mask"]

class SBERTSemantic:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
    @torch.no_grad()
    def encode(self, texts: List[str]):
        tok = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.model(**tok)
        last = out.last_hidden_state         # (B,T,H)
        mask = tok["attention_mask"].unsqueeze(-1)  # (B,T,1)
        # mean pooling
        emb = (last * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        emb = l2norm(emb)
        return emb

# ===================== Training & Eval =====================
def train_alignclip(
    model: AlignCLIP,
    text_pipe: TextPipeline,
    sbert: SBERTSemantic,
    loader: DataLoader,
    device="cuda",
    epochs=30, lr=1e-3, wd=0.1, warmup=10_000, total_steps=None,
    alpha=1.0, beta=0.5, cosine=True, save_dir="runs/alignclip_ckpt"
):
    os.makedirs(save_dir, exist_ok=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if total_steps is None:
        total_steps = epochs * len(loader)

    if cosine:
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    step = 0
    for ep in range(1, epochs+1):
        pbar = tqdm(loader, desc=f"Train {ep}/{epochs}")
        for images, caps in pbar:
            images = images.to(device, non_blocking=True)
            # text ids
            tid, am = text_pipe.encode(caps)
            # forward
            imgE = model.encode_image(images)
            txtE = model.encode_text(tid, am)
            # SBERT
            sE = sbert.encode(caps)
            # losses
            Lclip, _ = model.forward_clip_loss(imgE, txtE)
            Lsep, _ = model.forward_imsep_loss(imgE, txtE, sE)
            loss = alpha * Lclip + beta * Lsep

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        # save each epoch
        torch.save({"epoch": ep, "state_dict": model.state_dict()}, os.path.join(save_dir, f"ep_{ep:03d}.pt"))

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
def retrieval_eval(model: AlignCLIP, text_pipe: TextPipeline, loader: DataLoader, device="cuda"):
    model.eval()
    all_img, all_txt = [], []
    img2caps, cap2imgs = {}, {}
    img_idx = cap_idx = 0
    for images, caps in tqdm(loader, desc="Embed for retrieval", leave=False):
        images = images.to(device, non_blocking=True)
        tid, am = text_pipe.encode(caps)
        imgE = model.encode_image(images)
        txtE = model.encode_text(tid, am)
        for b in range(imgE.size(0)):
            all_img.append(imgE[b]); all_txt.append(txtE[b])
            img2caps[img_idx] = [cap_idx]; cap2imgs[cap_idx] = [img_idx]
            img_idx += 1; cap_idx += 1
    imgE = F.normalize(torch.stack(all_img), dim=-1)
    txtE = F.normalize(torch.stack(all_txt), dim=-1)

    sim = imgE @ txtE.t()
    i2t = recalls_from_sim(sim, img2caps)
    t2i = recalls_from_sim(sim.t(), cap2imgs)
    mg = modality_gap_metrics(imgE, txtE)
    return mg, i2t, t2i

# Zero-shot classification (CIFAR100, Tiny-ImageNet-200, DTD)
ZS_TEMPLATES = [
    "a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
    "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
    "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.", "a photo of the {}.",
    "a blurry photo of the {}.", "a black and white photo of the {}.", "a low contrast photo of the {}.",
    "a high contrast photo of the {}.", "a bad photo of the {}.", "a good photo of the {}.",
    "a photo of the small {}.", "a photo of the big {}."
]

@torch.no_grad()
def zeroshot_eval(model: AlignCLIP, text_pipe: TextPipeline, loader, classnames, device="cuda", templates=None):
    model.eval()
    T = templates or ZS_TEMPLATES
    # build class prototypes (mean over templates)
    proto = []
    for name in tqdm(classnames, desc="ZS: encode classes", leave=False):
        prompts = [t.format(name) for t in T]
        tid, am = text_pipe.encode(prompts)
        tE = model.encode_text(tid, am)          # (P, d)
        proto.append(F.normalize(tE, dim=-1))    # (P,d)
    C = len(classnames)
    correct1 = correct5 = n = 0

    img_bank = []
    for images, labels in tqdm(loader, desc="ZS infer", leave=False):
        images = images.to(device, non_blocking=True)
        fi = F.normalize(model.encode_image(images), dim=-1)  # (B, d)
        img_bank.append(fi)
        scores = torch.zeros(fi.size(0), C, device=device)
        for ci in range(C):
            p = proto[ci].to(device)
            scores[:, ci] = (fi @ p.t()).mean(dim=1)
        top5 = scores.topk(5, dim=-1).indices
        labels = labels.to(device)
        correct1 += (top5[:, 0] == labels).sum().item()
        for k in range(5):
            correct5 += (top5[:, k] == labels).sum().item()
        n += labels.size(0)

    img_cat = torch.cat(img_bank, 0)
    txt_cat = torch.cat([p for p in proto], 0).to(device)
    mg = modality_gap_metrics(img_cat, txt_cat)
    return correct1 / n, correct5 / n, mg

# ===================== Tiny-ImageNet-200 =====================
class TinyImageNet200(Dataset):
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    def __init__(self, root: str, split="val", transform=None, auto_download=True):
        self.root_base = Path(root); self.root = self.root_base / "tiny-imagenet-200"
        self.split = split; self.transform = transform
        assert split in ["train", "val"]
        self._ensure(auto_download)
        wnids = (self.root / "wnids.txt").read_text().splitlines()
        self.wnids = [w.strip() for w in wnids if w.strip()]
        words_map = {}
        for line in (self.root / "words.txt").read_text().splitlines():
            if not line.strip(): continue
            k, v = line.split("\t"); words_map[k] = v
        self.class_to_idx = {w: i for i, w in enumerate(self.wnids)}
        self.idx_to_name = {self.class_to_idx[w]: words_map.get(w, w).split(",")[0].split(";")[0] for w in self.wnids}
        self.samples = []
        if split == "train":
            tdir = self.root / "train"
            for wnid in self.wnids:
                idir = tdir / wnid / "images"
                self.samples += [(str(p), self.class_to_idx[wnid]) for p in idir.glob("*.JPEG")]
        else:
            vdir = self.root / "val"; ann = vdir / "val_annotations.txt"; mapping = {}
            for line in ann.read_text().splitlines():
                if not line.strip(): continue
                fn, wnid = line.split("\t")[:2]; mapping[fn] = wnid
            idir = vdir / "images"
            for p in idir.glob("*.JPEG"):
                self.samples.append((str(p), self.class_to_idx[mapping[p.name]]))
        if not self.samples: raise RuntimeError("Tiny-ImageNet-200 empty")
    def _ensure(self, auto_download):
        if self.root.exists() and (self.root / "wnids.txt").exists(): return
        if not auto_download: raise FileNotFoundError(self.root)
        self.root_base.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_base / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            import urllib.request, shutil
            with urllib.request.urlopen(self.URL) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z: z.extractall(self.root_base)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        fp, y = self.samples[i]
        img = Image.open(fp).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y
    @property
    def classes(self):
        return [self.idx_to_name[i] for i in range(len(self.idx_to_name))]

# ===================== Main =====================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    ap.add_argument("--save-dir", type=str, default="runs/alignclip_baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    # Pretrain data
    ap.add_argument("--cc12m-mode", type=str, choices=["index", "coco-train", "none"], default="coco-train",
                    help="index: use CSV/JSONL index of CC12M; coco-train: use COCO train2017 as quick smoke; none: skip pretrain")
    ap.add_argument("--cc12m-index", type=str, default="", help="CSV/JSONL with columns image_path,caption")
    ap.add_argument("--max-pretrain", type=int, default=None)

    # Model
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--txt-maxlen", type=int, default=77)
    ap.add_argument("--vocab-size", type=int, default=49408)

    # Train hyperparams (AlignCLIP paper)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)

    # Finetune retrieval (optional, per paper)
    ap.add_argument("--ft-coco-epochs", type=int, default=8)
    ap.add_argument("--ft-flickr-epochs", type=int, default=20)
    ap.add_argument("--ft-batch", type=int, default=128)
    ap.add_argument("--ft-lr", type=float, default=5e-6)
    ap.add_argument("--ft-wd", type=float, default=0.2)

    # Eval limits
    ap.add_argument("--max-coco-eval", type=int, default=None)
    ap.add_argument("--max-flickr-eval", type=int, default=None)

    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Build model & text pipelines
    model = AlignCLIP(img_size=args.img_size, patch=args.patch, vocab_size=args.vocab_size,
                      txt_maxlen=args.txt_maxlen, width=args.width, layers=args.layers, heads=args.heads).to(device)
    text_pipe = TextPipeline(model_name="roberta-base", max_len=args.txt_maxlen, device=device)
    sbert = SBERTSemantic(model_name="sentence-transformers/all-mpnet-base-v2", device=device)

    results = {"config":{
        "img_size":args.img_size,"patch":args.patch,"width":args.width,"layers":args.layers,"heads":args.heads,
        "txt_maxlen":args.txt_maxlen,"epochs":args.epochs,"batch_size":args.batch_size,"lr":args.lr,
        "weight_decay":args.weight_decay,"warmup":args.warmup,"alpha":args.alpha,"beta":args.beta,
        "device":device
    },"runtime_sec":{},"retrieval":{},"zero_shot":{},"modality_gap":{}}
    t_all = time.time()

    # ========== Pretrain on CC12M (or COCO train mock) ==========
    if args.cc12m_mode != "none":
        tf_train = build_tf(args.img_size, True)
        if args.cc12m_mode == "index":
            assert args.cc12m_index, "Please provide --cc12m-index"
            pre_ds = CC12MWebDataset(args.cc12m_index, transform=tf_train, max_items=args.max_pretrain)
        elif args.cc12m_mode == "coco-train":
            pre_ds = CocoTrainPairs(args.data_root, transform=tf_train, split="train2017")
            if args.max_pretrain is not None:
                class Sub(Dataset):
                    def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
                    def __len__(self): return self.n
                    def __getitem__(self, i): return self.base[i]
                pre_ds = Sub(pre_ds, args.max_pretrain)
        loader = DataLoader(pre_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            collate_fn=collate_images_captions)
        t0 = time.time()
        train_alignclip(model, text_pipe, sbert, loader, device=device,
                        epochs=args.epochs, lr=args.lr, wd=args.weight_decay, warmup=args.warmup,
                        total_steps=args.epochs*len(loader), alpha=args.alpha, beta=args.beta,
                        cosine=True, save_dir=os.path.join(args.save_dir, "ckpt"))
        results["runtime_sec"]["pretrain"] = time.time() - t0

    # ========== Retrieval Eval: COCO & Flickr30k ==========
    tf_eval = build_tf(args.img_size, False)
    coco_eval_raw = CocoCaptionsEval(args.data_root, split="val", transform=tf_eval)
    if args.max_coco_eval:
        class Sub(Dataset):
            def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.base[i]
        coco_eval_raw = Sub(coco_eval_raw, args.max_coco_eval)
    coco_loader = DataLoader(coco_eval_raw, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=collate_images_captions)
    t0 = time.time()
    mg_coco, i2t_coco, t2i_coco = retrieval_eval(model, text_pipe, coco_loader, device)
    results["retrieval"]["mscoco"] = {"I2T":i2t_coco,"T2I":t2i_coco,"modality_gap":mg_coco}
    results["runtime_sec"]["mscoco_eval"] = time.time() - t0

    flickr_eval_raw = Flickr30kEval(args.data_root, transform=tf_eval)
    if args.max_flickr_eval:
        class Sub(Dataset):
            def __init__(self, base, n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self, i): return self.base[i]
        flickr_eval_raw = Sub(flickr_eval_raw, args.max_flickr_eval)
    flickr_loader = DataLoader(flickr_eval_raw, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
                               collate_fn=collate_images_captions)
    t0 = time.time()
    mg_flickr, i2t_flickr, t2i_flickr = retrieval_eval(model, text_pipe, flickr_loader, device)
    results["retrieval"]["flickr30k"] = {"I2T":i2t_flickr,"T2I":t2i_flickr,"modality_gap":mg_flickr}
    results["runtime_sec"]["flickr30k_eval"] = time.time() - t0

    # ========== Zero-Shot: CIFAR100 / Tiny-ImageNet-200 / DTD ==========
    tf_zs = build_tf(args.img_size, False)

    # CIFAR100
    t0 = time.time()
    cifar_val = tvds.CIFAR100(root=str(Path(args.data_root)/"cifar100"), train=False, transform=tf_zs, download=True)
    cifar_train = tvds.CIFAR100(root=str(Path(args.data_root)/"cifar100"), train=True, transform=tf_zs, download=True)
    loader = DataLoader(cifar_val, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    top1, top5, mg = zeroshot_eval(model, text_pipe, loader, cifar_train.classes, device)
    results["zero_shot"]["cifar100"] = {"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["cifar100_zs"] = time.time() - t0

    # Tiny-ImageNet-200
    t0 = time.time()
    tiny = TinyImageNet200(args.data_root, split="val", transform=tf_zs, auto_download=True)
    loader = DataLoader(tiny, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    top1, top5, mg = zeroshot_eval(model, text_pipe, loader, tiny.classes, device)
    results["zero_shot"]["tiny_imagenet_200"] = {"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["tiny_imagenet_200_zs"] = time.time() - t0

    # DTD
    t0 = time.time()
    dtd_val = tvds.DTD(root=str(Path(args.data_root)/"dtd"), split="test", transform=tf_zs, download=True)
    dtd_train = tvds.DTD(root=str(Path(args.data_root)/"dtd"), split="train", transform=tf_zs, download=True)
    loader = DataLoader(dtd_val, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    top1, top5, mg = zeroshot_eval(model, text_pipe, loader, dtd_train.classes, device)
    results["zero_shot"]["dtd"] = {"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["dtd_zs"] = time.time() - t0

    results["runtime_sec"]["total"] = time.time() - t_all

    # Save
    js_path = os.path.join(args.save_dir, f"alignclip_baseline_{ts_now()}.json")
    with open(js_path, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)

    # also CSV (flat)
    rows = []
    def add_row(name, metrics, rt=None):
        row = {"dataset":name}
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
        if rt is not None: row["runtime_sec"] = rt
        rows.append(row)

    add_row("mscoco", results["retrieval"]["mscoco"], results["runtime_sec"]["mscoco_eval"])
    add_row("flickr30k", results["retrieval"]["flickr30k"], results["runtime_sec"]["flickr30k_eval"])
    add_row("cifar100", results["zero_shot"]["cifar100"], results["runtime_sec"]["cifar100_zs"])
    add_row("tiny_imagenet_200", results["zero_shot"]["tiny_imagenet_200"], results["runtime_sec"]["tiny_imagenet_200_zs"])
    add_row("dtd", results["zero_shot"]["dtd"], results["runtime_sec"]["dtd_zs"])
    rows.append({"dataset":"TOTAL","runtime_sec":results["runtime_sec"]["total"]})
    csv_path = os.path.join(args.save_dir, f"alignclip_baseline_{ts_now()}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved JSON -> {js_path}")
    print(f"Saved CSV  -> {csv_path}")
    print(f"Model params (trainable): {count_params(model):,}")

if __name__ == "__main__":
    main()
