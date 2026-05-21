# -*- coding: utf-8 -*-
"""
eval_modality_gap_and_zeroshot.py  (rev: coco2017 + deep dtd)

改动要点：
- 自动探测 COCO 2017 或 2014：
  * 2017:  data_root/coco2017/annotations/captions_val2017.json
           data_root/coco2017/images/val2017/*.jpg
  * 2014:  data_root/coco/annotations/captions_val2014.json
           data_root/coco/val2014/*.jpg
- 自动探测 DTD 多层目录：
  * 优先 torchvision DTD（split='test'）
  * 否则扫描：
      data_root/dtd/dtd/dtd/images/<class>/*
      或 data_root/dtd/images/<class>/*
- CIFAR100：优先 data_root/cifar100，fallback 到 data_root
- 其余逻辑与之前一致：计算 CD/FD/RMG/FMMD；分类 Top1/5；检索 R@1/5/10；输出 CSV/JSON
"""

import os, sys, math, json, argparse, random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import open_clip
from scipy.linalg import sqrtm

# ----------------------------
# 通用工具
# ----------------------------
def set_seed(sd=2025):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def to_np(t: torch.Tensor): return t.detach().cpu().float().numpy()

def l2norm_np(x, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def cosine_sim_np(A, B):
    A = l2norm_np(A, -1); B = l2norm_np(B, -1)
    return A @ B.T

def safe_mean_cov(X: np.ndarray, eps=1e-6):
    mu = X.mean(axis=0)
    Xc = X - mu
    if X.shape[0] > 1:
        C = (Xc.T @ Xc) / (X.shape[0]-1)
    else:
        C = np.eye(X.shape[1]) * eps
    return mu, C

def centroid_distance(X: np.ndarray, Y: np.ndarray) -> float:
    if X.size==0 or Y.size==0: return float("nan")
    mu_x = X.mean(axis=0); mu_y = Y.mean(axis=0)
    return float(np.linalg.norm(mu_x - mu_y))

def frechet_distance(X: np.ndarray, Y: np.ndarray, eps=1e-6) -> float:
    if X.size==0 or Y.size==0: return float("nan")
    mu_x, Sx = safe_mean_cov(X, eps)
    mu_y, Sy = safe_mean_cov(Y, eps)
    diff = mu_x - mu_y
    covmean = sqrtm(Sx @ Sy)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff @ diff + np.trace(Sx + Sy - 2*covmean))

def relative_modality_gap(X: np.ndarray, Y: np.ndarray, eps=1e-8) -> float:
    if X.size==0 or Y.size==0: return float("nan")
    mu_x = X.mean(axis=0); mu_y = Y.mean(axis=0)
    Xc = X - mu_x; Yc = Y - mu_y
    Sx = (Xc**2).sum()/max(1, X.shape[0]-1)
    Sy = (Yc**2).sum()/max(1, Y.shape[0]-1)
    return float(np.linalg.norm(mu_x-mu_y) / math.sqrt(Sx+Sy+eps))

# ----------------------------
# 分形核 + FMMD
# ----------------------------
class FractalKernel:
    def __init__(self, radii: List[float], weights: Optional[List[float]]=None, D: float=1.0):
        assert len(radii)>=1
        self.radii = np.array(radii, dtype=np.float32)
        if weights is None:
            w = np.ones_like(self.radii) / len(self.radii)
        else:
            w = np.array(weights, dtype=np.float32); w = w / (w.sum() + 1e-8)
        self.weights = w; self.D = float(D)

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        XX = (X**2).sum(axis=1, keepdims=True)
        YY = (Y**2).sum(axis=1, keepdims=True)
        sq = XX + YY.T - 2*(X @ Y.T)
        sq = np.maximum(sq, 0.0)
        K = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)
        for w, r in zip(self.weights, self.radii):
            denom = max(1e-8, (4.0*(r**self.D)))
            K += w * np.exp(-sq/denom)
        return K.astype(np.float32)

def fmmd2(X: np.ndarray, Y: np.ndarray, fk: FractalKernel) -> float:
    if X.size==0 or Y.size==0: return float("nan")
    Kxx = fk.kernel_matrix(X,X)
    Kyy = fk.kernel_matrix(Y,Y)
    Kxy = fk.kernel_matrix(X,Y)
    n = X.shape[0]; m = Y.shape[0]
    term_xx = (Kxx.sum() - np.trace(Kxx)) / (n*(n-1)) if n>1 else 0.0
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (m*(m-1)) if m>1 else 0.0
    term_xy = 2.0 * Kxy.mean()
    return float(max(0.0, term_xx + term_yy - term_xy))

# ----------------------------
# CLIP 编码
# ----------------------------
def build_clip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer

@torch.no_grad()
def encode_images(model, preprocess, device, paths: List[str], batch_size=256):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="encode_images"):
        imgs = []
        for p in paths[i:i+batch_size]:
            imgs.append(preprocess(Image.open(p).convert("RGB")))
        ims = torch.stack(imgs, 0).to(device)
        z = F.normalize(model.encode_image(ims), dim=-1)
        embs.append(to_np(z))
        del ims, z; torch.cuda.empty_cache()
    return np.concatenate(embs, 0) if embs else np.zeros((0, model.text_projection.shape[1]), dtype=np.float32)

@torch.no_grad()
def encode_texts(model, tokenizer, device, texts: List[str], batch_size=256):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode_texts"):
        toks = tokenizer(texts[i:i+batch_size]).to(device)
        z = F.normalize(model.encode_text(toks), dim=-1)
        embs.append(to_np(z))
        del toks, z; torch.cuda.empty_cache()
    return np.concatenate(embs, 0) if embs else np.zeros((0, model.text_projection.shape[1]), dtype=np.float32)

# ----------------------------
# CIFAR100
# ----------------------------
def load_cifar100_dataset(root: str):
    # 优先 data_root/cifar100
    try:
        ds = datasets.CIFAR100(root=os.path.join(root, "cifar100"), train=False, download=False)
    except Exception:
        ds = datasets.CIFAR100(root=root, train=False, download=False)
    return ds, ds.classes

class CIFARImageDataset(Dataset):
    def __init__(self, cifar_ds, preprocess):
        self.ds = cifar_ds; self.preprocess = preprocess
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, y = self.ds[i]
        return self.preprocess(img), int(y)

CLIP_PROMPT_TEMPLATES = [
    "a photo of a {}.", "a blurry photo of a {}.", "a close-up photo of a {}.",
    "a photo of a small {}.", "a photo of a large {}.", "a clean photo of a {}.",
    "a bright photo of a {}.", "a cropped photo of a {}.", "a good photo of a {}.",
    "a low resolution photo of a {}.", "a photo of the {}.", "a rendering of a {}.",
    "a bad photo of a {}.", "a black and white photo of a {}.", "a dark photo of a {}.",
    "a drawing of a {}.", "a photo of my {}.", "a photo of the cool {}.",
    "a photo of the small {}.", "a photo of the large {}."
]

def build_class_text_embeds(model, tokenizer, device, classnames: List[str], batch=256):
    texts = [tmpl.format(c) for c in classnames for tmpl in CLIP_PROMPT_TEMPLATES]
    Z = encode_texts(model, tokenizer, device, texts, batch)
    Z = Z.reshape(len(classnames), len(CLIP_PROMPT_TEMPLATES), -1).mean(axis=1)
    return l2norm_np(Z, -1)

@torch.no_grad()
def zero_shot_eval_images(model, preprocess, device,
                          image_paths: List[str], labels: List[int],
                          class_embeds: np.ndarray,
                          batch_size=256, num_workers=4):
    """
    Evaluate zero-shot classification accuracy (Top1 / Top5)
    using CLIP embeddings.
    """
    class PDS(Dataset):
        def __init__(self, paths, labels, t):
            self.paths = paths
            self.labels = labels
            self.t = t
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            return self.t(Image.open(self.paths[i]).convert("RGB")), int(self.labels[i])

    dl = DataLoader(PDS(image_paths, labels, preprocess),
                    batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    all_logits, all_labels = [], []
    for ims, ys in tqdm(dl, desc="zero-shot classify"):
        ims = ims.to(device)
        z = F.normalize(model.encode_image(ims), dim=-1)
        logits = z @ torch.from_numpy(class_embeds).to(device).T
        all_logits.append(logits.cpu())
        all_labels.append(ys)
        del ims, z, logits
        torch.cuda.empty_cache()

    logits = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)

    # ---- 正确计算 Top-1 / Top-5 ----
    top1 = (logits.argmax(dim=1) == labels_t).float().mean().item() * 100.0
    _, top5_idx = logits.topk(k=5, dim=1)
    correct_top5 = (top5_idx == labels_t.unsqueeze(1)).any(dim=1)
    top5 = correct_top5.float().mean().item() * 100.0

    return float(top1), float(top5)

# ----------------------------
# Tiny-ImageNet-200（若不存在自动跳过）
# ----------------------------
def load_tiny_imagenet_val(root: str):
    base = os.path.join(root, "tiny-imagenet-200")
    wnids = os.path.join(base, "wnids.txt")
    if not os.path.isfile(wnids): raise FileNotFoundError("tiny-imagenet-200 not found")
    wnids = open(wnids).read().strip().splitlines()
    wnid_to_idx = {w:i for i,w in enumerate(wnids)}
    words = {}
    with open(os.path.join(base,"words.txt"), "r") as f:
        for line in f:
            wid, name = line.strip().split("\t"); words[wid]=name
    images, labels = [], []
    ann = os.path.join(base, "val", "val_annotations.txt")
    with open(ann, "r") as f:
        for line in f:
            fn, wnid, *_ = line.strip().split()
            if wnid not in wnid_to_idx: continue
            images.append(os.path.join(base, "val", "images", fn))
            labels.append(wnid_to_idx[wnid])
    classnames = [words.get(w,w) for w in wnids]
    return images, labels, classnames

# ----------------------------
# DTD（自动探测深层目录）
# ----------------------------
def load_dtd_images(root: str):
    # 1) torchvision 的官方 split
    try:
        ds = datasets.DTD(root=root, split='test', download=False)
        imgs = [p for p,_ in ds._image_files]
        labels = list(ds._labels)
        classnames = list(ds.categories)
        return imgs, labels, classnames
    except Exception:
        pass
    # 2) 深层目录探测
    candidates = [
        os.path.join(root, "dtd", "dtd", "dtd", "images"),
        os.path.join(root, "dtd", "images"),
    ]
    images_dir = None
    for c in candidates:
        if os.path.isdir(c): images_dir = c; break
    if images_dir is None:
        raise FileNotFoundError("DTD images dir not found under candidates: dtd/dtd/dtd/images or dtd/images")

    classnames = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir,d))])
    cls_to_idx = {c:i for i,c in enumerate(classnames)}
    images, labels = [], []
    for c in classnames:
        cdir = os.path.join(images_dir, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                images.append(os.path.join(cdir, fn)); labels.append(cls_to_idx[c])
    return images, labels, classnames

# ----------------------------
# COCO（自动探测 2017/2014）
# ----------------------------
def load_coco_val_autodetect(root: str):
    """
    返回 (img_paths, texts, text_to_image_index)
    支持：
      - data_root/coco2017/annotations/captions_val2017.json + images/val2017
      - data_root/coco/annotations/captions_val2014.json    + val2014
    """
    # 1) 2017
    base17 = os.path.join(root, "coco2017")
    ann17 = os.path.join(base17, "annotations", "captions_val2017.json")
    img17 = os.path.join(base17, "images", "val2017")
    if os.path.isfile(ann17) and os.path.isdir(img17):
        with open(ann17, "r") as f:
            data = json.load(f)
        id_to_file = {img["id"]: os.path.join(img17, img["file_name"]) for img in data["images"]}
        imgids_to_caps = defaultdict(list)
        for a in data["annotations"]:
            imgids_to_caps[a["image_id"]].append(a["caption"].strip())
        img_paths, texts, img_index_for_text = [], [], []
        for img_id, caps in imgids_to_caps.items():
            p = id_to_file.get(img_id, None)
            if p is None or not os.path.isfile(p): continue
            img_paths.append(p)
            for c in caps:
                texts.append(c); img_index_for_text.append(len(img_paths)-1)
        return img_paths, texts, img_index_for_text

    # 2) 2014
    base14 = os.path.join(root, "coco")
    ann14 = os.path.join(base14, "annotations", "captions_val2014.json")
    img14 = os.path.join(base14, "val2014")
    if os.path.isfile(ann14) and os.path.isdir(img14):
        with open(ann14, "r") as f:
            data = json.load(f)
        id_to_file = {img["id"]: os.path.join(img14, img["file_name"]) for img in data["images"]}
        imgids_to_caps = defaultdict(list)
        for a in data["annotations"]:
            imgids_to_caps[a["image_id"]].append(a["caption"].strip())
        img_paths, texts, img_index_for_text = [], [], []
        for img_id, caps in imgids_to_caps.items():
            p = id_to_file.get(img_id, None)
            if p is None or not os.path.isfile(p): continue
            img_paths.append(p)
            for c in caps:
                texts.append(c); img_index_for_text.append(len(img_paths)-1)
        return img_paths, texts, img_index_for_text

    raise FileNotFoundError("COCO val split not found under coco2017/ or coco/")

# ----------------------------
# Flickr30K（与之前一致）
# ----------------------------
def load_flickr30k(root: str):
    base = os.path.join(root, "flickr30k")
    img_dir = os.path.join(base, "flickr30k-images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Flickr30K images dir missing: {img_dir}")
    ann_token = os.path.join(base, "results_20130124.token")
    ann_csv = os.path.join(base, "flickr_annotations_30k.csv")

    captions_map = defaultdict(list)
    if os.path.isfile(ann_token):
        with open(ann_token, "r") as f:
            for line in f:
                try:
                    key, cap = line.strip().split("\t", 1)
                    img_fn = key.split("#")[0]
                    captions_map[img_fn].append(cap.strip())
                except Exception:
                    continue
    elif os.path.isfile(ann_csv):
        try:
            df = pd.read_csv(ann_csv)
        except Exception:
            df = pd.read_csv(ann_csv, sep="\t")
        cols = [c.lower() for c in df.columns]
        lc2i = {c.lower():i for i,c in enumerate(df.columns)}
        if "image" in cols and "caption" in cols:
            for _,row in df.iterrows():
                img = str(row[df.columns[lc2i["image"]]]).strip()
                cap = str(row[df.columns[lc2i["caption"]]]).strip()
                if img and cap: captions_map[img].append(cap)
        else:
            sent_cols = [c for c in df.columns if c.lower().startswith("sentence")]
            if ("image" in cols) and sent_cols:
                for _,row in df.iterrows():
                    img = str(row[df.columns[lc2i["image"]]]).strip()
                    for sc in sent_cols:
                        cap = str(row[sc]).strip()
                        if img and cap and cap!='nan': captions_map[img].append(cap)
            else:
                raise RuntimeError("Unsupported CSV format for flickr_annotations_30k.csv")
    else:
        raise FileNotFoundError("Missing both results_20130124.token and flickr_annotations_30k.csv")

    img_files, texts, img_index_for_text = [], [], []
    for fn, caps in captions_map.items():
        p = os.path.join(img_dir, fn)
        if not os.path.isfile(p): continue
        img_files.append(p)
        for c in caps:
            texts.append(c); img_index_for_text.append(len(img_files)-1)
    return img_files, texts, img_index_for_text

# ----------------------------
# 检索评测
# ----------------------------
@torch.no_grad()
def retrieval_eval(model, preprocess, tokenizer, device,
                   img_paths: List[str], texts: List[str], img_index_for_text: List[int],
                   batch_size=256):
    ZI = encode_images(model, preprocess, device, img_paths, batch_size)
    ZT = encode_texts(model, tokenizer, device, texts, batch_size)
    S = cosine_sim_np(ZI, ZT)  # [Ni, Nt]

    img_to_texts = defaultdict(list)
    for t_idx, i_idx in enumerate(img_index_for_text):
        img_to_texts[i_idx].append(t_idx)

    # I2T
    ranks = []
    for i in range(len(img_paths)):
        pos = set(img_to_texts.get(i, []))
        if not pos: continue
        order = np.argsort(-S[i])
        rank = len(order)
        for k, t_idx in enumerate(order):
            if t_idx in pos: rank = k+1; break
        ranks.append(rank)
    R = lambda r, K: 100.0 * (np.array(r) <= K).mean() if r else float("nan")
    i2t = {"R@1": R(ranks,1), "R@5": R(ranks,5), "R@10": R(ranks,10)}

    # T2I
    ranks_t = []
    St = S.T
    for t in range(len(texts)):
        pos_img = img_index_for_text[t]
        order = np.argsort(-St[t])
        where = np.where(order==pos_img)[0]
        rank = int(where[0])+1 if where.size>0 else len(order)
        ranks_t.append(rank)
    t2i = {"R@1": R(ranks_t,1), "R@5": R(ranks_t,5), "R@10": R(ranks_t,10)}

    # 正对集合（用于 gap）
    Zi_pos = np.stack([ZI[i] for i in img_index_for_text], 0) if img_index_for_text else np.zeros((0, ZI.shape[1]), np.float32)
    Zt_pos = ZT

    return {"ZI": ZI, "ZT": ZT, "I2T": i2t, "T2I": t2i, "pos_pairs": {"ZI": Zi_pos, "ZT": Zt_pos}}

# ----------------------------
# 主流程
# ----------------------------
@dataclass
class Cfg:
    data_root: str
    out_dir: str
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 8
    seed: int = 2025

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    cfg = Cfg(**vars(args))

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir); ensure_dir(os.path.join(cfg.out_dir, "per_dataset_details"))

    device = cfg.device if (cfg.device=="cpu" or torch.cuda.is_available()) else "cpu"
    model, preprocess, tokenizer = build_clip(cfg.model_name, cfg.pretrained, device)
    fk = FractalKernel(radii=[0.25, 0.5, 1.0, 2.0, 4.0])

    summary = {}
    rows_cls, rows_ret = [], []

    # ---- CIFAR100
    print("\n==== CIFAR-100 ====")
    try:
        ds, classes = load_cifar100_dataset(cfg.data_root)
        ZT_cls = build_class_text_embeds(model, tokenizer, device, classes, cfg.batch_size)

        dl = DataLoader(CIFARImageDataset(ds, preprocess),
                        batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
        ZI_list, Y_list = [], []
        with torch.no_grad():
            for ims, ys in tqdm(dl, desc="CIFAR100 encode"):
                z = F.normalize(model.encode_image(ims.to(device)), dim=-1)
                ZI_list.append(to_np(z)); Y_list.append(ys.numpy())
                del ims; torch.cuda.empty_cache()
        ZI = np.concatenate(ZI_list, 0); Y = np.concatenate(Y_list, 0)

        logits = ZI @ ZT_cls.T
        top1 = (np.argmax(logits,1)==Y).mean()*100.0
        top5 = (np.topk(torch.from_numpy(logits), k=5, dim=1).indices.numpy()==Y[:,None]).any(1).mean()*100.0

        ZT_pair = ZT_cls[Y]
        cd = centroid_distance(ZI, ZT_pair)
        fd = frechet_distance(ZI, ZT_pair)
        rmg = relative_modality_gap(ZI, ZT_pair)
        fm = fmmd2(ZI, ZT_pair, fk)

        det = {"dataset":"CIFAR100","num_images":int(ZI.shape[0]),"num_classes":len(classes),
               "zero_shot":{"top1":float(top1),"top5":float(top5)},
               "gap":{"CD":float(cd),"FD":float(fd),"RMG":float(rmg),"FMMD":float(fm)}}
        json.dump(det, open(os.path.join(cfg.out_dir,"per_dataset_details","cifar100.json"),"w"), indent=2)
        summary["CIFAR100"]=det
        rows_cls.append(["CIFAR100", len(classes), ZI.shape[0], top1, top5, cd, fd, rmg, fm])
    except Exception as e:
        print("[WARN] CIFAR100 failed:", e)

    # ---- Tiny-ImageNet-200（若缺失自动跳过）
    print("\n==== Tiny-ImageNet-200 (val) ====")
    try:
        ti_images, ti_labels, ti_classes = load_tiny_imagenet_val(cfg.data_root)
        ZT_cls = build_class_text_embeds(model, tokenizer, device, ti_classes, cfg.batch_size)
        # top1/5
        def _eval(paths, labels):
            class PDS(Dataset):
                def __init__(self, p, y, t): self.p=p; self.y=y; self.t=t
                def __len__(self): return len(self.p)
                def __getitem__(self,i): return self.t(Image.open(self.p[i]).convert("RGB")), int(self.y[i])
            dl = DataLoader(PDS(ti_images, ti_labels, preprocess),
                            batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
            logits=[]; gts=[]
            for ims, ys in tqdm(dl, desc="Tiny-IN eval"):
                z = F.normalize(model.encode_image(ims.to(device)), dim=-1)
                logits.append(to_np(z) @ ZT_cls.T); gts.append(ys.numpy())
                del ims; torch.cuda.empty_cache()
            logits = np.concatenate(logits,0); gts = np.concatenate(gts,0)
            top1 = (np.argmax(logits,1)==gts).mean()*100.0
            top5 = (np.topk(torch.from_numpy(logits), k=5, dim=1).indices.numpy()==gts[:,None]).any(1).mean()*100.0
            return top1, top5
        top1, top5 = _eval(ti_images, ti_labels)
        ZI = encode_images(model, preprocess, device, ti_images, cfg.batch_size)
        ZT_pair = ZT_cls[np.array(ti_labels)]
        cd = centroid_distance(ZI, ZT_pair); fd = frechet_distance(ZI, ZT_pair)
        rmg = relative_modality_gap(ZI, ZT_pair); fm = fmmd2(ZI, ZT_pair, fk)
        det = {"dataset":"Tiny-ImageNet-200","num_images":int(ZI.shape[0]),"num_classes":len(ti_classes),
               "zero_shot":{"top1":float(top1),"top5":float(top5)},
               "gap":{"CD":float(cd),"FD":float(fd),"RMG":float(rmg),"FMMD":float(fm)}}
        json.dump(det, open(os.path.join(cfg.out_dir,"per_dataset_details","tiny_imagenet_200.json"),"w"), indent=2)
        summary["Tiny-ImageNet-200"]=det
        rows_cls.append(["Tiny-ImageNet-200", len(ti_classes), ZI.shape[0], top1, top5, cd, fd, rmg, fm])
    except Exception as e:
        print("[WARN] Tiny-ImageNet-200 skipped:", e)

    # ---- DTD
    print("\n==== DTD ====")
    try:
        dtd_images, dtd_labels, dtd_classes = load_dtd_images(cfg.data_root)
        ZT_cls = build_class_text_embeds(model, tokenizer, device, dtd_classes, cfg.batch_size)
        top1, top5 = zero_shot_eval_images(model, preprocess, device, dtd_images, dtd_labels, ZT_cls,
                                           batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        ZI = encode_images(model, preprocess, device, dtd_images, cfg.batch_size)
        ZT_pair = ZT_cls[np.array(dtd_labels)]
        cd = centroid_distance(ZI, ZT_pair); fd = frechet_distance(ZI, ZT_pair)
        rmg = relative_modality_gap(ZI, ZT_pair); fm = fmmd2(ZI, ZT_pair, fk)
        det = {"dataset":"DTD","num_images":int(ZI.shape[0]),"num_classes":len(dtd_classes),
               "zero_shot":{"top1":float(top1),"top5":float(top5)},
               "gap":{"CD":float(cd),"FD":float(fd),"RMG":float(rmg),"FMMD":float(fm)}}
        json.dump(det, open(os.path.join(cfg.out_dir,"per_dataset_details","dtd.json"),"w"), indent=2)
        summary["DTD"]=det
        rows_cls.append(["DTD", len(dtd_classes), ZI.shape[0], top1, top5, cd, fd, rmg, fm])
    except Exception as e:
        print("[WARN] DTD failed:", e)

    # 保存分类 CSV
    if rows_cls:
        pd.DataFrame(rows_cls, columns=[
            "dataset","num_classes","num_images","top1","top5","CD","FD","RMG","FMMD"
        ]).to_csv(os.path.join(cfg.out_dir,"classification_metrics.csv"), index=False)

    # ---- MS-COCO（自动 2017/2014）
    print("\n==== MS-COCO Retrieval (auto-detect 2017/2014) ====")
    try:
        coco_imgs, coco_caps, coco_imgidx = load_coco_val_autodetect(cfg.data_root)
        coco_res = retrieval_eval(model, preprocess, tokenizer, device,
                                  coco_imgs, coco_caps, coco_imgidx,
                                  batch_size=cfg.batch_size)
        Zi_pos, Zt_pos = coco_res["pos_pairs"]["ZI"], coco_res["pos_pairs"]["ZT"]
        cd = centroid_distance(Zi_pos, Zt_pos); fd = frechet_distance(Zi_pos, Zt_pos)
        rmg = relative_modality_gap(Zi_pos, Zt_pos); fm = fmmd2(Zi_pos, Zt_pos, fk)
        det = {"dataset":"MS-COCO (auto)","num_images":len(coco_imgs),"num_texts":len(coco_caps),
               "retrieval":{"I2T":coco_res["I2T"],"T2I":coco_res["T2I"]},
               "gap_on_positive_pairs":{"CD":float(cd),"FD":float(fd),"RMG":float(rmg),"FMMD":float(fm)}}
        json.dump(det, open(os.path.join(cfg.out_dir,"per_dataset_details","mscoco_val.json"),"w"), indent=2)
        summary["MS-COCO"]=det
        rows_ret.append(["MS-COCO", len(coco_imgs), len(coco_caps),
                         coco_res["I2T"]["R@1"], coco_res["I2T"]["R@5"], coco_res["I2T"]["R@10"],
                         coco_res["T2I"]["R@1"], coco_res["T2I"]["R@5"], coco_res["T2I"]["R@10"],
                         cd, fd, rmg, fm])
    except Exception as e:
        print("[WARN] COCO retrieval failed:", e)

    # ---- Flickr30K
    print("\n==== Flickr30K Retrieval ====")
    try:
        fk_imgs, fk_caps, fk_imgidx = load_flickr30k(cfg.data_root)
        fk_res = retrieval_eval(model, preprocess, tokenizer, device,
                                fk_imgs, fk_caps, fk_imgidx,
                                batch_size=cfg.batch_size)
        Zi_pos, Zt_pos = fk_res["pos_pairs"]["ZI"], fk_res["pos_pairs"]["ZT"]
        cd = centroid_distance(Zi_pos, Zt_pos); fd = frechet_distance(Zi_pos, Zt_pos)
        rmg = relative_modality_gap(Zi_pos, Zt_pos); fm = fmmd2(Zi_pos, Zt_pos, fk)
        det = {"dataset":"Flickr30K","num_images":len(fk_imgs),"num_texts":len(fk_caps),
               "retrieval":{"I2T":fk_res["I2T"],"T2I":fk_res["T2I"]},
               "gap_on_positive_pairs":{"CD":float(cd),"FD":float(fd),"RMG":float(rmg),"FMMD":float(fm)}}
        json.dump(det, open(os.path.join(cfg.out_dir,"per_dataset_details","flickr30k.json"),"w"), indent=2)
        summary["Flickr30K"]=det
        rows_ret.append(["Flickr30K", len(fk_imgs), len(fk_caps),
                         fk_res["I2T"]["R@1"], fk_res["I2T"]["R@5"], fk_res["I2T"]["R@10"],
                         fk_res["T2I"]["R@1"], fk_res["T2I"]["R@5"], fk_res["T2I"]["R@10"],
                         cd, fd, rmg, fm])
    except Exception as e:
        print("[WARN] Flickr30K retrieval failed:", e)

    # 保存检索 CSV + 总表
    if rows_ret:
        pd.DataFrame(rows_ret, columns=[
            "dataset","num_images","num_texts",
            "I2T_R@1","I2T_R@5","I2T_R@10",
            "T2I_R@1","T2I_R@5","T2I_R@10",
            "CD_pos","FD_pos","RMG_pos","FMMD_pos"
        ]).to_csv(os.path.join(cfg.out_dir,"retrieval_metrics.csv"), index=False)

    json.dump(summary, open(os.path.join(cfg.out_dir,"results_summary.json"),"w"), indent=2)
    print("\nAll done. Results saved to:", cfg.out_dir)

if __name__ == "__main__":
    main()
