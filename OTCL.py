#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill-the-Gap Baseline (Spectral & OT) — 完整实现
- 后处理：
  * Spectral（谱）：按论文构造二部图 A = [[0, W],[W^T, 0]]，W = X Y^T（X/Y 为单位范数特征），
    计算随机游走拉普拉斯 L_rw = D^{-1}(D-A) 的最小非零 k 个特征向量，F 的每行即节点新表示。
    取前 N 行作为图像新嵌入 X', 后 N 行作为文本新嵌入 Y'。（k 默认 60）
  * OT（最优传输，Laplacian 正则 EMDLaplaceTransport）：学习 γ，使源分布（图像）运输到目标（文本），
    同时用模态内相似图正则平滑位移。输出 X' = γ @ Y，Y' = γ^T @ X（再单位化）。
- 指标：
  * Modality Gap：centroid distance、Fréchet distance（FID 公式）、relative modality gap
  * IR 异质性指标：ITR/TIR（同模态最近邻偏置比），IMR/TMR（跨模态“最优项”的平均名次）
  * 标准检索：I2T/T2I 的 R@1/5/10（top-K 召回）
  * Zero-Shot（CIFAR100/Tiny-ImageNet-200/DTD）：Top-1/Top-5（18 模板）
- 评测对齐前述基线；同时输出 ORIG、SPECT、OT 版本的结果与计时。

参考实现依据：
- 谱方法步骤（W、二部图 A、Laplacian 与特征向量用作新表示）:contentReference[oaicite:0]{index=0}
- OT 目标（拉普拉斯正则 EMDLaplaceTransport）与符号定义（γ、S^s/S^t 等）:contentReference[oaicite:1]{index=1}
- IR 指标 ITR/TIR、IMR/TMR 的定义与动机 :contentReference[oaicite:2]{index=2}
- FID 公式（用于分布距离）:contentReference[oaicite:3]{index=3}
"""

import os, math, json, time, random, warnings
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

import open_clip

# =============== 工具 ===============
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor, dim=-1, eps=1e-9):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def to_device(x, device):
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x

def ensure_pot():
    try:
        import ot, ot.da  # noqa
        return True
    except Exception:
        return False

# =============== 变换（统一） ===============
def build_tf(image_size=224, train=True, rrc_scale=(0.5,1.0)):
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

# =============== 数据集封装（与前几条基线一致） ===============
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
        img,caps = self.ds[i]; return img, caps

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

# =============== ZS 模板（18 个） ===============
ZS_TEMPLATES = [
 "a photo of a {}.","a blurry photo of a {}.","a black and white photo of a {}.",
 "a low contrast photo of a {}.","a high contrast photo of a {}.","a bad photo of a {}.",
 "a good photo of a {}.","a photo of a small {}.","a photo of a big {}.","a photo of the {}.",
 "a blurry photo of the {}.","a black and white photo of the {}.","a low contrast photo of the {}.",
 "a high contrast photo of the {}.","a bad photo of the {}.","a good photo of the {}.",
 "a photo of the small {}.","a photo of the big {}."
]

# =============== 指标 ===============
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

def ir_measures_concat(imgE: torch.Tensor, txtE: torch.Tensor) -> Dict[str, float]:
    """
    ITR/TIR、IMR/TMR（论文定义）。输入需单位范数。
    - ITR = (#image->image 最近邻) / (#image->text 最近邻)
    - TIR = (#text->text 最近邻) / (#text->image 最近邻)
    - IMR = 图像查询时，最靠前的“文本”名次的平均
    - TMR = 文本查询时，最靠前的“图像”名次的平均
    """
    X = imgE; Y = txtE
    X = F.normalize(X, dim=-1); Y = F.normalize(Y, dim=-1)
    Z = torch.cat([X, Y], dim=0)  # (N+M, d)
    S = (Z @ Z.t()).cpu().numpy()
    N = X.size(0); M = Y.size(0)
    # 对每行找“除自身外”的最大
    np.fill_diagonal(S, -1e9)
    nn_idx = S.argmax(axis=1)
    img_nn = nn_idx[:N]; txt_nn = nn_idx[N:]
    # ITR / TIR
    img2img = np.sum(img_nn < N)
    img2txt = np.sum(img_nn >= N)
    txt2txt = np.sum(txt_nn >= N)
    txt2img = np.sum(txt_nn < N)
    ITR = float(img2img) / max(1, float(img2txt))
    TIR = float(txt2txt) / max(1, float(txt2img))
    # IMR / TMR：找跨模态的最优名次
    # 对每个 i（图像行），按相似度对所有列排序，找第一个“文本列”的排名
    order = np.argsort(-S, axis=1)
    imr = []; tmr = []
    for i in range(N):
        ranks = order[i]
        # 找第一个 >= N 的列
        pos = np.where(ranks >= N)[0]
        imr.append(int(pos[0])+1 if len(pos)>0 else N+M)
    for i in range(N, N+M):
        ranks = order[i]
        pos = np.where(ranks < N)[0]
        tmr.append(int(pos[0])+1 if len(pos)>0 else N+M)
    IMR = float(np.mean(imr)) if len(imr)>0 else float('nan')
    TMR = float(np.mean(tmr)) if len(tmr)>0 else float('nan')
    return {"ITR": ITR, "TIR": TIR, "IMR": IMR, "TMR": TMR}

def recalls_from_sim(sim: np.ndarray, gt: Dict[int,List[int]], ks=(1,5,10)):
    order = np.argsort(-sim, axis=1); res={}
    for k in ks:
        ok=0
        for i in range(order.shape[0]):
            if any(t in set(order[i,:k]) for t in gt[i]): ok+=1
        res[f"R@{k}"]=ok/order.shape[0]
    return res

# =============== 后处理：Spectral ===============
def spectral_transform(imgE: torch.Tensor, txtE: torch.Tensor, k: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入：单位化后的 X=imgE, Y=txtE
    W = X Y^T, A = [[0,W],[W^T,0]], L_rw = D^{-1}(D-A)，取最小非零的 k 个特征向量作为 F
    返回：X' = F[:N], Y' = F[N:]
    """
    X = F.normalize(imgE, dim=-1).cpu().numpy()
    Y = F.normalize(txtE, dim=-1).cpu().numpy()
    N, d = X.shape; M = Y.shape[0]
    W = X @ Y.T  # 相似度
    # 构造稀疏 A
    A_upper = np.concatenate([np.zeros((N,N)), W], axis=1)
    A_lower = np.concatenate([W.T, np.zeros((M,M))], axis=1)
    A = np.concatenate([A_upper, A_lower], axis=0)
    # 度矩阵对角
    deg = A.sum(axis=1)
    Dinv = np.diag(1.0/np.maximum(deg, 1e-12))
    Lrw = np.eye(N+M) - Dinv @ A  # D^{-1}(D-A)
    # 取最小的 k+1 个特征（包含 0 特征），去掉第一个 0 特征
    # 用对称化保证实数谱：S = (Lrw + Lrw.T)/2
    S = 0.5*(Lrw + Lrw.T)
    # 稀疏化以加速
    S_sp = csr_matrix(S)
    vals, vecs = eigsh(S_sp, k=min(k+1, N+M-1), which='SM')  # 最小特征值
    # 去掉最小（~0）的那一列
    order = np.argsort(vals)
    vecs = vecs[:, order]
    # 丢掉第一个（对应近 0 特征值）
    if vecs.shape[1] > k:
        Femb = vecs[:, 1:k+1]
    else:
        Femb = vecs[:, 1:]
    # 单位化
    Femb = Femb / (np.linalg.norm(Femb, axis=1, keepdims=True) + 1e-9)
    Xp = torch.tensor(Femb[:N], dtype=torch.float32)
    Yp = torch.tensor(Femb[N:], dtype=torch.float32)
    return Xp, Yp

# =============== 后处理：Optimal Transport (POT) ===============
def knn_graph_cosine(feats: np.ndarray, k: int = 10) -> np.ndarray:
    """构造对称 kNN 图的相似度矩阵（余弦），用于拉普拉斯正则。"""
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    sim = feats @ feats.T
    N = sim.shape[0]
    G = np.zeros_like(sim)
    idx = np.argpartition(-sim, kth=min(k, N-1), axis=1)[:, :k+1]  # 包含自身
    for i in range(N):
        G[i, idx[i]] = sim[i, idx[i]]
    G = np.maximum(G, G.T)  # 对称
    np.fill_diagonal(G, 0.0)
    return G

def ot_laplace_align(imgE: torch.Tensor, txtE: torch.Tensor,
                     k_graph: int = 10, reg_e: float = 0.0, reg_lap: float = 1.0,
                     max_iter: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 POT 的 EMDLaplaceTransport 实现拉普拉斯正则的最优传输，
    超参可调：k_graph（图邻接），reg_e（熵正则，可 0），reg_lap（拉普拉斯正则强度）
    返回：X' = γY，Y' = γ^T X
    """
    if not ensure_pot():
        raise RuntimeError("需要安装 POT：pip install pot")
    import ot, ot.da  # type: ignore

    X = F.normalize(imgE, dim=-1).cpu().numpy()
    Y = F.normalize(txtE, dim=-1).cpu().numpy()
    ns, nt = X.shape[0], Y.shape[0]
    xs_graph = knn_graph_cosine(X, k=k_graph)
    xt_graph = knn_graph_cosine(Y, k=k_graph)

    # 统一权重
    p = np.ones((ns,)) / ns
    q = np.ones((nt,)) / nt

    # 代价矩阵（1 - cosine）
    C = 1.0 - (X @ Y.T)

    # EMDLaplaceTransport
    trans = ot.da.EMDLaplaceTransport(reg_e=reg_e, reg_cl=reg_lap, reg_cu=reg_lap,
                                      max_iter=max_iter, verbose=False)
    trans.fit(Xs=X, Xt=Y, xs_graph=xs_graph, xt_graph=xt_graph, p=p, q=q, M=C)
    # γ: (ns, nt)
    G = trans.coupling_
    Xp = G @ Y  # (ns, nt)@(nt,d) -> (ns,d)
    Yp = G.T @ X
    Xp = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-9)
    Yp = Yp / (np.linalg.norm(Yp, axis=1, keepdims=True) + 1e-9)
    return torch.tensor(Xp, dtype=torch.float32), torch.tensor(Yp, dtype=torch.float32)

# =============== 编码与评测通用 ===============
@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device, batch=256):
    outs=[]
    for i in range(0,len(texts),batch):
        toks = tokenizer(texts[i:i+batch]).to(device)
        ft = l2norm(model.encode_text(toks), -1)
        outs.append(ft)
    return torch.cat(outs,0)

@torch.no_grad()
def encode_captions(model, tokenizer, caps_list: List[str], device, batch=256):
    return encode_texts(model, tokenizer, caps_list, device, batch=batch)

@torch.no_grad()
def zeroshot_eval(model, tokenizer, loader, classnames, device, templates=None):
    T = templates or ZS_TEMPLATES
    cls_bank=[]
    for cname in tqdm(classnames, desc="ZS: encode classes", leave=False):
        prompts=[t.format(cname) for t in T]
        ft = encode_texts(model, tokenizer, prompts, device)
        cls_bank.append(ft)  # (P, D)
    correct1=correct5=n=0
    img_all=[]; txt_all=[]
    for images,labels in tqdm(loader, desc="Zero-shot", leave=False):
        images=to_device(images, device); labels=to_device(labels, device)
        fi = l2norm(model.encode_image(images), -1).cpu(); img_all.append(fi)
        B=fi.size(0); C=len(classnames)
        scores = torch.zeros(B,C, device="cpu")
        for ci in range(C):
            txtc = cls_bank[ci].to(device)  # (P,D)
            sc = (fi.to(device) @ txtc.t()).mean(1).cpu()
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

@torch.no_grad()
def retrieval_eval(model, tokenizer, ds, device, batch_size=64, max_items=None,
                   postproc: Optional[str]=None, spectral_k: int=60,
                   ot_k_graph: int=10, ot_reg_e: float=0.0, ot_reg_lap: float=1.0, ot_max_iter: int=200):
    """
    返回：dict{ORIG, SPECT, OT} 版本的 {mg, ir, i2t, t2i}
    """
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
    imgE = torch.cat(imgE,0)
    # 文本
    ft=[]
    for i in range(0,len(all_caps),256):
        toks = tokenizer(all_caps[i:i+256]).to(device)
        ft.append(l2norm(model.encode_text(toks), -1).cpu())
    txtE = torch.cat(ft,0) if ft else torch.empty(0, imgE.size(1))

    def eval_block(X, Y):
        mg = modality_gap(X, Y)
        ir = ir_measures_concat(X, Y)
        sim = (X @ Y.t()).numpy()
        i2t = recalls_from_sim(sim, img2caps, ks=(1,5,10))
        t2i = recalls_from_sim(sim.T, cap2imgs, ks=(1,5,10))
        return mg, ir, i2t, t2i

    out = {}

    # ORIG
    out["ORIG"] = eval_block(imgE, txtE)

    # Spectral
    if postproc is None or postproc.lower() in ["spectral", "both", "all"]:
        try:
            Xi, Yi = spectral_transform(imgE, txtE, k=spectral_k)
            out["SPECT"] = eval_block(Xi, Yi)
        except Exception as e:
            warnings.warn(f"Spectral failed: {e}")

    # OT
    if postproc is None or postproc.lower() in ["ot", "both", "all"]:
        try:
            Xi, Yi = ot_laplace_align(imgE, txtE, k_graph=ot_k_graph, reg_e=ot_reg_e,
                                      reg_lap=ot_reg_lap, max_iter=ot_max_iter)
            out["OT"] = eval_block(Xi, Yi)
        except Exception as e:
            warnings.warn(f"OT failed: {e}")

    return out

# =============== 主流程 ===============
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
    ap.add_argument("--save-dir", type=str, default="runs/fill_the_gap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-coco", type=int, default=None)
    ap.add_argument("--max-flickr", type=int, default=None)

    # 后处理与其超参（用于检索/IR/MG；ZS 默认用 ORIG）
    ap.add_argument("--postproc", type=str, default="all", choices=["all","spectral","ot","none"],
                    help="检索评测中启用的后处理（同时也总会保存 ORIG）")
    ap.add_argument("--spectral-k", type=int, default=60, help="谱方法的维度 k（论文建议 ~60-120）")
    ap.add_argument("--ot-k-graph", type=int, default=10)
    ap.add_argument("--ot-reg-e", type=float, default=0.0, help="OT 熵正则（可 0）")
    ap.add_argument("--ot-reg-lap", type=float, default=1.0, help="OT 拉普拉斯正则强度")
    ap.add_argument("--ot-max-iter", type=int, default=200)

    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 模型与 tokenizer
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)

    tf_eval  = build_tf(args.image_size, False)

    results={"config":{
        "model":args.model,"pretrained":args.pretrained,"image_size":args.image_size,
        "batch_size":args.batch_size,"device":device,
        "postproc":args.postproc,"spectral_k":args.spectral_k,
        "ot_k_graph":args.ot_k_graph,"ot_reg_e":args.ot_reg_e,"ot_reg_lap":args.ot_reg_lap
    },"zero_shot":{},"retrieval":{},"runtime_sec":{}}

    t_all=time.time()

    # -------- Zero-shot: CIFAR-100 --------
    t0=time.time()
    print("\n[Eval] CIFAR-100 (ZS)")
    cifar_val = tvds.CIFAR100(root=os.path.join(args.data_root,"cifar100"), train=False, transform=tf_eval, download=True)
    cifar_train = tvds.CIFAR100(root=os.path.join(args.data_root,"cifar100"), train=True, transform=tf_eval, download=True)
    cifar_classes = cifar_train.classes
    loader = DataLoader(cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval(model, tokenizer, loader, cifar_classes, device, templates=ZS_TEMPLATES)
    results["zero_shot"]["cifar100"]={"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["cifar100"]=time.time()-t0
    print(f"CIFAR100: top1={top1:.4f} top5={top5:.4f} time={results['runtime_sec']['cifar100']:.1f}s")

    # -------- Zero-shot: Tiny-ImageNet-200 --------
    t0=time.time()
    print("\n[Eval] Tiny-ImageNet-200 (ZS)")
    tiny = TinyImageNet200(args.data_root, split="val", transform=tf_eval, auto_download=True)
    loader = DataLoader(tiny, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval(model, tokenizer, loader, tiny.classes, device, templates=ZS_TEMPLATES)
    results["zero_shot"]["tiny_imagenet_200"]={"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["tiny_imagenet_200"]=time.time()-t0
    print(f"Tiny-ImageNet-200: top1={top1:.4f} top5={top5:.4f} time={results['runtime_sec']['tiny_imagenet_200']:.1f}s")

    # -------- Zero-shot: DTD --------
    t0=time.time()
    print("\n[Eval] DTD (ZS)")
    dtd_val = tvds.DTD(root=os.path.join(args.data_root,"dtd"), split="test", transform=tf_eval, download=True)
    dtd_train = tvds.DTD(root=os.path.join(args.data_root,"dtd"), split="train", transform=tf_eval, download=True)
    dtd_classes = dtd_train.classes
    loader = DataLoader(dtd_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    top1, top5, mg, *_ = zeroshot_eval(model, tokenizer, loader, dtd_classes, device, templates=ZS_TEMPLATES)
    results["zero_shot"]["dtd"]={"top1":top1,"top5":top5,"modality_gap":mg}
    results["runtime_sec"]["dtd"]=time.time()-t0
    print(f"DTD: top1={top1:.4f} top5={top5:.4f} time={results['runtime_sec']['dtd']:.1f}s")

    # -------- Retrieval: MSCOCO（ORIG+后处理） --------
    t0=time.time()
    print("\n[Eval] MSCOCO (I2T/T2I + IR/MG)  ORIG / SPECT / OT")
    coco = CocoCaptionsEval(args.data_root, split="val", transform=tf_eval)
    if args.max_coco is not None:
        class Sub(Dataset):
            def __init__(self, base,n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self,i): return self.base[i]
        coco = Sub(coco, args.max_coco)
    coco_out = retrieval_eval(model, tokenizer, coco, device, batch_size=args.batch_size, max_items=args.max_coco,
                              postproc=(None if args.postproc=="all" else args.postproc),
                              spectral_k=args.spectral_k,
                              ot_k_graph=args.ot_k_graph, ot_reg_e=args.ot_reg_e, ot_reg_lap=args.ot_reg_lap,
                              ot_max_iter=args.ot_max_iter)
    results["retrieval"]["mscoco"]= {}
    for key, (mg, ir, i2t, t2i) in coco_out.items():
        results["retrieval"]["mscoco"][key] = {"I2T":i2t,"T2I":t2i,"modality_gap":mg,"ir_indices":ir}
    results["runtime_sec"]["mscoco"]=time.time()-t0
    print(f"MSCOCO done in {results['runtime_sec']['mscoco']:.1f}s")

    # -------- Retrieval: Flickr30k（ORIG+后处理） --------
    t0=time.time()
    print("\n[Eval] Flickr30k (I2T/T2I + IR/MG)  ORIG / SPECT / OT")
    flickr = Flickr30kEval(args.data_root, split="test", transform=tf_eval)
    if args.max_flickr is not None:
        class SubF(Dataset):
            def __init__(self, base,n): self.base, self.n = base, min(n, len(base))
            def __len__(self): return self.n
            def __getitem__(self,i): return self.base[i]
        flickr = SubF(flickr, args.max_flickr)
    flickr_out = retrieval_eval(model, tokenizer, flickr, device, batch_size=args.batch_size, max_items=args.max_flickr,
                                postproc=(None if args.postproc=="all" else args.postproc),
                                spectral_k=args.spectral_k,
                                ot_k_graph=args.ot_k_graph, ot_reg_e=args.ot_reg_e, ot_reg_lap=args.ot_reg_lap,
                                ot_max_iter=args.ot_max_iter)
    results["retrieval"]["flickr30k"]= {}
    for key, (mg, ir, i2t, t2i) in flickr_out.items():
        results["retrieval"]["flickr30k"][key] = {"I2T":i2t,"T2I":t2i,"modality_gap":mg,"ir_indices":ir}
    results["runtime_sec"]["flickr30k"]=time.time()-t0
    print(f"Flickr30k done in {results['runtime_sec']['flickr30k']:.1f}s")

    # -------- 保存 JSON/CSV --------
    total=time.time()-t_all
    results["runtime_sec"]["total"]=total
    ts=time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.save_dir, exist_ok=True)
    json_path=os.path.join(args.save_dir, f"fill_the_gap_{ts}.json")
    with open(json_path,"w",encoding="utf-8") as f: json.dump(results,f,ensure_ascii=False,indent=2)

    # CSV 摘要
    rows=[]
    def add_row(name, metrics, rt=None, prefix=""):
        row={"dataset":name}
        for key,v in metrics.items():
            if isinstance(v,dict):
                for kk,vv in v.items():
                    if isinstance(vv,dict):
                        for kkk,vvv in vv.items():
                            row[f"{prefix}{key}.{kk}.{kkk}"]=vvv
                    else:
                        row[f"{prefix}{key}.{kk}"]=vv
            else:
                row[f"{prefix}{key}"]=v
        if rt is not None: row["runtime_sec"]=rt
        rows.append(row)

    add_row("cifar100", results["zero_shot"]["cifar100"], results["runtime_sec"]["cifar100"])
    add_row("tiny_imagenet_200", results["zero_shot"]["tiny_imagenet_200"], results["runtime_sec"]["tiny_imagenet_200"])
    add_row("dtd", results["zero_shot"]["dtd"], results["runtime_sec"]["dtd"])

    # mscoco / flickr30k 的 ORIG/SPECT/OT 分开摊平
    for ds_name in ["mscoco","flickr30k"]:
        for ver, pack in results["retrieval"][ds_name].items():
            add_row(f"{ds_name}.{ver}.I2T", {"I2T":pack["I2T"], "mg":pack["modality_gap"], "ir":pack["ir_indices"]},
                    results["runtime_sec"][ds_name], prefix="")
            add_row(f"{ds_name}.{ver}.T2I", {"T2I":pack["T2I"], "mg":pack["modality_gap"], "ir":pack["ir_indices"]},
                    None, prefix="")

    rows.append({"dataset":"TOTAL","runtime_sec":results["runtime_sec"]["total"]})
    csv_path=os.path.join(args.save_dir, f"fill_the_gap_{ts}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nSaved JSON -> {json_path}")
    print(f"Saved CSV  -> {csv_path}")

if __name__ == "__main__":
    main()
