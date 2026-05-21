# -*- coding: utf-8 -*-
# fractal_manifold_demo.py
# 验证版：直接基于 CLIP 的 patch/token embedding 构建分形层级流形（Global→Entities→Relations）
# 并在同一嵌入空间内，做多尺度扩散 + 软聚合 + 匹配，对齐并可视化输出。

import os, json, math, argparse, itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

import open_clip
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.linalg import sqrtm

# --------------------
# 基础工具
# --------------------
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def l2norm_np(x, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def cosine_np(a, b):
    a = l2norm_np(a, -1); b = l2norm_np(b, -1)
    return (a @ b.T)

def to_np(t: torch.Tensor):
    return t.detach().cpu().float().numpy()

def draw_boxes(img: Image.Image, boxes, labels, fill_color=None, outline=(0,255,0), w=3):
    im = img.copy()
    d = ImageDraw.Draw(im, "RGBA")
    for (x0,y0,x1,y1), lab in zip(boxes, labels):
        if fill_color is not None:
            d.rectangle([x0,y0,x1,y1], fill=fill_color)
        d.rectangle([x0,y0,x1,y1], outline=outline, width=w)
        if lab:
            d.text((x0+2, y0+2), lab, fill=(255,0,0,255))
    return im

# --------------------
# CLIP 低层访问：提取 patch / token 表示（ViT-B/L 系）
# --------------------
@torch.no_grad()
def build_clip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer

class CLIPIntrospector:
    """
    直接复现 open_clip VisionTransformer / TextTransformer 的前向，拿到：
    - 图像：CLS 全局向量 z_img_global，末层 patch token 表示 H (P, d)
    - 文本：全局向量 z_txt_global，末层 token 表示 Y (L, d)，以及有效 token mask
    """
    def __init__(self, clip_model):
        self.m = clip_model
        assert hasattr(self.m, "visual"), "model.visual missing"
        assert hasattr(self.m, "transformer"), "text transformer missing?"

    def image_tokens(self, image_tensor: torch.Tensor):
        # image_tensor: [1,3,H,W] 已经是 preprocess 之后
        visual = self.m.visual
        # 下面逻辑参考 openai-clip 的 ViT 前向
        x = visual.conv1(image_tensor)              # [1, C, h, w]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [1, C, P]
        x = x.permute(0, 2, 1)                      # [1, P, C]
        x = torch.cat(
            [visual.class_embedding.to(x.dtype)
             + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)                             # [1, 1+P, C]
        x = x + visual.positional_embedding.to(x.dtype)  # [1, 1+P, C]
        x = visual.ln_pre(x)

        # transformer blocks
        x = x.permute(1,0,2)   # [1+P, 1, C]  (seq,batch,embed) for MultiheadAttention
        for blk in visual.transformer.resblocks:
            x = blk(x)
        x = x.permute(1,0,2)   # [1, 1+P, C]
        x = visual.ln_post(x)

        # 全局向量（cls）
        if visual.proj is not None:
            z_global = x[:,0,:] @ visual.proj
        else:
            z_global = x[:,0,:]
        z_global = F.normalize(z_global, dim=-1)

        # patch token（去掉 cls）
        H = x[:,1:,:].squeeze(0)   #[P, C]
        if visual.proj is not None:
            H_proj = H @ visual.proj
        else:
            H_proj = H
        H_proj = F.normalize(H_proj, dim=-1)        # [P, d]
        return z_global.squeeze(0), H_proj          # [d], [P,d]

    def text_tokens(self, token_ids: torch.Tensor):
        # token_ids: [1, L]
        text = self.m.transformer
        x = self.m.token_embedding(token_ids)       # [1, L, d]
        x = x + self.m.positional_embedding[:x.size(1)]
        x = x.permute(1,0,2)                        # [L,1,d]
        for blk in text.resblocks:
            x = blk(x)
        x = x.permute(1,0,2)                        # [1,L,d]
        x = self.m.ln_final(x)
        # eot position
        attn_mask = token_ids.ne(0).float()         # [1,L]
        # 全局文本向量：取 eot 位置
        # open_clip 的做法：取最后一个非 pad token（eot）对应的向量再乘 text_projection
        eot_idx = attn_mask.sum(dim=-1).long()-1    # [1]
        z = x[torch.arange(x.size(0)), eot_idx] @ self.m.text_projection
        z = F.normalize(z, dim=-1).squeeze(0)       # [d]
        # token 级投影（便于 phrase 编码）
        Y = x @ self.m.text_projection              # [1,L,d]
        Y = F.normalize(Y, dim=-1).squeeze(0)       # [L,d]
        return z, Y, attn_mask.squeeze(0)           # [d], [L,d], [L]

# --------------------
# 文本解析（实体/关系）
# --------------------
def parse_text_entities_relations(caption: str):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(caption.strip())

    # 实体（名词短语）
    ents = []
    seen = set()
    for nc in doc.noun_chunks:
        s = nc.text.strip()
        if s and s.lower() not in seen:
            ents.append(s)
            seen.add(s.lower())

    # 关系（head, rel, tail）
    rels = []
    for tok in doc:
        if tok.dep_ in ("ROOT","relcl") and tok.pos_ in ("VERB","AUX"):
            subj = [w for w in tok.lefts if w.dep_ in ("nsubj","nsubjpass")]
            obj  = [w for w in tok.rights if w.dep_ in ("dobj","attr","oprd","pobj","dative")]
            head = " ".join([w.text for w in (subj[0].subtree if subj else [])]) if subj else ""
            relw = " ".join([tok.lemma_] + [w.text for w in tok.children if w.dep_ in ("prep","prt")])
            tail = " ".join([w.text for w in (obj[0].subtree if obj else [])]) if obj else ""
            if head and relw and tail:
                rels.append((head, relw, tail))

    # 兜底：with/in/on 等介词结构
    if not rels:
        txt = caption.strip()
        for pre in ["with","in","on","at","to","from","near","beside","between","while"]:
            if f" {pre} " in txt:
                s = txt.split(f" {pre} ", 1)
                if len(s)==2:
                    rels.append((s[0], pre, s[1]))
                    break
    return ents[:16], rels[:20]

# --------------------
# 图像侧：构图、扩散、实体聚合
# --------------------
def grid_adjacency(W, H):
    # ViT-B/32@224 → P=7*7；一般 patch 网格
    # 返回边索引列表（4-邻接）
    idx = lambda x,y: y*W + x
    edges = []
    for y in range(H):
        for x in range(W):
            u = idx(x,y)
            if x+1 < W: edges.append((u, idx(x+1,y)))
            if y+1 < H: edges.append((u, idx(x,y+1)))
    return edges

def build_patch_graph(Hp: torch.Tensor, grid_W: int, grid_H: int, tau_feat=0.5):
    # Hp: [P,d], 已 L2 norm
    P, d = Hp.shape
    assert P == grid_W*grid_H
    edges = grid_adjacency(grid_W, grid_H)
    # 余弦相似转权重：w = exp( (cos-1)/tau )
    Hp_np = to_np(Hp)
    W = np.zeros((P,P), dtype=np.float32)
    for (u,v) in edges:
        sim = (Hp_np[u]*Hp_np[v]).sum()
        w = math.exp((sim-1.0)/max(1e-6, tau_feat))
        W[u,v] = W[v,u] = w
    # 自环
    for i in range(P): W[i,i] = 1.0
    # 归一化邻接 \tilde A = D^{-1} W
    D = W.sum(axis=1, keepdims=True) + 1e-8
    Ahat = W / D
    return Ahat  # [P,P]

def text_conditioned_saliency(Hp: torch.Tensor, z_img_global: torch.Tensor,
                              z_txt_global: torch.Tensor) -> np.ndarray:
    # 主分量：对 z_img·z_txt 的相似度，求对 Hp 的梯度范数（反映 patch 对全局相似的贡献）
    Hp = Hp.clone().detach().requires_grad_(True)   # [P,d]
    z_img_global = z_img_global.clone().detach().requires_grad_(True)
    z_txt_global = z_txt_global.clone().detach()
    sim = (z_img_global * z_txt_global).sum()
    sim.backward(retain_graph=True)
    g_patch = Hp.grad   # None（我们不直接用 Hp 参与相似度公式） -> fallback: 用 z_img_global 的 grad 当导引
    if g_patch is None:
        g = z_img_global.grad
        # 用与 patch 的夹角来近似敏感度（退而求其次但稳定）
        g = F.normalize(g, dim=-1)  # [d]
        Hp_n = F.normalize(Hp, dim=-1)
        sal = to_np((Hp_n @ g).abs())  # [P]
    else:
        sal = to_np(g_patch.norm(dim=-1))
    # 次分量：每个 patch 与文本全局余弦
    Hp_n = F.normalize(Hp.detach(), dim=-1)
    sal2 = to_np((Hp_n @ z_txt_global.detach()).clamp(min=0))  # 负值无视
    s = sal / (sal.max()+1e-8) * 0.6 + sal2 / (sal2.max()+1e-8) * 0.4
    return s  # [P]

def multi_scale_diffusion(Ahat: np.ndarray, s0: np.ndarray, scales=(1,2,4,8)):
    # 用 (Ahat)^r 做“离散热扩散”的近似，多尺度返回
    P = Ahat.shape[0]
    S = []
    M = np.eye(P, dtype=np.float32)
    for r in range(1, max(scales)+1):
        M = M @ Ahat
        if r in scales:
            S.append((r, (M @ s0)))
    return S  # list of (scale, s_vector)

def peak_and_aggregate(Hp: torch.Tensor, sal_map: np.ndarray, grid_W: int, grid_H: int,
                       top_k=8, thr_rel=0.35):
    # 在 sal_map 上取 top_k 局部峰并做连通聚合（按邻接逐步吸收 > thr * center 的 patch）
    P, d = Hp.shape
    sal = sal_map.copy()
    # 找候选：top_k 全局峰
    idxs = np.argsort(-sal)[:top_k].tolist()

    # 邻居索引函数
    def neigh(p):
        x, y = p % grid_W, p // grid_W
        out = []
        if x>0: out.append(p-1)
        if x+1<grid_W: out.append(p+1)
        if y>0: out.append(p-grid_W)
        if y+1<grid_H: out.append(p+grid_W)
        return out

    used = np.zeros(P, dtype=bool)
    clusters = []
    for c in idxs:
        if used[c]: continue
        center_val = sal[c]
        mask = np.zeros(P, dtype=bool)
        # BFS 吸收
        q = [c]; mask[c]=True; used[c]=True
        while q:
            u = q.pop()
            for v in neigh(u):
                if not used[v] and sal[v] >= thr_rel*center_val:
                    used[v]=True; mask[v]=True; q.append(v)
        if mask.sum()>=1:
            clusters.append(mask)

    # 聚合实体向量
    Hp_np = to_np(Hp)
    ents = []
    for m in clusters:
        w = (sal[m] + 1e-8)
        z = (Hp_np[m] * (w[:,None]/w.sum())).sum(axis=0)
        z = z / (np.linalg.norm(z)+1e-8)
        ents.append(z)
    Z_ents = np.asarray(ents, dtype=np.float32) if ents else np.zeros((0,Hp.shape[1]), dtype=np.float32)

    # 生成 bbox
    boxes = []
    for m in clusters:
        ys, xs = np.where(m.reshape(grid_H, grid_W))
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        boxes.append((x0, y0, x1, y1))
    return Z_ents, clusters, boxes

# --------------------
# 关系层构造（图像与文本）
# --------------------
def relation_embed_from_entities(Z_ent: np.ndarray, boxes_grid, proj_geom: np.ndarray):
    # 简单几何 + 语义融合
    if len(boxes_grid)==0 or Z_ent.shape[0]==0:
        return np.zeros((0,Z_ent.shape[1]), dtype=np.float32), []

    H = max([b[3] for b in boxes_grid])+1
    W = max([b[2] for b in boxes_grid])+1

    R = []
    rel_info = []
    for i,j in itertools.combinations(range(Z_ent.shape[0]), 2):
        (x0,y0,x1,y1) = boxes_grid[i]
        (u0,v0,u1,v1) = boxes_grid[j]
        cx_i, cy_i = (x0+x1)/2.0, (y0+y1)/2.0
        cx_j, cy_j = (u0+u1)/2.0, (v0+v1)/2.0
        dx, dy = (cx_j - cx_i)/max(1.0,W), (cy_j - cy_i)/max(1.0,H)
        sx = math.log(max(1,(x1-x0))/max(1,(u1-u0)))
        sy = math.log(max(1,(y1-y0))/max(1,(v1-v0)))
        geom = np.array([dx, dy, sx, sy], dtype=np.float32) @ proj_geom  # [d]
        z = l2norm_np(Z_ent[i] + Z_ent[j] + geom[None,:])[0]
        R.append(z); rel_info.append((i,j))
    return np.asarray(R, dtype=np.float32), rel_info

def text_entities_relations_embeds(clip_model, tokenizer, device, caption: str, entities: List[str], relations: List[Tuple[str,str,str]]):
    with torch.no_grad():
        # 实体：名词短语
        if entities:
            toks = tokenizer(entities).to(device)
            z_ent = clip_model.encode_text(toks)
            z_ent = F.normalize(z_ent, dim=-1).cpu().numpy()
        else:
            z_ent = np.zeros((0, clip_model.text_projection.shape[1]), dtype=np.float32)

        # 关系：把 (h,r,t) 拼成短语
        rel_txts = [f"{h} {r} {t}" for (h,r,t) in relations]
        if rel_txts:
            toks2 = tokenizer(rel_txts).to(device)
            z_rel = clip_model.encode_text(toks2)
            z_rel = F.normalize(z_rel, dim=-1).cpu().numpy()
        else:
            z_rel = np.zeros((0, clip_model.text_projection.shape[1]), dtype=np.float32)
    return z_ent, z_rel

# --------------------
# 评测指标（统一空间，前/后）
# --------------------
def centroid_distance(X, Y):
    if X.size==0 or Y.size==0: return float("nan")
    mu_x = X.mean(axis=0); mu_y = Y.mean(axis=0)
    return float(np.linalg.norm(mu_x - mu_y))

def frechet_distance(X, Y, eps=1e-6):
    if X.size==0 or Y.size==0: return float("nan")
    mu_x = X.mean(axis=0); mu_y = Y.mean(axis=0)
    Xc = X - mu_x; Yc = Y - mu_y
    Sx = (Xc.T @ Xc) / max(1, X.shape[0]-1)
    Sy = (Yc.T @ Yc) / max(1, Y.shape[0]-1)
    diff = mu_x - mu_y
    Id = np.eye(Sx.shape[0])
    Sx = Sx + eps*Id; Sy = Sy + eps*Id
    covmean = sqrtm(Sx @ Sy)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff @ diff + np.trace(Sx + Sy - 2*covmean))

def rmg(X, Y, eps=1e-8):
    if X.size==0 or Y.size==0: return float("nan")
    mu_x = X.mean(axis=0); mu_y = Y.mean(axis=0)
    Xc = X - mu_x; Yc = Y - mu_y
    Sx = (Xc**2).sum()/max(1, X.shape[0]-1)
    Sy = (Yc**2).sum()/max(1, Y.shape[0]-1)
    return float(np.linalg.norm(mu_x-mu_y) / math.sqrt(Sx+Sy+eps))

# --------------------
# 可视化：散点 / 覆盖框
# --------------------
def pca_2d_list(list_arrays: List[np.ndarray]):
    X = np.vstack([A for A in list_arrays if A.size>0])
    if X.shape[0] < 2:
        return [np.zeros((A.shape[0],2)) for A in list_arrays]
    p = PCA(n_components=2).fit(X)
    return [p.transform(A) if A.size>0 else np.zeros((0,2)) for A in list_arrays]

def scatter_two(Xa, Xb, la, lb, title, path):
    plt.figure(figsize=(6,6))
    if Xa.size: plt.scatter(Xa[:,0], Xa[:,1], s=30, alpha=0.9, label=la, marker="o")
    if Xb.size: plt.scatter(Xb[:,0], Xb[:,1], s=30, alpha=0.9, label=lb, marker="x")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# --------------------
# 主流程
# --------------------
@dataclass
class Args:
    image: str
    caption: str
    out_dir: str = "fractal_outputs"
    device: str = "cuda"
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    tau_feat: float = 0.5
    num_entities: int = 10         # 每尺度最多实体数（聚合中心上限）
    thr_rel: float = 0.35
    grid_w: Optional[int] = None   # 若 None，按 ViT-B/32@224 -> 7
    grid_h: Optional[int] = None

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--image", required=True)
    pa.add_argument("--caption", required=True)
    pa.add_argument("--out_dir", default="fractal_outputs")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--model_name", default="ViT-B-32")
    pa.add_argument("--pretrained", default="openai")
    pa.add_argument("--tau_feat", type=float, default=0.5)
    pa.add_argument("--num_entities", type=int, default=10)
    pa.add_argument("--thr_rel", type=float, default=0.35)
    pa.add_argument("--grid_w", type=int, default=0)
    pa.add_argument("--grid_h", type=int, default=0)
    args = pa.parse_args()
    cfg = Args(
        image=args.image, caption=args.caption, out_dir=args.out_dir, device=args.device,
        model_name=args.model_name, pretrained=args.pretrained, tau_feat=args.tau_feat,
        num_entities=args.num_entities, thr_rel=args.thr_rel,
        grid_w=(args.grid_w or None), grid_h=(args.grid_h or None)
    )
    ensure_dir(cfg.out_dir)

    device = cfg.device if (cfg.device=="cpu" or torch.cuda.is_available()) else "cpu"

    # 1) 模型与输入
    model, preprocess, tokenizer = build_clip(cfg.model_name, cfg.pretrained, device)
    insp = CLIPIntrospector(model)

    img = Image.open(cfg.image).convert("RGB")
    im_ = preprocess(img).unsqueeze(0).to(device)    # [1,3,H,W]
    toks = tokenizer([cfg.caption]).to(device)       # [1,L]

    # 2) 取全局与 patch/token 表示
    with torch.no_grad():
        z_img_g, H = insp.image_tokens(im_)          # [d], [P,d]
        z_txt_g, Y, mask = insp.text_tokens(toks)    # [d], [L,d], [L]
    d = H.shape[1]

    # ViT-B-32@224 缺省 7x7；B/16@224 是 14x14。我们根据视觉模型 pos embedding 推断网格
    if cfg.grid_w is None or cfg.grid_h is None:
        P = H.shape[0]
        # 尝试平方根（7/14/16/24...）
        s = int(round(math.sqrt(P)))
        gw = gh = s if s*s==P else (P,1)
    else:
        gw, gh = cfg.grid_w, cfg.grid_h

    # 3) 图像侧 patch 图 + 文本条件显著性 + 多尺度扩散（S=1,2,4,8）
    Ahat = build_patch_graph(torch.from_numpy(to_np(H)).to(device), gw, gh, tau_feat=cfg.tau_feat)
    Ahat = Ahat  # numpy
    s0 = text_conditioned_saliency(H, z_img_g, z_txt_g)          # [P]
    scales = (1,2,4,8)
    S = multi_scale_diffusion(Ahat, s0, scales=scales)           # list of (r, s_r)

    # 4) 各尺度做实体聚合（取 top_k 峰 + 连通聚合）
    #    把多个尺度的实体集合并（去重），形成 Level-2 的实体/部件节点
    ZI_ents = []
    clusters_all = []
    boxes_all = []
    for r, s_r in S:
        Zr, Clr, Bxr = peak_and_aggregate(H, s_r, gw, gh, top_k=cfg.num_entities, thr_rel=cfg.thr_rel)
        if Zr.size:
            ZI_ents.append(Zr)
            clusters_all += Clr
            boxes_all += Bxr
    ZI_ents = np.vstack(ZI_ents) if ZI_ents else np.zeros((0,d), dtype=np.float32)

    # 5) 文本侧解析（名词短语、依存关系） + 嵌入
    txt_entities, txt_rel_trip = parse_text_entities_relations(cfg.caption)
    ZT_ent_b, ZT_rel_b = text_entities_relations_embeds(model, tokenizer, device, cfg.caption, txt_entities, txt_rel_trip)

    # 6) 关系层（图像）：从实体对构造
    #    用一个固定的几何投影（高斯随机）把几何注入关系
    rng = np.random.default_rng(2025)
    proj_geom = rng.standard_normal((4, d)).astype(np.float32) * 0.05
    ZI_rel, rel_pairs = relation_embed_from_entities(ZI_ents, boxes_all, proj_geom)

    # 7) 文本侧全局向量
    ZI_g = z_img_g.unsqueeze(0).cpu().numpy()
    ZT_g_b = z_txt_g.unsqueeze(0).cpu().numpy()

    # 8) Fractal 对齐（我们不移动 embedding，只在匹配时使用）
    #    先做“实体层”匈牙利匹配取锚点，再做一个最小刚体以便可视化对齐前后（embedding 不被修改，用作 AFTER 可视化）
    def orth_procrustes(X, Y):
        Xc = X - X.mean(axis=0, keepdims=True); Yc = Y - Y.mean(axis=0, keepdims=True)
        U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
        R = U @ Vt; t = Y.mean(axis=0) - X.mean(axis=0) @ R
        return R, t

    if ZI_ents.size and ZT_ent_b.size:
        S_ent = cosine_np(ZT_ent_b, ZI_ents)
        ri, cj = linear_sum_assignment(-S_ent)
        A_t, A_i = [], []
        for r_idx, c_idx in zip(ri, cj):
            if S_ent[r_idx, c_idx] > 0:  # 只保留正相似
                A_t.append(ZT_ent_b[r_idx]); A_i.append(ZI_ents[c_idx])
        if len(A_t)==0:
            A_t = [ZT_g_b[0]]; A_i = [ZI_g[0]]
        else:
            A_t.append(ZT_g_b[0]); A_i.append(ZI_g[0])
        A_t = np.vstack(A_t); A_i = np.vstack(A_i)
        R, t = orth_procrustes(A_t, A_i)
        def map_text(Z): 
            return l2norm_np(Z @ R + t, -1) if Z.size else Z
        ZT_ent_a = map_text(ZT_ent_b)
        ZT_rel_a = map_text(ZT_rel_b)
        ZT_g_a   = map_text(ZT_g_b)
    else:
        ZT_ent_a, ZT_rel_a, ZT_g_a = ZT_ent_b, ZT_rel_b, ZT_g_b

    # 9) 评测指标（Before/After），层：global / entities / relations / all
    def pack_metrics(name, Xi, Xt_b, Xt_a):
        return {
            name: {
                "CD": {"before": centroid_distance(Xi,Xt_b), "after": centroid_distance(Xi,Xt_a)},
                "FD": {"before": frechet_distance(Xi,Xt_b),   "after": frechet_distance(Xi,Xt_a)},
                "RMG":{"before": rmg(Xi,Xt_b),                "after": rmg(Xi,Xt_a)}
            }
        }

    metrics = {}
    Xi_all = np.vstack([ZI_g, ZI_ents, ZI_rel]) if (ZI_ents.size or ZI_rel.size) else ZI_g
    Xt_b_all = np.vstack([ZT_g_b, ZT_ent_b, ZT_rel_b]) if (ZT_ent_b.size or ZT_rel_b.size) else ZT_g_b
    Xt_a_all = np.vstack([ZT_g_a, ZT_ent_a, ZT_rel_a]) if (ZT_ent_a.size or ZT_rel_a.size) else ZT_g_a

    metrics.update(pack_metrics("global",   ZI_g,     ZT_g_b,   ZT_g_a))
    metrics.update(pack_metrics("entities", ZI_ents,  ZT_ent_b, ZT_ent_a))
    metrics.update(pack_metrics("relations",ZI_rel,   ZT_rel_b, ZT_rel_a))
    metrics.update(pack_metrics("all",      Xi_all,   Xt_b_all, Xt_a_all))

    # 10) 匹配输出（实体层与关系层）
    matches = {"entities": [], "relations": []}
    if ZI_ents.size and ZT_ent_b.size:
        S_ent_b = cosine_np(ZT_ent_b, ZI_ents)
        ri_b, cj_b = linear_sum_assignment(-S_ent_b)
        for r_idx,c_idx in zip(ri_b,cj_b):
            matches["entities"].append({
                "text": txt_entities[r_idx] if r_idx < len(txt_entities) else f"text_ent_{r_idx}",
                "image_ent_idx": int(c_idx),
                "similarity_before": float(S_ent_b[r_idx, c_idx]),
            })
        if ZT_ent_a.size:
            S_ent_a = cosine_np(ZT_ent_a, ZI_ents)
            for k,m in enumerate(matches["entities"]):
                r_idx, c_idx = ri_b[k], m["image_ent_idx"]
                m["similarity_after"] = float(S_ent_a[r_idx, c_idx])

    if ZI_rel.size and ZT_rel_b.size:
        S_rel_b = cosine_np(ZT_rel_b, ZI_rel)
        ri_b, cj_b = linear_sum_assignment(-S_rel_b)
        rel_texts = [f"{h} {r} {t}" for (h,r,t) in txt_rel_trip]
        for r_idx,c_idx in zip(ri_b,cj_b):
            matches["relations"].append({
                "text": rel_texts[r_idx] if r_idx < len(rel_texts) else f"text_rel_{r_idx}",
                "image_rel_pair": rel_pairs[c_idx] if c_idx < len(rel_pairs) else [-1,-1],
                "similarity_before": float(S_rel_b[r_idx, c_idx]),
            })
        if ZT_rel_a.size:
            S_rel_a = cosine_np(ZT_rel_a, ZI_rel)
            for k,m in enumerate(matches["relations"]):
                r_idx, c_idx = ri_b[k], k
                m["similarity_after"] = float(S_rel_a[r_idx, cj_b[k]])

    # 11) 可视化：PCA-2D（前/后）
    Xi_e_2d, Xt_e_b_2d, Xt_e_a_2d = pca_2d_list([ZI_ents, ZT_ent_b, ZT_ent_a])
    Xi_r_2d, Xt_r_b_2d, Xt_r_a_2d = pca_2d_list([ZI_rel, ZT_rel_b, ZT_rel_a])
    scatter_two(Xi_e_2d, Xt_e_b_2d, "Image entities", "Text entities (before)",
                "Entities BEFORE", os.path.join(cfg.out_dir, "entities_before.png"))
    scatter_two(Xi_e_2d, Xt_e_a_2d, "Image entities", "Text entities (after)",
                "Entities AFTER", os.path.join(cfg.out_dir, "entities_after.png"))
    scatter_two(Xi_r_2d, Xt_r_b_2d, "Image relations", "Text relations (before)",
                "Relations BEFORE", os.path.join(cfg.out_dir, "relations_before.png"))
    scatter_two(Xi_r_2d, Xt_r_a_2d, "Image relations", "Text relations (after)",
                "Relations AFTER", os.path.join(cfg.out_dir, "relations_after.png"))

    # 12) 图像覆盖显示（把实体簇的网格框映射到原图像坐标）
    #     计算网格到像素的放缩：
    Wimg, Himg = img.size
    cell_w = Wimg // gw
    cell_h = Himg // gh
    boxes_px = []
    labels = []
    # 为每个实体挑一个最佳匹配的文本（AFTER）
    best_txt_for_ent = [""] * len(boxes_all)
    if ZI_ents.size and ZT_ent_a.size:
        S_ea = cosine_np(ZI_ents, ZT_ent_a).T  # [N_txt, N_img] -> transpose => [N_img, N_txt]
        for i in range(len(boxes_all)):
            if S_ea.shape[0]==0: continue
            j = int(np.argmax(S_ea[i]))
            lab = txt_entities[j] if j < len(txt_entities) else f"t{j}"
            best_txt_for_ent[i] = lab

    for (x0,y0,x1,y1), lab in zip(boxes_all, best_txt_for_ent):
        px0, py0 = x0*cell_w, y0*cell_h
        px1, py1 = (x1+1)*cell_w-1, (y1+1)*cell_h-1
        boxes_px.append((px0,py0,px1,py1))
        labels.append(lab)

    draw_boxes(img, boxes_px, labels, fill_color=(0,255,0,40), outline=(0,255,0), w=3)\
        .save(os.path.join(cfg.out_dir, "image_entities_overlay.png"))

    # 13) 输出 JSON
    with open(os.path.join(cfg.out_dir, "matches.json"), "w") as f:
        json.dump({
            "caption": cfg.caption,
            "text_entities": txt_entities,
            "text_relations": txt_rel_trip,
            "entity_matches": matches["entities"],
            "relation_matches": matches["relations"]
        }, f, indent=2)

    with open(os.path.join(cfg.out_dir, "manifold_stats.json"), "w") as f:
        json.dump({
            "grid": {"W": gw, "H": gh, "P": gw*gh},
            "scales": list(scales),
            "num_image_entities": int(ZI_ents.shape[0]),
            "num_image_relations": int(ZI_rel.shape[0]),
            "metrics": metrics
        }, f, indent=2)

    print("[DONE] Outputs saved to:", cfg.out_dir)
    print(" - image_entities_overlay.png")
    print(" - entities_before.png / entities_after.png")
    print(" - relations_before.png / relations_after.png")
    print(" - matches.json / manifold_stats.json")

if __name__ == "__main__":
    main()
