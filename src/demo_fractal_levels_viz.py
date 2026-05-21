import os, re, math, argparse, json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import open_clip
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def l2_normalize(x, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def cosine_sim(A, B):
    A = l2_normalize(A, -1)
    B = l2_normalize(B, -1)
    return A @ B.T

def draw_boxes(img: Image.Image, boxes: List[Tuple[int,int,int,int]], outline=(255,0,0), width=3):
    img2 = img.copy()
    d = ImageDraw.Draw(img2)
    for (x0,y0,x1,y1) in boxes:
        d.rectangle([x0,y0,x1,y1], outline=outline, width=width)
    return img2

# -----------------------------
# Fractal diffusion kernel (simplified)
# -----------------------------
class FractalKernel:
    def __init__(self, num_scales=6, r_min=0.1, r_max=3.0, Q=3.0):
        self.num_scales = num_scales
        self.r = np.exp(np.linspace(np.log(r_min), np.log(r_max), num_scales)).astype(np.float32)
        self.alpha = 1.0 + 0.5 * Q  # spectral dimension -> exponent
        w = self.r ** (-self.alpha)
        self.w = w / (w.sum() + 1e-8)

    def kernel(self, Z, Zp):
        # Z, Zp: (n,d), (m,d)
        # use ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        Z2 = (Z**2).sum(axis=1, keepdims=True)        # (n,1)
        Zp2 = (Zp**2).sum(axis=1, keepdims=True).T    # (1,m)
        dot = Z @ Zp.T                                 # (n,m)
        sq = Z2 + Zp2 - 2.0*dot
        K = np.zeros_like(sq, dtype=np.float32)
        for k in range(self.num_scales):
            rk = self.r[k]
            K += self.w[k] * np.exp(- sq / (4.0 * rk + 1e-8))
        return K

    def sim(self, Z, Zp):
        # similarity = - fractal distance ~ 2*K - const
        # const = 2*sum w_k = 2
        K = self.kernel(Z, Zp)
        return 2.0*K - 2.0

# -----------------------------
# Image multi-scale crops
# -----------------------------
@dataclass
class CropConfig:
    grid: int
    crop_ratio: float   # relative to shorter side

def generate_crops(img: Image.Image, cfg: CropConfig) -> Tuple[List[Image.Image], List[Tuple[int,int,int,int]]]:
    W, H = img.size
    s = min(W, H)
    crop_size = int(s * cfg.crop_ratio)
    crop_size = max(16, crop_size)
    # stride so that grid^2 crops roughly cover image
    if cfg.grid <= 1:
        xs = [W//2 - crop_size//2]
        ys = [H//2 - crop_size//2]
    else:
        xs = np.linspace(0, W - crop_size, cfg.grid).astype(int).tolist()
        ys = np.linspace(0, H - crop_size, cfg.grid).astype(int).tolist()
    crops, boxes = [], []
    for y in ys:
        for x in xs:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + crop_size, y0 + crop_size
            x1 = min(x1, W); y1 = min(y1, H)
            x0 = x1 - crop_size; y0 = y1 - crop_size
            patch = img.crop((x0,y0,x1,y1))
            crops.append(patch)
            boxes.append((x0,y0,x1,y1))
    return crops, boxes

# -----------------------------
# Text multi-scale phrases
# -----------------------------
def tokenize_levels(caption: str):
    # Level-1: words likely describing local attributes/parts
    cap = caption.strip()
    # 简单规则：拆词后，保留含字母的 token；挑出可能是局部属性的词（hair, hand, shirt, yard 等）
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", cap)
    tokens_lower = [t.lower() for t in tokens]

    # 一些常见局部词的优先表（可扩展）
    l1_keywords = {"hair","hand","hands","shirt","yard","guy","guys","man","men","boy","boys","girl","girls","hat","shoe","shoes","arm","arms","face","head"}
    level1 = [t for t in tokens if t.lower() in l1_keywords]
    if not level1:
        # 兜底：选长度较短的名词样式词
        level1 = [t for t in tokens if len(t) <= 5][:6]

    # Level-2: 短语（用逗号/and/with/between 等切分；再过滤过短片段）
    # 简单启发式切分
    phrases = re.split(r"(?:,| and | with | while )", cap)
    level2 = [p.strip() for p in phrases if len(p.strip().split()) >= 2]
    level2 = level2[:6] if len(level2) > 6 else level2

    # Level-3: 整句
    level3 = [cap]

    return level1, level2, level3

# -----------------------------
# Encode with CLIP
# -----------------------------
def build_clip(model_name="ViT-B-32", pretrained="openai", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer

def encode_image_list(model, preprocess, device, images: List[Image.Image], batch_size=64):
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            ims = torch.stack([preprocess(im) for im in batch]).to(device)
            f = model.encode_image(ims)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)

def encode_text_list(model, tokenizer, device, texts: List[str], batch_size=128):
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            toks = tokenizer(texts[i:i+batch_size]).to(device)
            f = model.encode_text(toks)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)

# -----------------------------
# Alignment (Procrustes)
# -----------------------------
def orthogonal_procrustes(X, Y):
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    t = Y.mean(axis=0) - X.mean(axis=0) @ R
    return R, t

# -----------------------------
# Plotting utils
# -----------------------------
def scatter_2d(X_img, X_txt, labels_img, labels_txt, title, path):
    plt.figure(figsize=(6,6))
    plt.scatter(X_img[:,0], X_img[:,1], s=14, alpha=0.75, label=labels_img)
    plt.scatter(X_txt[:,0], X_txt[:,1], s=14, alpha=0.75, marker="x", label=labels_txt)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def connect_top_pairs(XA, XB, sim_mat, topk=10, title="", path="pair.png"):
    # XA: (n,2), XB:(m,2), sim_mat:(n,m)
    plt.figure(figsize=(6,6))
    plt.scatter(XA[:,0], XA[:,1], s=12, alpha=0.8, label="A")
    plt.scatter(XB[:,0], XB[:,1], s=12, alpha=0.8, marker="x", label="B")
    # 画 topk 条最大相似度连线
    idxs = np.dstack(np.unravel_index(np.argsort(sim_mat.ravel())[::-1][:topk], sim_mat.shape))[0]
    for i,j in idxs:
        plt.plot([XA[i,0], XB[j,0]], [XA[i,1], XB[j,1]], linewidth=1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--out_dir", default="outputs_demo")
    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # 0) Load CLIP
    device = args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu"
    model, preprocess, tokenizer = build_clip(args.clip_model, args.clip_pretrained, device)

    # 1) Load image + caption
    img = Image.open(args.image).convert("RGB")
    caption = args.caption.strip()

    # 2) Build multi-scale crops for image: L1/L2/L3
    # L1: small patches (fine)    ; L2: medium patches (relations); L3: full image (global)
    L1_cfg = CropConfig(grid=6, crop_ratio=0.20)
    L2_cfg = CropConfig(grid=3, crop_ratio=0.45)
    # L3: single full crop
    crops_L1, boxes_L1 = generate_crops(img, L1_cfg)
    crops_L2, boxes_L2 = generate_crops(img, L2_cfg)
    crops_L3, boxes_L3 = [img.copy()], [(0,0,img.size[0], img.size[1])]

    # 3) Text multi-scale: L1 words, L2 phrases, L3 full sentence
    T1_words, T2_phrases, T3_full = tokenize_levels(caption)

    # 4) Encode with CLIP
    feats_i_L1 = encode_image_list(model, preprocess, device, crops_L1)
    feats_i_L2 = encode_image_list(model, preprocess, device, crops_L2)
    feats_i_L3 = encode_image_list(model, preprocess, device, crops_L3)

    feats_t_L1 = encode_text_list(model, tokenizer, device, T1_words if T1_words else ["."])
    feats_t_L2 = encode_text_list(model, tokenizer, device, T2_phrases if T2_phrases else ["."])
    feats_t_L3 = encode_text_list(model, tokenizer, device, T3_full)

    # 5) “构建前”——各自空间降维（PCA）
    pca_img = PCA(n_components=2).fit(np.vstack([feats_i_L1, feats_i_L2, feats_i_L3]))
    pca_txt = PCA(n_components=2).fit(np.vstack([feats_t_L1, feats_t_L2, feats_t_L3]))

    Xi_before = pca_img.transform(np.vstack([feats_i_L1, feats_i_L2, feats_i_L3]))
    Xt_before = pca_txt.transform(np.vstack([feats_t_L1, feats_t_L2, feats_t_L3]))

    n1_i, n2_i, n3_i = len(feats_i_L1), len(feats_i_L2), len(feats_i_L3)
    n1_t, n2_t, n3_t = len(feats_t_L1), len(feats_t_L2), len(feats_t_L3)

    Xi_L1_b = Xi_before[:n1_i]
    Xi_L2_b = Xi_before[n1_i:n1_i+n2_i]
    Xi_L3_b = Xi_before[n1_i+n2_i:]

    Xt_L1_b = Xt_before[:n1_t]
    Xt_L2_b = Xt_before[n1_t:n1_t+n2_t]
    Xt_L3_b = Xt_before[n1_t+n2_t:]

    # Scatter before
    scatter_2d(
        np.vstack([Xi_L1_b, Xi_L2_b, Xi_L3_b]),
        np.vstack([Xt_L1_b, Xt_L2_b, Xt_L3_b]),
        "Image (L1+L2+L3)", "Text (L1+L2+L3)",
        "BEFORE: Raw Spaces (modality gap)",
        os.path.join(args.out_dir, "01_before_gap.png")
    )

    # 6) “构建后”：先把文本特征映射到图像空间（正交 Procrustes）
    #    （模拟我们论文里的“多尺度对齐思想”的几何版示意）
    # 用 L3（全局）对齐，再验证 L2/L1 的重合度
    R, t = orthogonal_procrustes(feats_t_L3, feats_i_L3)
    def align_txt(feats_t):
        return (feats_t @ R) + t

    feats_t_L1_a = align_txt(feats_t_L1)
    feats_t_L2_a = align_txt(feats_t_L2)
    feats_t_L3_a = align_txt(feats_t_L3)

    Xi_after = pca_img.transform(np.vstack([feats_i_L1, feats_i_L2, feats_i_L3]))
    Xt_after = pca_img.transform(np.vstack([feats_t_L1_a, feats_t_L2_a, feats_t_L3_a]))

    Xt_L1_a = Xt_after[:n1_t]
    Xt_L2_a = Xt_after[n1_t:n1_t+n2_t]
    Xt_L3_a = Xt_after[n1_t+n2_t:]

    scatter_2d(
        np.vstack([Xi_L1_b, Xi_L2_b, Xi_L3_b]),
        np.vstack([Xt_L1_a, Xt_L2_a, Xt_L3_a]),
        "Image (L1+L2+L3)", "Text (aligned L1+L2+L3)",
        "AFTER: Shared (fractal) manifold alignment",
        os.path.join(args.out_dir, "02_after_alignment.png")
    )

    # 7) 分形扩散核匹配（多尺度）
    fk_small = FractalKernel(num_scales=6, r_min=0.05, r_max=0.5, Q=2.5)  # 偏局部
    fk_mid   = FractalKernel(num_scales=6, r_min=0.1,  r_max=1.2, Q=3.0)  # 中等
    fk_large = FractalKernel(num_scales=6, r_min=0.3,  r_max=3.0, Q=3.5)  # 偏全局

    # Level-1: 局部词 ↔ 小裁剪
    S_L1 = fk_small.sim(feats_t_L1_a, feats_i_L1)  # (n_t1, n_i1)
    # Level-2: 短语 ↔ 中裁剪
    S_L2 = fk_mid.sim(feats_t_L2_a, feats_i_L2)
    # Level-3: 整句 ↔ 全图
    S_L3 = fk_large.sim(feats_t_L3_a, feats_i_L3)

    # 8) 可视化：在降维后的图上连接 top 匹配
    Xi_L1_2d = Xi_L1_b
    Xt_L1_2d = pca_img.transform(feats_t_L1_a)
    connect_top_pairs(Xt_L1_2d, Xi_L1_2d, S_L1, topk=min(15, S_L1.size), title="Level-1 (local) matches", path=os.path.join(args.out_dir, "03_level1_pairs.png"))

    Xi_L2_2d = Xi_L2_b
    Xt_L2_2d = pca_img.transform(feats_t_L2_a)
    connect_top_pairs(Xt_L2_2d, Xi_L2_2d, S_L2, topk=min(12, S_L2.size), title="Level-2 (phrase/relations) matches", path=os.path.join(args.out_dir, "04_level2_pairs.png"))

    # L3 基本是一对一（句子 ↔ 全图），直接输出分数
    with open(os.path.join(args.out_dir, "05_level3_score.txt"), "w") as f:
        f.write(f"Level-3 sentence-to-image fractal similarity: {float(S_L3[0,0]):.4f}\n")
        f.write(f"Caption: {caption}\n")

    # 9) 在原图上高亮 L1/L2 的显著区域
    #    取每个文本元素在对应尺度的 top-k 裁剪，合并去重画框
    def top_boxes(sim, boxes, k_each=2, max_total=12):
        chosen = []
        for i in range(sim.shape[0]):
            idx = np.argsort(-sim[i])[:k_each]
            for j in idx:
                chosen.append(tuple(boxes[j]))
        # 去重
        uniq = []
        seen = set()
        for b in chosen:
            if b not in seen:
                uniq.append(b); seen.add(b)
        return uniq[:max_total]

    boxes_L1_top = top_boxes(S_L1, boxes_L1, k_each=2, max_total=12)
    boxes_L2_top = top_boxes(S_L2, boxes_L2, k_each=1, max_total=8)

    vis_L1 = draw_boxes(img, boxes_L1_top, outline=(255,0,0), width=3)
    vis_L2 = draw_boxes(img, boxes_L2_top, outline=(0,255,0), width=3)

    vis_L1.save(os.path.join(args.out_dir, "06_img_L1_highlight.png"))
    vis_L2.save(os.path.join(args.out_dir, "07_img_L2_highlight.png"))

    # 10) 文本侧：把 L1/L2 对应的词/短语列出来（与可视化对应）
    with open(os.path.join(args.out_dir, "08_text_levels.json"), "w") as f:
        json.dump({
            "Level-1 words (local)": T1_words,
            "Level-2 phrases (relations)": T2_phrases,
            "Level-3 sentence": T3_full
        }, f, indent=2)

    # 11) Gap 直观量化：对齐前后，文本与图像 Lk 的质心距离（PCA 空间）
    def centroid(X):
        return X.mean(axis=0, keepdims=True)

    before_dist = {
        "L1": float(np.linalg.norm(centroid(Xi_L1_b) - centroid(Xt_L1_b))),
        "L2": float(np.linalg.norm(centroid(Xi_L2_b) - centroid(Xt_L2_b))),
        "L3": float(np.linalg.norm(centroid(Xi_L3_b) - centroid(Xt_L3_b))),
    }
    after_dist = {
        "L1": float(np.linalg.norm(centroid(Xi_L1_b) - centroid(Xt_L1_a))),
        "L2": float(np.linalg.norm(centroid(Xi_L2_b) - centroid(Xt_L2_a))),
        "L3": float(np.linalg.norm(centroid(Xi_L3_b) - centroid(Xt_L3_a))),
    }
    with open(os.path.join(args.out_dir, "09_gap_reduction.json"), "w") as f:
        json.dump({"before": before_dist, "after": after_dist}, f, indent=2)

    print("\n[Done] Outputs saved to", args.out_dir)
if __name__ == "__main__":
    main()
