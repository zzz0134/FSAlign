# src/eval_retrieval.py
import os, math, torch, numpy as np
from tqdm import tqdm
from .config import get_eval_args
from .utils.common import load_embeddings
from .models.heads import ProjectionHead
from .models.fractal_kernel import FractalKernel

def build_ground_truth(img_ids, txt_ids):
    fn2img = {fn:i for i,fn in enumerate(img_ids)}
    txt_to_img = [fn2img[meta['image_fn']] for meta in txt_ids]
    img_to_txt = {}
    for j, meta in enumerate(txt_ids):
        i = fn2img[meta['image_fn']]
        img_to_txt.setdefault(i, []).append(j)
    return txt_to_img, img_to_txt

@torch.inference_mode()
def fractal_kernel_block(Zq, Zg, fk):
    """
    计算 fractal 核：K(z,z') = sum_k w_k * exp(-(||z||^2 + ||z'||^2 - 2 z z'^T)/(4 r_k))
    仅使用 2D 矩阵，不产生 (n,m,d) 的 3D 张量。返回 (n_query, n_gallery)。
    """
    # 预计算范数
    q2 = (Zq**2).sum(dim=1).unsqueeze(1)      # (nq,1)
    g2 = (Zg**2).sum(dim=1).unsqueeze(0)      # (1,ng)
    # 分块 matmul，避免一次性巨大矩阵
    # 这里返回整块矩阵，调用方再做更细粒度控制
    # 注意：为了节省显存，尽量让 Zg 常驻 GPU，Zq 分块送入
    nq, ng = Zq.size(0), Zg.size(0)
    device = Zq.device
    K = torch.zeros(nq, ng, device=device)
    w = fk.weights()           # (K,)
    r = fk.r                   # (K,)
    # 我们一次性做完所有尺度：先算出 dot，再在标量尺度上逐层加和
    # (Zq @ Zg.T) 仍然很大 -> 必须分块
    # 经验：按 query 维度切块，gallery 常驻
    step = max(1, 2048 if nq*ng <= 80_000_000 else 512)  # 自适应一点
    for a in range(0, nq, step):
        b = min(a + step, nq)
        # (b-a, ng)
        dot = Zq[a:b] @ Zg.t()
        sq = q2[a:b] + g2 - 2.0 * dot
        # 按尺度聚合
        K_block = torch.zeros_like(dot)
        for k in range(fk.num_scales):
            K_block += w[k] * torch.exp(- sq / (4.0 * r[k] + 1e-8))
        K[a:b] = K_block
        del dot, sq, K_block
        torch.cuda.empty_cache()
    return K

@torch.inference_mode()
def dfrac_point_point_streaming(Zq, Zg, fk, block_q=512):
    """
    返回 -D_frac 的相似度矩阵，但**不要**一次性保留全矩阵：
    调用方可按需处理（如只做 topK 或 rank 计数）。
    这里提供一个生成器风格的接口：分块产出 (idx_slice, S_block)。
    """
    wsum = fk.weights().sum()
    nq = Zq.size(0)
    for a in range(0, nq, block_q):
        b = min(a + block_q, nq)
        K_block = fractal_kernel_block(Zq[a:b], Zg, fk)        # (b-a, ng)
        # D = 2*wsum - 2*K
        S_block = -(2.0 * wsum - 2.0 * K_block)                # 相似度 = -D
        yield slice(a, b), S_block

def recall_at_k_streaming_text2image(Zt, Zi, fk, txt_to_img, Ks=(1,5,10), block_q=256):
    """
    只统计 R@K / MedR / MeanR，而不是存完整相似度矩阵。
    算法：
      1) 先算每个文本与 GT 图片的相似度 S_gt（一次性向量，内存小）
      2) 再遍历 gallery 的分块相似度，统计每个文本比 S_gt 更高的候选个数 -> 得到 rank
    """
    device = Zi.device
    # step0: 先把 Zt 分块映射到 shared space（如果已是投影后则略）
    # 这里假定 Zt/Zi 已经是 projection head 的输出
    n_txt = Zt.size(0)
    n_img = Zi.size(0)

    # 第1步：计算每个文本对其 GT 图片的分形核相似度
    # 构建索引 (j, txt_to_img[j])，用批量 gather 方式避免全矩阵
    gt_idx = torch.tensor(txt_to_img, device=device, dtype=torch.long)   # (n_txt,)
    # 分块计算 S_gt
    S_gt = torch.empty(n_txt, device=device)
    step = 1024
    for a in tqdm(range(0, n_txt, step), desc="T2I: S_gt"):
        b = min(a + step, n_txt)
        Kj = fractal_kernel_block(Zt[a:b], Zi[gt_idx[a:b]], fk)    # (b-a, 1)
        wsum = fk.weights().sum()
        Sj = -(2.0 * wsum - 2.0 * Kj.squeeze(1))                  # -D = - (2wsum - 2K)
        S_gt[a:b] = Sj
        del Kj, Sj
        torch.cuda.empty_cache()

    # 第2步：遍历 gallery，并统计每个文本有多少图片的分数 > S_gt
    higher_counts = torch.zeros(n_txt, dtype=torch.int64, device=device)
    g_step = 2000  # 视显存调小/大
    for c in tqdm(range(0, n_img, g_step), desc="T2I: scan gallery"):
        d = min(c + g_step, n_img)
        # 计算 Zt vs Zi[c:d] 的相似度分块，逐个文本与 S_gt 比较
        for slc, S_block in dfrac_point_point_streaming(Zt, Zi[c:d], fk, block_q=block_q):
            # S_block: (len(slc), d-c)
            # 与对应的 S_gt[slc] 比较
            comp = (S_block > S_gt[slc].unsqueeze(1)).sum(dim=1)   # 每个文本此块里有多少更高
            higher_counts[slc] += comp.to(torch.int64)
            del S_block, comp
        torch.cuda.empty_cache()

    # rank = 1 + higher_counts
    ranks = (higher_counts + 1).to(torch.int64).cpu().numpy()
    recalls = {f"R@{K}": float((ranks <= K).mean()) for K in Ks}
    recalls["MedR"] = float(np.median(ranks))
    recalls["MeanR"] = float(np.mean(ranks))
    return recalls

def recall_at_k_streaming_image2text(Zi, Zt, fk, img_to_txt, Ks=(1,5,10), block_q=128, topK_cap=1000):
    """
    I2T 的 rank 计算略复杂，因为每张图有 5 个 GT 文本。
    我们采用**两阶段**：
      1) 先算每张图与其5个GT文本中的最大 S_gt_max（更“乐观”的 GT 分数）
      2) 流式扫描所有文本块，统计有多少文本的得分 > S_gt_max
    为了效率，这里对文本端采用 topK 裁剪（可选），但为保证准确 rank，我们仍逐块比较所有文本。
    """
    device = Zt.device
    n_img = Zi.size(0)
    n_txt = Zt.size(0)
    # 1) 计算每张图对其GT文本集合的最大 S_gt
    S_gt_max = torch.full((n_img,), -1e30, device=device)
    wsum = fk.weights().sum()
    for i, gts in tqdm(enumerate(img_to_txt), total=n_img, desc="I2T: S_gt"):
        if not gts:
            continue
        P = Zt[torch.tensor(gts, device=device, dtype=torch.long)]  # (5,d)
        Ki = fractal_kernel_block(Zi[i:i+1], P, fk)                 # (1,5)
        S = -(2.0 * wsum - 2.0 * Ki.squeeze(0))
        S_gt_max[i] = torch.max(S)
        del P, Ki, S
        if i % 1000 == 0:
            torch.cuda.empty_cache()

    # 2) 扫描所有文本，统计每张图有多少文本 > S_gt_max
    higher_counts = torch.zeros(n_img, dtype=torch.int64, device=device)
    t_step = 2000
    for c in tqdm(range(0, n_txt, t_step), desc="I2T: scan texts"):
        d = min(c + t_step, n_txt)
        for slc, S_block in dfrac_point_point_streaming(Zi, Zt[c:d], fk, block_q=block_q):
            # S_block: (len(slc), d-c) 对每张图，这一块里有多少文本更高
            comp = (S_block > S_gt_max[slc].unsqueeze(1)).sum(dim=1)
            higher_counts[slc] += comp.to(torch.int64)
            del S_block, comp
        torch.cuda.empty_cache()

    ranks = (higher_counts + 1).to(torch.int64).cpu().numpy()
    recalls = {f"R@{K}": float((ranks <= K).mean()) for K in Ks}
    recalls["MedR"] = float(np.median(ranks))
    recalls["MeanR"] = float(np.mean(ranks))
    return recalls

def main():
    args = get_eval_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    img_feats, txt_feats, img_ids, txt_ids, splits = load_embeddings(args.emb_dir)
    txt_to_img, img_to_txt = build_ground_truth(img_ids, txt_ids)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    d_img = img_feats.size(1)
    d_txt = txt_feats.size(1)
    proj_dim = ckpt['args']['proj_dim']

    img_head = ProjectionHead(d_img, out_dim=proj_dim).to(device)
    txt_head = ProjectionHead(d_txt, out_dim=proj_dim).to(device)
    img_head.load_state_dict(ckpt['img_head']); img_head.eval()
    txt_head.load_state_dict(ckpt['txt_head']); txt_head.eval()

    fk = FractalKernel(num_scales=args.num_scales, r_min=args.r_min, r_max=args.r_max,
                       alpha_mode=args.alpha_mode, alpha_fixed=args.alpha_fixed,
                       learn_Q=False, Q_init=ckpt['fk'].get('Q', 3.0), device=device).to(device)
    fk.load_state_dict(ckpt['fk'], strict=False); fk.eval()

    with torch.inference_mode():
        Zi = img_head(img_feats.to(device))
        Zt = txt_head(txt_feats.to(device))

        # 可按显存调整 block 大小
        r_ti = recall_at_k_streaming_text2image(Zt, Zi, fk, txt_to_img, Ks=(1,5,10), block_q=512)
        r_it = recall_at_k_streaming_image2text(Zi, Zt, fk, img_to_txt, Ks=(1,5,10), block_q=256)

    print('Text->Image:', r_ti)
    print('Image->Text:', r_it)

if __name__ == '__main__':
    main()
