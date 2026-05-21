#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stable version:
Local Flickr30k -> CLIP embeddings -> baseline UMAP -> train OUR fractal alignment (J_frac + J_match) -> after UMAP

Key stability fixes:
  - Keep self-loop in W (W_ii = self_loop_weight, default 1.0)
  - Safe eigendecomposition: float64 + (symmetrize + diagonal jitter + retry)
  - Optional spectral computations on CPU (default) to avoid GPU eigh non-convergence

Run:
  python fk30k_fractal_align_umap_stable.py \
    --root /work/was598/modilty_gap/tools/data/flickr30k \
    --split test --n 400 \
    --epochs 200 --lr 1e-3 \
    --dtype float64 \
    --spectral_device cpu \
    --out_prefix fk30k_fractal_stable

Outputs:
  {out_prefix}_baseline_umap.png
  {out_prefix}_after_umap.png
  {out_prefix}_ckpt.pt
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import umap
from transformers import CLIPProcessor, CLIPModel


# -----------------------------
# Dataset utilities (local Flickr30k)
# -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_dir(root: Path) -> Path:
    candidates = []
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        cnt = 0
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                cnt += 1
        if cnt > 0:
            candidates.append((cnt, d))
    if not candidates:
        raise FileNotFoundError(f"No images found under: {root}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_caption_file(root: Path) -> Path:
    token = list(root.rglob("*.token"))
    if token:
        token.sort(key=lambda p: p.stat().st_size, reverse=True)
        return token[0]
    patterns = ["*caption*.txt", "*captions*.txt", "*.csv", "*.json"]
    cands = []
    for pat in patterns:
        cands += list(root.rglob(pat))
    cands = [p for p in cands if p.is_file() and p.stat().st_size > 0]
    if not cands:
        raise FileNotFoundError(
            f"No caption file found under: {root}\n"
            "Expected something like results_20130124.token (image.jpg#k<TAB>caption)."
        )
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def parse_captions(caption_path: Path):
    """
    Returns dict[image_basename] -> list[captions]
    Supports:
      *.token : image.jpg#0<TAB>caption
      *.csv   : tries to find columns containing image + caption/text/sentence
      *.json  : best-effort for {image, caption} structures
      others  : heuristics for "img<TAB>cap" or "img|cap"
    """
    ext = caption_path.suffix.lower()
    caps = {}

    def add(img_name, cap):
        img_name = Path(img_name).name
        caps.setdefault(img_name, []).append(cap.strip())

    if ext == ".token":
        with caption_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                left, cap = line.split("\t", 1)
                img = left.split("#", 1)[0]
                add(img, cap)

    elif ext == ".csv":
        with caption_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {caption_path}")
            cols_lower = [c.lower() for c in reader.fieldnames]
            img_key_l = None
            cap_key_l = None
            for c in cols_lower:
                if img_key_l is None and ("image" in c or "filename" in c or "file" in c):
                    img_key_l = c
                if cap_key_l is None and ("caption" in c or "text" in c or "sentence" in c):
                    cap_key_l = c
            if img_key_l is None or cap_key_l is None:
                raise ValueError(f"CSV columns not recognized: {reader.fieldnames}")
            name_map = {c.lower(): c for c in reader.fieldnames}
            img_key = name_map[img_key_l]
            cap_key = name_map[cap_key_l]
            for row in reader:
                add(row[img_key], row[cap_key])

    elif ext == ".json":
        with caption_path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)

        def handle_item(item):
            img = item.get("image") or item.get("filename") or item.get("file") or item.get("img")
            cap = item.get("caption") or item.get("text") or item.get("sentence")
            if img is None or cap is None:
                return
            if isinstance(cap, list):
                for c in cap:
                    add(img, str(c))
            else:
                add(img, str(cap))

        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    handle_item(it)
        elif isinstance(obj, dict):
            if "annotations" in obj and isinstance(obj["annotations"], list):
                for it in obj["annotations"]:
                    if isinstance(it, dict):
                        handle_item(it)
            else:
                for k, v in obj.items():
                    if isinstance(v, list):
                        for c in v:
                            add(k, str(c))
                    else:
                        add(k, str(v))
        else:
            raise ValueError("Unsupported JSON structure.")
    else:
        with caption_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    img, cap = line.split("\t", 1)
                    add(img, cap)
                elif "|" in line:
                    img, cap = line.split("|", 1)
                    add(img, cap)

    if not caps:
        raise ValueError(f"Parsed 0 captions from: {caption_path}")
    return caps


def find_split_file(root: Path, split: str):
    split = split.lower()
    cands = []
    for p in root.rglob("*.txt"):
        name = p.name.lower()
        if split in name and ("train" in name or "val" in name or "test" in name):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def read_split_list(split_path: Path):
    names = []
    with split_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            names.append(Path(s).name)
    return names


# -----------------------------
# Model: learned metric per modality
# -----------------------------

class MetricNet(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, metric_type: str, mlp_hidden: int, init_identity: bool):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.metric_type = metric_type

        if metric_type == "linear":
            self.A = nn.Linear(dim_in, dim_out, bias=False)
            if init_identity and dim_in == dim_out:
                with torch.no_grad():
                    self.A.weight.copy_(torch.eye(dim_in))
        elif metric_type == "mlp":
            self.fc1 = nn.Linear(dim_in, mlp_hidden, bias=True)
            self.fc2 = nn.Linear(mlp_hidden, dim_out, bias=False)
            if init_identity and dim_in == dim_out and mlp_hidden >= dim_in:
                with torch.no_grad():
                    self.fc1.weight.zero_()
                    self.fc1.bias.zero_()
                    self.fc1.weight[:dim_in, :].copy_(torch.eye(dim_in))
                    self.fc2.weight.zero_()
                    self.fc2.weight[:, :dim_in].copy_(torch.eye(dim_in))
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor, l2_normalize: bool) -> torch.Tensor:
        if self.metric_type == "linear":
            y = self.A(z)
        else:
            y = self.fc2(F.gelu(self.fc1(z)))
        y = y * torch.exp(self.log_scale)
        if l2_normalize:
            y = F.normalize(y, p=2, dim=-1)
        return y

    def identity_reg(self) -> torch.Tensor:
        if self.metric_type == "linear" and self.dim_in == self.dim_out:
            I = torch.eye(self.dim_in, device=self.A.weight.device, dtype=self.A.weight.dtype)
            return ((self.A.weight - I) ** 2).mean()
        return torch.tensor(0.0, device=self.log_scale.device, dtype=self.log_scale.dtype)


# -----------------------------
# Core math
# -----------------------------

def pairwise_distances(y: torch.Tensor, eps: float) -> torch.Tensor:
    yy = (y * y).sum(dim=1, keepdim=True)
    dist2 = yy + yy.t() - 2.0 * (y @ y.t())
    dist2 = dist2.clamp_min(0.0)
    return torch.sqrt(dist2 + eps)


def soft_ball_mass(dist: torch.Tensor, r: torch.Tensor, ball_eps: float) -> torch.Tensor:
    m = torch.sigmoid((r - dist) / ball_eps)
    return m.mean(dim=1)


def build_weight_matrix_from_dist(dist: torch.Tensor, bandwidth: torch.Tensor, self_loop_weight: float, eps: float) -> torch.Tensor:
    bw2 = (bandwidth * bandwidth).clamp_min(eps)
    W = torch.exp(-(dist * dist) / (2.0 * bw2))
    # enforce symmetry
    W = 0.5 * (W + W.t())

    # stabilize: keep self-loop (diagonal) as a fixed positive weight
    if self_loop_weight is not None:
        diag = torch.full((W.shape[0],), float(self_loop_weight), device=W.device, dtype=W.dtype)
        W = W.clone()
        W.fill_diagonal_(0.0)
        W = W + torch.diag(diag)

    return W


def normalized_laplacian(W: torch.Tensor, eps: float) -> torch.Tensor:
    N = W.shape[0]
    deg = W.sum(dim=1)
    inv_sqrt = torch.rsqrt(deg + eps)
    S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
    I = torch.eye(N, device=W.device, dtype=W.dtype)
    L = I - S
    L = 0.5 * (L + L.t())
    return L


def safe_eigh(L: torch.Tensor, base_jitter: float, retries: int):
    """
    Safe symmetric eigendecomposition with increasing diagonal jitter.
    Returns evals, evecs, used_jitter.
    """
    L = 0.5 * (L + L.t())
    if not torch.isfinite(L).all():
        raise RuntimeError("L has non-finite entries (NaN/Inf) before eigh.")

    N = L.shape[0]
    I = torch.eye(N, device=L.device, dtype=L.dtype)

    last_err = None
    for t in range(retries):
        jitter = float(base_jitter) * (10.0 ** t)
        try:
            evals, evecs = torch.linalg.eigh(L + jitter * I)
            if torch.isfinite(evals).all() and torch.isfinite(evecs).all():
                return evals, evecs, jitter
            last_err = RuntimeError("Non-finite eigen outputs.")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"safe_eigh failed after {retries} retries. Last error: {last_err}")


def heat_trace_and_diag(
    L: torch.Tensor,
    s: torch.Tensor,
    alpha: torch.Tensor,
    spectral_device: str,
    main_device: torch.device,
    main_dtype: torch.dtype,
    base_jitter: float,
    retries: int,
    pow_eps: float,
):
    """
    Returns Theta(s), p_diag(s;i) on main_device/main_dtype.

    We perform eigh in float64 on spectral_device for stability.
    """
    # move to spectral device & float64 for eigendecomposition stability
    L_eval = L.to(dtype=torch.float64)
    if spectral_device == "cpu":
        L_eval = L_eval.cpu()
    else:
        L_eval = L_eval.to("cuda")

    evals, evecs, used_j = safe_eigh(L_eval, base_jitter=base_jitter, retries=retries)

    evals = evals.clamp_min(0.0)
    evals_a = torch.pow(evals + pow_eps, alpha.to(dtype=torch.float64, device=evals.device))
    exp_term = torch.exp(-s.to(dtype=torch.float64, device=evals.device) * evals_a)

    Theta = exp_term.sum()

    U2 = evecs * evecs
    p_diag = U2 @ exp_term  # (N,)

    # cast back to main device/dtype
    Theta = Theta.to(device=main_device, dtype=main_dtype)
    p_diag = p_diag.to(device=main_device, dtype=main_dtype)
    return Theta, p_diag, used_j


def trapezoid_weights(scales: torch.Tensor) -> torch.Tensor:
    K = scales.numel()
    w = torch.zeros_like(scales)
    if K == 1:
        w[0] = scales[0]
        return w
    w[0] = 0.5 * (scales[1] - scales[0])
    for k in range(1, K - 1):
        w[k] = 0.5 * (scales[k + 1] - scales[k - 1])
    w[K - 1] = 0.5 * (scales[K - 1] - scales[K - 2])
    return w


def local_zeta_hat(p_diags: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor, q: torch.Tensor, eps: float):
    gamma_q = torch.exp(torch.lgamma(q))
    s_pow = torch.pow(scales + eps, q - 1.0)
    coeff = (weights * s_pow)[:, None]
    zeta = (coeff * p_diags).sum(dim=0) / (gamma_q + eps)
    return zeta


# -----------------------------
# Loss assembly
# -----------------------------

class FractalAlignObjective(nn.Module):
    def __init__(self, args, dim_in: int):
        super().__init__()
        self.args = args

        dim_out = dim_in if args.dim_out <= 0 else args.dim_out
        self.metric_img = MetricNet(dim_in, dim_out, args.metric_type, args.mlp_hidden, args.init_identity)
        self.metric_txt = MetricNet(dim_in, dim_out, args.metric_type, args.mlp_hidden, args.init_identity)

        if args.learn_df:
            self.df_raw = nn.Parameter(torch.tensor(float(args.df_init)))
        else:
            self.register_buffer("df_raw", torch.tensor(float(args.df_init)))

        if args.learn_ds:
            self.ds_raw = nn.Parameter(torch.tensor(float(args.ds_init)))
        else:
            self.register_buffer("ds_raw", torch.tensor(float(args.ds_init)))

        self.register_buffer("alpha", torch.tensor(float(args.alpha)))

        s = np.logspace(np.log10(args.s_min), np.log10(args.s_max), args.num_scales).astype(np.float64)
        self.register_buffer("scales", torch.tensor(s, dtype=torch.float64))
        self.register_buffer("weights", trapezoid_weights(self.scales))

        R = np.logspace(np.log10(args.r_min), np.log10(args.r_max), args.num_radii).astype(np.float64)
        self.register_buffer("radii", torch.tensor(R, dtype=torch.float64))

        rho_max = min(args.r0 / float(r) for r in R)
        rho_max = max(rho_max, 1.0000001)
        rho_hi = min(args.rho_max_cap, rho_max)
        lam = np.logspace(np.log10(args.rho_min), np.log10(rho_hi), args.num_rhos).astype(np.float64)
        self.register_buffer("rhos", torch.tensor(lam, dtype=torch.float64))

    def df_value(self, dtype, device):
        if self.args.learn_df:
            return (F.softplus(self.df_raw.to(dtype=dtype, device=device)) + self.args.df_floor)
        return self.df_raw.to(dtype=dtype, device=device)

    def ds_value(self, dtype, device):
        if self.args.learn_ds:
            return (F.softplus(self.ds_raw.to(dtype=dtype, device=device)) + self.args.ds_floor)
        return self.ds_raw.to(dtype=dtype, device=device)

    def forward(self, z_img: torch.Tensor, z_txt: torch.Tensor):
        dtype = z_img.dtype
        device = z_img.device

        df = self.df_value(dtype, device)
        ds = self.ds_value(dtype, device)
        alpha = self.alpha.to(dtype=dtype, device=device)

        scales = self.scales.to(dtype=dtype, device=device)
        weights = self.weights.to(dtype=dtype, device=device)
        radii = self.radii.to(dtype=dtype, device=device)
        rhos = self.rhos.to(dtype=dtype, device=device)

        q = (ds / (2.0 * alpha) + self.args.q_margin).clamp_min(1e-6)

        y_img = self.metric_img(z_img, l2_normalize=self.args.l2_normalize)
        y_txt = self.metric_txt(z_txt, l2_normalize=self.args.l2_normalize)

        dist_img = pairwise_distances(y_img, eps=self.args.dist_eps)
        dist_txt = pairwise_distances(y_txt, eps=self.args.dist_eps)

        # ---- L_dbl
        def L_dbl_one(dist: torch.Tensor):
            total = torch.zeros((), device=device, dtype=dtype)
            mass_r = []
            for r in radii:
                mass_r.append(soft_ball_mass(dist, r, ball_eps=self.args.ball_eps))
            mass_r = torch.stack(mass_r, dim=0)

            Rn = radii.numel()
            Hn = rhos.numel()
            for ri in range(Rn):
                mr = mass_r[ri]
                r = radii[ri]
                for hj in range(Hn):
                    rho = rhos[hj]
                    rr = rho * r
                    mrr = soft_ball_mass(dist, rr, ball_eps=self.args.ball_eps)
                    num = (mrr - torch.pow(rho, df) * mr)
                    den = (mrr + torch.pow(rho, df) * mr).clamp_min(self.args.den_eps)
                    frac = num / den
                    total = total + (frac * frac).mean()
            total = total / (float(Rn) * float(Hn))
            return total

        L_dbl_img = L_dbl_one(dist_img)
        L_dbl_txt = L_dbl_one(dist_txt)
        L_dbl = 0.5 * (L_dbl_img + L_dbl_txt)

        # ---- spectral objects
        def spectral_objects(dist: torch.Tensor):
            K = scales.numel()
            Thetas = []
            pdiags = []
            used_jitters = []
            for s in scales:
                W = build_weight_matrix_from_dist(
                    dist=dist,
                    bandwidth=s,
                    self_loop_weight=self.args.self_loop_weight,
                    eps=self.args.band_eps,
                )
                L = normalized_laplacian(W, eps=self.args.lap_eps)

                Theta, p_diag, used_j = heat_trace_and_diag(
                    L=L,
                    s=s,
                    alpha=alpha,
                    spectral_device=self.args.spectral_device,
                    main_device=device,
                    main_dtype=dtype,
                    base_jitter=self.args.eigh_jitter,
                    retries=self.args.eigh_retries,
                    pow_eps=self.args.pow_eps,
                )
                Thetas.append(Theta)
                pdiags.append(p_diag)
                used_jitters.append(used_j)

            Thetas = torch.stack(Thetas, dim=0)  # (K,)
            pdiags = torch.stack(pdiags, dim=0)  # (K,N)
            return Thetas, pdiags, used_jitters

        Theta_img, pdiag_img, jit_img = spectral_objects(dist_img)
        Theta_txt, pdiag_txt, jit_txt = spectral_objects(dist_txt)

        # ---- L_spec
        def L_spec_one(Theta: torch.Tensor):
            K = Theta.numel()
            loss = torch.zeros((), device=device, dtype=dtype)
            for k in range(K - 1):
                ratio = Theta[k + 1] / Theta[k].clamp_min(self.args.trace_eps)
                target = torch.pow(scales[k + 1] / scales[k], -ds / (2.0 * alpha))
                loss = loss + (ratio - target) * (ratio - target)
            return loss / float(K - 1)

        L_spec_img = L_spec_one(Theta_img)
        L_spec_txt = L_spec_one(Theta_txt)
        L_spec = 0.5 * (L_spec_img + L_spec_txt)

        J_frac = L_dbl + L_spec

        # ---- J_match (local zeta)
        zeta_img = local_zeta_hat(pdiag_img, scales, weights, q, eps=self.args.zeta_eps)
        zeta_txt = local_zeta_hat(pdiag_txt, scales, weights, q, eps=self.args.zeta_eps)
        J_match = ((zeta_img - zeta_txt) ** 2).mean()

        reg_id = self.metric_img.identity_reg() + self.metric_txt.identity_reg()
        reg_scale = (self.metric_img.log_scale ** 2 + self.metric_txt.log_scale ** 2)

        J_total = J_frac + self.args.match_weight * J_match \
                  + self.args.reg_identity * reg_id + self.args.reg_scale * reg_scale

        return {
            "J_total": J_total,
            "J_frac": J_frac,
            "L_dbl": L_dbl,
            "L_spec": L_spec,
            "J_match": J_match,
            "df": df.detach(),
            "ds": ds.detach(),
            "q": q.detach(),
            "y_img": y_img,
            "y_txt": y_txt,
            "jit_img": float(max(jit_img)) if len(jit_img) > 0 else 0.0,
            "jit_txt": float(max(jit_txt)) if len(jit_txt) > 0 else 0.0,
        }


# -----------------------------
# Visualization
# -----------------------------

def umap_plot_pairs(img_emb: np.ndarray, txt_emb: np.ndarray, out_path: str, title: str, seed: int):
    assert img_emb.shape[0] == txt_emb.shape[0]
    N = img_emb.shape[0]
    all_emb = np.vstack([img_emb, txt_emb])

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.10,
        metric="cosine",
        random_state=seed,
        n_jobs=1,  # avoids the warning you saw (random_state forces n_jobs=1 anyway)
    )
    coords = reducer.fit_transform(all_emb)
    img_xy = coords[:N]
    txt_xy = coords[N:]

    plt.figure(figsize=(9, 7))
    plt.scatter(img_xy[:, 0], img_xy[:, 1], marker="^", s=28, alpha=0.85, label="Image")
    plt.scatter(txt_xy[:, 0], txt_xy[:, 1], marker="o", s=18, alpha=0.85, label="Text")
    for i in range(N):
        plt.plot([img_xy[i, 0], txt_xy[i, 0]], [img_xy[i, 1], txt_xy[i, 1]],
                 linewidth=0.6, alpha=0.35)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--root", type=str, default="/work/was598/modilty_gap/tools/data/flickr30k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--caption_idx", type=int, default=0)

    # CLIP
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch_size", type=int, default=32)

    # training device/dtype
    ap.add_argument("--device", type=str, default="cuda", help="device for training: cuda or cpu")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 means no clipping")

    # metric network
    ap.add_argument("--metric_type", type=str, default="linear", choices=["linear", "mlp"])
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--dim_out", type=int, default=-1)
    ap.add_argument("--init_identity", action="store_true")
    ap.add_argument("--l2_normalize", action="store_true")

    # fractal params
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--learn_df", action="store_true")
    ap.add_argument("--learn_ds", action="store_true")
    ap.add_argument("--df_init", type=float, default=2.0)
    ap.add_argument("--ds_init", type=float, default=2.0)
    ap.add_argument("--df_floor", type=float, default=1e-3)
    ap.add_argument("--ds_floor", type=float, default=1e-3)

    # R and Lambda
    ap.add_argument("--r0", type=float, default=2.0)
    ap.add_argument("--r_min", type=float, default=0.05)
    ap.add_argument("--r_max", type=float, default=0.6)
    ap.add_argument("--num_radii", type=int, default=6)

    ap.add_argument("--rho_min", type=float, default=1.25)
    ap.add_argument("--rho_max_cap", type=float, default=8.0)
    ap.add_argument("--num_rhos", type=int, default=5)

    # S
    ap.add_argument("--s_min", type=float, default=0.10)
    ap.add_argument("--s_max", type=float, default=0.60)
    ap.add_argument("--num_scales", type=int, default=6)

    # zeta q
    ap.add_argument("--q_margin", type=float, default=1.0)

    # loss weights
    ap.add_argument("--match_weight", type=float, default=1.0)

    # regularization
    ap.add_argument("--reg_identity", type=float, default=1e-4)
    ap.add_argument("--reg_scale", type=float, default=1e-6)

    # numeric eps
    ap.add_argument("--dist_eps", type=float, default=1e-12)
    ap.add_argument("--ball_eps", type=float, default=0.01)
    ap.add_argument("--den_eps", type=float, default=1e-8)
    ap.add_argument("--band_eps", type=float, default=1e-12)
    ap.add_argument("--lap_eps", type=float, default=1e-12)
    ap.add_argument("--pow_eps", type=float, default=1e-12)
    ap.add_argument("--trace_eps", type=float, default=1e-12)
    ap.add_argument("--zeta_eps", type=float, default=1e-12)

    # spectral stability controls
    ap.add_argument("--self_loop_weight", type=float, default=1.0, help="W_ii value (stabilizes degrees)")
    ap.add_argument("--spectral_device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--eigh_jitter", type=float, default=1e-8, help="base diagonal jitter for safe_eigh")
    ap.add_argument("--eigh_retries", type=int, default=6, help="retries with jitter * 10^t")

    # outputs
    ap.add_argument("--out_prefix", type=str, default="fk30k_fractal_stable")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, switching training to cpu.")
        args.device = "cpu"

    if args.spectral_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, switching spectral_device to cpu.")
        args.spectral_device = "cpu"

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # ---- load local flickr30k
    root = Path(args.root)
    img_dir = find_image_dir(root)
    cap_file = find_caption_file(root)
    caps = parse_captions(cap_file)

    disk_imgs = {p.name: p for p in img_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS}
    common = sorted(set(disk_imgs.keys()) & set(caps.keys()))
    if not common:
        raise RuntimeError(
            f"No overlap between images in {img_dir} and captions in {cap_file}.\n"
            f"Example image names: {list(disk_imgs.keys())[:5]}\n"
            f"Example caption keys: {list(caps.keys())[:5]}"
        )

    split_path = find_split_file(root, args.split)
    if split_path is not None:
        split_names = set(read_split_list(split_path))
        common = [n for n in common if n in split_names]
        if not common:
            raise RuntimeError(f"Split file found ({split_path}) but no matching images after filtering.")

    N = min(args.n, len(common))
    chosen = common[:N]

    def pick_cap(img_name: str):
        lst = caps[img_name]
        idx = max(0, min(args.caption_idx, len(lst) - 1))
        return lst[idx]

    # ---- CLIP encoding (on device for speed)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip.eval()

    img_embeds = []
    txt_embeds = []

    for start in tqdm(range(0, N, args.batch_size), desc="CLIP encoding"):
        batch_names = chosen[start:start + args.batch_size]
        images = []
        texts = []
        for nm in batch_names:
            images.append(Image.open(disk_imgs[nm]).convert("RGB"))
            texts.append(pick_cap(nm))

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clip(**inputs)
            im = out.image_embeds
            tx = out.text_embeds
            im = im / im.norm(dim=-1, keepdim=True)
            tx = tx / tx.norm(dim=-1, keepdim=True)

        img_embeds.append(im.detach().cpu())
        txt_embeds.append(tx.detach().cpu())

    z_img = torch.cat(img_embeds, dim=0).to(device=device, dtype=dtype)
    z_txt = torch.cat(txt_embeds, dim=0).to(device=device, dtype=dtype)

    # ---- baseline UMAP
    base_path = f"{args.out_prefix}_baseline_umap.png"
    umap_plot_pairs(
        img_emb=z_img.detach().cpu().numpy().astype(np.float32),
        txt_emb=z_txt.detach().cpu().numpy().astype(np.float32),
        out_path=base_path,
        title=f"Flickr30k Paired CLIP Embeddings + UMAP (N={N})",
        seed=args.seed,
    )
    print(f"[OK] baseline plot saved: {base_path}")

    # ---- train objective
    obj = FractalAlignObjective(args, dim_in=z_img.shape[1]).to(device=device, dtype=dtype)
    optim = torch.optim.Adam(obj.parameters(), lr=args.lr)

    log_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        optim.zero_grad(set_to_none=True)

        out = obj(z_img, z_txt)
        loss = out["J_total"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became non-finite at epoch {epoch}: {loss.item()}")

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(obj.parameters(), args.grad_clip)
        optim.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[epoch {epoch:4d}/{args.epochs}] "
                f"J_total={out['J_total'].item():.6e} | "
                f"J_frac={out['J_frac'].item():.6e} | "
                f"L_dbl={out['L_dbl'].item():.6e} | "
                f"L_spec={out['L_spec'].item():.6e} | "
                f"J_match={out['J_match'].item():.6e} | "
                f"d_f={out['df'].item():.4f} d_s={out['ds'].item():.4f} q={out['q'].item():.4f} | "
                f"max_jitter(img/txt)={out['jit_img']:.1e}/{out['jit_txt']:.1e}"
            )

    # ---- after-training UMAP (use f_theta(z))
    with torch.no_grad():
        y_img = obj.metric_img(z_img, l2_normalize=args.l2_normalize)
        y_txt = obj.metric_txt(z_txt, l2_normalize=args.l2_normalize)

    after_path = f"{args.out_prefix}_after_umap.png"
    umap_plot_pairs(
        img_emb=y_img.detach().cpu().numpy().astype(np.float32),
        txt_emb=y_txt.detach().cpu().numpy().astype(np.float32),
        out_path=after_path,
        title=f"Flickr30k OUR Method (Fractal + Local Zeta Match) + UMAP (N={N})",
        seed=args.seed,
    )
    print(f"[OK] after plot saved: {after_path}")

    ckpt_path = f"{args.out_prefix}_ckpt.pt"
    torch.save(
        {
            "args": vars(args),
            "state_dict": obj.state_dict(),
            "chosen_images": chosen,
            "image_dir": str(img_dir),
            "caption_file": str(cap_file),
        },
        ckpt_path,
    )
    print(f"[OK] checkpoint saved: {ckpt_path}")
    print(f"[OK] image_dir = {img_dir}")
    print(f"[OK] caption_file = {cap_file}")


if __name__ == "__main__":
    main()
