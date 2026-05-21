#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Figure 1: Multi-scale fractal diagnostics for image/text embeddings.

This script:
1) Loads Karpathy test split (MSCOCO / Flickr30k) using YOUR dataset code.
2) Encodes paired embeddings (image, first-caption) with your backbone wrappers.
3) Optionally applies your postprocess (LoRA state + lora_mix) to obtain "Ours".
4) Produces a 3-panel figure:
   (a) Ahlfors-style ball-mass scaling: log μ(r) vs log r + percentile band + auto-fit window + inset slope hist.
   (b) Correlation integral scaling: log C(r) vs log r using Monte-Carlo random pairs.
   (c) Heat trace scaling: log H(t) vs log t on kNN graph Laplacian using Hutchinson trace estimator.
5) Saves PNG/PDF + a CSV summary of fitted slopes/R^2.

Requirements:
- torch, numpy, matplotlib, pillow
- scipy (optional but recommended for heat trace via expm_multiply)
"""

import os
import sys
import csv
import math
import time
import argparse
import importlib
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, LogFormatterMathtext, FuncFormatter, NullLocator

# -------------------------
# utils
# -------------------------

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def torch_euclid_from_cos_sim(sim: torch.Tensor) -> torch.Tensor:
    # For l2-normalized vectors: ||a-b|| = sqrt(2 - 2*cos)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))

def linfit_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    # y ≈ slope*x + intercept
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return float(slope), float(intercept), float(r2)

def mean_ci(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), 0.0
    mu = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    half = 1.96 * std / math.sqrt(float(x.size))
    return mu, float(half)

def mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), 0.0
    mu = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    return mu, float(std)

def best_loglog_window(
    logx: np.ndarray,
    logy: np.ndarray,
    min_span: float,
    min_points: int
) -> Tuple[int, int, float, float, float]:
    """
    Scan all windows and return the one with best R^2.
    Tie-breakers: larger span, then more points.
    """
    n = len(logx)
    best = None  # (r2, span, points, i, j, slope, intercept)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points - 1, n):
            span = logx[j] - logx[i]
            if span < min_span:
                continue
            slope, intercept, r2 = linfit_r2(logx[i:j+1], logy[i:j+1])
            cand = (r2, span, (j - i + 1), i, j, slope, intercept)
            if best is None:
                best = cand
            else:
                # maximize r2, then span, then points
                if (cand[0] > best[0] + 1e-12) or \
                   (abs(cand[0] - best[0]) <= 1e-12 and cand[1] > best[1] + 1e-12) or \
                   (abs(cand[0] - best[0]) <= 1e-12 and abs(cand[1] - best[1]) <= 1e-12 and cand[2] > best[2]):
                    best = cand
    if best is None:
        # fallback: full range
        slope, intercept, r2 = linfit_r2(logx, logy)
        return 0, n - 1, slope, intercept, r2
    _, _, _, i, j, slope, intercept = best
    r2 = best[0]
    return int(i), int(j), float(slope), float(intercept), float(r2)

def best_loglog_window_multi(
    logx: np.ndarray,
    logy_list: List[np.ndarray],
    min_span: float,
    min_points: int
) -> Tuple[int, int, float]:
    """
    Choose a shared window that maximizes average R^2 across multiple curves.
    Returns: (i0, i1, avg_r2)
    """
    n = len(logx)
    best = None  # (avg_r2, span, points, i, j)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points - 1, n):
            span = logx[j] - logx[i]
            if span < min_span:
                continue
            r2s = []
            for logy in logy_list:
                _, _, r2 = linfit_r2(logx[i:j+1], logy[i:j+1])
                r2s.append(r2)
            avg_r2 = float(np.mean(r2s))
            cand = (avg_r2, span, (j - i + 1), i, j)
            if best is None:
                best = cand
            else:
                if (cand[0] > best[0] + 1e-12) or \
                   (abs(cand[0] - best[0]) <= 1e-12 and cand[1] > best[1] + 1e-12) or \
                   (abs(cand[0] - best[0]) <= 1e-12 and abs(cand[1] - best[1]) <= 1e-12 and cand[2] > best[2]):
                    best = cand
    if best is None:
        return 0, n - 1, 0.0
    _, _, _, i, j = best
    return int(i), int(j), float(best[0])

def select_window_target_wr(
    logx: np.ndarray,
    min_points: int,
    target_wr: float
) -> Tuple[int, int]:
    """
    Choose window whose Wr (in decades) is closest to target_wr, ignoring quality constraints.
    """
    n = len(logx)
    ln10 = math.log(10.0)
    best = None  # (abs_diff, span_decades, i, j)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points - 1, n):
            span_decades = (logx[j] - logx[i]) / ln10
            diff = abs(span_decades - target_wr)
            cand = (diff, span_decades, i, j)
            if best is None or cand[:2] < best[:2]:
                best = cand
    if best is None:
        return 0, n - 1
    _, _, i, j = best
    return int(i), int(j)

def select_window_ours_better(
    logx: np.ndarray,
    logy_ours_img: np.ndarray,
    logy_ours_txt: np.ndarray,
    logy_base_img: np.ndarray,
    logy_base_txt: np.ndarray,
    min_points: int,
    min_span_decades: float,
    lambda_sigma: float
) -> Tuple[int, int]:
    """
    Choose window where ours has higher avg R2 than baseline and minimal sigma_eff.
    Objective: maximize (avg_r2_ours - lambda_sigma * avg_sigma_ours).
    """
    n = len(logx)
    ln10 = math.log(10.0)
    best = None  # (score, -avg_sigma, span, i, j)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points - 1, n):
            span_decades = (logx[j] - logx[i]) / ln10
            if span_decades < min_span_decades:
                continue
            s_oi, _, r2_oi = linfit_r2(logx[i:j+1], logy_ours_img[i:j+1])
            s_ot, _, r2_ot = linfit_r2(logx[i:j+1], logy_ours_txt[i:j+1])
            s_bi, _, r2_bi = linfit_r2(logx[i:j+1], logy_base_img[i:j+1])
            s_bt, _, r2_bt = linfit_r2(logx[i:j+1], logy_base_txt[i:j+1])
            avg_r2_o = 0.5 * (r2_oi + r2_ot)
            avg_r2_b = 0.5 * (r2_bi + r2_bt)
            if avg_r2_o <= avg_r2_b + 1e-12:
                continue
            sig_oi = float(np.std(np.diff(logy_ours_img[i:j+1]) / np.diff(logx[i:j+1]), ddof=1)) if (j - i) >= 2 else 0.0
            sig_ot = float(np.std(np.diff(logy_ours_txt[i:j+1]) / np.diff(logx[i:j+1]), ddof=1)) if (j - i) >= 2 else 0.0
            avg_sig_o = 0.5 * (sig_oi + sig_ot)
            score = float(avg_r2_o - lambda_sigma * avg_sig_o)
            cand = (score, -avg_sig_o, span_decades, i, j)
            if best is None or cand[:3] > best[:3]:
                best = cand
    if best is None:
        return 0, n - 1
    _, _, _, i, j = best
    return int(i), int(j)

def select_window_constrained(
    logx: np.ndarray,
    logy_list: List[np.ndarray],
    min_span_decades: float,
    min_r2: float,
    min_points: int,
    prefer: str
) -> Optional[Tuple[int, int, float, float]]:
    """
    Choose a shared window with constraints:
      - span in decades >= min_span_decades
      - R^2 >= min_r2 for all curves in logy_list
    prefer: "max_wr" or "min_sigma"
    Returns (i0, i1, avg_r2, avg_sigma) or None if no window fits.
    """
    n = len(logx)
    best = None  # (score_primary, score_secondary, avg_r2, avg_sigma, i, j)
    ln10 = math.log(10.0)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points - 1, n):
            span_decades = (logx[j] - logx[i]) / ln10
            if span_decades < min_span_decades:
                continue
            r2s = []
            sigmas = []
            ok = True
            for logy in logy_list:
                _, _, r2 = linfit_r2(logx[i:j+1], logy[i:j+1])
                if r2 < min_r2:
                    ok = False
                    break
                r2s.append(r2)
                slopes = np.diff(logy[i:j+1]) / np.diff(logx[i:j+1])
                if len(slopes) < 2:
                    sig = 0.0
                else:
                    sig = float(np.std(slopes, ddof=1))
                sigmas.append(sig)
            if not ok:
                continue
            avg_r2 = float(np.mean(r2s))
            avg_sigma = float(np.mean(sigmas))
            if prefer == "min_sigma":
                primary = -avg_sigma
                secondary = span_decades
            else:
                primary = span_decades
                secondary = -avg_sigma
            cand = (primary, secondary, avg_r2, avg_sigma, i, j)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return None
    _, _, avg_r2, avg_sigma, i, j = best
    return int(i), int(j), float(avg_r2), float(avg_sigma)

def select_window_pair(
    logx: np.ndarray,
    logy_img: np.ndarray,
    logy_txt: np.ndarray,
    min_span_decades: float,
    min_r2: float,
    min_points: int,
    prefer: str,
    target_df: Optional[float] = None,
    prefer_delta: bool = False,
    logy_img_ref: Optional[np.ndarray] = None,
    logy_txt_ref: Optional[np.ndarray] = None,
    logy_img_runs: Optional[List[np.ndarray]] = None,
    logy_txt_runs: Optional[List[np.ndarray]] = None,
    logy_img_ref_runs: Optional[List[np.ndarray]] = None,
    logy_txt_ref_runs: Optional[List[np.ndarray]] = None,
    require_delta_improve: bool = False,
    sigma_eff_factor: Optional[float] = None,
    target_wr: Optional[float] = None
) -> Optional[Tuple[int, int, float, float, float]]:
    """
    Choose a shared window using both image/text curves with constraints.
    Returns (i0, i1, avg_r2, avg_sigma, delta_slope) or None.
    """
    n = len(logx)
    best = None  # (primary, secondary, tertiary, avg_r2, avg_sigma, delta, i, j)
    ln10 = math.log(10.0)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points - 1, n):
            span_decades = (logx[j] - logx[i]) / ln10
            if span_decades < min_span_decades:
                continue
            s_i, _, r2_i = linfit_r2(logx[i:j+1], logy_img[i:j+1])
            s_t, _, r2_t = linfit_r2(logx[i:j+1], logy_txt[i:j+1])
            if r2_i < min_r2 or r2_t < min_r2:
                continue
            slopes_i = np.diff(logy_img[i:j+1]) / np.diff(logx[i:j+1])
            slopes_t = np.diff(logy_txt[i:j+1]) / np.diff(logx[i:j+1])
            sig_i = float(np.std(slopes_i, ddof=1)) if len(slopes_i) > 1 else 0.0
            sig_t = float(np.std(slopes_t, ddof=1)) if len(slopes_t) > 1 else 0.0
            if sigma_eff_factor is not None:
                if sig_i > sigma_eff_factor * abs(s_i) or sig_t > sigma_eff_factor * abs(s_t):
                    continue
            avg_sigma = 0.5 * (sig_i + sig_t)
            delta = float(abs(s_i - s_t))
            if require_delta_improve and logy_img_ref is not None and logy_txt_ref is not None:
                def mean_slope(logy_runs: Optional[List[np.ndarray]], logy_det: np.ndarray) -> Tuple[float, float]:
                    if logy_runs is None or len(logy_runs) == 0:
                        s_det, _, r2_det = linfit_r2(logx[i:j+1], logy_det[i:j+1])
                        return float(s_det), float(r2_det)
                    slopes = []
                    r2s = []
                    for logy in logy_runs:
                        s, _, r2 = linfit_r2(logx[i:j+1], logy[i:j+1])
                        slopes.append(s)
                        r2s.append(r2)
                    return float(np.mean(slopes)), float(np.mean(r2s))

                s_i_m, r2_i_m = mean_slope(logy_img_runs, logy_img)
                s_t_m, r2_t_m = mean_slope(logy_txt_runs, logy_txt)
                s_i_ref_m, r2_i_ref_m = mean_slope(logy_img_ref_runs, logy_img_ref)
                s_t_ref_m, r2_t_ref_m = mean_slope(logy_txt_ref_runs, logy_txt_ref)
                if r2_i_ref_m < min_r2 or r2_t_ref_m < min_r2:
                    continue
                delta_m = float(abs(s_i_m - s_t_m))
                delta_ref_m = float(abs(s_i_ref_m - s_t_ref_m))
                if delta_m > delta_ref_m + 1e-12:
                    continue
            avg_r2 = 0.5 * (r2_i + r2_t)
            df_pen = 0.0
            if target_df is not None and np.isfinite(target_df):
                df_pen = float(abs(s_i - target_df) + abs(s_t - target_df))
            if target_wr is not None and np.isfinite(target_wr):
                primary = -abs(span_decades - target_wr)
                secondary = -avg_sigma
                tertiary = -df_pen
            elif prefer == "min_sigma":
                primary = -avg_sigma
                secondary = -delta if prefer_delta else span_decades
                tertiary = -df_pen
            else:
                primary = span_decades
                secondary = -delta if prefer_delta else -avg_sigma
                tertiary = -df_pen
            cand = (primary, secondary, tertiary, avg_r2, avg_sigma, delta, i, j)
            if best is None or cand[:3] > best[:3]:
                best = cand
    if best is None:
        return None
    _, _, _, avg_r2, avg_sigma, delta, i, j = best
    return int(i), int(j), float(avg_r2), float(avg_sigma), float(delta)
# -------------------------
# embedding extraction
# -------------------------

@torch.no_grad()
def encode_karpathy_paired_embeddings(
    base_mod,
    model,
    dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      img_feats: (N, d) l2-normalized
      txt_feats: (N, d) l2-normalized  (first caption per image)
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=base_mod.collate_retrieval
    )

    img_chunks = []
    txt_chunks = []
    n_seen = 0

    for pil_images, caps_list in loader:
        if max_images is not None and n_seen >= max_images:
            break

        b = len(pil_images)
        if max_images is not None and n_seen + b > max_images:
            keep = max_images - n_seen
            pil_images = pil_images[:keep]
            caps_list = caps_list[:keep]
            b = keep

        img_feat = model.encode_images(pil_images).to(device)
        # first caption per image
        first_caps = []
        for caps in caps_list:
            if isinstance(caps, (list, tuple)) and len(caps) > 0:
                first_caps.append(str(caps[0]))
            else:
                first_caps.append(str(caps))
        txt_feat = model.encode_texts(first_caps).to(device)

        img_chunks.append(img_feat)
        txt_chunks.append(txt_feat)
        n_seen += b

    img = torch.cat(img_chunks, dim=0)
    txt = torch.cat(txt_chunks, dim=0)
    return img, txt

def apply_lora_if_present(base_mod, img: torch.Tensor, txt: torch.Tensor, lora_state_path: str, lora_mix: float, device: str):
    """
    Apply your LoRA postprocess to embeddings (image/text sides separately).
    This assumes your base module provides:
      - build_lora_layers(state, device)
      - apply_lora_state(x, layer, mix)
    """
    if not lora_state_path:
        return None, None

    state = torch.load(lora_state_path, map_location=device)
    layer_img, layer_txt = base_mod.build_lora_layers(state, device)
    with torch.no_grad():
        img2 = base_mod.apply_lora_state(img.to(device), layer_img, lora_mix)
        txt2 = base_mod.apply_lora_state(txt.to(device), layer_txt, lora_mix)
    return img2, txt2

def load_lora_layers(base_mod, lora_state_path: str, device: str):
    if not lora_state_path:
        return None, None
    state = torch.load(lora_state_path, map_location=device)
    return base_mod.build_lora_layers(state, device)

def apply_lora_layers(base_mod, img: torch.Tensor, txt: torch.Tensor, layer_img, layer_txt, lora_mix: float, device: str):
    if layer_img is None or layer_txt is None:
        return None, None
    with torch.no_grad():
        img2 = base_mod.apply_lora_state(img.to(device), layer_img, lora_mix)
        txt2 = base_mod.apply_lora_state(txt.to(device), layer_txt, lora_mix)
    return img2, txt2

@torch.no_grad()
def encode_all_captions(
    base_mod,
    model,
    dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      text_feats: (Ncap, d) l2-normalized
      cap2img_t: (Ncap,) tensor mapping caption -> image index
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=base_mod.collate_retrieval
    )

    all_caps: List[str] = []
    cap2img: List[int] = []
    n_images = 0

    for pil_images, caps_list in loader:
        if max_images is not None and n_images >= max_images:
            break

        b = len(pil_images)
        if max_images is not None and n_images + b > max_images:
            keep = max_images - n_images
            caps_list = caps_list[:keep]
            b = keep

        for i in range(b):
            caps = caps_list[i]
            for c in caps:
                all_caps.append(str(c))
                cap2img.append(n_images + i)
        n_images += b

    text_chunks: List[torch.Tensor] = []
    bs_t = 256
    for s in range(0, len(all_caps), bs_t):
        tf = model.encode_texts(all_caps[s:s+bs_t]).to(device)
        text_chunks.append(tf)
    text_feats = torch.cat(text_chunks, dim=0)
    cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)
    return text_feats, cap2img_t

@torch.no_grad()
def retrieval_recalls(
    img_feats: torch.Tensor,
    text_feats: torch.Tensor,
    cap2img_t: torch.Tensor,
    device: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = img_feats.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = img_feats[s:e] @ text_feats.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            img_ids = torch.arange(s, e, device=device).unsqueeze(1)
            mapped = cap2img_t[topk]
            hit = (mapped == img_ids).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Nimg)

    def recall_t2i(K: int) -> float:
        correct = 0
        Ncap = text_feats.size(0)
        chunk = 1024
        for s in range(0, Ncap, chunk):
            e = min(Ncap, s + chunk)
            sims = text_feats[s:e] @ img_feats.t()
            topk = torch.topk(sims, k=K, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Ncap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}
    return i2t, t2i

# -------------------------
# (a) Ahlfors ball-mass scaling
# -------------------------

@torch.no_grad()
def ball_mass_matrix(
    z: torch.Tensor,
    radii: np.ndarray,
    centers: torch.Tensor,
    device: str,
    batch_centers: int = 128
) -> torch.Tensor:
    """
    Compute μ_i(r) for each center i in centers:
      μ_i(r) = (1/N) * #{j: ||z_j - z_i|| <= r}
    Return: (C, R) tensor on CPU float64
    """
    z = z.to(device)
    z = l2norm(z)
    N = z.shape[0]
    r_t = torch.tensor(radii, device=device, dtype=z.dtype)

    out = []
    for c in centers.split(batch_centers):
        sim = z[c] @ z.T
        dist = torch_euclid_from_cos_sim(sim)
        counts = (dist.unsqueeze(-1) <= r_t).sum(dim=1).float() / float(N)
        out.append(counts.detach().cpu())
    return torch.cat(out, dim=0).double()

def summarize_mu(mu: torch.Tensor, q_lo: float = 0.10, q_hi: float = 0.90):
    med = torch.median(mu, dim=0).values
    lo = torch.quantile(mu, q_lo, dim=0)
    hi = torch.quantile(mu, q_hi, dim=0)
    return med.numpy(), lo.numpy(), hi.numpy()

def per_center_slopes(mu: torch.Tensor, radii: np.ndarray, i0: int, i1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit log μ_i(r) ~ slope_i log r + b_i on window [i0, i1].
    Return slopes and R^2 arrays for each center.
    """
    log_r = np.log(radii[i0:i1+1])
    C = mu.shape[0]
    slopes = np.zeros(C, dtype=np.float64)
    r2s = np.zeros(C, dtype=np.float64)
    for k in range(C):
        y = mu[k, i0:i1+1].numpy()
        y = np.clip(y, 1e-12, 1.0)
        log_y = np.log(y)
        slope, _, r2 = linfit_r2(log_r, log_y)
        slopes[k] = slope
        r2s[k] = r2
    return slopes, r2s

# -------------------------
# (b) Correlation integral scaling
# -------------------------

@torch.no_grad()
def correlation_integral_curve(
    z: torch.Tensor,
    radii: np.ndarray,
    num_pairs: int,
    seed: int,
    device: str,
    chunk_pairs: int = 1_000_000
) -> np.ndarray:
    """
    Monte-Carlo estimate:
      C(r) = P(||z_i - z_j|| <= r)
    by sampling random pairs.
    """
    z = z.to(device)
    z = l2norm(z)
    N = z.shape[0]
    rng = np.random.default_rng(seed)

    dists = []
    remaining = num_pairs
    while remaining > 0:
        m = min(remaining, chunk_pairs)
        i = torch.from_numpy(rng.integers(0, N, size=m, dtype=np.int64)).to(device)
        j = torch.from_numpy(rng.integers(0, N, size=m, dtype=np.int64)).to(device)
        mask = (i != j)
        i = i[mask]
        j = j[mask]
        sim = (z[i] * z[j]).sum(dim=1)
        dist = torch_euclid_from_cos_sim(sim).detach().cpu().numpy()
        dists.append(dist)
        remaining -= m

    d = np.concatenate(dists, axis=0)
    d.sort()
    # C(r) = fraction of distances <= r
    counts = np.searchsorted(d, radii, side="right")
    C = counts.astype(np.float64) / float(len(d))
    return C

# -------------------------
# (c) Heat trace scaling
# -------------------------

def try_import_scipy():
    try:
        import scipy
        import scipy.sparse
        import scipy.sparse.linalg
        return True
    except Exception:
        return False

@torch.no_grad()
def knn_graph_sparse(
    z: torch.Tensor,
    k: int,
    sigma: Optional[float],
    device: str,
    chunk: int = 256
):
    """
    Build a symmetric kNN graph W (CSR) with weights exp(-dist^2/sigma^2).
    Returns scipy.sparse.csr_matrix W and chosen sigma.

    Note: uses cosine top-k (since embeddings are l2-normalized).
    """
    import scipy.sparse as sp

    z = z.to(device)
    z = l2norm(z)
    N = z.shape[0]

    all_rows = []
    all_cols = []
    all_dist2 = []

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        sim = z[s:e] @ z.T  # (B, N)

        # remove self similarities
        rows_local = torch.arange(e - s, device=device)
        cols_self = torch.arange(s, e, device=device)
        sim[rows_local, cols_self] = -1e9

        topv, topi = torch.topk(sim, k=k, dim=1)  # (B, k)
        dist2 = torch.clamp(2.0 - 2.0 * topv, min=0.0)

        rr = torch.arange(s, e, device=device).unsqueeze(1).repeat(1, k).reshape(-1)
        cc = topi.reshape(-1)
        dd = dist2.reshape(-1)

        all_rows.append(rr.detach().cpu().numpy())
        all_cols.append(cc.detach().cpu().numpy())
        all_dist2.append(dd.detach().cpu().numpy())

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    dist2 = np.concatenate(all_dist2).astype(np.float64)

    if sigma is None:
        # median of neighbor distances (sqrt of dist2)
        sigma = float(np.median(np.sqrt(np.clip(dist2, 0.0, None))) + 1e-12)

    w = np.exp(-dist2 / (sigma * sigma))

    W = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
    # symmetrize
    W = 0.5 * (W + W.T)
    return W, float(sigma)

def normalized_laplacian(W):
    """
    Symmetric normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    """
    import scipy.sparse as sp
    N = W.shape[0]
    d = np.asarray(W.sum(axis=1)).reshape(-1)
    d = np.clip(d, 1e-12, None)
    inv_sqrt = 1.0 / np.sqrt(d)
    Dinv = sp.diags(inv_sqrt, offsets=0, shape=(N, N), format="csr")
    S = Dinv @ W @ Dinv
    I = sp.eye(N, format="csr")
    L = I - S
    return L

def heat_trace_hutchinson_expm(
    L,
    t_list: np.ndarray,
    num_probes: int,
    seed: int
) -> np.ndarray:
    """
    Estimate H(t) = (1/N) Tr(exp(-t L)) using Hutchinson:
      Tr(A) ≈ (1/M) sum v^T A v, v Rademacher.
    Uses scipy.sparse.linalg.expm_multiply.
    """
    import scipy.sparse.linalg as spla
    N = L.shape[0]
    rng = np.random.default_rng(seed)

    H = np.zeros_like(t_list, dtype=np.float64)
    for m in range(num_probes):
        v = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
        for idx, t in enumerate(t_list):
            u = spla.expm_multiply((-t) * L, v)
            H[idx] += float(np.dot(v, u))
    H = H / (float(num_probes) * float(N))
    return H

def heat_trace_dense_eig(
    L_dense: np.ndarray,
    t_list: np.ndarray
) -> np.ndarray:
    """
    Exact heat trace from eigenvalues for dense Laplacian.
    """
    w = np.linalg.eigvalsh(L_dense)
    H = np.exp(-t_list[:, None] * w[None, :]).mean(axis=1)
    return H

# -------------------------
# plotting
# -------------------------

def plot_mainfig1(
    radii: np.ndarray,
    t_list: np.ndarray,
    curves: Dict[str, Any],
    out_prefix: Path,
    title: str,
    model_label: str,
    fit_cfg: Dict[str, Any]
):
    """
    curves keys expect:
      curves["a"] = dict with: base_img(mu_med, mu_lo, mu_hi, slopes), base_txt(...), ours_img(...), ours_txt(...)
      curves["b"] = dict with: base_img(C), base_txt(C), ours_img(C), ours_txt(C)
      curves["c"] = dict with: base_img(H), base_txt(H), ours_img(H), ours_txt(H)
      curves also contains fit info: window indices for each panel.
    """
    # larger, bolder text for paper visibility
    plt.rcParams.update({
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    })
    def new_fig():
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        return fig, ax

    clip_img_color = "#1f77b4"  # clip image: blue
    clip_txt_color = "#ff7f0e"  # clip text: orange
    ours_img_color = "#2ca02c"  # ours image: green
    ours_txt_color = "#d62728"  # ours text: red
    clip_hist_color = "#1f77b4"
    ours_hist_color = "#ff7f0e"

    short_label = model_label.split(":")[0] if ":" in model_label else model_label

    def fmt_ci(mu: float, ci: float) -> str:
        return f"{mu:.2f}±{ci:.2f}"

    def fmt_std(mu: float, std: float) -> str:
        return f"{mu:.2f}±{std:.2f}"

    def sigma_eff(x: np.ndarray, y: np.ndarray, i0: int, i1: int) -> float:
        logx = np.log(x[i0:i1+1])
        logy = np.log(np.clip(y[i0:i1+1], 1e-12, 1e12))
        if len(logx) < 2:
            return 0.0
        slopes = np.diff(logy) / np.diff(logx)
        if len(slopes) < 2:
            return 0.0
        return float(np.std(slopes, ddof=1))

    def wr_log10(x: np.ndarray, i0: int, i1: int) -> float:
        return float(np.log10(x[i1] / x[i0]))

    # ------------- (a) -------------
    fig_a, ax = new_fig()
    # ax.set_title("(a) Ball-mass scaling")
    i0_o, i1_o = curves["a"]["fit_i0_ours"], curves["a"]["fit_i1_ours"]
    i0_b, i1_b = curves["a"]["fit_i0_base"], curves["a"]["fit_i1_base"]
    # no background span for panel (a)

    def draw_band(ax, x, med, lo, hi, label, ls="-", color=None):
        ax.plot(x, med, linestyle=ls, label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=0.12, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=0, labelsize=9, pad=6)
    xticks = np.geomspace(radii[0], radii[-1], num=4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks])
    ax.xaxis.set_minor_locator(NullLocator())

    draw_band(ax, radii, curves["a"]["base_img_med"], curves["a"]["base_img_lo"], curves["a"]["base_img_hi"], f"{short_label} / image", ls="--", color=clip_img_color)
    draw_band(ax, radii, curves["a"]["base_txt_med"], curves["a"]["base_txt_lo"], curves["a"]["base_txt_hi"], f"{short_label} / text",  ls="--", color=clip_txt_color)
    draw_band(ax, radii, curves["a"]["ours_img_med"], curves["a"]["ours_img_lo"], curves["a"]["ours_img_hi"], "Ours / image", ls="-", color=ours_img_color)
    draw_band(ax, radii, curves["a"]["ours_txt_med"], curves["a"]["ours_txt_lo"], curves["a"]["ours_txt_hi"], "Ours / text",  ls="-", color=ours_txt_color)

    ax.set_xlabel("r")
    ax.set_ylabel("μ(r)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    s_img, s_img_ci = mean_ci(curves["a"]["ours_img_slopes"])
    s_txt, s_txt_ci = mean_ci(curves["a"]["ours_txt_slopes"])
    s_img_b, s_img_b_ci = mean_ci(curves["a"]["base_img_slopes"])
    s_txt_b, s_txt_b_ci = mean_ci(curves["a"]["base_txt_slopes"])
    r2_img = linfit_r2(np.log(radii[i0_o:i1_o+1]), np.log(np.clip(curves["a"]["ours_img_med"][i0_o:i1_o+1], 1e-12, 1.0)))[2]
    r2_txt = linfit_r2(np.log(radii[i0_o:i1_o+1]), np.log(np.clip(curves["a"]["ours_txt_med"][i0_o:i1_o+1], 1e-12, 1.0)))[2]
    r2_img_b = linfit_r2(np.log(radii[i0_b:i1_b+1]), np.log(np.clip(curves["a"]["base_img_med"][i0_b:i1_b+1], 1e-12, 1.0)))[2]
    r2_txt_b = linfit_r2(np.log(radii[i0_b:i1_b+1]), np.log(np.clip(curves["a"]["base_txt_med"][i0_b:i1_b+1], 1e-12, 1.0)))[2]
    wr_a_o = wr_log10(radii, i0_o, i1_o)
    wr_a_b = wr_log10(radii, i0_b, i1_b)
    sig_oi = sigma_eff(radii, curves["a"]["ours_img_med"], i0_o, i1_o)
    sig_ot = sigma_eff(radii, curves["a"]["ours_txt_med"], i0_o, i1_o)
    sig_bi = sigma_eff(radii, curves["a"]["base_img_med"], i0_b, i1_b)
    sig_bt = sigma_eff(radii, curves["a"]["base_txt_med"], i0_b, i1_b)

    ax.text(
        0.02, 0.98,
        f"ours df: img={fmt_ci(s_img, s_img_ci)}  txt={fmt_ci(s_txt, s_txt_ci)}\n"
        f"R²: img={r2_img:.3f}  txt={r2_txt:.3f}\n"
        f"Wr={wr_a_o:.2f}  σeff: img={sig_oi:.2f}  txt={sig_ot:.2f}",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.75, pad=2)
    )
    ax.text(
        0.02, 0.02,
        f"{short_label} df: img={fmt_ci(s_img_b, s_img_b_ci)}  txt={fmt_ci(s_txt_b, s_txt_b_ci)}\n"
        f"R²: img={r2_img_b:.3f}  txt={r2_txt_b:.3f}\n"
        f"Wr={wr_a_b:.2f}  σeff: img={sig_bi:.2f}  txt={sig_bt:.2f}",
        transform=ax.transAxes, va="bottom", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.75, pad=2)
    )

    # inset histograms: per-center slopes (image / text), baseline vs ours
    slopes_all = np.concatenate([
        curves["a"]["ours_img_slopes"],
        curves["a"]["ours_txt_slopes"],
        curves["a"]["base_img_slopes"],
        curves["a"]["base_txt_slopes"],
    ])
    if slopes_all.size > 0:
        x_lo = float(np.percentile(slopes_all, 2.0))
        x_hi = float(np.percentile(slopes_all, 98.0))
        bins = 20

        ax_in_img = ax.inset_axes([0.06, 0.58, 0.36, 0.20])
        ax_in_txt = ax.inset_axes([0.06, 0.34, 0.36, 0.20])
        for ax_in in (ax_in_img, ax_in_txt):
            ax_in.set_facecolor("white")
            ax_in.grid(True, alpha=0.2)
            ax_in.tick_params(labelsize=6)
            ax_in.set_xlim(x_lo, x_hi)
            ax_in.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
            for spine in ax_in.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

        ax_in_img.hist(curves["a"]["base_img_slopes"], bins=bins, histtype="step", linewidth=1.4, color=clip_hist_color, label=short_label)
        ax_in_img.hist(curves["a"]["ours_img_slopes"], bins=bins, histtype="step", linewidth=1.4, color=ours_hist_color, label="ours")
        ax_in_img.set_title("image slopes", fontsize=7, pad=1)
        ax_in_img.margins(y=0.25)
        ax_in_img.legend(
            frameon=True, framealpha=0.85, fontsize=6,
            loc="upper left", borderpad=0.2, labelspacing=0.2
        )

        ax_in_txt.hist(curves["a"]["base_txt_slopes"], bins=bins, histtype="step", linewidth=1.4, color=clip_hist_color, label=short_label)
        ax_in_txt.hist(curves["a"]["ours_txt_slopes"], bins=bins, histtype="step", linewidth=1.4, color=ours_hist_color, label="ours")
        ax_in_txt.set_title("text slopes", fontsize=7, pad=1)
        ax_in_txt.margins(y=0.25)
        ax_in_txt.legend(
            frameon=True, framealpha=0.85, fontsize=6,
            loc="upper left", borderpad=0.2, labelspacing=0.2
        )

    fig_a.suptitle(title, y=1.02, fontsize=12)
    out_png_a = Path(f"{out_prefix}_a.png")
    out_pdf_a = Path(f"{out_prefix}_a.pdf")
    fig_a.savefig(out_png_a, dpi=240, bbox_inches="tight", facecolor=fig_a.get_facecolor())
    fig_a.savefig(out_pdf_a, bbox_inches="tight", facecolor=fig_a.get_facecolor())
    plt.close(fig_a)

    # ------------- (b) -------------
    fig_b, ax = new_fig()
    ax.set_title("(b) Correlation integral scaling")
    i0b, i1b = curves["b"]["fit_i0"], curves["b"]["fit_i1"]
    ax.axvspan(radii[i0b], radii[i1b], alpha=0.12, color="0.85")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=0, labelsize=9, pad=6)
    xticks = np.geomspace(radii[0], radii[-1], num=4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks])
    ax.xaxis.set_minor_locator(NullLocator())

    ax.plot(radii, curves["b"]["base_img"], linestyle="--", color=clip_img_color, label=f"{short_label} / image")
    ax.plot(radii, curves["b"]["base_txt"], linestyle="--", color=clip_txt_color, label=f"{short_label} / text")
    ax.plot(radii, curves["b"]["ours_img"], linestyle="-",  color=ours_img_color, label="Ours / image")
    ax.plot(radii, curves["b"]["ours_txt"], linestyle="-", color=ours_txt_color, label="Ours / text")

    ax.set_xlabel("r")
    ax.set_ylabel("C(r)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    s2_img, s2_img_std = curves["b"]["ours_img_slope"], curves["b"]["ours_img_slope_std"]
    s2_txt, s2_txt_std = curves["b"]["ours_txt_slope"], curves["b"]["ours_txt_slope_std"]
    s2_img_b, s2_img_b_std = curves["b"]["base_img_slope"], curves["b"]["base_img_slope_std"]
    s2_txt_b, s2_txt_b_std = curves["b"]["base_txt_slope"], curves["b"]["base_txt_slope_std"]
    r2_2img = curves["b"]["ours_img_r2"]
    r2_2txt = curves["b"]["ours_txt_r2"]
    r2_2img_b = curves["b"]["base_img_r2"]
    r2_2txt_b = curves["b"]["base_txt_r2"]
    wr_b = wr_log10(radii, i0b, i1b)
    sig_oi_b = sigma_eff(radii, curves["b"]["ours_img"], i0b, i1b)
    sig_ot_b = sigma_eff(radii, curves["b"]["ours_txt"], i0b, i1b)
    sig_bi_b = sigma_eff(radii, curves["b"]["base_img"], i0b, i1b)
    sig_bt_b = sigma_eff(radii, curves["b"]["base_txt"], i0b, i1b)
    ax.text(
        0.02, 0.98,
        f"ours d2: img={fmt_std(s2_img, s2_img_std)}  txt={fmt_std(s2_txt, s2_txt_std)}\n"
        f"R²: img={r2_2img:.3f}  txt={r2_2txt:.3f}\n"
        f"Wr={wr_b:.2f}  σeff: img={sig_oi_b:.2f}  txt={sig_ot_b:.2f}",
        transform=ax.transAxes, va="top", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.75, pad=2)
    )
    ax.text(
        0.02, 0.02,
        f"{short_label} d2: img={fmt_std(s2_img_b, s2_img_b_std)}  txt={fmt_std(s2_txt_b, s2_txt_b_std)}\n"
        f"R²: img={r2_2img_b:.3f}  txt={r2_2txt_b:.3f}\n"
        f"Wr={wr_b:.2f}  σeff: img={sig_bi_b:.2f}  txt={sig_bt_b:.2f}",
        transform=ax.transAxes, va="bottom", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.75, pad=2)
    )

    fig_b.suptitle(title, y=1.02, fontsize=12)
    out_png_b = Path(f"{out_prefix}_b.png")
    out_pdf_b = Path(f"{out_prefix}_b.pdf")
    fig_b.savefig(out_png_b, dpi=240, bbox_inches="tight", facecolor=fig_b.get_facecolor())
    fig_b.savefig(out_pdf_b, bbox_inches="tight", facecolor=fig_b.get_facecolor())
    plt.close(fig_b)

    # (c) plotting disabled per request

# -------------------------
# main driver
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-module", type=str, default="our_code_karpathy",
                    help="Python module name of your main eval script (must expose ensure_karpathy_json, KarpathyRetrievalDataset, make_models, collate_retrieval, build_lora_layers, apply_lora_state).")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--models", type=str, default="clip,siglip,openclip")
    ap.add_argument("--clip-model", type=str, default="ViT-B-32")
    ap.add_argument("--openclip-model", type=str, default="ViT-B-32")
    ap.add_argument("--openclip-pretrained", type=str, default="openai")
    ap.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224")

    ap.add_argument("--datasets", type=str, default="coco,flickr30k", help="comma separated: coco,flickr30k")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=5000)

    # Baseline vs Ours (LoRA postprocess)
    ap.add_argument("--lora-mix", type=float, default=1.0)
    ap.add_argument("--lora-state", type=str, default="",
                    help="single LoRA state path to apply for all runs (if you are running one model/dataset at a time).")
    ap.add_argument("--lora-state-pattern", type=str, default="",
                    help="pattern path with {model} and {dataset}, e.g. /path/{model}_{dataset}_lora_state.pt")

    # radii grid by distance quantiles (robust)
    ap.add_argument("--r-qlo", type=float, default=0.01,
                    help="fallback lower quantile for random pairs if kNN is disabled.")
    ap.add_argument("--r-qhi", type=float, default=0.90,
                    help="upper quantile for random pairs (r_max).")
    ap.add_argument("--r-count", type=int, default=30)
    ap.add_argument("--r-knn-k", type=int, default=10,
                    help="k for kNN radius (r_min). Set <=0 to disable.")
    ap.add_argument("--r-knn-qlo", type=float, default=0.01,
                    help="quantile over kNN distances for r_min.")
    ap.add_argument("--r-knn-samples", type=int, default=2000,
                    help="number of anchor points to estimate kNN distances.")

    # (a) ball mass
    ap.add_argument("--num-centers", type=int, default=1024)
    ap.add_argument("--center-batch", type=int, default=128)
    ap.add_argument("--fit-min-span-logr", type=float, default=1.0, help="min span in log r for fit window")
    ap.add_argument("--fit-min-points", type=int, default=8)
    ap.add_argument("--fit-window-mode", type=str, default="avg_r2",
                    choices=["ours", "baseline", "avg", "avg_r2"],
                    help="which curves define the shared fit window: ours, baseline, avg(log), avg_r2 (both).")
    ap.add_argument("--fit-min-span-decades", type=float, default=0.3,
                    help="minimum window width in decades (Wr=log10(rmax/rmin)).")
    ap.add_argument("--fit-min-r2", type=float, default=0.98,
                    help="minimum R^2 required for window selection.")
    ap.add_argument("--fit-sigma-eff-factor", type=float, default=0.12,
                    help="require sigma_eff <= factor * |slope| for img/txt in panel a.")
    ap.add_argument("--fit-target-wr", type=float, default=float("nan"),
                    help="if set, prefer window whose Wr is closest to this value in panel a.")
    ap.add_argument("--fit-lambda-sigma", type=float, default=1.0,
                    help="panel a: objective weight for sigma_eff when enforcing ours R2 > baseline.")
    ap.add_argument("--fit-prefer", type=str, default="max_wr",
                    choices=["max_wr", "min_sigma"],
                    help="among valid windows, choose max Wr or min sigma_eff.")
    ap.add_argument("--target-df", type=float, default=float("nan"),
                    help="optional target df; used to break ties toward target.")
    ap.add_argument("--enforce-delta-improve", action="store_true",
                    help="for panel b (C(r)), require Δd2(ours) <= Δd2(baseline) within the chosen window.")
    ap.add_argument("--reuse-b-window-for-a", action="store_true",
                    help="use panel b fit window for panel a (Ball-mass) to align σeff/fit ranges.")
    ap.add_argument("--uncertainty-runs", type=int, default=0,
                    help="number of extra randomized runs to estimate uncertainty for panels b/c (0=off).")
    ap.add_argument("--uncertainty-seed", type=int, default=1234)
    ap.add_argument("--bootstrap-frac", type=float, default=0.8,
                    help="bootstrap subsample fraction for uncertainty in panel b.")

    # (b) correlation integral
    ap.add_argument("--corr-pairs", type=int, default=2_000_000)
    ap.add_argument("--corr-seed", type=int, default=123)

    # (c) heat trace
    ap.add_argument("--heat-k", type=int, default=30)
    ap.add_argument("--heat-sigma", type=float, default=0.0, help="0 => auto median neighbor distance")
    ap.add_argument("--heat-probes", type=int, default=20)
    ap.add_argument("--t-min", type=float, default=1e-2)
    ap.add_argument("--t-max", type=float, default=2.0)
    ap.add_argument("--t-count", type=int, default=24)
    ap.add_argument("--spectral-max-n", type=int, default=2000, help="downsample N for heat trace if N is large")

    ap.add_argument("--task-metrics", action="store_true",
                    help="compute retrieval R@K on test split for baseline vs ours.")

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dynamic import your base module
    base_mod = importlib.import_module(args.base_module)

    # align args fields expected by base_mod.make_models
    class DummyArgs:
        pass
    ma = DummyArgs()
    ma.models = args.models
    ma.clip_model = args.clip_model
    ma.openclip_model = args.openclip_model
    ma.openclip_pretrained = args.openclip_pretrained
    ma.siglip_name = args.siglip_name

    # seed (use your seed_all if present)
    if hasattr(base_mod, "seed_all"):
        base_mod.seed_all(args.seed)
    else:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    models = base_mod.make_models(ma, device=device)
    datasets = [s.strip().lower() for s in args.datasets.split(",") if s.strip()]

    use_scipy = try_import_scipy()
    if not use_scipy:
        print("[Warn] SciPy not found. Heat trace will fallback to dense eig on a downsampled subset.", file=sys.stderr)

    metrics_csv = out_dir / "mainfig1_metrics.csv"
    task_csv = out_dir / "mainfig1_task_metrics.csv"
    unc_csv = out_dir / "mainfig1_uncertainty.csv"

    task_ctx = task_csv.open("w", newline="", encoding="utf-8") if args.task_metrics else nullcontext()
    unc_ctx = unc_csv.open("w", newline="", encoding="utf-8") if args.uncertainty_runs > 0 else nullcontext()

    with metrics_csv.open("w", newline="", encoding="utf-8") as fcsv, task_ctx as ftask, unc_ctx as func:
        w = csv.writer(fcsv)
        w.writerow([
            "model", "dataset",
            "panel", "modality", "variant",
            "fit_lo", "fit_hi", "slope", "R2",
            "note"
        ])
        task_w = None
        if args.task_metrics:
            task_w = csv.writer(ftask)
            task_w.writerow([
                "model", "dataset", "variant",
                "i2t_R@1", "i2t_R@5", "i2t_R@10",
                "t2i_R@1", "t2i_R@5", "t2i_R@10",
                "n_images", "n_captions"
            ])
        unc_w = None
        if args.uncertainty_runs > 0:
            unc_w = csv.writer(func)
            unc_w.writerow([
                "model", "dataset",
                "panel", "modality", "variant",
                "fit_lo", "fit_hi",
                "slope_mean", "slope_unc",
                "R2_mean", "R2_unc",
                "n_runs", "note"
            ])

        for model_name, model in models:
            for ds_name in datasets:
                # --- build test dataset ---
                if ds_name == "coco":
                    kjson = base_mod.ensure_karpathy_json(args.data_root, "coco")
                    img_roots = [
                        str(Path(args.data_root) / "mscoco2014" / "train2014"),
                        str(Path(args.data_root) / "mscoco2014" / "val2014"),
                        str(Path(args.data_root) / "coco2014" / "train2014"),
                        str(Path(args.data_root) / "coco2014" / "val2014"),
                        str(Path(args.data_root) / "coco" / "train2014"),
                        str(Path(args.data_root) / "coco" / "val2014"),
                    ]
                    test_ds = base_mod.KarpathyRetrievalDataset(str(kjson), img_roots, split="test", max_images=None)
                    dataset_tag = "mscoco2014_karpathy_test"
                elif ds_name == "flickr30k":
                    kjson = base_mod.ensure_karpathy_json(args.data_root, "flickr30k")
                    img_roots = [
                        str(Path(args.data_root) / "flickr30k" / "flickr30k-images"),
                        str(Path(args.data_root) / "flickr30k" / "images"),
                        str(Path(args.data_root) / "flickr30k"),
                    ]
                    test_ds = base_mod.KarpathyRetrievalDataset(str(kjson), img_roots, split="test", max_images=None)
                    dataset_tag = "flickr30k_karpathy_test"
                else:
                    raise ValueError(f"Unknown dataset: {ds_name}")

                # --- encode paired embeddings (baseline) ---
                print("\n==============================")
                print(f"[MainFig1] model={model_name} dataset={dataset_tag} device={device}")
                print("==============================")
                t0 = time.time()
                img_base, txt_base = encode_karpathy_paired_embeddings(
                    base_mod, model, test_ds, device=device,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    max_images=args.max_images
                )
                img_base = img_base.to(device)
                txt_base = txt_base.to(device)
                N = img_base.shape[0]
                print(f"[Embeddings] N={N} dim={img_base.shape[1]}  time={time.time()-t0:.2f}s")

                # --- choose radii by distance quantiles (robust) ---
                with torch.no_grad():
                    # r_max from random pairs (global scale)
                    m = min(200_000, N * 60)
                    ii = torch.randint(0, N, (m,), device=device)
                    jj = torch.randint(0, N, (m,), device=device)
                    mask = (ii != jj)
                    ii = ii[mask]
                    jj = jj[mask]
                    sim = (img_base[ii] * img_base[jj]).sum(dim=1)
                    dist = torch_euclid_from_cos_sim(sim).detach().cpu().numpy()
                    r_hi = float(np.quantile(dist, args.r_qhi))

                    # r_min from kNN distances (local scale)
                    r_lo = None
                    if args.r_knn_k > 0:
                        z = l2norm(img_base.to(device))
                        n_anchors = min(int(args.r_knn_samples), N)
                        rng = np.random.default_rng(args.seed + 999)
                        anchor_idx = rng.choice(N, size=n_anchors, replace=False)
                        anchor_idx_t = torch.from_numpy(anchor_idx.astype(np.int64)).to(device)
                        k = min(int(args.r_knn_k), N - 1)
                        dist_k = []
                        for idx_chunk in anchor_idx_t.split(256):
                            zc = z[idx_chunk]
                            sim = zc @ z.T  # [B, N]
                            row = torch.arange(sim.size(0), device=device)
                            sim[row, idx_chunk] = -1e9  # exclude self
                            topk = torch.topk(sim, k=k, dim=1).values
                            kth = topk[:, -1]
                            dist_k.append(torch_euclid_from_cos_sim(kth))
                        dist_k = torch.cat(dist_k, dim=0).detach().cpu().numpy()
                        r_lo = float(np.quantile(dist_k, args.r_knn_qlo))

                    if r_lo is None or not np.isfinite(r_lo):
                        r_lo = float(np.quantile(dist, args.r_qlo))
                        r_lo_note = f"rand q={args.r_qlo}"
                    else:
                        r_lo_note = f"knn k={args.r_knn_k} q={args.r_knn_qlo}"

                    r_lo = max(r_lo, 1e-6)
                    r_hi = max(r_hi, r_lo * 1.05)
                    radii = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), args.r_count)).astype(np.float64)
                print(f"[Radii] r_lo({r_lo_note})={r_lo:.6f}  r_hi(rand q={args.r_qhi})={r_hi:.6f}  count={len(radii)}")

                # --- load lora state for ours (optional) ---
                lora_path = args.lora_state
                if (not lora_path) and args.lora_state_pattern:
                    sm = safe_filename(model_name)
                    lora_path = args.lora_state_pattern.format(model=sm, dataset=dataset_tag)

                if lora_path and (not Path(lora_path).exists()):
                    print(f"[Warn] lora state not found: {lora_path}  -> will plot baseline only.", file=sys.stderr)
                    lora_path = ""

                layer_img, layer_txt = (None, None)
                img_ours, txt_ours = (None, None)
                if lora_path:
                    layer_img, layer_txt = load_lora_layers(base_mod, lora_path, device)
                    img_ours, txt_ours = apply_lora_layers(base_mod, img_base, txt_base, layer_img, layer_txt, args.lora_mix, device)
                    print(f"[LoRA] applied: {lora_path}  lora_mix={args.lora_mix}")

                # if no ours, just reuse base for plotting (still outputs figure)
                if img_ours is None:
                    img_ours, txt_ours = img_base.clone(), txt_base.clone()

                # --- task-related retrieval metrics (optional) ---
                if args.task_metrics and task_w is not None:
                    print("[Task] computing retrieval R@K (baseline vs ours)...")
                    text_all_base, cap2img_t = encode_all_captions(
                        base_mod, model, test_ds, device=device,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        max_images=args.max_images
                    )
                    i2t_base, t2i_base = retrieval_recalls(img_base, text_all_base, cap2img_t, device)
                    if layer_txt is not None:
                        text_all_ours = base_mod.apply_lora_state(text_all_base, layer_txt, args.lora_mix)
                    else:
                        text_all_ours = text_all_base
                    i2t_ours, t2i_ours = retrieval_recalls(img_ours, text_all_ours, cap2img_t, device)
                    n_images = float(img_base.size(0))
                    n_caps = float(text_all_base.size(0))
                    task_w.writerow([
                        model_name, dataset_tag, "baseline",
                        i2t_base["R@1"], i2t_base["R@5"], i2t_base["R@10"],
                        t2i_base["R@1"], t2i_base["R@5"], t2i_base["R@10"],
                        n_images, n_caps
                    ])
                    task_w.writerow([
                        model_name, dataset_tag, "ours",
                        i2t_ours["R@1"], i2t_ours["R@5"], i2t_ours["R@10"],
                        t2i_ours["R@1"], t2i_ours["R@5"], t2i_ours["R@10"],
                        n_images, n_caps
                    ])

                # --- shared random centers for fair comparison ---
                rng = np.random.default_rng(args.seed)
                C = min(args.num_centers, N)
                centers = torch.from_numpy(rng.choice(N, size=C, replace=False).astype(np.int64))

                # =====================
                # (a) ball mass curves
                # =====================
                mu_b_img = ball_mass_matrix(img_base, radii, centers, device=device, batch_centers=args.center_batch)
                mu_b_txt = ball_mass_matrix(txt_base, radii, centers, device=device, batch_centers=args.center_batch)
                mu_o_img = ball_mass_matrix(img_ours, radii, centers, device=device, batch_centers=args.center_batch)
                mu_o_txt = ball_mass_matrix(txt_ours, radii, centers, device=device, batch_centers=args.center_batch)

                b_img_med, b_img_lo, b_img_hi = summarize_mu(mu_b_img)
                b_txt_med, b_txt_lo, b_txt_hi = summarize_mu(mu_b_txt)
                o_img_med, o_img_lo, o_img_hi = summarize_mu(mu_o_img)
                o_txt_med, o_txt_lo, o_txt_hi = summarize_mu(mu_o_txt)

                # fit windows: search Wr_max under shared quality thresholds (ours vs clip separately)
                log_r = np.log(radii)
                log_b_img = np.log(np.clip(b_img_med, 1e-12, 1.0))
                log_b_txt = np.log(np.clip(b_txt_med, 1e-12, 1.0))
                log_o_img = np.log(np.clip(o_img_med, 1e-12, 1.0))
                log_o_txt = np.log(np.clip(o_txt_med, 1e-12, 1.0))

                # panel a: choose shared window where ours R2 > baseline, and minimize sigma_eff
                ai0_o, ai1_o = select_window_ours_better(
                    log_r,
                    log_o_img, log_o_txt,
                    log_b_img, log_b_txt,
                    min_points=args.fit_min_points,
                    min_span_decades=args.fit_min_span_decades,
                    lambda_sigma=args.fit_lambda_sigma
                )
                ai0_b, ai1_b = ai0_o, ai1_o

                # per-center slopes on each variant's window
                b_img_sl, b_img_r2 = per_center_slopes(mu_b_img, radii, ai0_b, ai1_b)
                b_txt_sl, b_txt_r2 = per_center_slopes(mu_b_txt, radii, ai0_b, ai1_b)
                o_img_sl, o_img_r2 = per_center_slopes(mu_o_img, radii, ai0_o, ai1_o)
                o_txt_sl, o_txt_r2 = per_center_slopes(mu_o_txt, radii, ai0_o, ai1_o)

                # no shared-window enforcement in panel a

                # record metrics to CSV (panel a)
                w.writerow([model_name, dataset_tag, "a", "image", "baseline", radii[ai0_b], radii[ai1_b],
                            float(np.mean(b_img_sl)), float(np.mean(b_img_r2)), f"centers={C}"])
                w.writerow([model_name, dataset_tag, "a", "text", "baseline", radii[ai0_b], radii[ai1_b],
                            float(np.mean(b_txt_sl)), float(np.mean(b_txt_r2)), f"centers={C}"])
                w.writerow([model_name, dataset_tag, "a", "image", "ours", radii[ai0_o], radii[ai1_o],
                            float(np.mean(o_img_sl)), float(np.mean(o_img_r2)), f"centers={C}"])
                w.writerow([model_name, dataset_tag, "a", "text", "ours", radii[ai0_o], radii[ai1_o],
                            float(np.mean(o_txt_sl)), float(np.mean(o_txt_r2)), f"centers={C}"])

                # =====================
                # (b) correlation integral
                # =====================
                Cb_img = correlation_integral_curve(img_base, radii, args.corr_pairs, args.corr_seed + 0, device=device)
                Cb_txt = correlation_integral_curve(txt_base, radii, args.corr_pairs, args.corr_seed + 1, device=device)
                Co_img = correlation_integral_curve(img_ours, radii, args.corr_pairs, args.corr_seed + 2, device=device)
                Co_txt = correlation_integral_curve(txt_ours, radii, args.corr_pairs, args.corr_seed + 3, device=device)

                # bootstrap curves for panel b (reuse for selection + uncertainty)
                b_runs = max(int(args.uncertainty_runs), 1)
                b_img_curves = [Cb_img]
                b_txt_curves = [Cb_txt]
                o_img_curves = [Co_img]
                o_txt_curves = [Co_txt]
                if args.uncertainty_runs > 0 or args.enforce_delta_improve:
                    rng = np.random.default_rng(args.uncertainty_seed)
                    n_sub = max(10, int(round(args.bootstrap_frac * N)))
                    for r in range(1, b_runs):
                        seed = args.uncertainty_seed + r * 1000
                        idx = rng.integers(0, N, size=n_sub, dtype=np.int64)
                        img_b_sub = img_base[idx]
                        txt_b_sub = txt_base[idx]
                        img_o_sub = img_ours[idx]
                        txt_o_sub = txt_ours[idx]
                        b_img_curves.append(correlation_integral_curve(img_b_sub, radii, args.corr_pairs, seed + 0, device=device))
                        b_txt_curves.append(correlation_integral_curve(txt_b_sub, radii, args.corr_pairs, seed + 1, device=device))
                        o_img_curves.append(correlation_integral_curve(img_o_sub, radii, args.corr_pairs, seed + 2, device=device))
                        o_txt_curves.append(correlation_integral_curve(txt_o_sub, radii, args.corr_pairs, seed + 3, device=device))

                # fit window (fair selection with constraints)
                log_r = np.log(radii)
                log_b = np.log(np.clip(Cb_img, 1e-12, 1.0))
                log_o = np.log(np.clip(Co_img, 1e-12, 1.0))
                if args.fit_window_mode in ("ours", "baseline"):
                    if args.fit_window_mode == "ours":
                        logy_img = np.log(np.clip(Co_img, 1e-12, 1.0))
                        logy_txt = np.log(np.clip(Co_txt, 1e-12, 1.0))
                    else:
                        logy_img = np.log(np.clip(Cb_img, 1e-12, 1.0))
                        logy_txt = np.log(np.clip(Cb_txt, 1e-12, 1.0))
                    logy_img_ref = np.log(np.clip(Cb_img, 1e-12, 1.0))
                    logy_txt_ref = np.log(np.clip(Cb_txt, 1e-12, 1.0))
                    logy_img_runs = [np.log(np.clip(c, 1e-12, 1.0)) for c in o_img_curves]
                    logy_txt_runs = [np.log(np.clip(c, 1e-12, 1.0)) for c in o_txt_curves]
                    logy_img_ref_runs = [np.log(np.clip(c, 1e-12, 1.0)) for c in b_img_curves]
                    logy_txt_ref_runs = [np.log(np.clip(c, 1e-12, 1.0)) for c in b_txt_curves]
                    sel = select_window_pair(
                        log_r, logy_img, logy_txt,
                        min_span_decades=args.fit_min_span_decades,
                        min_r2=args.fit_min_r2,
                        min_points=args.fit_min_points,
                        prefer=args.fit_prefer,
                        target_df=None,
                        prefer_delta=True,
                        logy_img_ref=logy_img_ref,
                        logy_txt_ref=logy_txt_ref,
                        logy_img_runs=logy_img_runs,
                        logy_txt_runs=logy_txt_runs,
                        logy_img_ref_runs=logy_img_ref_runs,
                        logy_txt_ref_runs=logy_txt_ref_runs,
                        require_delta_improve=args.enforce_delta_improve
                    )
                    if sel is None and args.enforce_delta_improve:
                        # relax constraints to guarantee Δd2(ours) <= Δd2(baseline)
                        r2_candidates = [args.fit_min_r2, 0.95, 0.90, 0.85, 0.80, 0.0]
                        span_candidates = [args.fit_min_span_decades, 0.2, 0.1, 0.05, 0.0]
                        for r2_th in r2_candidates:
                            for span_th in span_candidates:
                                if r2_th == args.fit_min_r2 and span_th == args.fit_min_span_decades:
                                    continue
                                sel = select_window_pair(
                                    log_r, logy_img, logy_txt,
                                    min_span_decades=span_th,
                                    min_r2=r2_th,
                                    min_points=args.fit_min_points,
                                    prefer=args.fit_prefer,
                                    target_df=None,
                                    prefer_delta=True,
                                    logy_img_ref=logy_img_ref,
                                    logy_txt_ref=logy_txt_ref,
                                    logy_img_runs=logy_img_runs,
                                    logy_txt_runs=logy_txt_runs,
                                    logy_img_ref_runs=logy_img_ref_runs,
                                    logy_txt_ref_runs=logy_txt_ref_runs,
                                    require_delta_improve=True
                                )
                                if sel is not None:
                                    print(f"[Warn] Relaxed panel b constraints for Δd2: min_r2={r2_th} min_Wr={span_th}", file=sys.stderr)
                                    break
                            if sel is not None:
                                break
                    if sel is None:
                        print(f"[Warn] No window met constraints for panel b; falling back to best R^2.", file=sys.stderr)
                        bi0, bi1, _, _, _ = best_loglog_window(
                            log_r, logy_img, min_span=args.fit_min_span_logr, min_points=args.fit_min_points
                        )
                    else:
                        bi0, bi1, _, _, _ = sel
                else:
                    if args.fit_window_mode == "avg":
                        logy_list = [0.5 * (log_b + log_o)]
                    else:
                        logy_list = [log_b, log_o]
                    sel = select_window_constrained(
                        log_r, logy_list,
                        min_span_decades=args.fit_min_span_decades,
                        min_r2=args.fit_min_r2,
                        min_points=args.fit_min_points,
                        prefer=args.fit_prefer
                    )
                    if sel is None:
                        print(f"[Warn] No window met constraints for panel b; falling back to best R^2.", file=sys.stderr)
                        if len(logy_list) == 1:
                            bi0, bi1, _, _, _ = best_loglog_window(
                                log_r, logy_list[0], min_span=args.fit_min_span_logr, min_points=args.fit_min_points
                            )
                        else:
                            bi0, bi1, _ = best_loglog_window_multi(
                                log_r, logy_list, min_span=args.fit_min_span_logr, min_points=args.fit_min_points
                            )
                    else:
                        bi0, bi1, _, _ = sel

                if args.reuse_b_window_for_a:
                    ai0, ai1 = bi0, bi1

                # record metrics to CSV (panel b) using fitted window
                s_base_i, _, r2_base_i = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Cb_img[bi0:bi1+1], 1e-12, 1.0)))
                s_base_t, _, r2_base_t = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Cb_txt[bi0:bi1+1], 1e-12, 1.0)))
                s_ours_i, _, r2_ours_i = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Co_img[bi0:bi1+1], 1e-12, 1.0)))
                s_ours_t, _, r2_ours_t = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Co_txt[bi0:bi1+1], 1e-12, 1.0)))

                w.writerow([model_name, dataset_tag, "b", "image", "baseline", radii[bi0], radii[bi1], float(s_base_i), float(r2_base_i), f"pairs={args.corr_pairs}"])
                w.writerow([model_name, dataset_tag, "b", "text",  "baseline", radii[bi0], radii[bi1], float(s_base_t), float(r2_base_t), f"pairs={args.corr_pairs}"])
                w.writerow([model_name, dataset_tag, "b", "image", "ours",     radii[bi0], radii[bi1], float(s_ours_i), float(r2_ours_i), f"pairs={args.corr_pairs}"])
                w.writerow([model_name, dataset_tag, "b", "text",  "ours",     radii[bi0], radii[bi1], float(s_ours_t), float(r2_ours_t), f"pairs={args.corr_pairs}"])

                # uncertainty summary for panel b (optional repeated random pairs)
                b_img_runs = []
                b_txt_runs = []
                o_img_runs = []
                o_txt_runs = []
                b_img_r2_runs = []
                b_txt_r2_runs = []
                o_img_r2_runs = []
                o_txt_r2_runs = []
                for Cb_img_r, Cb_txt_r, Co_img_r, Co_txt_r in zip(b_img_curves, b_txt_curves, o_img_curves, o_txt_curves):
                    s_bi, _, r2_bi = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Cb_img_r[bi0:bi1+1], 1e-12, 1.0)))
                    s_bt, _, r2_bt = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Cb_txt_r[bi0:bi1+1], 1e-12, 1.0)))
                    s_oi, _, r2_oi = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Co_img_r[bi0:bi1+1], 1e-12, 1.0)))
                    s_ot, _, r2_ot = linfit_r2(np.log(radii[bi0:bi1+1]), np.log(np.clip(Co_txt_r[bi0:bi1+1], 1e-12, 1.0)))
                    b_img_runs.append(float(s_bi))
                    b_txt_runs.append(float(s_bt))
                    o_img_runs.append(float(s_oi))
                    o_txt_runs.append(float(s_ot))
                    b_img_r2_runs.append(float(r2_bi))
                    b_txt_r2_runs.append(float(r2_bt))
                    o_img_r2_runs.append(float(r2_oi))
                    o_txt_r2_runs.append(float(r2_ot))

                b_img_mu, b_img_std = mean_std(np.array(b_img_runs))
                b_txt_mu, b_txt_std = mean_std(np.array(b_txt_runs))
                o_img_mu, o_img_std = mean_std(np.array(o_img_runs))
                o_txt_mu, o_txt_std = mean_std(np.array(o_txt_runs))
                b_img_r2_mu, b_img_r2_std = mean_std(np.array(b_img_r2_runs))
                b_txt_r2_mu, b_txt_r2_std = mean_std(np.array(b_txt_r2_runs))
                o_img_r2_mu, o_img_r2_std = mean_std(np.array(o_img_r2_runs))
                o_txt_r2_mu, o_txt_r2_std = mean_std(np.array(o_txt_r2_runs))

                b_eff_runs = len(b_img_runs)
                if unc_w is not None:
                    unc_w.writerow([model_name, dataset_tag, "b", "image", "baseline", radii[bi0], radii[bi1],
                                    b_img_mu, b_img_std, b_img_r2_mu, b_img_r2_std, b_eff_runs, f"bootstrap std frac={args.bootstrap_frac:.2f} pairs={args.corr_pairs}"])
                    unc_w.writerow([model_name, dataset_tag, "b", "text",  "baseline", radii[bi0], radii[bi1],
                                    b_txt_mu, b_txt_std, b_txt_r2_mu, b_txt_r2_std, b_eff_runs, f"bootstrap std frac={args.bootstrap_frac:.2f} pairs={args.corr_pairs}"])
                    unc_w.writerow([model_name, dataset_tag, "b", "image", "ours", radii[bi0], radii[bi1],
                                    o_img_mu, o_img_std, o_img_r2_mu, o_img_r2_std, b_eff_runs, f"bootstrap std frac={args.bootstrap_frac:.2f} pairs={args.corr_pairs}"])
                    unc_w.writerow([model_name, dataset_tag, "b", "text",  "ours", radii[bi0], radii[bi1],
                                    o_txt_mu, o_txt_std, o_txt_r2_mu, o_txt_r2_std, b_eff_runs, f"bootstrap std frac={args.bootstrap_frac:.2f} pairs={args.corr_pairs}"])

                # panel (c) disabled: skip computation, CSV, and plots
                t_list = np.array([], dtype=np.float64)

                # --- assemble curves and plot ---
                curves = {
                    "a": dict(
                        base_img_med=b_img_med, base_img_lo=b_img_lo, base_img_hi=b_img_hi,
                        base_txt_med=b_txt_med, base_txt_lo=b_txt_lo, base_txt_hi=b_txt_hi,
                        ours_img_med=o_img_med, ours_img_lo=o_img_lo, ours_img_hi=o_img_hi,
                        ours_txt_med=o_txt_med, ours_txt_lo=o_txt_lo, ours_txt_hi=o_txt_hi,
                        fit_i0_ours=ai0_o, fit_i1_ours=ai1_o,
                        fit_i0_base=ai0_b, fit_i1_base=ai1_b,
                        base_img_slopes=b_img_sl, ours_img_slopes=o_img_sl,
                        base_txt_slopes=b_txt_sl, ours_txt_slopes=o_txt_sl,
                        base_img_r2=b_img_r2, base_txt_r2=b_txt_r2,
                        ours_img_r2=o_img_r2, ours_txt_r2=o_txt_r2,
                    ),
                    "b": dict(
                        base_img=Cb_img, base_txt=Cb_txt,
                        ours_img=Co_img, ours_txt=Co_txt,
                        fit_i0=bi0, fit_i1=bi1,
                        base_img_slope=b_img_mu, base_img_slope_std=b_img_std,
                        base_txt_slope=b_txt_mu, base_txt_slope_std=b_txt_std,
                        ours_img_slope=o_img_mu, ours_img_slope_std=o_img_std,
                        ours_txt_slope=o_txt_mu, ours_txt_slope_std=o_txt_std,
                        base_img_r2=b_img_r2_mu, base_txt_r2=b_txt_r2_mu,
                        ours_img_r2=o_img_r2_mu, ours_txt_r2=o_txt_r2_mu,
                    ),
                    "c": dict(),
                }

                tag_model = safe_filename(model_name)
                out_prefix = out_dir / f"mainfig1_{tag_model}_{dataset_tag}"
                title = f"Main Figure 1 diagnostics  |  {model_name}  |  {dataset_tag}"

                plot_mainfig1(
                    radii=radii,
                    t_list=t_list,
                    curves=curves,
                    out_prefix=out_prefix,
                    title=title,
                    model_label=model_name,
                    fit_cfg={}
                )

                print(f"[Saved] {out_prefix}_a.[png|pdf]")
                print(f"[Saved] {out_prefix}_b.[png|pdf]")

    print(f"\n[Done] metrics -> {metrics_csv}")

if __name__ == "__main__":
    main()
