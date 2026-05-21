#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline #1 (multi-backbone): CLIP / OpenCLIP / SigLIP
- Datasets:
  * Retrieval: MSCOCO (coco2017), Flickr30k
  * Zero-shot classification: CIFAR100, Tiny-ImageNet-200, DTD
- Metrics (modality gap, retrieval):
  * centroid_distance
  * relative_modality_gap (RMG)
  * NAS@K  (per paper screenshot: bidirectional top-k hit on paired samples)
  * CMAS   (mean paired cosine similarity)
- Retrieval metrics:
  * I2T and T2I: R@1, R@5, R@10
- Zero-shot:
  * top1, top5 accuracy
- Save per-model per-dataset results to JSON + CSV.

Expected dataset layout under --data-root (default: /work/was598/modilty_gap/tools/data):
  coco2017/
    images/train2017, val2017
    annotations/captions_train2017.json, captions_val2017.json
  flickr30k/
    flickr30k-images/xxxx.jpg
    results_20130124.token
  cifar100/    (torchvision will download if missing)
  dtd/         (torchvision will download if missing)
  tiny-imagenet-200/ (either already prepared OR auto-download enabled)
"""

from __future__ import annotations
import os, json, time, math, random, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets as tvds
import torchvision.transforms as T

from tqdm import tqdm

# --------- Optional deps ----------
# OpenCLIP/CLIP
try:
    import open_clip
except Exception as e:
    open_clip = None

# SigLIP (Transformers)
try:
    from transformers import SiglipModel, SiglipProcessor
except Exception:
    SiglipModel = None
    SiglipProcessor = None


# =========================
# Repro / Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def gap_metrics_centroid_rmg(imgE: torch.Tensor, txtE: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """
    centroid_distance and relative_modality_gap (RMG).
    Uses trace(cov) proxy without materializing full covariance matrices.
    """
    imgE = l2norm(imgE)
    txtE = l2norm(txtE)
    mu_x = imgE.mean(dim=0)
    mu_y = txtE.mean(dim=0)
    centroid = torch.norm(mu_x - mu_y)

    Xc = imgE - mu_x
    Yc = txtE - mu_y
    trCx = (Xc * Xc).sum(dim=-1).mean()
    trCy = (Yc * Yc).sum(dim=-1).mean()
    denom = torch.sqrt(0.5 * (trCx + trCy) + eps)
    rmg = centroid / denom
    return {
        "centroid_distance": float(centroid.item()),
        "relative_modality_gap": float(rmg.item()),
    }

@torch.no_grad()
def cmas_nas_metrics(
    imgE: torch.Tensor,
    txtE: torch.Tensor,
    img2caps: Dict[int, List[int]],
    cap2imgs: Dict[int, List[int]],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Exactly as screenshot:
      CMAS = average cosine similarity of paired samples.
      NAS@k = average over paired samples of 1[y in topk(I->T)] * 1[x in topk(T->I)].
    Supports 1 image -> multiple captions by expanding paired samples (i, c).
    """
    imgE = l2norm(imgE)
    txtE = l2norm(txtE)
    S = imgE @ txtE.t()  # (N_img, N_cap)

    pairs_i = []
    pairs_c = []
    for i, caps in img2caps.items():
        for c in caps:
            pairs_i.append(i)
            pairs_c.append(c)
    if len(pairs_i) == 0:
        out = {"CMAS": float("nan")}
        for k in ks:
            out[f"NAS@{k}"] = float("nan")
        return out

    pairs_i = torch.tensor(pairs_i, device=S.device, dtype=torch.long)
    pairs_c = torch.tensor(pairs_c, device=S.device, dtype=torch.long)

    pair_sims = S[pairs_i, pairs_c]
    CMAS = pair_sims.mean()

    out = {"CMAS": float(CMAS.item())}
    maxk = max(ks)
    top_caps = torch.topk(S, k=maxk, dim=1, largest=True, sorted=True).indices  # (N_img,maxk)
    top_imgs = torch.topk(S.t(), k=maxk, dim=1, largest=True, sorted=True).indices  # (N_cap,maxk)

    for k in ks:
        caps_topk = top_caps[pairs_i, :k]
        imgs_topk = top_imgs[pairs_c, :k]
        hit_i2t = (caps_topk == pairs_c.unsqueeze(1)).any(dim=1)
        hit_t2i = (imgs_topk == pairs_i.unsqueeze(1)).any(dim=1)
        nas = (hit_i2t & hit_t2i).float().mean()
        out[f"NAS@{k}"] = float(nas.item())

    return out

@torch.no_grad()
def recalls_from_sim(sim: torch.Tensor, gt: Dict[int, List[int]], ks=(1,5,10)) -> Dict[str, float]:
    """
    sim: (N_query, N_gallery) on GPU
    gt: mapping query -> list of correct gallery indices
    """
    maxk = max(ks)
    topk = torch.topk(sim, k=maxk, dim=1, largest=True, sorted=True).indices
    topk = topk.cpu().numpy()
    out = {}
    for k in ks:
        ok = 0
        for i in range(topk.shape[0]):
            preds = set(topk[i, :k].tolist())
            if any(t in preds for t in gt[i]):
                ok += 1
        out[f"R@{k}"] = ok / topk.shape[0]
    return out


# =========================
# Datasets
# =========================
class CocoCaptionsEval(Dataset):
    """
    Returns (image_tensor, captions_list[str]) from torchvision CocoCaptions.
    """
    def __init__(self, data_root: str, split: str = "val", transform=None):
        droot = Path(data_root) / "coco2017"
        img_dir = droot / "images" / ("val2017" if split == "val" else "train2017")
        ann = droot / "annotations" / f"captions_{'val2017' if split=='val' else 'train2017'}.json"
        assert img_dir.exists(), f"COCO images not found: {img_dir}"
        assert ann.exists(), f"COCO captions not found: {ann}"
        self.ds = tvds.CocoCaptions(root=str(img_dir), annFile=str(ann), transform=transform)

    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, caps = self.ds[i]
        caps = caps if isinstance(caps, list) else [str(caps)]
        return img, caps

class Flickr30kEval(Dataset):
    """
    Old torchvision API:
      torchvision.datasets.Flickr30k(root=images_root, ann_file=token_file)
    Expected:
      <data_root>/flickr30k/flickr30k-images/*.jpg
      <data_root>/flickr30k/results_20130124.token
    """
    def __init__(self, data_root: str, transform=None):
        droot = Path(data_root) / "flickr30k"
        images_root = droot / "flickr30k-images"
        ann_file = droot / "results_20130124.token"
        assert images_root.exists(), f"Flickr images not found: {images_root}"
        assert ann_file.exists(), f"Flickr token file not found: {ann_file}"
        self.ds = tvds.Flickr30k(root=str(images_root), ann_file=str(ann_file), transform=transform)

    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, caps = self.ds[i]
        caps = caps if isinstance(caps, list) else [str(caps)]
        return img, caps

class TinyImageNet200(Dataset):
    """
    If not found under <data_root>/tiny-imagenet-200 and --auto-download-tiny is set,
    downloads and extracts the official zip into <data_root>.
    """
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    def __init__(self, data_root: str, split="val", transform=None, auto_download=False):
        self.data_root = Path(data_root)
        self.root = self.data_root / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        assert split in ["train", "val"]
        self._ensure(auto_download)

        wnids_path = self.root / "wnids.txt"
        words_path = self.root / "words.txt"
        assert wnids_path.exists(), f"wnids.txt not found under {self.root}"
        assert words_path.exists(), f"words.txt not found under {self.root}"

        wnids = [w.strip() for w in wnids_path.read_text().splitlines() if w.strip()]
        self.wnids = wnids
        self.class_to_idx = {w:i for i,w in enumerate(wnids)}

        words = {}
        for line in words_path.read_text().splitlines():
            if not line.strip(): continue
            k, v = line.split("\t")
            words[k] = v
        self.idx_to_name = {self.class_to_idx[w]: words.get(w, w).split(",")[0].split(";")[0] for w in wnids}

        self.samples: List[Tuple[str,int]] = []
        if split == "train":
            tdir = self.root / "train"
            for wnid in wnids:
                idir = tdir / wnid / "images"
                for p in idir.glob("*.JPEG"):
                    self.samples.append((str(p), self.class_to_idx[wnid]))
        else:
            vdir = self.root / "val"
            ann = vdir / "val_annotations.txt"
            assert ann.exists(), f"val_annotations.txt not found under {vdir}"
            mapping = {}
            for line in ann.read_text().splitlines():
                if not line.strip(): continue
                parts = line.split("\t")
                fn, wnid = parts[0], parts[1]
                mapping[fn] = wnid
            idir = vdir / "images"
            for p in idir.glob("*.JPEG"):
                wnid = mapping.get(p.name, None)
                if wnid is None: continue
                self.samples.append((str(p), self.class_to_idx[wnid]))

        if len(self.samples) == 0:
            raise RuntimeError("Tiny-ImageNet-200 appears empty or not prepared correctly.")

    def _ensure(self, auto_download: bool):
        if self.root.exists() and (self.root / "wnids.txt").exists():
            return
        if not auto_download:
            raise FileNotFoundError(
                f"Tiny-ImageNet-200 not found at {self.root}. "
                f"Either place it there or run with --auto-download-tiny."
            )
        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            import urllib.request, shutil
            print(f"[TinyImageNet] Downloading {self.URL} -> {zip_path}")
            with urllib.request.urlopen(self.URL) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)
        import zipfile
        print(f"[TinyImageNet] Extracting {zip_path} -> {self.data_root}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.data_root)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        fp, y = self.samples[i]
        img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    @property
    def classes(self) -> List[str]:
        return [self.idx_to_name[i] for i in range(len(self.idx_to_name))]


# =========================
# Prompt templates (classic CLIP)
# =========================
ZS_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
]


# =========================
# Model wrappers
# =========================
class VLModel:
    """
    Unified interface:
      encode_image(images[B,3,H,W]) -> feats[B,d] (L2-normed)
      encode_text(texts[List[str]]) -> feats[B,d] (L2-normed)
    """
    name: str
    device: str

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class OpenCLIPWrapper(VLModel):
    def __init__(self, model_name: str, pretrained: str, device: str):
        assert open_clip is not None, "open_clip is not installed. pip install open_clip_torch"
        self.name = f"open_clip:{model_name}:{pretrained}"
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(device).eval()
        self.preprocess = preprocess  # not used directly here (we use our own transforms)
        self.tokenizer = tokenizer
        # infer dim
        with torch.no_grad():
            d = self.model.text_projection.shape[1] if hasattr(self.model, "text_projection") else self.model.embed_dim
        self._dim = int(d)

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        feats = self.model.encode_image(images)
        feats = feats.float()
        feats = l2norm(feats)
        return feats

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts)
        if isinstance(tokens, dict):
            # should not happen for open_clip tokenizer, but keep safe
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            feats = self.model.encode_text(**tokens)
        else:
            tokens = tokens.to(self.device)
            feats = self.model.encode_text(tokens)
        feats = feats.float()
        feats = l2norm(feats)
        return feats


class SigLIPWrapper(VLModel):
    def __init__(self, hf_name: str, device: str, native_preprocess: bool = False):
        assert SiglipModel is not None and SiglipProcessor is not None, \
            "transformers SiglipModel not available. pip install transformers"
        self.name = f"siglip:{hf_name}"
        self.device = device
        self.native_preprocess = native_preprocess

        self.model = SiglipModel.from_pretrained(hf_name).to(device).eval()
        self.proc = SiglipProcessor.from_pretrained(hf_name)

        # ---- Robust dim inference (no config-field assumption) ----
        # Try common config names first, then fallback to actual forward output.
        d = None
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for key in ["projection_dim", "embed_dim", "hidden_size", "vision_embed_dim", "text_embed_dim"]:
                if hasattr(cfg, key):
                    val = getattr(cfg, key)
                    if isinstance(val, int) and val > 0:
                        d = val
                        break

        if d is None:
            # Fallback: run a tiny forward to infer dim.
            # Use processor to build correct pixel_values/input_ids shapes.
            with torch.no_grad():
                dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                if self.native_preprocess:
                    inputs = self.proc(images=dummy_img, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                else:
                    # If not native preprocess, we still can use proc to get pixel_values
                    # for dim inference only (no harm).
                    inputs = self.proc(images=dummy_img, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)

                feat = self.model.get_image_features(pixel_values=pixel_values)
                d = int(feat.shape[-1])

        self._dim = int(d)

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        If native_preprocess:
          images is expected to be a batch of PIL images (list) OR a torch tensor in [0,1] (unnormalized).
          We'll use SiglipProcessor.
        Else:
          images is a torch tensor already normalized by our transform; we pass it as pixel_values directly
          (fast, but not SigLIP-official normalization).
        """
        if self.native_preprocess:
            # images may come as torch tensor from DataLoader; convert to list of PIL for processor
            # This branch is slower; use only if you want strict SigLIP preprocess.
            if isinstance(images, torch.Tensor):
                # tensor is (B,3,H,W) in normalized space -> cannot faithfully invert.
                # So require you to build dataset transform that outputs raw PIL or unnormalized tensor.
                raise RuntimeError(
                    "SigLIP native_preprocess requires dataset to return PIL images (or unnormalized tensors). "
                    "Please set dataset transform=None and handle preprocess here, or add a flag to switch transforms."
                )
            inputs = self.proc(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            out = self.model.get_image_features(pixel_values=pixel_values)
        else:
            # Fast path: assume images is torch tensor already on GPU
            images = images.to(self.device, non_blocking=True)
            out = self.model.get_image_features(pixel_values=images)

        out = out.float()
        out = l2norm(out)
        return out

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.proc(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.get_text_features(**inputs)
        out = out.float()
        out = l2norm(out)
        return out


# =========================
# Transforms
# =========================
def build_clip_style_tf(image_size=224, train: bool=False) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])


# =========================
# Eval: Retrieval
# =========================
def retrieval_collate_fn(batch):
    """
    batch: list of (image_tensor, captions_list[str])
    return:
      images: (B,3,H,W) tensor
      caps_list: List[List[str]] length B, each inner list variable length
    """
    images = [b[0] for b in batch]
    caps_list = [b[1] for b in batch]

    # stack images
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)
    else:
        # if somehow PIL slips in
        raise TypeError("retrieval_collate_fn expects image tensors. Check your dataset transform.")

    # normalize caps_list to List[List[str]]
    norm_caps = []
    for caps in caps_list:
        if isinstance(caps, (tuple, list)):
            norm_caps.append([str(c) for c in caps])
        else:
            norm_caps.append([str(caps)])
    return images, norm_caps

@torch.no_grad()
def retrieval_eval(
    model: VLModel,
    ds: Dataset,
    device: str,
    batch_size: int = 128,
    max_images: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[Dict[str,float], Dict[str,float], Dict[str,float]]:
    """
    For COCO/Flickr datasets which return (image_tensor, captions_list[str]).
    We expand captions: each image has M captions (COCO/Flickr typically 5).
    Return:
      gap metrics (centroid, rmg, cmas, nas@k)
      i2t recalls R@1/5/10
      t2i recalls R@1/5/10
    """
    # IMPORTANT: avoid CUDA-in-fork issues -> num_workers=0 recommended on your setup.
    # loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    loader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
    collate_fn=retrieval_collate_fn,
    )

    img_feats = []
    txt_feats = []
    img2caps: Dict[int, List[int]] = {}
    cap2imgs: Dict[int, List[int]] = {}

    img_index = 0
    cap_index = 0

    it = tqdm(loader, desc=f"Embed retrieval [{model.name}]", leave=False)
    for batch in it:
        images, caps_list = batch  # caps_list is list of lists (len=B)
        if isinstance(caps_list, tuple):
            caps_list = list(caps_list)

        # encode images
        f_img = model.encode_image(images)  # (B,d)
        f_img = f_img.to(device)

        # expand captions
        flat_caps: List[str] = []
        cap_owner_img: List[int] = []
        for b in range(len(caps_list)):
            caps = caps_list[b]
            caps = caps if isinstance(caps, list) else [str(caps)]
            # keep all captions
            for c in caps:
                flat_caps.append(str(c))
                cap_owner_img.append(img_index + b)

        f_txt = model.encode_text(flat_caps)  # (B*M,d) on GPU
        f_txt = f_txt.to(device)

        # append image feats
        for b in range(f_img.size(0)):
            img_feats.append(f_img[b])

        # append text feats + mappings
        for j in range(len(flat_caps)):
            txt_feats.append(f_txt[j])

        # build mappings
        # image indices for this batch are [img_index, img_index+B)
        # caption indices are [cap_index, cap_index+len(flat_caps))
        # cap_owner_img contains global img indices
        # fill img2caps/cap2imgs
        for b in range(f_img.size(0)):
            img2caps[img_index + b] = []
        for j, owner in enumerate(cap_owner_img):
            cidx = cap_index + j
            img2caps[owner].append(cidx)
            cap2imgs[cidx] = [owner]

        img_index += f_img.size(0)
        cap_index += len(flat_caps)

        if max_images is not None and img_index >= max_images:
            break

    imgE = torch.stack(img_feats, dim=0).to(device)
    txtE = torch.stack(txt_feats, dim=0).to(device)

    # similarities
    S = imgE @ txtE.t()

    # retrieval metrics
    i2t = recalls_from_sim(S, img2caps, ks=(1,5,10))
    t2i = recalls_from_sim(S.t(), cap2imgs, ks=(1,5,10))

    # gap metrics: centroid+rmg + cmas+nas
    gap_basic = gap_metrics_centroid_rmg(imgE, txtE)
    gap_align = cmas_nas_metrics(imgE, txtE, img2caps, cap2imgs, ks=(1,5,10))
    gap = {**gap_basic, **gap_align}

    return gap, i2t, t2i


# =========================
# Eval: Zero-shot classification
# =========================
@torch.no_grad()
def build_zeroshot_prototypes(
    model: VLModel,
    classnames: List[str],
    templates: List[str],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Returns prototypes: (C,d) normalized
    Using mean over template text embeddings per class.
    """
    all_proto = []
    for cname in tqdm(classnames, desc=f"ZS class text [{model.name}]", leave=False):
        prompts = [t.format(cname) for t in templates]
        # encode in batches if prompts are many
        feats = []
        for i in range(0, len(prompts), batch_size):
            feats.append(model.encode_text(prompts[i:i+batch_size]))
        tE = torch.cat(feats, dim=0)  # (P,d)
        tE = l2norm(tE)
        proto = tE.mean(dim=0)
        proto = l2norm(proto.unsqueeze(0))[0]
        all_proto.append(proto)
    return torch.stack(all_proto, dim=0).to(device)

@torch.no_grad()
def zeroshot_eval(
    model: VLModel,
    ds: Dataset,
    classnames: List[str],
    device: str,
    batch_size: int = 256,
    num_workers: int = 0,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    proto = build_zeroshot_prototypes(model, classnames, ZS_TEMPLATES, device)

    top1 = 0
    top5 = 0
    n = 0

    it = tqdm(loader, desc=f"ZS infer [{model.name}]", leave=False)
    for images, labels in it:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        f = model.encode_image(images)  # (B,d)
        # logits: cosine similarity
        logits = f @ proto.t()  # (B,C)
        pred5 = logits.topk(5, dim=1).indices
        top1 += (pred5[:, 0] == labels).sum().item()
        top5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)
        if max_items is not None and n >= max_items:
            break

    return {
        "top1": top1 / max(1, n),
        "top5": top5 / max(1, n),
    }


# =========================
# Main
# =========================
def make_models(args, device: str) -> List[VLModel]:
    models: List[VLModel] = []

    # "CLIP" via open_clip with pretrained='openai'
    if "clip" in args.models:
        models.append(OpenCLIPWrapper(model_name=args.clip_model, pretrained="openai", device=device))

    # "openclip" via open_clip with laion pretrained
    if "openclip" in args.models:
        models.append(OpenCLIPWrapper(model_name=args.openclip_model, pretrained=args.openclip_pretrained, device=device))

    # "siglip" via transformers
    if "siglip" in args.models:
        models.append(SigLIPWrapper(hf_name=args.siglip_name, device=device, native_preprocess=args.siglip_native_preprocess))


    return models

def flatten_results_to_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    results[model][dataset] -> row-wise CSV flattening.
    """
    rows: List[Dict[str, Any]] = []
    for mname, md in results.items():
        for dname, metrics in md.items():
            row = {"model": mname, "dataset": dname}
            def add(prefix, obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        add(f"{prefix}{k}.", v)
                else:
                    row[prefix[:-1]] = obj
            add("", metrics)
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/work/was598/modilty_gap/tools/data")
    ap.add_argument("--out-dir", type=str, default="/work/was598/modilty_gap/results/baseline1_multi_backbone")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    # which models to run
    ap.add_argument("--models", type=str, default="clip,openclip,siglip",
                    help="comma-separated: clip,openclip,siglip")

    # CLIP/OpenCLIP configs (open_clip_torch)
    ap.add_argument("--clip-model", type=str, default="ViT-B-32",
                    help="open_clip model name for CLIP(openai) weights")
    ap.add_argument("--openclip-model", type=str, default="ViT-B-32",
                    help="open_clip model name for OpenCLIP(laion) weights")
    ap.add_argument("--openclip-pretrained", type=str, default="laion2b_s34b_b79k",
                    help="open_clip pretrained tag for OpenCLIP")

    # SigLIP configs (transformers)
    ap.add_argument("--siglip-name", type=str, default="google/siglip-base-patch16-224",
                    help="HF model id for SigLIP")
    ap.add_argument("--siglip-native-preprocess", action="store_true")


    # runtime
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--zs-batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Keep 0 on your cluster to avoid CUDA fork issues.")
    ap.add_argument("--max-coco-images", type=int, default=5000)
    ap.add_argument("--max-flickr-images", type=int, default=5000)
    ap.add_argument("--max-zs-items", type=int, default=None)
    

    # tiny imagenet
    ap.add_argument("--auto-download-tiny", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    args.models = [s.strip().lower() for s in args.models.split(",") if s.strip()]
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # transforms
    tf_eval = build_clip_style_tf(args.image_size, train=False)

    # datasets
    # Retrieval
    coco_ds = CocoCaptionsEval(args.data_root, split="val", transform=tf_eval)
    flickr_ds = Flickr30kEval(args.data_root, transform=tf_eval)

    # Zero-shot
    # CIFAR100 / DTD can download if missing; this is baseline evaluation.
    cifar_train = tvds.CIFAR100(root=str(Path(args.data_root) / "cifar100"), train=True, download=True)
    cifar_test = tvds.CIFAR100(root=str(Path(args.data_root) / "cifar100"), train=False, download=True, transform=tf_eval)

    dtd_train = tvds.DTD(root=str(Path(args.data_root) / "dtd"), split="train", download=True)
    dtd_test = tvds.DTD(root=str(Path(args.data_root) / "dtd"), split="test", download=True, transform=tf_eval)

    tiny_val = TinyImageNet200(args.data_root, split="val", transform=tf_eval, auto_download=args.auto_download_tiny)

    # models
    models = make_models(args, device=device)

    # run
    results: Dict[str, Any] = {}
    for m in models:
        results[m.name] = {}
        print(f"\n==============================")
        print(f"[Model] {m.name}")
        print(f"==============================")

        # ---- COCO retrieval ----
        t0 = time.time()
        gap, i2t, t2i = retrieval_eval(
            m, coco_ds, device=device, batch_size=args.batch_size,
            max_images=args.max_coco_images, num_workers=args.num_workers
        )
        results[m.name]["mscoco"] = {
            "modality_gap": gap,
            "I2T": i2t,
            "T2I": t2i,
            "runtime_sec": time.time() - t0,
        }

        # ---- Flickr retrieval ----
        t0 = time.time()
        gap, i2t, t2i = retrieval_eval(
            m, flickr_ds, device=device, batch_size=args.batch_size,
            max_images=args.max_flickr_images, num_workers=args.num_workers
        )
        results[m.name]["flickr30k"] = {
            "modality_gap": gap,
            "I2T": i2t,
            "T2I": t2i,
            "runtime_sec": time.time() - t0,
        }

        # ---- Zero-shot CIFAR100 ----
        t0 = time.time()
        zs = zeroshot_eval(
            m, cifar_test, classnames=cifar_train.classes, device=device,
            batch_size=args.zs_batch, num_workers=args.num_workers, max_items=args.max_zs_items
        )
        results[m.name]["cifar100"] = {
            "zero_shot": zs,
            "runtime_sec": time.time() - t0,
        }

        # ---- Zero-shot Tiny-ImageNet-200 ----
        t0 = time.time()
        zs = zeroshot_eval(
            m, tiny_val, classnames=tiny_val.classes, device=device,
            batch_size=args.zs_batch, num_workers=args.num_workers, max_items=args.max_zs_items
        )
        results[m.name]["tiny_imagenet_200"] = {
            "zero_shot": zs,
            "runtime_sec": time.time() - t0,
        }

        # ---- Zero-shot DTD ----
        t0 = time.time()
        zs = zeroshot_eval(
            m, dtd_test, classnames=dtd_train.classes, device=device,
            batch_size=args.zs_batch, num_workers=args.num_workers, max_items=args.max_zs_items
        )
        results[m.name]["dtd"] = {
            "zero_shot": zs,
            "runtime_sec": time.time() - t0,
        }

    # save
    stamp = now_ts()
    json_path = out_dir / f"baseline1_clip_openclip_siglip_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = flatten_results_to_rows(results)
    csv_path = out_dir / f"baseline1_clip_openclip_siglip_{stamp}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
