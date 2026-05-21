#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-1: CLIP / SigLIP(2) / OpenCLIP
- Karpathy split for MSCOCO + Flickr30k retrieval
- Zero-shot image classification on CIFAR100 / Tiny-ImageNet-200 / DTD
- Gap metrics: centroid distance, Relative Modality Gap (RMG), NAS(k), CMAS
- Record runtime per (model, dataset)
- Save JSONL + CSV

Data expected under --data-root (examples):
  MSCOCO-2014 images:
    {data_root}/mscoco2014/train2014/*.jpg
    {data_root}/mscoco2014/val2014/*.jpg
  Flickr30k images:
    {data_root}/flickr30k/flickr30k-images/*.jpg  (or images/ or directly under flickr30k/)
  CIFAR100:
    {data_root}/cifar100  (torchvision layout, already downloaded)
  DTD:
    {data_root}/dtd       (torchvision layout, already downloaded)
  Tiny-ImageNet-200:
    {data_root}/tiny-imagenet-200/wnids.txt, words.txt, val/val_annotations.txt, val/images/*.JPEG

Karpathy JSON:
  auto-downloaded & cached under:
    {data_root}/mscoco2014/karpathy/dataset_coco.json
    {data_root}/flickr30k/karpathy/dataset_flickr30k.json

Run:
  python baseline1_ground_truth_karpathy_mscoco2014.py \
    --data-root /work/was598/modilty_gap/tools/data \
    --out-dir /work/was598/modilty_gap/results/baseline1 \
    --models clip,siglip,openclip \
    --batch-size 128 --num-workers 8 \
    --max-coco 5000 --max-flickr 5000 --max-cls 10000 \
    --nas-k 10
"""

import os
import csv
import json
import time
import math
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import torchvision.datasets as tvds

# optional deps
try:
    import open_clip
except Exception:
    open_clip = None

try:
    from transformers import SiglipModel, SiglipProcessor
except Exception:
    SiglipModel = None
    SiglipProcessor = None

try:
    from transformers import Siglip2Model, Siglip2Processor
except Exception:
    Siglip2Model = None
    Siglip2Processor = None

try:
    from transformers import Siglip2ImageProcessor
except Exception:
    Siglip2ImageProcessor = None

try:
    from transformers import AutoConfig
except Exception:
    AutoConfig = None


# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def download_file(url: str, dst: Path, timeout: int = 180):
    import urllib.request
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"[Download] {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    tmp.write_bytes(data)
    tmp.replace(dst)
    print(f"[Download] saved -> {dst} ({dst.stat().st_size/1e6:.2f} MB)")

def _siglip2_sanity_check(model: "SigLIPWrapper", sample_items: List[Tuple[str, List[str]]]):
    if len(sample_items) == 0:
        print("[SigLIP2 Sanity] No sample items available.")
        return
    take = min(4, len(sample_items))
    imgs: List[Image.Image] = []
    texts: List[str] = []
    for p, caps in sample_items[:take]:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        imgs.append(img)
        if isinstance(caps, (list, tuple)) and len(caps) > 0:
            texts.append(str(caps[0]))
        else:
            texts.append("a photo of something.")

    if len(imgs) == 0:
        print("[SigLIP2 Sanity] No images could be loaded.")
        return

    with torch.no_grad():
        inp = model._image_inputs(imgs)
        pix = inp.get("pixel_values", None)
        if pix is not None:
            print(f"[SigLIP2 Sanity] pixel_values shape: {tuple(pix.shape)}")
        if "spatial_shapes" in inp:
            ss = inp["spatial_shapes"]
            try:
                print(f"[SigLIP2 Sanity] spatial_shapes: {ss.tolist()}")
            except Exception:
                print(f"[SigLIP2 Sanity] spatial_shapes type: {type(ss)}")

        x = model.encode_images(imgs)
        y = model.encode_texts(texts)
        print(f"[SigLIP2 Sanity] image_feats: shape={tuple(x.shape)} norm_mean={x.norm(dim=1).mean().item():.4f}")
        print(f"[SigLIP2 Sanity] text_feats:  shape={tuple(y.shape)} norm_mean={y.norm(dim=1).mean().item():.4f}")
        sim = (x @ y.t()).cpu().numpy()
        print(f"[SigLIP2 Sanity] sim: mean={sim.mean():.4f} max={sim.max():.4f} min={sim.min():.4f}")

def _siglip_sanity_check(model: "SigLIPWrapper", sample_items: List[Tuple[str, List[str]]]):
    if len(sample_items) == 0:
        print("[SigLIP Sanity] No sample items available.")
        return
    take = min(4, len(sample_items))
    imgs: List[Image.Image] = []
    texts: List[str] = []
    for p, caps in sample_items[:take]:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        imgs.append(img)
        if isinstance(caps, (list, tuple)) and len(caps) > 0:
            texts.append(str(caps[0]))
        else:
            texts.append("a photo of something.")

    if len(imgs) == 0:
        print("[SigLIP Sanity] No images could be loaded.")
        return

    with torch.no_grad():
        inp = model._image_inputs(imgs)
        pix = inp.get("pixel_values", None)
        if pix is not None:
            print(f"[SigLIP Sanity] pixel_values shape: {tuple(pix.shape)}")
        x = model.encode_images(imgs)
        y = model.encode_texts(texts)
        print(f"[SigLIP Sanity] image_feats: shape={tuple(x.shape)} norm_mean={x.norm(dim=1).mean().item():.4f}")
        print(f"[SigLIP Sanity] text_feats:  shape={tuple(y.shape)} norm_mean={y.norm(dim=1).mean().item():.4f}")
        sim = (x @ y.t()).cpu().numpy()
        diag = sim.diagonal()
        off = sim[~np.eye(sim.shape[0], dtype=bool)]
        print(f"[SigLIP Sanity] sim: mean={sim.mean():.4f} max={sim.max():.4f} min={sim.min():.4f}")
        print(f"[SigLIP Sanity] sim: diag_mean={diag.mean():.4f} off_mean={off.mean():.4f}")

def _cached_hf_file(repo_id: str, filename: str) -> Optional[str]:
    try:
        from transformers.utils.hub import cached_file
        return cached_file(repo_id, filename)
    except Exception:
        return None

def _load_hf_tensors(repo_id: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_NAME

    index_path = _cached_hf_file(repo_id, SAFE_WEIGHTS_INDEX_NAME)
    if index_path is None:
        index_path = _cached_hf_file(repo_id, WEIGHTS_INDEX_NAME)

    if index_path is not None:
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        target_key = next((k for k in keys if k in weight_map), None)
        if target_key is None:
            return {}
        shard_file = weight_map[target_key]
        shard_path = _cached_hf_file(repo_id, shard_file)
        if shard_path is None:
            return {}
        if shard_path.endswith(".safetensors"):
            try:
                from safetensors.torch import safe_open
                out: Dict[str, torch.Tensor] = {}
                with safe_open(shard_path, framework="pt") as f:
                    for k in keys:
                        if k in f.keys():
                            out[k] = f.get_tensor(k)
                return out
            except Exception:
                from safetensors.torch import load_file
                state = load_file(shard_path)
        else:
            state = torch.load(shard_path, map_location="cpu", weights_only=True)
        return {k: state[k] for k in keys if k in state}

    weights_path = _cached_hf_file(repo_id, SAFE_WEIGHTS_NAME)
    if weights_path is None:
        weights_path = _cached_hf_file(repo_id, WEIGHTS_NAME)
    if weights_path is None:
        return {}
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weights_path)
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    return {k: state[k] for k in keys if k in state}

def _fix_siglip2_patch_embedding(model: "Siglip2Model", repo_id: str) -> bool:
    key = "vision_model.embeddings.patch_embedding.weight"
    bias_key = "vision_model.embeddings.patch_embedding.bias"
    tensors = _load_hf_tensors(repo_id, [key, bias_key])
    if key not in tensors:
        return False
    w = tensors[key]
    if w.ndim == 4:
        w = w.reshape(w.shape[0], -1)
    target = model.vision_model.embeddings.patch_embedding.weight
    if w.shape != target.shape:
        return False
    target.data.copy_(w.to(dtype=target.dtype))
    if bias_key in tensors:
        bias = getattr(model.vision_model.embeddings.patch_embedding, "bias", None)
        if bias is not None and tensors[bias_key].shape == bias.shape:
            bias.data.copy_(tensors[bias_key].to(dtype=bias.dtype))
    return True

def _load_siglip2_model(repo_id: str, device: str):
    try:
        model = Siglip2Model.from_pretrained(repo_id, output_loading_info=True)
        if isinstance(model, tuple):
            model, info = model
            if info.get("missing_keys") or info.get("unexpected_keys"):
                print(f"[SigLIP2 Load] missing_keys={len(info.get('missing_keys', []))} "
                      f"unexpected_keys={len(info.get('unexpected_keys', []))}")
        return model.to(device).eval()
    except RuntimeError as e:
        if "size mismatch for weight" not in str(e):
            raise
        model = Siglip2Model.from_pretrained(
            repo_id,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
        info = {}
        if isinstance(model, tuple):
            model, info = model
        model = model.to(device).eval()
        if not _fix_siglip2_patch_embedding(model, repo_id):
            raise
        if info.get("missing_keys") or info.get("unexpected_keys"):
            print(f"[SigLIP2 Load] missing_keys={len(info.get('missing_keys', []))} "
                  f"unexpected_keys={len(info.get('unexpected_keys', []))}")
        return model

# ============================================================
# Karpathy JSON download (Stanford zip + fallback mirror)
# ============================================================

KARPATHY_CAPTION_ZIP_URLS = [
    "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
]
KARPATHY_SPLITS_MIRROR = {
    "coco": "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download=",
    "flickr30k": "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_flickr30k.json?download=",
}

def download_and_extract_from_zip(urls: List[str], dst_json: Path, member_name: str, timeout: int = 240):
    import urllib.request
    import zipfile

    ensure_dir(dst_json.parent)
    if dst_json.exists() and dst_json.stat().st_size > 0:
        return

    zip_path = dst_json.parent / "caption_datasets.zip"
    last_err = None

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        for u in urls:
            try:
                print(f"[Download] {u}")
                with urllib.request.urlopen(u, timeout=timeout) as resp:
                    data = resp.read()
                tmp = zip_path.with_suffix(".zip.tmp")
                tmp.write_bytes(data)
                tmp.replace(zip_path)
                print(f"[Download] saved -> {zip_path} ({zip_path.stat().st_size/1e6:.2f} MB)")
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None and (not zip_path.exists() or zip_path.stat().st_size == 0):
            raise RuntimeError(f"Failed to download caption_datasets.zip. Last error: {last_err}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        if member_name not in names:
            cand = [n for n in zf.namelist() if n.endswith("/" + member_name) or n.endswith(member_name)]
            if len(cand) == 0:
                raise RuntimeError(f"{member_name} not found in {zip_path}.")
            member = cand[0]
        else:
            member = member_name

        with zf.open(member, "r") as f:
            data = f.read()

    tmp_json = dst_json.with_suffix(dst_json.suffix + ".tmp")
    tmp_json.write_bytes(data)
    tmp_json.replace(dst_json)
    print(f"[Extract] {member_name} -> {dst_json} ({dst_json.stat().st_size/1e6:.2f} MB)")

def ensure_karpathy_json(data_root: str, which: str) -> Path:
    root = Path(data_root)

    if which == "coco":
        dst = root / "mscoco2014" / "karpathy" / "dataset_coco.json"
        member = "dataset_coco.json"
        mirror = KARPATHY_SPLITS_MIRROR["coco"]
    elif which == "flickr30k":
        dst = root / "flickr30k" / "karpathy" / "dataset_flickr30k.json"
        member = "dataset_flickr30k.json"
        mirror = KARPATHY_SPLITS_MIRROR["flickr30k"]
    else:
        raise ValueError(which)

    if dst.exists() and dst.stat().st_size > 0:
        return dst

    try:
        download_and_extract_from_zip(KARPATHY_CAPTION_ZIP_URLS, dst_json=dst, member_name=member)
        if dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception as e:
        print(f"[Warn] Stanford zip failed: {e}")

    try:
        download_file(mirror, dst)
        if dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception as e:
        print(f"[Warn] Mirror json failed: {e}")

    raise RuntimeError(f"Failed to obtain Karpathy json for '{which}'.")

# ============================================================
# Datasets
# ============================================================

class KarpathyRetrievalDataset(Dataset):
    """
    From Karpathy json. Each item yields:
      (PIL.Image, captions_list[str])
    """
    def __init__(self, karpathy_json: str, image_roots: List[str], split: str, max_images: Optional[int] = None):
        super().__init__()
        self.karpathy_json = Path(karpathy_json)
        assert self.karpathy_json.exists(), f"Karpathy json not found: {self.karpathy_json}"
        self.image_roots = [Path(p) for p in image_roots]
        self.split = split

        data = json.loads(self.karpathy_json.read_text(encoding="utf-8"))
        images = data["images"]

        items = []
        missing = 0
        for img in images:
            if img.get("split", "") != split:
                continue
            fn = img.get("filename", None)
            if fn is None:
                continue

            caps = []
            for s in img.get("sentences", []):
                if "raw" in s:
                    caps.append(s["raw"])
                elif "tokens" in s:
                    caps.append(" ".join(s["tokens"]))
            if len(caps) == 0:
                continue

            p = self._resolve_path(fn)
            if p is None:
                missing += 1
                continue

            items.append((str(p), caps))
            if max_images is not None and len(items) >= max_images:
                break

        if len(items) == 0:
            raise AssertionError(
                f"No items for split='{split}' from {karpathy_json}. "
                f"Check image roots. (missing_paths={missing})"
            )

        print(f"[KarpathyDataset] split={split} items={len(items)} (missing_paths={missing})")
        self.items = items

    def _resolve_path(self, filename: str) -> Optional[Path]:
        # direct join
        for r in self.image_roots:
            if not r.exists():
                continue
            p = r / filename
            if p.exists():
                return p

        # cheap 1-level common subfolders
        common_subs = ["images", "flickr30k-images", "train2014", "val2014", "train", "val"]
        for r in self.image_roots:
            if not r.exists():
                continue
            for sub in common_subs:
                p = r / sub / filename
                if p.exists():
                    return p

        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, caps = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return img, caps

class TinyImageNet200Val(Dataset):
    """
    Tiny-ImageNet-200 val split.
    Uses wnids.txt + words.txt to build classnames.
    """
    def __init__(self, data_root: str):
        super().__init__()
        self.root = Path(data_root) / "tiny-imagenet-200"
        assert self.root.exists(), f"Tiny-ImageNet root not found: {self.root}"

        wnids_path = self.root / "wnids.txt"
        words_path = self.root / "words.txt"
        ann_path = self.root / "val" / "val_annotations.txt"
        img_dir = self.root / "val" / "images"

        assert wnids_path.exists(), f"wnids.txt not found under {self.root}"
        assert words_path.exists(), f"words.txt not found under {self.root}"
        assert ann_path.exists(), f"val_annotations.txt not found under {ann_path}"
        assert img_dir.exists(), f"val/images not found under {img_dir}"

        self.wnids = [l.strip() for l in wnids_path.read_text().splitlines() if l.strip()]
        wnid_to_words = {}
        for line in words_path.read_text().splitlines():
            # format: n01443537\tgoldfish, Carassius auratus
            parts = line.split("\t")
            if len(parts) >= 2:
                wnid_to_words[parts[0]] = parts[1].split(",")[0].strip()

        self.classnames = [wnid_to_words.get(w, w) for w in self.wnids]

        img2wnid = {}
        for line in ann_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                img2wnid[parts[0]] = parts[1]

        samples = []
        for img_name, wnid in img2wnid.items():
            p = img_dir / img_name
            if p.exists() and wnid in self.wnids:
                y = self.wnids.index(wnid)
                samples.append((str(p), y))

        assert len(samples) > 0, f"No val samples found under {img_dir}"
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return img, y

# ============================================================
# Collate functions (avoid variable-length caption issues)
# ============================================================

def collate_retrieval(batch):
    # batch: [(PIL, [caps]), ...]
    images = [b[0] for b in batch]
    caps_list = []
    for _, caps in batch:
        if isinstance(caps, (list, tuple)):
            caps_list.append([str(c) for c in caps])
        else:
            caps_list.append([str(caps)])
    return images, caps_list

def collate_cls(batch):
    # batch: [(PIL, y), ...]
    images = [b[0] for b in batch]
    ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return images, ys

# ============================================================
# Backbones: CLIP / OpenCLIP / SigLIP(2) (preprocess inside wrapper)
# ============================================================

class VLBackbone(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

class OpenCLIPWrapper(VLBackbone):
    def __init__(self, model_name: str, pretrained: str, device: str):
        super().__init__(device=device)
        assert open_clip is not None, "open_clip is not installed."
        self.model_name = model_name
        self.pretrained = pretrained

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(device).eval()
        self.preprocess = preprocess_val
        self.tokenizer = tokenizer

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feat = self.model.encode_image(dummy)
            self._dim = int(feat.shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        # preprocess on CPU then move to GPU
        tens = torch.stack([self.preprocess(im) for im in pil_images], dim=0)
        tens = tens.to(self.device, non_blocking=True)
        feat = self.model.encode_image(tens).float()
        return l2norm(feat)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts)
        if isinstance(toks, dict):
            toks = {k: v.to(self.device) for k, v in toks.items()}
            feat = self.model.encode_text(**toks)
        else:
            toks = toks.to(self.device)
            feat = self.model.encode_text(toks)
        feat = feat.float()
        return l2norm(feat)

class CLIPWrapper(OpenCLIPWrapper):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name=model_name, pretrained="openai", device=device)

class SigLIPWrapper(VLBackbone):
    def __init__(self, hf_name: str, device: str):
        super().__init__(device=device)
        self.hf_name = hf_name
        use_siglip2 = "siglip2" in hf_name.lower()
        if AutoConfig is not None:
            try:
                cfg = AutoConfig.from_pretrained(hf_name)
                if getattr(cfg, "model_type", None) == "siglip2":
                    use_siglip2 = True
                elif getattr(cfg, "model_type", None) == "siglip":
                    use_siglip2 = False
            except Exception:
                pass
        self._use_siglip2 = use_siglip2
        if use_siglip2:
            assert Siglip2Model is not None and Siglip2Processor is not None, "SigLIP2 requires transformers with Siglip2Model."
            self.model = _load_siglip2_model(hf_name, device=device)
            self.proc = Siglip2Processor.from_pretrained(hf_name)
            self.img_proc = None
            if Siglip2ImageProcessor is not None:
                try:
                    self.img_proc = Siglip2ImageProcessor.from_pretrained(hf_name)
                except Exception:
                    vcfg = getattr(self.model.config, "vision_config", None)
                    patch_size = getattr(vcfg, "patch_size", 16) if vcfg is not None else 16
                    max_num_patches = getattr(vcfg, "num_patches", 256) if vcfg is not None else 256
                    self.img_proc = Siglip2ImageProcessor(patch_size=patch_size, max_num_patches=max_num_patches)
            self._text_padding = None
            self._text_max_length = None
            self._lowercase_text = False
        else:
            assert SiglipModel is not None and SiglipProcessor is not None, "SigLIP requires transformers>=4.40 with SiglipModel."
            self.model = SiglipModel.from_pretrained(hf_name).to(device).eval()
            self.proc = SiglipProcessor.from_pretrained(hf_name)
            self.img_proc = None
            self._text_padding = True
            self._text_max_length = None
            self._lowercase_text = False
            self._v1_force_maxlen = False

        # robust dim inference (do NOT rely on config.projection_dim)
        with torch.no_grad():
            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            inp = self._image_inputs([dummy_img])
            inp = {k: v.to(device) for k, v in inp.items()}
            feat = self.model.get_image_features(**inp)
            self._dim = int(feat.shape[-1])

    def _image_inputs(self, pil_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        if self._use_siglip2:
            if self.img_proc is not None:
                inp = self.img_proc(images=pil_images, return_tensors="pt")
            else:
                inp = self.proc(images=pil_images, return_tensors="pt")
            if "pixel_values" in inp and inp["pixel_values"].dim() == 4:
                raise RuntimeError("SigLIP2 image processor returned unpatchified pixel_values. "
                                   "Please ensure Siglip2ImageProcessor is available.")
            return inp
        return self.proc(images=pil_images, return_tensors="pt")

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inp = self._image_inputs(pil_images)
        inp = {k: v.to(self.device, non_blocking=True) for k, v in inp.items()}
        feat = self.model.get_image_features(**inp).float()
        return l2norm(feat)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if self._use_siglip2:
            inp = self.proc(text=texts, return_tensors="pt")
        else:
            if self._lowercase_text:
                texts = [t.lower() for t in texts]
            if self._v1_force_maxlen:
                inp = self.proc(
                    text=texts,
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
            elif self._text_max_length is None:
                inp = self.proc(text=texts, padding=self._text_padding, truncation=True, return_tensors="pt")
            else:
                inp = self.proc(
                    text=texts,
                    padding=self._text_padding,
                    truncation=True,
                    max_length=self._text_max_length,
                    return_tensors="pt",
                )
        inp = {k: v.to(self.device, non_blocking=True) for k, v in inp.items()}
        feat = self.model.get_text_features(**inp).float()
        return l2norm(feat)

def make_models(args, device: str) -> List[Tuple[str, VLBackbone]]:
    keys = [k.strip().lower() for k in args.models.split(",") if k.strip()]
    out: List[Tuple[str, VLBackbone]] = []
    for k in keys:
        if k == "clip":
            out.append((f"clip:{args.clip_model}:openai", CLIPWrapper(args.clip_model, device=device)))
        elif k == "openclip":
            out.append((f"open_clip:{args.openclip_model}:{args.openclip_pretrained}",
                        OpenCLIPWrapper(args.openclip_model, args.openclip_pretrained, device=device)))
        elif k == "siglip":
            out.append((f"siglip:{args.siglip_name}", SigLIPWrapper(args.siglip_name, device=device)))
        else:
            raise ValueError(f"Unknown model key: {k}")
    return out

# ============================================================
# Gap metrics
# ============================================================

@torch.no_grad()
def centroid_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    return float((mx - my).norm().item())

@torch.no_grad()
def relative_modality_gap(x: torch.Tensor, y: torch.Tensor, intra_samples: int = 20000) -> float:
    """
    RMG = D_pair / (D_pair + D_intra)
    where D_pair uses paired (x_i, y_i), and D_intra is average intra-modality distance.
    """
    m = min(x.shape[0], y.shape[0])
    x = x[:m]
    y = y[:m]
    d_pair = (x - y).norm(dim=1).mean()

    def sample_mean_dist(z: torch.Tensor) -> torch.Tensor:
        N = z.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=z.device)
        s = min(intra_samples, max(2000, N * 50))
        i = torch.randint(0, N, (s,), device=z.device)
        j = torch.randint(0, N, (s,), device=z.device)
        mask = i != j
        if mask.any():
            i = i[mask]
            j = j[mask]
        return (z[i] - z[j]).norm(dim=1).mean()

    d_intra = 0.5 * (sample_mean_dist(x) + sample_mean_dist(y))
    return float((d_pair / (d_pair + d_intra + 1e-12)).item())

@torch.no_grad()
def cmas(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    CMAS = mean cosine similarity of paired samples (x,y must be l2-normalized)
    """
    m = min(x.shape[0], y.shape[0])
    return float((x[:m] * y[:m]).sum(dim=1).mean().item())

@torch.no_grad()
def nas_k(x: torch.Tensor, y: torch.Tensor, k: int = 10, max_items: int = 5000) -> float:
    """
    NAS(k) = (1/N) sum_i |Nk(x_i) ∩ Nk(y_i)| / k
    where Nk uses within-modality neighbors among first N items.
    """
    n = min(x.shape[0], y.shape[0], max_items)
    if n <= k + 1:
        return 0.0
    x = x[:n]
    y = y[:n]

    sx = x @ x.t()
    sy = y @ y.t()
    diag = torch.arange(n, device=x.device)
    sx[diag, diag] = -1e9
    sy[diag, diag] = -1e9

    nx = torch.topk(sx, k=k, dim=1).indices
    ny = torch.topk(sy, k=k, dim=1).indices

    inter = (nx.unsqueeze(2) == ny.unsqueeze(1)).any(dim=2).sum(dim=1)
    return float((inter.float().mean() / float(k)).item())

# ============================================================
# Retrieval eval (Karpathy)
# ============================================================

@torch.no_grad()
def retrieval_eval(
    model: VLBackbone,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Dataset item: (PIL, [captions])
    We compute:
      - image feats for images
      - text feats for all captions
      - i2t recall: image -> any caption that belongs to that image
      - t2i recall: caption -> its image
      - gap metrics using paired (image, first-caption) for each image
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # we keep PIL on CPU; GPU transfer happens in wrapper
        collate_fn=collate_retrieval
    )

    image_feats_chunks: List[torch.Tensor] = []
    all_caps: List[str] = []
    cap2img: List[int] = []

    n_images = 0
    for pil_images, caps_list in loader:
        if max_images is not None and n_images >= max_images:
            break

        b = len(pil_images)
        if max_images is not None and n_images + b > max_images:
            keep = max_images - n_images
            pil_images = pil_images[:keep]
            caps_list = caps_list[:keep]
            b = keep

        feats = model.encode_images(pil_images)  # (b, d) on GPU
        image_feats_chunks.append(feats)

        for i in range(b):
            caps = caps_list[i]
            for c in caps:
                all_caps.append(c)
                cap2img.append(n_images + i)

        n_images += b

    image_feats = torch.cat(image_feats_chunks, dim=0)  # (Nimg, d) GPU
    n_caps = len(all_caps)

    # encode captions
    text_feats_chunks: List[torch.Tensor] = []
    bs_t = 256
    for s in range(0, n_caps, bs_t):
        tf = model.encode_texts(all_caps[s:s+bs_t])  # (bs, d) GPU
        text_feats_chunks.append(tf)
    text_feats = torch.cat(text_feats_chunks, dim=0)  # (Ncap, d) GPU

    # paired text = first caption per image
    first_cap = [-1] * image_feats.size(0)
    for cap_idx, img_idx in enumerate(cap2img):
        if first_cap[img_idx] < 0:
            first_cap[img_idx] = cap_idx
    pair_map = torch.tensor(first_cap, dtype=torch.long, device=device)
    paired_text = text_feats[pair_map]

    gap = {
        "centroid_distance": centroid_distance(image_feats, paired_text),
        "relative_modality_gap": relative_modality_gap(image_feats, paired_text, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(image_feats, paired_text, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(image_feats, paired_text),
    }

    cap2img_t = torch.tensor(cap2img, dtype=torch.long, device=device)

    # chunked recall for GPU memory control
    def recall_i2t(K: int) -> float:
        correct = 0
        Nimg = image_feats.size(0)
        chunk = 512
        for s in range(0, Nimg, chunk):
            e = min(Nimg, s + chunk)
            sims = image_feats[s:e] @ text_feats.t()  # GPU
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
            sims = text_feats[s:e] @ image_feats.t()  # GPU
            topk = torch.topk(sims, k=K, dim=1).indices
            true_img = cap2img_t[s:e].unsqueeze(1)
            hit = (topk == true_img).any(dim=1)
            correct += int(hit.sum().item())
        return 100.0 * correct / float(Ncap)

    i2t = {"R@1": recall_i2t(1), "R@5": recall_i2t(5), "R@10": recall_i2t(10)}
    t2i = {"R@1": recall_t2i(1), "R@5": recall_t2i(5), "R@10": recall_t2i(10)}
    extra = {"n_images": float(image_feats.size(0)), "n_captions": float(text_feats.size(0))}
    return gap, i2t, t2i, extra

# ============================================================
# Zero-shot classification
# ============================================================

CIFAR100_TEMPLATES = [
    "a photo of a {c}.",
    "a photo of the {c}.",
    "a blurry photo of a {c}.",
    "a photo of a small {c}.",
    "a photo of a big {c}.",
    "a low resolution photo of a {c}.",
    "a close-up photo of a {c}.",
    "a bright photo of a {c}.",
    "a dark photo of a {c}.",
]

DTD_TEMPLATES = [
    "a photo of a {c} texture.",
    "a close-up photo of a {c} texture.",
    "a photo of the {c} pattern.",
    "a close-up photo of the {c} pattern.",
]

@torch.no_grad()
def build_zeroshot_weights(model: VLBackbone, classnames: List[str], templates: List[str], device: str) -> torch.Tensor:
    ws = []
    bs = 128
    for cname in classnames:
        texts = [t.format(c=cname) for t in templates]
        feats_chunks: List[torch.Tensor] = []
        for s in range(0, len(texts), bs):
            feats_chunks.append(model.encode_texts(texts[s:s+bs]))
        feats = torch.cat(feats_chunks, dim=0)
        w = l2norm(feats.mean(dim=0, keepdim=True)).squeeze(0)
        ws.append(w)
    W = torch.stack(ws, dim=0).to(device)
    return W

@torch.no_grad()
def zeroshot_eval(
    model: VLBackbone,
    dataset: Dataset,
    classnames: List[str],
    templates: List[str],
    device: str,
    batch_size: int,
    num_workers: int,
    max_items: Optional[int],
    nas_k_val: int,
    nas_max_items: int,
    intra_samples: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    W = build_zeroshot_weights(model, classnames, templates, device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_cls
    )

    correct1 = 0
    correct5 = 0
    total = 0

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    for pil_images, labels in loader:
        if max_items is not None and total >= max_items:
            break
        b = len(pil_images)
        if max_items is not None and total + b > max_items:
            keep = max_items - total
            pil_images = pil_images[:keep]
            labels = labels[:keep]
            b = keep

        x = model.encode_images(pil_images)  # (b, d) GPU
        logits = x @ W.t()                   # (b, C) GPU

        top1 = torch.argmax(logits, dim=1).cpu()
        correct1 += int((top1 == labels).sum().item())

        top5 = torch.topk(logits, k=5, dim=1).indices.cpu()
        correct5 += int(sum([labels[i].item() in top5[i].tolist() for i in range(b)]))

        total += b

        xs.append(x)
        ys.append(W[labels.to(device, non_blocking=True)])

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)

    gap = {
        "centroid_distance": centroid_distance(x_all, y_all),
        "relative_modality_gap": relative_modality_gap(x_all, y_all, intra_samples=intra_samples),
        f"NAS@{nas_k_val}": nas_k(x_all, y_all, k=nas_k_val, max_items=nas_max_items),
        "CMAS": cmas(x_all, y_all),
    }
    acc = {"top1": 100.0 * correct1 / float(total), "top5": 100.0 * correct5 / float(total), "n": float(total)}
    return gap, acc

# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--models", type=str, default="clip,siglip,openclip")
    ap.add_argument("--model-size", type=int, choices=[16, 32], default=None)
    ap.add_argument("--clip-model", type=str, default="ViT-B-32")
    ap.add_argument("--openclip-model", type=str, default="ViT-B-32")
    ap.add_argument("--openclip-pretrained", type=str, default="openai")
    ap.add_argument("--siglip-name", type=str, default="google/siglip2-base-patch16-224")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--max-coco", type=int, default=5000)
    ap.add_argument("--max-flickr", type=int, default=5000)
    ap.add_argument("--max-cls", type=int, default=10000)

    ap.add_argument("--nas-k", type=int, default=10)
    ap.add_argument("--nas-max-items", type=int, default=5000)
    ap.add_argument("--intra-samples", type=int, default=20000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--siglip2-sanity", action="store_true",
                    help="Run a tiny SigLIP2 sanity check and exit.")
    ap.add_argument("--siglip-sanity", action="store_true",
                    help="Run a tiny SigLIP (v1) sanity check and exit.")
    ap.add_argument("--siglip-v1-maxlen", action="store_true",
                    help="Force SigLIP v1 text padding='max_length' and max_length=64.")

    args = ap.parse_args()
    seed_all(args.seed)

    # optional unified model size for CLIP/OpenCLIP
    if args.model_size is not None:
        size_tag = "ViT-B-16" if args.model_size == 16 else "ViT-B-32"
        if args.clip_model == "ViT-B-32":
            args.clip_model = size_tag
        if args.openclip_model == "ViT-B-32":
            args.openclip_model = size_tag

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_jsonl = out_dir / "baseline1_results.jsonl"
    out_csv = out_dir / "baseline1_results.csv"

    # ---- Karpathy json ----
    coco_kjson = ensure_karpathy_json(args.data_root, "coco")
    flickr_kjson = ensure_karpathy_json(args.data_root, "flickr30k")

    # ---- COCO2014 image roots (DIRECT MSCOCO-2014) ----
    coco_img_roots = [
        str(Path(args.data_root) / "mscoco2014" / "train2014"),
        str(Path(args.data_root) / "mscoco2014" / "val2014"),
        str(Path(args.data_root) / "coco2014" / "train2014"),
        str(Path(args.data_root) / "coco2014" / "val2014"),
        str(Path(args.data_root) / "coco" / "train2014"),
        str(Path(args.data_root) / "coco" / "val2014"),
    ]

    # ---- Flickr image roots ----
    flickr_img_roots = [
        str(Path(args.data_root) / "flickr30k" / "flickr30k-images"),
        str(Path(args.data_root) / "flickr30k" / "images"),
        str(Path(args.data_root) / "flickr30k"),
    ]

    # ---- Build Karpathy TEST datasets ----
    coco_test = KarpathyRetrievalDataset(str(coco_kjson), coco_img_roots, split="test", max_images=None)
    flickr_test = KarpathyRetrievalDataset(str(flickr_kjson), flickr_img_roots, split="test", max_images=None)

    # ---- Classification datasets (PIL output) ----
    # CIFAR100/DTD from torchvision return PIL by default if transform=None
    cifar_root = Path(args.data_root) / "cifar100"
    dtd_root = Path(args.data_root) / "dtd"

    cifar_test = tvds.CIFAR100(root=str(cifar_root), train=False, download=False, transform=None)
    dtd_test = tvds.DTD(root=str(dtd_root), split="test", download=False, transform=None)

    tiny_val_ds = TinyImageNet200Val(args.data_root)

    cifar_classes = cifar_test.classes
    dtd_classes = dtd_test.classes
    tiny_classes = tiny_val_ds.classnames
    tiny_templates = ["a photo of a {c}.", "a photo of the {c}."]

    # ---- Models ----
    models = make_models(args, device=device)
    if args.siglip_v1_maxlen:
        for _, m in models:
            if isinstance(m, SigLIPWrapper) and not getattr(m, "_use_siglip2", False):
                m._v1_force_maxlen = True
    if args.siglip2_sanity:
        for _, m in models:
            if isinstance(m, SigLIPWrapper) and getattr(m, "_use_siglip2", False):
                _siglip2_sanity_check(m, coco_test.items[:8])
                return
    if args.siglip_sanity:
        for _, m in models:
            if isinstance(m, SigLIPWrapper) and not getattr(m, "_use_siglip2", False):
                _siglip_sanity_check(m, coco_test.items[:8])
                return

    # ---- Output schemas ----
    header = [
        "model", "dataset",
        "centroid_distance", "relative_modality_gap", f"NAS@{args.nas_k}", "CMAS",
        "I2T_R1", "I2T_R5", "I2T_R10",
        "T2I_R1", "T2I_R5", "T2I_R10",
        "top1", "top5",
        "eval_time_sec",
    ]

    rows = []
    jf = out_jsonl.open("w", encoding="utf-8")

    for model_name, model in models:
        print("\n==============================")
        print(f"[Model] {model_name}")
        print("==============================")

        # ------------------------------------------------------------
        # COCO retrieval
        # ------------------------------------------------------------
        print("[Eval] MSCOCO Karpathy test (I2T/T2I R@K + gap)")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval(
            model, coco_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_coco,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "mscoco2014_karpathy_test",
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "mscoco2014_karpathy_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "",
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Flickr retrieval
        # ------------------------------------------------------------
        print("[Eval] Flickr30k Karpathy test (I2T/T2I R@K + gap)")
        t0 = time.time()
        gap, i2t, t2i, extra = retrieval_eval(
            model, flickr_test, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_flickr,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "flickr30k_karpathy_test",
            "gap": gap,
            "i2t": i2t,
            "t2i": t2i,
            "extra": extra,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "flickr30k_karpathy_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            i2t["R@1"], i2t["R@5"], i2t["R@10"],
            t2i["R@1"], t2i["R@5"], t2i["R@10"],
            "", "",
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # CIFAR100 zero-shot
        # ------------------------------------------------------------
        print("[Eval] CIFAR100 zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval(
            model, cifar_test,
            cifar_classes, CIFAR100_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "cifar100_test",
            "gap": gap,
            "acc": acc,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "cifar100_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # DTD zero-shot
        # ------------------------------------------------------------
        print("[Eval] DTD zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval(
            model, dtd_test,
            dtd_classes, DTD_TEMPLATES,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "dtd_test",
            "gap": gap,
            "acc": acc,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "dtd_test",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            float(t1 - t0),
        ])

        # ------------------------------------------------------------
        # Tiny-ImageNet-200 zero-shot (val)
        # ------------------------------------------------------------
        print("[Eval] Tiny-ImageNet-200 val zero-shot (top1/top5 + gap)")
        t0 = time.time()
        gap, acc = zeroshot_eval(
            model, tiny_val_ds,
            tiny_classes, tiny_templates,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_items=args.max_cls,
            nas_k_val=args.nas_k,
            nas_max_items=args.nas_max_items,
            intra_samples=args.intra_samples
        )
        t1 = time.time()
        rec = {
            "model": model_name,
            "dataset": "tiny-imagenet-200_val",
            "gap": gap,
            "acc": acc,
            "eval_time_sec": float(t1 - t0),
        }
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jf.flush()

        rows.append([
            model_name, "tiny-imagenet-200_val",
            gap["centroid_distance"], gap["relative_modality_gap"], gap[f"NAS@{args.nas_k}"], gap["CMAS"],
            "", "", "",
            "", "", "",
            acc["top1"], acc["top5"],
            float(t1 - t0),
        ])

    jf.close()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    print(f"\n[Done] JSONL -> {out_jsonl}")
    print(f"[Done] CSV   -> {out_csv}")


if __name__ == "__main__":
    main()
