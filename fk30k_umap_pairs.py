"""
Visualize Flickr30k image-text paired embeddings in 2D with UMAP (lines connect pairs).

Install:
  pip install torch torchvision transformers datasets umap-learn matplotlib pillow tqdm

Run:
  python fk30k_umap_pairs.py --root /work/was598/modilty_gap/tools/data/flickr30k --split test --n 400 --out fk30k_umap.png
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import umap
from transformers import CLIPProcessor, CLIPModel


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_dir(root: Path) -> Path:
    # pick the directory with the most image files
    candidates = []
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        cnt = sum(1 for _ in d.glob("*") if _.suffix.lower() in IMG_EXTS)
        if cnt > 0:
            candidates.append((cnt, d))
    if not candidates:
        raise FileNotFoundError(f"No images found under: {root}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_caption_file(root: Path) -> Path:
    # Prefer Flickr30k token file: results_20130124.token
    token = list(root.rglob("*.token"))
    if token:
        # choose the largest token file
        token.sort(key=lambda p: p.stat().st_size, reverse=True)
        return token[0]

    # Fallbacks: common caption files
    patterns = ["*caption*.txt", "*captions*.txt", "*.csv", "*.json"]
    cands = []
    for pat in patterns:
        cands += list(root.rglob(pat))
    # Filter obvious non-caption files crudely
    cands = [p for p in cands if p.is_file() and p.stat().st_size > 0]
    if not cands:
        raise FileNotFoundError(
            f"No caption file found under: {root}\n"
            "Expected something like results_20130124.token (lines: image.jpg#0\\tcaption).\n"
            "If your captions are in a custom format, tell me the filename and format."
        )
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def parse_captions(caption_path: Path):
    """
    Returns: dict[str, list[str]] mapping image filename -> list of captions
    Supports:
      - .token:  image.jpg#0<TAB>caption
      - .txt:    try: image.jpg<TAB>caption  OR  image.jpg|caption  (heuristic)
      - .csv:    columns containing image + caption
      - .json:   a list/dict containing image/caption fields (best-effort)
    """
    ext = caption_path.suffix.lower()
    caps = {}

    def add(img_name, cap):
        img_name = Path(img_name).name  # keep basename
        caps.setdefault(img_name, []).append(cap.strip())

    if ext == ".token":
        with caption_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # expected: "123.jpg#0\tA man ..."
                if "\t" not in line:
                    continue
                left, cap = line.split("\t", 1)
                img = left.split("#", 1)[0]
                add(img, cap)

    elif ext == ".csv":
        with caption_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            # try to guess column names
            cols = [c.lower() for c in reader.fieldnames or []]
            img_key = None
            cap_key = None
            for c in cols:
                if img_key is None and ("image" in c or "filename" in c or "file" in c):
                    img_key = c
                if cap_key is None and ("caption" in c or "text" in c or "sentence" in c):
                    cap_key = c
            if img_key is None or cap_key is None:
                raise ValueError(f"CSV columns not recognized: {reader.fieldnames}")
            # map back to original case
            name_map = {c.lower(): c for c in (reader.fieldnames or [])}
            img_key = name_map[img_key]
            cap_key = name_map[cap_key]
            for row in reader:
                add(row[img_key], row[cap_key])

    elif ext == ".json":
        with caption_path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)

        def handle_item(item):
            # best-effort keys
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
            # maybe dict: {image: [caps]} or has "annotations"
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
        # .txt or others: heuristic parsing
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
                else:
                    # cannot parse; skip
                    continue

    if not caps:
        raise ValueError(f"Parsed 0 captions from: {caption_path}")
    return caps


def find_split_file(root: Path, split: str):
    # common filenames: train.txt / val.txt / test.txt / train_images.txt, etc.
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
            # keep basename, strip possible paths
            names.append(Path(s).name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/work/was598/modilty_gap/tools/data/flickr30k")
    ap.add_argument("--split", type=str, default="test", help="train/val/test; if no split file found, uses all images")
    ap.add_argument("--n", type=int, default=400, help="number of pairs to visualize")
    ap.add_argument("--caption_idx", type=int, default=0, help="which caption index to pick per image (0..4)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="fk30k_umap.png")
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_min_dist", type=float, default=0.10)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)

    img_dir = find_image_dir(root)
    cap_file = find_caption_file(root)
    caps = parse_captions(cap_file)

    # candidate images: intersection of images on disk and images with captions
    disk_imgs = {p.name: p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS}
    common = sorted(set(disk_imgs.keys()) & set(caps.keys()))
    if not common:
        raise RuntimeError(
            f"No overlap between images in {img_dir} and captions in {cap_file}.\n"
            f"Example image names: {list(disk_imgs.keys())[:5]}\n"
            f"Example caption keys: {list(caps.keys())[:5]}"
        )

    # optionally filter by split list
    split_path = find_split_file(root, args.split)
    if split_path is not None:
        split_names = set(read_split_list(split_path))
        common = [n for n in common if n in split_names]
        if not common:
            raise RuntimeError(f"Split file found ({split_path}) but no matching images after filtering.")

    # sample N
    n = min(args.n, len(common))
    chosen = common[:n]  # deterministic
    # If you want random sampling, uncomment:
    # chosen = random.sample(common, n)

    # load CLIP
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()

    img_embeds, txt_embeds = [], []

    def pick_cap(img_name: str):
        lst = caps[img_name]
        idx = max(0, min(args.caption_idx, len(lst) - 1))
        return lst[idx]

    for start in tqdm(range(0, n, args.batch_size), desc="Encoding"):
        batch_names = chosen[start : start + args.batch_size]
        images = []
        texts = []
        for nm in batch_names:
            p = disk_imgs[nm]
            images.append(Image.open(p).convert("RGB"))
            texts.append(pick_cap(nm))

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            im = out.image_embeds
            tx = out.text_embeds
            im = im / im.norm(dim=-1, keepdim=True)
            tx = tx / tx.norm(dim=-1, keepdim=True)

        img_embeds.append(im.detach().cpu().numpy())
        txt_embeds.append(tx.detach().cpu().numpy())

    img_embeds = np.concatenate(img_embeds, axis=0)
    txt_embeds = np.concatenate(txt_embeds, axis=0)

    # UMAP on combined space
    all_embeds = np.vstack([img_embeds, txt_embeds])
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    coords = reducer.fit_transform(all_embeds)
    img_xy = coords[:n]
    txt_xy = coords[n:]

    # plot
    plt.figure(figsize=(9, 7))
    plt.scatter(img_xy[:, 0], img_xy[:, 1], marker="^", s=28, alpha=0.85, label="Image")
    plt.scatter(txt_xy[:, 0], txt_xy[:, 1], marker="o", s=18, alpha=0.85, label="Text")

    for i in range(n):
        plt.plot([img_xy[i, 0], txt_xy[i, 0]], [img_xy[i, 1], txt_xy[i, 1]], linewidth=0.6, alpha=0.35)

    plt.title(f"Flickr30k (local) Paired CLIP Embeddings + UMAP (N={n})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[OK] image_dir = {img_dir}")
    print(f"[OK] caption_file = {cap_file}")
    print(f"[OK] saved = {args.out}")


if __name__ == "__main__":
    main()