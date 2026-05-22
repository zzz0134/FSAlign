import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Optional heavy downloads
COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images":   "http://images.cocodataset.org/zips/val2017.zip",
    "annotations":  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _maybe_download(url: str, dst: Path):
    import urllib.request
    if dst.exists():
        return dst
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return dst

def _unzip(zip_path: Path, dst_dir: Path):
    import zipfile
    print(f"Extracting {zip_path.name} -> {dst_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir)

def prepare_coco2017(root: Path, download: bool, coco_images: Optional[str], coco_ann: Optional[str]):
    coco_root = root / "coco2017"
    img_root = coco_root / "images"
    ann_root = coco_root / "annotations"
    _ensure_dir(img_root); _ensure_dir(ann_root)

    if download:
        tmp = root / "_tmp"
        _ensure_dir(tmp)
        train_zip = _maybe_download(COCO_URLS["train_images"], tmp / "train2017.zip")
        val_zip   = _maybe_download(COCO_URLS["val_images"], tmp / "val2017.zip")
        ann_zip   = _maybe_download(COCO_URLS["annotations"], tmp / "annotations_trainval2017.zip")
        _unzip(train_zip, coco_root)
        _unzip(val_zip, coco_root)
        _unzip(ann_zip, coco_root)
        # Move into expected layout
        _ensure_dir(img_root)
        if (coco_root / "train2017").exists():
            shutil.move(str(coco_root / "train2017"), str(img_root / "train2017"))
        if (coco_root / "val2017").exists():
            shutil.move(str(coco_root / "val2017"), str(img_root / "val2017"))
        if (coco_root / "annotations").exists():
            # already at coco_root/annotations
            pass
        print("COCO2017 prepared.")
    else:
        # Use user-provided dirs
        if coco_images:
            coco_images = Path(coco_images)
            # Expect contains train2017/ and val2017/
            assert (coco_images / "train2017").exists() and (coco_images / "val2017").exists(), \
                f"{coco_images} must contain train2017/ and val2017/"
            _ensure_dir(img_root)
            if not (img_root / "train2017").exists():
                shutil.copytree(coco_images / "train2017", img_root / "train2017")
            if not (img_root / "val2017").exists():
                shutil.copytree(coco_images / "val2017", img_root / "val2017")
        if coco_ann:
            coco_ann = Path(coco_ann)
            assert coco_ann.exists(), f"annotations path not found: {coco_ann}"
            # Expect captions_*.json and instances_*.json
            for f in ["instances_train2017.json", "instances_val2017.json",
                      "captions_train2017.json", "captions_val2017.json"]:
                src = coco_ann / f
                if src.exists():
                    shutil.copy2(src, ann_root / f)
        print("COCO2017 linked/copied from provided paths.")

def prepare_cifar100(root: Path):
    # Let torchvision handle the download into <root>/cifar100
    from torchvision.datasets import CIFAR100
    data_dir = root / "cifar100"
    _ensure_dir(data_dir)
    _ = CIFAR100(root=str(data_dir), train=True, download=True)
    _ = CIFAR100(root=str(data_dir), train=False, download=True)
    print("CIFAR-100 prepared.")

def verify_imagenet(root: Path):
    # Expect standard ImageFolder layout
    imgnet = root / "imagenet"
    train = imgnet / "train"
    val = imgnet / "val"
    if not train.exists() or not val.exists():
        print(f"[WARN] Expected ImageNet folders not found under: {imgnet}")
        print("       Place folders as imagenet/train/<wnid>/*.JPEG and imagenet/val/<wnid>/*.JPEG")
        return
    # Do a light check: at least 100 classes and a few images
    cls_train = [p for p in train.iterdir() if p.is_dir()]
    cls_val = [p for p in val.iterdir() if p.is_dir()]
    n_train, n_val = len(cls_train), len(cls_val)
    n_train_imgs = sum(len(list(p.glob('*.JPEG'))) for p in cls_train[:5])
    n_val_imgs = sum(len(list(p.glob('*.JPEG'))) for p in cls_val[:5])
    print(f"ImageNet appears present. Classes: train={n_train}, val={n_val}. Sample counts (subset check): train~{n_train_imgs}, val~{n_val_imgs}")

def prepare_dtd(root: Path):
    from torchvision.datasets import DTD
    data_dir = root / "dtd"
    _ensure_dir(data_dir)
    for split in ["train", "val", "test"]:
        _ = DTD(root=str(data_dir), split=split, download=True)
    print("DTD prepared.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Target dataset root, e.g., /work/was598/modilty_gap/tools/data")
    ap.add_argument("--dataset", type=str, required=True, choices=["coco2017", "dtd", "cifar100", "imagenet", "all"])
    ap.add_argument("--download", action="store_true", help="For COCO 2017: download official zips")
    ap.add_argument("--coco-images", type=str, default=None, help="Existing COCO images dir containing train2017/ and val2017/")
    ap.add_argument("--coco-ann", type=str, default=None, help="Existing COCO annotations dir containing *.json")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("coco2017", "all"):
        prepare_coco2017(root, download=args.download, coco_images=args.coco_images, coco_ann=args.coco_ann)
    if args.dataset in ("cifar100", "all"):
        prepare_cifar100(root)
    if args.dataset in ("imagenet", "all"):
        verify_imagenet(root)
    if args.dataset in ("dtd", "all"):
        prepare_dtd(root)

if __name__ == "__main__":
    main()
