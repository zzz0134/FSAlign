import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tvds

from .transforms import build_transforms
from .text_utils import SST2Tokenizer, sst2_collate

# -------- Paths --------

def _resolve_root(root: Optional[str]) -> Path:
    if root is None:
        root = "/work/was598/modilty_gap/tools/data"
    return Path(root).expanduser().resolve()

# -------- Vision Datasets --------

def _cifar100(root: str | None, split: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    r = _resolve_root(root) / "cifar100"
    is_train = split == "train"
    ds = tvds.CIFAR100(
        root=str(r),
        train=is_train,
        transform=build_transforms(image_size, is_train),
        download=True,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if is_train else False,
                      num_workers=num_workers, pin_memory=True)

def _imagenet(root: str | None, split: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    r = _resolve_root(root) / "imagenet" / split
    if not r.exists():
        raise FileNotFoundError(f"ImageNet split folder not found: {r}. Expected standard ImageFolder layout.")
    ds = tvds.ImageFolder(
        root=str(r),
        transform=build_transforms(image_size, is_train=(split=='train'))
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if split == "train" else False,
                      num_workers=num_workers, pin_memory=True)

def _coco2017(root: str | None, split: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool, task: str = "captions") -> DataLoader:
    """
    task: 'captions' uses CocoCaptions; 'det' uses CocoDetection (instances).
    """
    r = _resolve_root(root) / "coco2017"
    images = r / "images" / (f"{'train2017' if split=='train' else 'val2017'}")
    ann_dir = r / "annotations"
    if task == "captions":
        ann = ann_dir / f"captions_{'train2017' if split=='train' else 'val2017'}.json"
        ds = tvds.CocoCaptions(str(images), str(ann), transform=build_transforms(image_size, is_train=(split=='train')))
    else:
        ann = ann_dir / f"instances_{'train2017' if split=='train' else 'val2017'}.json"
        ds = tvds.CocoDetection(str(images), str(ann), transform=build_transforms(image_size, is_train=(split=='train')))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if split == "train" else False,
                      num_workers=num_workers, pin_memory=True)

def _dtd(root: str | None, split: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    r = _resolve_root(root) / "dtd"
    if split not in {"train", "val", "test"}:
        raise ValueError("DTD split must be one of 'train', 'val', or 'test'")
    ds = tvds.DTD(
        root=str(r),
        split=split,
        transform=build_transforms(image_size, is_train=(split=='train')),
        download=True
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if split == "train" else False,
                      num_workers=num_workers, pin_memory=True)


# -------- Text Dataset (SST-2) --------

class _JSONLDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

def _sst2(root: str | None, split: str, batch_size: int, num_workers: int, tokenizer_name: str, max_length: int) -> DataLoader:
    r = _resolve_root(root) / "sst2"
    fp = r / (f"{'train' if split=='train' else ('dev' if split=='dev' else 'test')}.jsonl")
    if not fp.exists():
        raise FileNotFoundError(f"SST-2 file not found: {fp}. Please run the prepare step first.")
    ds = _JSONLDataset(str(fp))
    tok = SST2Tokenizer(model_name=tokenizer_name, max_length=max_length)
    collate = lambda batch: sst2_collate(batch, tok)
    return DataLoader(ds, batch_size=batch_size, shuffle=True if split=='train' else False,
                      num_workers=num_workers, collate_fn=collate, pin_memory=True)

# -------- Public Factory --------

def create_dataloader(
    name: str,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    shuffle: bool = True,
    root: Optional[str] = None,
    # text options
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    # coco options
    coco_task: str = "captions",
):
    name = name.lower()
    if name == "cifar100":
        return _cifar100(root, split, image_size, batch_size, num_workers, shuffle)
    elif name == "imagenet":
        return _imagenet(root, split, image_size, batch_size, num_workers, shuffle)
    elif name in ("coco", "coco2017", "mscoco"):
        return _coco2017(root, split, image_size, batch_size, num_workers, shuffle, task=coco_task)
    elif name in ("dtd", "textures"):
        return _dtd(root, split, image_size, batch_size, num_workers, shuffle)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
