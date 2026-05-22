#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple


def exists(p: Path) -> Tuple[bool, str]:
    return (p.exists(), str(p))


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="data root, e.g. tools/data")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    checks: List[Tuple[str, bool, str]] = []

    # COCO2017
    checks.append(("coco2017 train images dir", *exists(root / "coco2017/images/train2017")))
    checks.append(("coco2017 val images dir", *exists(root / "coco2017/images/val2017")))
    checks.append(("coco2017 captions_train2017.json", *exists(root / "coco2017/annotations/captions_train2017.json")))
    checks.append(("coco2017 captions_val2017.json", *exists(root / "coco2017/annotations/captions_val2017.json")))

    # CIFAR100
    checks.append(("cifar100 train file", *exists(root / "cifar100/cifar-100-python/train")))
    checks.append(("cifar100 test file", *exists(root / "cifar100/cifar-100-python/test")))

    # DTD
    checks.append(("dtd images dir", *exists(root / "dtd/dtd/images")))
    checks.append(("dtd labels dir", *exists(root / "dtd/dtd/labels")))

    # Tiny-ImageNet-200
    checks.append(("tiny-imagenet-200 train dir", *exists(root / "tiny-imagenet-200/train")))
    checks.append(("tiny-imagenet-200 val dir", *exists(root / "tiny-imagenet-200/val")))

    # Flickr30k / MSCOCO2014 (manual in most setups)
    checks.append(("flickr30k karpathy split json", *exists(root / "flickr30k/karpathy_splits.json")))
    checks.append(("flickr30k token file", *exists(root / "flickr30k/results_20130124.token")))
    checks.append(("mscoco2014 dir", *exists(root / "mscoco2014")))

    ok = 0
    fail = 0
    print(f"[verify] root={root}")
    for name, passed, path in checks:
        mark = "OK" if passed else "MISS"
        print(f"[{mark}] {name}: {path}")
        if passed:
            ok += 1
        else:
            fail += 1

    # lightweight cardinality hints
    def safe_count(d: Path, pat: str) -> int:
        return count_files(d, pat) if d.exists() else 0

    print("\n[counts]")
    print(f"coco2017 train *.jpg: {safe_count(root / 'coco2017/images/train2017', '*.jpg')}")
    print(f"coco2017 val *.jpg:   {safe_count(root / 'coco2017/images/val2017', '*.jpg')}")
    print(f"tiny train *.JPEG:    {safe_count(root / 'tiny-imagenet-200/train', '*.JPEG')}")
    print(f"tiny val *.JPEG:      {safe_count(root / 'tiny-imagenet-200/val', '*.JPEG')}")

    print(f"\n[summary] OK={ok} MISS={fail}")
    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

