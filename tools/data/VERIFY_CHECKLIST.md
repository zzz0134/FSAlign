# FSAlign Data Download & Verification Checklist

This checklist is for reproducing the release run with:
- `FSAlign_release.py`
- `tools/run_release_gap_embedded_20260521.sh`

## 1) One-command bootstrap

```bash
bash tools/download_data_release.sh
```

Optional:

```bash
DATA_ROOT=/your/path/to/data bash tools/download_data_release.sh
```

## 2) What is auto-downloaded

- COCO2017 images + annotations
- CIFAR100
- DTD
- Tiny-ImageNet-200

## 3) What is usually manual/license-gated

- Flickr30k raw data/resources under `tools/data/flickr30k/`
- MSCOCO2014/Karpathy resources under `tools/data/mscoco2014/`

## 4) Verification command

```bash
python tools/data/verify_release_data.py --root tools/data
```

Pass condition:
- Script exits with code `0`
- Summary reports `MISS=0`

## 5) Expected key paths

- `tools/data/coco2017/images/train2017`
- `tools/data/coco2017/images/val2017`
- `tools/data/coco2017/annotations/captions_train2017.json`
- `tools/data/coco2017/annotations/captions_val2017.json`
- `tools/data/cifar100/cifar-100-python/train`
- `tools/data/cifar100/cifar-100-python/test`
- `tools/data/dtd/dtd/images`
- `tools/data/dtd/dtd/labels`
- `tools/data/tiny-imagenet-200/train`
- `tools/data/tiny-imagenet-200/val`
- `tools/data/flickr30k/karpathy_splits.json`
- `tools/data/flickr30k/results_20130124.token`
- `tools/data/mscoco2014`

