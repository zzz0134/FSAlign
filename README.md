# FSAlign (ICML Release Package)

This repository provides the release package for **FSAlign**, a modality-gap-aware cross-modal alignment method for image-text retrieval and transfer zero-shot evaluation.

## 1. What Is Included

The package contains the exact files used for the current release run:

- `FSAlign_release.py`: main algorithm/evaluation entrypoint (renamed from the internal `our_code` filename).
- `MG.py`: modality-gap metric and MG-shift utilities.
- `vqav2_eval.py`: VQAv2-related evaluation utilities used by the main script.
- `tools/run_release_gap_embedded_20260521.sh`: end-to-end run script.
- `tools/data/`: data root used by the run script.
- `results/clip_full_plus_zs_releasegap_20260521/our_code_final_results.jsonl`: result file requested for release.

## 2. Task Setting

The script evaluates:

- Cross-modal retrieval:
  - MS-COCO Karpathy split
  - Flickr30k Karpathy split
- Zero-shot classification:
  - CIFAR100
  - DTD
  - Tiny-ImageNet-200

Reported metrics include:

- Retrieval: I2T/T2I R@1, R@5, R@10
- Modality gap: CD, RMG, NAS@100, CMAS
- Classification: Top-1 / Top-5

## 3. Reproducibility Notes

- The release run uses CLIP (`ViT-B-32`, OpenAI weights).
- Training/evaluation hyperparameters are set in the provided run script.
- Gap computation is embedded in the main code path (not a separate post-hoc JSON rewrite).

## 4. Run

```bash
bash tools/run_release_gap_embedded_20260521.sh
```

Outputs are written under:

- `results/clip_full_plus_zs_releasegap_embedded_20260521/` (new run outputs)
- `results/clip_full_plus_zs_releasegap_20260521/our_code_final_results.jsonl` (included release result file)

## 5. Environment

Recommended environment:

- Python 3.10+
- PyTorch + CUDA-enabled GPU
- `transformers`, `open_clip_torch`, `torchvision`, `Pillow`, `numpy`

Install dependencies according to your local setup before running.

## 6. Contact

For questions about this release package, please open an issue in this repository.
