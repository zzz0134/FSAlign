# FSAlign (ICML Release)
# Abstract
Vision–language models such as CLIP embed images and text into a shared space, but still suffer from a modality gap, where image and text features cluster separately and nearest neighbors are dominated by same-modality rather than true cross-modal matches. Existing works alleviate the modality gap by strengthening cross-modal losses, post-processing embeddings or similarities, or imposing geometric regularization, but they primarily enforce global alignment and can distort local geometry, limiting gains in local ranking and zero-shot accuracy. We propose Fractal Spectral Alignment (FSAlign), which reduces the modality gap by shaping and matching the multi-scale geometry of image and text embeddings. By enforcing Ahlfors-regularity and sub-Gaussian heat kernel bounds, FSAlign constructs a shared fractal multi-scale structure for multiple modalities. This structure captures geometry across scales, from local neighborhoods to global structure, and ensures shared fractal spectral geometry across modalities. Based on this structure, we introduce a fractal spectral zeta score derived from multi-scale heat kernels and minimize the discrepancy between pairwise image–text samples to align their multi-scale neighborhoods. We theoretically demonstrate that FSAlign can guarantee the alignment of local spectral measures and global fractional Dirichlet energies.

## Included Files
- `FSAlign_release.py` (main entrypoint)
- `MG.py` (modality-gap metrics)
- `vqav2_eval.py` (VQAv2 utility)
- `tools/run_release_gap_embedded_20260521.sh` (run script)
- `tools/data/` (lightweight data scripts + metadata)
- `results/clip_full_plus_zs_releasegap_20260521/our_code_final_results.jsonl` (release result)

## Run
```bash
bash tools/run_release_gap_embedded_20260521.sh
```

## Data
Please prepare datasets locally under `tools/data` using `tools/data/prepare_datasets.py`.
Large raw datasets are not stored in this GitHub repo.

## Citation
If you use this codebase in academic work, please cite as:

```bibtex
@inproceedings{
anonymous2026mitigating,
title={Mitigating the Modality Gap in Vision{\textendash}Language Models with Fractal Spectral Geometry},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=pGkM5BjfD1}
}
```

