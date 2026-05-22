# FSAlign (ICML Release)

Official release package for FSAlign.

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
