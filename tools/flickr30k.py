import os, json, zipfile, argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", required=True, help="Path to flickr30k-images.zip")
    ap.add_argument("--csv_path", required=True, help="Path to flickr_annotations_30k.csv")
    ap.add_argument("--out_root", default="data/flickr30k", help="Output root folder")
    args = ap.parse_args()

    OUT_ROOT = Path(args.out_root)
    IMG_DIR  = OUT_ROOT / "flickr30k-images"
    TOKEN    = OUT_ROOT / "results_20130124.token"
    SPLITJS  = OUT_ROOT / "karpathy_splits.json"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 解压图片
    print(f"[+] Extracting images from {args.zip_path} ...")
    with zipfile.ZipFile(args.zip_path, 'r') as z:
        members = [m for m in z.namelist() if m.lower().endswith(".jpg")]
        for m in members:
            name = Path(m).name
            out = IMG_DIR / name
            if not out.exists():
                with z.open(m) as src, open(out, "wb") as dst:
                    dst.write(src.read())
    print(f"[+] Images extracted to {IMG_DIR}")

    # ---- 读取 CSV
    print(f"[+] Reading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)

    # ---- 写 results_20130124.token
    print(f"[+] Writing {TOKEN} ...")
    with open(TOKEN, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            fn = str(row["filename"])
            caps = json.loads(row["raw"])
            for k, c in enumerate(caps):
                f.write(f"{fn}#{k}\t{c}\n")

    # ---- 生成 karpathy_splits.json
    print(f"[+] Writing {SPLITJS} ...")
    filenames = sorted(df["filename"].astype(str).unique())
    fn2imgidx = {fn: i for i, fn in enumerate(filenames)}

    text_indices_per_fn = {fn: [] for fn in filenames}
    running = 0
    with open(TOKEN, "r", encoding="utf-8") as f:
        for line in f:
            fn_k, _ = line.strip().split("\t", 1)
            fn, _ = fn_k.split("#")
            text_indices_per_fn[fn].append(running)
            running += 1

    splits = {"train": {"images": [], "texts": []},
              "val":   {"images": [], "texts": []},
              "test":  {"images": [], "texts": []}}

    df_first = df.drop_duplicates(subset=["filename"])
    sp_map = dict(zip(df_first["filename"].astype(str), df_first["split"].astype(str)))

    for fn in filenames:
        sp = sp_map.get(fn, "train").lower()
        if sp not in splits:
            sp = "train"
        splits[sp]["images"].append(fn2imgidx[fn])
        splits[sp]["texts"].extend(text_indices_per_fn.get(fn, []))

    with open(SPLITJS, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    print("\n[OK] Prepared dataset for the project:")
    print(f" - {IMG_DIR}/")
    print(f" - {TOKEN}")
    print(f" - {SPLITJS}")

if __name__ == "__main__":
    main()
