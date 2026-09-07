#!/usr/bin/env python3
"""
Fix Kaggle absolute paths in CSVs to local relative paths.
Run this AFTER extracting your zips to data/images/
"""
import os
import sys
import pandas as pd
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def fix_path(path: str, dataset_name: str) -> str:
    """
    Convert Kaggle absolute path to local relative path.
    """
    if pd.isna(path):
        return ""
    
    path = str(path).strip()
    
    # Recovery: /kaggle/working/images/recovery/0.jpg → data/images/recovery/0.jpg
    if "recovery" in path.lower() or dataset_name == "recovery":
        # Extract just the filename
        filename = os.path.basename(path)
        return f"data/images/recovery/{filename}"
    
    # FakeHealth: /kaggle/working/images/fakehealth/... or /kaggle/working/fakehealth/...
    if "fakehealth" in path.lower() or dataset_name == "fakehealth":
        filename = os.path.basename(path)
        return f"data/images/fakehealth/{filename}"
    
    # Honest OOC: long messy path like /kaggle/working/.../bbc/images/0000/694.jpg
    if "honest" in path.lower() or "ooc" in path.lower() or dataset_name == "honest_ooc":
        # Extract just the filename (e.g., 694.jpg)
        filename = os.path.basename(path)
        # Honest OOC images often have subfolders, but we'll flatten to one folder
        return f"data/images/honest_ooc/{filename}"
    
    # Fallback: just take the filename
    return f"data/images/{dataset_name}/{os.path.basename(path)}"


def process_csv(input_path: str, output_path: str, dataset_name: str):
    """Read CSV, fix image paths, save new CSV."""
    if not os.path.exists(input_path):
        print(f" Not found: {input_path}")
        return False
    
    print(f"\nProcessing {dataset_name}...")
    df = pd.read_csv(input_path)
    
    # Detect image column
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image", "url"]:
        if c in df.columns:
            img_col = c
            break
    
    if img_col is None:
        print(f"  No image column found. Columns: {list(df.columns)}")
        return False
    
    print(f"  Found column: '{img_col}'")
    print(f"  Sample old path: {df[img_col].iloc[0]}")
    
    # Apply fix
    df[img_col] = df[img_col].apply(lambda x: fix_path(x, dataset_name))
    
    print(f"  Sample new path: {df[img_col].iloc[0]}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved fixed CSV to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fakehealth", default="data/raw/fakehealth_content.csv")
    parser.add_argument("--recovery", default="data/raw/recovery-news-data.csv")
    parser.add_argument("--honest_ooc", default="data/raw/honest_ooc_clean_long.csv")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FIXING KAGGLE PATHS → LOCAL PATHS")
    print("=" * 70)
    print("\n BEFORE running this:")
    print("   1. Extract fakehealth.zip → data/images/fakehealth/")
    print("   2. Extract recovery.zip → data/images/recovery/")
    print("   3. If you have honest_ooc images, put them in data/images/honest_ooc/")
    
    # Process each dataset
    process_csv(args.fakehealth, "data/raw/fakehealth_fixed.csv", "fakehealth")
    process_csv(args.recovery, "data/raw/recovery_fixed.csv", "recovery")
    process_csv(args.honest_ooc, "data/raw/honest_ooc_fixed.csv", "honest_ooc")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Update config.yaml to point to the _fixed.csv files:")
    print("   fakehealth: 'data/raw/fakehealth_fixed.csv'")
    print("   recovery: 'data/raw/recovery_fixed.csv'")
    print("   honest_ooc: 'data/raw/honest_ooc_fixed.csv'")
    print("2. Run: python scripts/preprocess.py --skip-missing")
    print("3. Run: python src/train.py ...")


if __name__ == "__main__":
    main()