#!/usr/bin/env python3
"""
================================================================================
SIMPLE LAUNCHER: run_all.py
Place this in phase1/ folder (next to config.yaml)
Run:  python run_all.py
================================================================================
Assumes you already have:
  data/raw/*.csv  and  data/images/*/*.jpg
================================================================================
"""
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def check_structure():
    """Verify folders and files exist."""
    banner("CHECKING PROJECT STRUCTURE")
    
    required_folders = [
        "data/raw",
        "data/images/honest_ooc",
        "data/images/fakehealth",
        "data/images/recovery",
        "src",
        "scripts",
    ]
    
    for f in required_folders:
        path = PROJECT_ROOT / f
        if not path.exists():
            print(f"    Missing folder: {f}")
            return False
        print(f"   {f}")
    
    # Check CSVs
    csv_files = {
        "honest_ooc": "data/raw/honest_ooc_fixed.csv",
        "fakehealth": "data/raw/fakehealth_fixed.csv",
        "recovery": "data/raw/recovery_fixed.csv",
    }
    
    found_csvs = {}
    for name, path in csv_files.items():
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            print(f"    CSV found: {path}")
            found_csvs[name] = full_path
        else:
            # Try alternate names
            alt = PROJECT_ROOT / "data" / "raw" / f"{name}_content.csv"
            if alt.exists():
                print(f"   CSV found (alt): {alt}")
                found_csvs[name] = alt
            else:
                print(f"     CSV not found: {path}")
    
    if not found_csvs:
        print("\n No dataset CSVs found in data/raw/!")
        return False
    
    return found_csvs


def standardize_csv(csv_path, dataset_name, image_folder):
    """
    Ensure CSV has exactly these columns:
    claim_text, image_path, label, source_dataset
    """
    print(f"\n   Standardizing {dataset_name}...")
    df = pd.read_csv(csv_path)
    original_cols = list(df.columns)
    print(f"      Original columns: {original_cols}")
    
    # --- Find image path column ---
    img_col = None
    for c in ["local_image_path", "image_path", "img_path", "image", "url"]:
        if c in df.columns:
            img_col = c
            break
    
    if img_col is None:
        print(f"      No image column found!")
        return False
    
    # --- Find text column ---
    text_col = None
    for c in ["claim_text", "caption", "content", "title", "text", "news_title"]:
        if c in df.columns:
            text_col = c
            break
    
    if text_col is None:
        print(f"     No text column found!")
        return False
    
    # --- Find label column ---
    label_col = None
    for c in ["label", "rating", "verdict", "reliability", "is_fake", "is_reliable"]:
        if c in df.columns:
            label_col = c
            break
    
    if label_col is None:
        print(f"      No label column found!")
        return False
    
    # --- Build clean dataframe ---
    clean = pd.DataFrame()
    clean["claim_text"] = df[text_col].astype(str)
    
    # Fix image paths: ensure they point to data/images/{folder}/
    def fix_path(p):
        p = str(p).strip().replace("\\", "/")
        # If already correct, keep it
        if p.startswith("data/images/"):
            return p
        # Otherwise, take filename and rebuild path
        filename = Path(p).name
        return f"data/images/{image_folder}/{filename}"
    
    clean["image_path"] = df[img_col].apply(fix_path)
    
    # Convert label to int (0 or 1)
    clean["label"] = pd.to_numeric(df[label_col], errors="coerce")
    # Handle non-numeric labels (e.g., "true"/"false")
    if clean["label"].isna().sum() > 0:
        unique_vals = df[label_col].unique()
        if set(unique_vals).issubset({"true", "false", "real", "fake", 0, 1, "0", "1"}):
            mapping = {"true": 1, "false": 0, "real": 1, "fake": 0, "1": 1, "0": 0}
            clean["label"] = df[label_col].astype(str).str.lower().map(mapping)
        else:
            # Fallback: median split
            clean["label"] = (df[label_col].rank() > len(df)/2).astype(int)
    
    clean["label"] = clean["label"].astype(int)
    clean["source_dataset"] = dataset_name
    
    # Verify images exist
    exists = clean["image_path"].apply(lambda x: (PROJECT_ROOT / x).exists())
    found = exists.sum()
    print(f"      Images found locally: {found}/{len(clean)}")
    
    if found == 0:
        print(f"        WARNING: No images found! Check data/images/{image_folder}/")
    
    # Save standardized CSV
    out_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}_clean.csv"
    clean.to_csv(out_path, index=False)
    print(f"       Saved: {out_path} ({len(clean)} rows)")
    return True


def update_config():
    """Update config.yaml to point to _clean.csv files."""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        print("    config.yaml not found. Skipping auto-update.")
        return
    
    with open(config_path, "r") as f:
        lines = f.readlines()
    
    # Simple text replacement
    new_lines = []
    for line in lines:
        if "honest_ooc:" in line and ".csv" in line:
            new_lines.append('  honest_ooc: "data/raw/honest_ooc_clean.csv"\n')
        elif "fakehealth:" in line and ".csv" in line:
            new_lines.append('  fakehealth: "data/raw/fakehealth_clean.csv"\n')
        elif "recovery:" in line and ".csv" in line:
            new_lines.append('  recovery: "data/raw/recovery_clean.csv"\n')
        else:
            new_lines.append(line)
    
    with open(config_path, "w") as f:
        f.writelines(new_lines)
    
    print("   Updated config.yaml paths")


def run_preprocess():
    banner("STEP 1: PREPROCESSING")
    cmd = [sys.executable, "scripts/preprocess.py", "--skip-missing"]
    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_train():
    banner("STEP 2: TRAINING")
    cmd = [
        sys.executable, "src/train.py",
        "--mode", "binary",
        "--epochs", "15",
        "--unfreeze_biobert", "4",
        "--unfreeze_vit", "2",
        "--fusion", "concat"
    ]
    print(f"   Running: {' '.join(cmd)}")
    print("\n   💡 If you get CUDA Out of Memory, stop and run with:")
    print("      python src/train.py --mode binary --epochs 15 --freeze_backbones --fusion concat --batch_size 8")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    print("""
     ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗     ██╗
     ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ███║
     ██████╔╝███████║███████║█████╗  █████╗      ╚██║
     ██╔═══╝ ██╔══██║██╔══██║██╔══╝  ██╔══╝       ██║
     ██║     ██║  ██║██║  ██║██║     ███████╗     ██║
     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝     ╚═╝
    """)
    
    # 1. Check structure
    csvs = check_structure()
    if not csvs:
        sys.exit(1)
    
    # 2. Standardize each CSV
    banner("STANDARDIZING CSVs")
    mappings = {
        "honest_ooc": ("honest_ooc", "honest_ooc"),
        "fakehealth": ("fakehealth", "fakehealth"),
        "recovery": ("recovery", "recovery"),
    }
    
    for key, (name, folder) in mappings.items():
        if key in csvs:
            standardize_csv(csvs[key], name, folder)
    
    # 3. Update config
    update_config()
    
    # 4. Preprocess
    if not run_preprocess():
        print("\n Preprocessing failed.")
        sys.exit(1)
    
    # 5. Train
    if not run_train():
        print("\n Training failed.")
        sys.exit(1)
    
    banner("ALL DONE! ")
    print("Check checkpoints/ for your saved models.")


if __name__ == "__main__":
    main()