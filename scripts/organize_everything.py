#!/usr/bin/env python3
"""
One-time organizer. Run this in your my_project/ folder.
It will:
1. Copy Honest OOC images from deep nested folders → data/images/honest_ooc/
2. Update honest_ooc_FIXED.csv → data/raw/honest_ooc_clean_long.csv
3. Guide you through FakeHealth and ReCOVery
"""
import os
import shutil
import pandas as pd
from pathlib import Path

print("=" * 70)
print("DATASET ORGANIZER")
print("=" * 70)

# =============================================================================
# 1. ORGANIZE HONEST OOC
# =============================================================================
print("\n[1/3] Organizing Honest OOC...")

ho_csv = "honest_ooc_FIXED.csv"
if not os.path.exists(ho_csv):
    print(f" {ho_csv} not found in current folder.")
    print("   Make sure you run this script from the same folder where")
    print("   download_and_fix.py created honest_ooc_FIXED.csv")
    exit(1)

df = pd.read_csv(ho_csv)
print(f"   Loaded CSV: {len(df)} rows")

# Detect image path column
img_col = None
for c in ["local_image_path", "image_path", "img_path"]:
    if c in df.columns:
        img_col = c
        break

if img_col is None:
    print(f"   Columns found: {list(df.columns)}")
    img_col = input("   Which column has the image paths? ").strip()

# Create destination
os.makedirs("data/images/honest_ooc", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# Copy images and build new paths
new_paths = []
copied = 0
missing = 0

for idx, row in df.iterrows():
    old_path = str(row[img_col]).strip()
    
    # Handle Windows backslashes
    old_path = old_path.replace("\\", "/")
    if old_path.startswith("./"):
        old_path = old_path[2:]
    
    # Get just the filename
    filename = os.path.basename(old_path)
    
    # Source and destination
    src = os.path.join(os.getcwd(), old_path.replace("/", os.sep))
    dst = os.path.join("data", "images", "honest_ooc", filename)
    
    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        new_paths.append(f"data/images/honest_ooc/{filename}")
        copied += 1
    else:
        new_paths.append("")  # Will be filtered out later
        missing += 1
    
    if (idx + 1) % 1000 == 0:
        print(f"   Processed {idx + 1}/{len(df)} images...")

print(f"  Copied: {copied} |  Missing: {missing}")

# Update CSV
df["image_path"] = new_paths
df = df[df["image_path"] != ""].reset_index(drop=True)  # Drop missing

# Keep only needed columns
keep_cols = ["claim_text", "caption", "label", "image_path", "source_dataset"]
existing_cols = [c for c in keep_cols if c in df.columns]
df = df[existing_cols]

# Ensure source_dataset column exists
if "source_dataset" not in df.columns:
    df["source_dataset"] = "honest_ooc"

# Rename caption to claim_text if needed
if "caption" in df.columns and "claim_text" not in df.columns:
    df = df.rename(columns={"caption": "claim_text"})

output_csv = "data/raw/honest_ooc_clean_long.csv"
df.to_csv(output_csv, index=False)
print(f"  Saved organized CSV: {output_csv} ({len(df)} rows)")

# =============================================================================
# 2. FAKEHEALTH & RECOVERY
# =============================================================================
print("\n" + "=" * 70)
print("[2/3] FakeHealth & ReCOVery")
print("=" * 70)

print("""
⚠️  You told me you have FakeHealth and ReCOVery in ZIP files.

   BEFORE continuing, you MUST:

   1. Extract your FakeHealth ZIP file
   2. Move ALL images to: data/images/fakehealth/
   3. Extract your ReCOVery ZIP file  
   4. Move ALL images to: data/images/recovery/

   Then place your original CSVs in data/raw/:
      - data/raw/fakehealth_content.csv
      - data/raw/recovery-news-data.csv
""")

input("\n   Press ENTER after you have done the above...")

# =============================================================================
# 3. FIX FAKEHEALTH & RECOVERY PATHS
# =============================================================================
print("\n[3/3] Fixing FakeHealth & ReCOVery paths...")

def fix_dataset_csv(csv_name, dataset_name, image_folder):
    csv_path = f"data/raw/{csv_name}"
    if not os.path.exists(csv_path):
        print(f"   {csv_path} not found. Skipping {dataset_name}.")
        return
    
    df = pd.read_csv(csv_path)
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image"]:
        if c in df.columns:
            img_col = c
            break
    
    if img_col is None:
        print(f"  No image column in {csv_name}")
        return
    
    print(f"   Processing {dataset_name}: {len(df)} rows")
    
    # Check how many images exist
    df["image_path_fixed"] = df[img_col].apply(
        lambda x: f"data/images/{image_folder}/{os.path.basename(str(x))}"
    )
    
    exists = df["image_path_fixed"].apply(lambda x: os.path.exists(x))
    found = exists.sum()
    
    print(f"      Images found: {found}/{len(df)}")
    
    if found == 0:
        print(f"  No images found! Check your extraction.")
        return
    
    # Update the CSV
    df["image_path"] = df["image_path_fixed"]
    df = df.drop(columns=["image_path_fixed"])
    
    # Ensure standard columns
    if "source_dataset" not in df.columns:
        df["source_dataset"] = dataset_name.lower()
    
    if "claim_text" not in df.columns:
        # Try to find text column
        for c in ["content", "title", "text", "claim"]:
            if c in df.columns:
                df = df.rename(columns={c: "claim_text"})
                break
    
    df.to_csv(csv_path, index=False)
    print(f" Updated: {csv_path}")

fix_dataset_csv("fakehealth_content.csv", "FakeHealth", "fakehealth")
fix_dataset_csv("recovery-news-data.csv", "ReCOVery", "recovery")

print("\n" + "=" * 70)
print("ORGANIZATION COMPLETE!")
print("=" * 70)
print("""
NEXT STEPS:
-----------
1. Update your config.yaml:

   data:
     raw_dir: "data/raw"
     processed_dir: "data/processed"
     image_root: "data/images"
     honest_ooc: "data/raw/honest_ooc_clean_long.csv"
     fakehealth: "data/raw/fakehealth_content.csv"
     recovery: "data/raw/recovery-news-data.csv"

2. Run: python scripts/preprocess.py --skip-missing

3. Run: python src/train.py --mode binary --epochs 15 --unfreeze_biobert 4 --unfreeze_vit 2 --fusion concat
""")