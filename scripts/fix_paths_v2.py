#!/usr/bin/env python3
"""
PATH FIXER v2 - Corrected for your folder structure:
  Images are in: data/raw/images/{dataset}/
  CSVs are in:   data/raw/
"""
import os
import pandas as pd
from pathlib import Path


def scan_images(folder_path):
    """Recursively scan folder for images."""
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')
    found = {}
    
    full_path = Path(folder_path)
    if not full_path.exists():
        print(f"  ❌ Folder not found: {folder_path}")
        return found
    
    count = 0
    for ext in exts:
        for file_path in full_path.rglob(f"*{ext}"):
            filename = file_path.name
            rel_path = str(file_path.relative_to(Path("."))).replace("\\", "/")
            found[filename] = rel_path
            count += 1
    
    print(f"  ✅ Found {count} images in {folder_path}")
    return found


def fix_csv(csv_path, dataset_name, image_scan):
    if not os.path.exists(csv_path):
        print(f"\n❌ CSV not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"CSV rows: {len(df)}")
    
    # Find image column
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image", "url", "local_image_path"]:
        if c in df.columns:
            img_col = c
            break
    
    if img_col is None:
        print(f"❌ No image column. Available: {list(df.columns)}")
        return False
    
    print(f"Image column: '{img_col}'")
    print(f"Sample old path: {df[img_col].iloc[0]}")
    
    # Match to real files
    new_paths = []
    matched = 0
    unmatched = 0
    
    for old_path in df[img_col]:
        old_path = str(old_path).strip()
        filename = Path(old_path).name
        
        if filename in image_scan:
            new_paths.append(image_scan[filename])
            matched += 1
        else:
            # Try case-insensitive match
            filename_lower = filename.lower()
            found = False
            for fname, fpath in image_scan.items():
                if fname.lower() == filename_lower:
                    new_paths.append(fpath)
                    matched += 1
                    found = True
                    break
            
            if not found:
                # Build expected path anyway
                new_paths.append(f"data/raw/images/{dataset_name}/{filename}")
                unmatched += 1
    
    print(f"  ✅ Matched to real files: {matched}/{len(df)}")
    print(f"  ❌ Still missing: {unmatched}/{len(df)}")
    
    # Update dataframe
    df["image_path"] = new_paths
    
    # Standardize
    if "source_dataset" not in df.columns:
        df["source_dataset"] = dataset_name
    
    if "claim_text" not in df.columns:
        for c in ["caption", "content", "title", "text", "news_title"]:
            if c in df.columns:
                df = df.rename(columns={c: "claim_text"})
                break
    
    keep = ["claim_text", "image_path", "label", "source_dataset"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    
    # Overwrite CSV
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved: {csv_path}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("PATH FIXER v2 (for data/raw/images/ structure)")
    print("=" * 60)
    
    # Scan the CORRECT folders
    honest_scan = scan_images("data/raw/images/honest_ooc")
    fake_scan = scan_images("data/raw/images/fakehealth")
    recov_scan = scan_images("data/raw/images/recovery")
    
    total = len(honest_scan) + len(fake_scan) + len(recov_scan)
    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES FOUND: {total}")
    print(f"  Honest OOC:  {len(honest_scan)}")
    print(f"  FakeHealth:  {len(fake_scan)}")
    print(f"  ReCOVery:    {len(recov_scan)}")
    print(f"{'='*60}")
    
    if total == 0:
        print("\n❌ STILL NO IMAGES FOUND")
        print("Please run this command and paste the output:")
        print("  Get-ChildItem data/raw/images -Recurse | Select-Object -First 50")
        exit(1)
    
    # Fix CSVs (they are in data/raw/)
    fix_csv("data/raw/honest_ooc_clean_long.csv", "honest_ooc", honest_scan)
    fix_csv("data/raw/fakehealth_content.csv", "fakehealth", fake_scan)
    fix_csv("data/raw/recovery-news-data.csv", "recovery", recov_scan)
    
    print("\n" + "=" * 60)
    print("NEXT: Run preprocessing")
    print("  python scripts/preprocess.py --skip-missing")
    print("=" * 60)