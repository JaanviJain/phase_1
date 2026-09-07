#!/usr/bin/env python3
"""Download missing images from URLs in dataset CSVs."""
import os
import sys
import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import urlparse
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config


def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def extract_urls_from_csv(csv_path: str, image_root: str):
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image", "url"]:
        if c in df.columns:
            img_col = c
            break
    if img_col is None:
        print(f"No image column in {csv_path}. Columns: {list(df.columns)}")
        return []
    
    downloads = []
    for idx, row in df.iterrows():
        path_or_url = str(row[img_col]).strip()
        if os.path.exists(path_or_url):
            continue
        if path_or_url.startswith(("http://", "https://")):
            parsed = urlparse(path_or_url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"img_{idx}.jpg"
            target_path = os.path.join(image_root, filename)
            downloads.append((path_or_url, target_path))
        elif "http" in path_or_url:
            url = path_or_url[path_or_url.find("http"):]
            filename = os.path.basename(urlparse(url).path) or f"img_{idx}.jpg"
            target_path = os.path.join(image_root, filename)
            downloads.append((url, target_path))
    return downloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", choices=["honest_ooc", "fakehealth", "recovery", "all"], default="all")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    os.makedirs(cfg.data.image_root, exist_ok=True)
    
    print("=" * 70)
    print("IMAGE DOWNLOADER")
    print("=" * 70)
    
    datasets_to_check = []
    if args.dataset in ("honest_ooc", "all"):
        datasets_to_check.append(("Honest OOC", cfg.data.honest_ooc))
    if args.dataset in ("fakehealth", "all"):
        datasets_to_check.append(("FakeHealth", cfg.data.fakehealth))
    if args.dataset in ("recovery", "all"):
        datasets_to_check.append(("ReCOVery", cfg.data.recovery))
    
    all_downloads = []
    for name, path in datasets_to_check:
        if os.path.exists(path):
            print(f"\nScanning {name}...")
            downloads = extract_urls_from_csv(path, cfg.data.image_root)
            print(f"  Found {len(downloads)} missing images to download")
            all_downloads.extend(downloads)
        else:
            print(f"\nSkipping {name} (file not found: {path})")
    
    if not all_downloads:
        print("\n All images already exist locally. Nothing to download.")
        return
    
    print(f"\nDownloading {len(all_downloads)} images to {cfg.data.image_root}...")
    success = 0
    failed = 0
    for url, save_path in tqdm(all_downloads):
        if os.path.exists(save_path):
            success += 1
            continue
        if download_image(url, save_path):
            success += 1
        else:
            failed += 1
            if failed <= 5:
                print(f"   Failed: {url[:80]}...")
    
    print(f"\n{'=' * 70}")
    print(f"Download complete: {success} success, {failed} failed")
    print(f"Images saved to: {cfg.data.image_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()