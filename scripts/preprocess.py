#!/usr/bin/env python3
"""Preprocess datasets for Phase 1."""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config


def standardize_honest_ooc(df: pd.DataFrame, cap_per_label: int = 1000) -> pd.DataFrame:
    img_col = "image_path" if "image_path" in df.columns else "image"
    cap_col = "caption" if "caption" in df.columns else "claim_text"
    df = df.copy()
    df["image_path"] = df[img_col].astype(str)
    df["claim_text"] = df[cap_col].astype(str)
    df["label"] = df["label"].astype(int)
    df["label"] = df["label"].map({1: 1, 2: 0})
    df["source_dataset"] = "honest_ooc"
    if cap_per_label is not None:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), cap_per_label), random_state=42)
        ).reset_index(drop=True)
        print(f"  Honest OOC capped: {len(df)} rows")
    return df[["claim_text", "image_path", "label", "source_dataset"]]


def standardize_fakehealth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_col = None
    for c in ["content", "claim_text", "title", "text"]:
        if c in df.columns:
            text_col = c
            break
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image"]:
        if c in df.columns:
            img_col = c
            break
    rating_col = None
    for c in ["rating", "label", "verdict", "is_fake"]:
        if c in df.columns:
            rating_col = c
            break
    if text_col is None or img_col is None or rating_col is None:
        print(f"  Warning: Auto-detect fallback. Columns: {list(df.columns)}")
        text_col = text_col or df.columns[0]
        img_col = img_col or df.columns[1]
        rating_col = rating_col or df.columns[2]
    df["claim_text"] = df[text_col].astype(str)
    df["image_path"] = df[img_col].astype(str)
    unique_ratings = df[rating_col].unique()
    if set(unique_ratings).issubset({0, 1}):
        df["label"] = df[rating_col].astype(int)
    else:
        df["label"] = (df[rating_col] > df[rating_col].median()).astype(int)
    df["source_dataset"] = "fakehealth"
    return df[["claim_text", "image_path", "label", "source_dataset"]]


def standardize_recovery(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_col = None
    for c in ["title", "claim_text", "content", "text", "news_title"]:
        if c in df.columns:
            text_col = c
            break
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image"]:
        if c in df.columns:
            img_col = c
            break
    label_col = None
    for c in ["reliability", "label", "rating", "is_reliable"]:
        if c in df.columns:
            label_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    if img_col is None:
        img_col = df.columns[1]
    if label_col is None:
        label_col = df.columns[2]
    df["claim_text"] = df[text_col].astype(str)
    df["image_path"] = df[img_col].astype(str)
    unique_vals = df[label_col].unique()
    if set(unique_vals).issubset({0, 1}):
        df["label"] = df[label_col].astype(int)
    else:
        df["label"] = df[label_col].astype(int)
    df["source_dataset"] = "recovery"
    return df[["claim_text", "image_path", "label", "source_dataset"]]


def stratified_split(df: pd.DataFrame, val_ratio: float, test_ratio: float, random_state: int):
    trainval, test = train_test_split(
        df, test_size=test_ratio, stratify=df["label"], random_state=random_state
    )
    val_size = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        trainval, test_size=val_size, stratify=trainval["label"], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-missing", action="store_true",
                       help="Skip rows with missing images instead of failing")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print("=" * 70)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 70)
    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    
    datasets = []
    
    if os.path.exists(cfg.data.honest_ooc):
        print(f"\nLoading Honest OOC from {cfg.data.honest_ooc}...")
        df_ho = pd.read_csv(cfg.data.honest_ooc)
        df_ho = standardize_honest_ooc(df_ho, cap_per_label=cfg.data.honest_ooc_cap_per_label)
        datasets.append(df_ho)
        print(f"  Loaded: {len(df_ho)} rows | Labels: {df_ho['label'].value_counts().to_dict()}")
    else:
        print(f"\n WARNING: {cfg.data.honest_ooc} NOT FOUND. Skipping Honest OOC.")
        print("   This is your PRIMARY dataset. Your results will be weaker.")
    
    if os.path.exists(cfg.data.fakehealth):
        print(f"\nLoading FakeHealth from {cfg.data.fakehealth}...")
        df_fh = pd.read_csv(cfg.data.fakehealth)
        df_fh = standardize_fakehealth(df_fh)
        datasets.append(df_fh)
        print(f"  Loaded: {len(df_fh)} rows | Labels: {df_fh['label'].value_counts().to_dict()}")
    else:
        print(f"\nWARNING: {cfg.data.fakehealth} not found. Skipping.")
    
    if os.path.exists(cfg.data.recovery):
        print(f"\nLoading ReCOVery from {cfg.data.recovery}...")
        df_rec = pd.read_csv(cfg.data.recovery)
        df_rec = standardize_recovery(df_rec)
        datasets.append(df_rec)
        print(f"  Loaded: {len(df_rec)} rows | Labels: {df_rec['label'].value_counts().to_dict()}")
    else:
        print(f"\n⚠️ WARNING: {cfg.data.recovery} not found. Skipping.")
    
    if not datasets:
        raise ValueError("No datasets found! Check your config.yaml paths.")
    
    print("\n" + "=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)
    merged = pd.concat(datasets, ignore_index=True)
    print(f"Total merged (before validation): {len(merged)}")
    print(f"Sources:\n{merged['source_dataset'].value_counts()}")
    print(f"Labels:\n{merged['label'].value_counts().sort_index()}")
    
    print("\nValidating image paths...")
    merged["exists"] = merged["image_path"].apply(lambda x: os.path.exists(str(x)))
    missing_count = (~merged["exists"]).sum()
    
    if missing_count > 0:
        print(f"  {missing_count} rows have MISSING images!")
        missing_by_source = merged[~merged["exists"]].groupby("source_dataset").size()
        for src, count in missing_by_source.items():
            print(f"     {src}: {count} missing")
        if args.skip_missing:
            print("   → Skipping missing rows (as requested by --skip-missing)")
            merged = merged[merged["exists"]].drop(columns=["exists"]).reset_index(drop=True)
        else:
            print("\n ERROR: Missing images found.")
            print("   Fix this by:")
            print("   1. Running: python scripts/download_images.py")
            print("   2. Or placing images in correct paths")
            print("   3. Or run with --skip-missing")
            raise ValueError(f"{missing_count} images are missing. Cannot proceed.")
    else:
        merged = merged.drop(columns=["exists"]).reset_index(drop=True)
        print(f"   All {len(merged)} image paths validated!")
    
    print("\nSplitting into train/val/test...")
    train, val, test = stratified_split(
        merged, cfg.data.val_ratio, cfg.data.test_ratio, cfg.data.random_state
    )
    
    train_path = os.path.join(cfg.data.processed_dir, "train_multimodal.csv")
    val_path = os.path.join(cfg.data.processed_dir, "val_multimodal.csv")
    test_path = os.path.join(cfg.data.processed_dir, "test_multimodal.csv")
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train)} samples)")
    print(f"  Val:   {val_path} ({len(val)} samples)")
    print(f"  Test:  {test_path} ({len(test)} samples)")
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()